"""Ticket 010 §2: Dev-A 손잡이 탐침 실행기(`evidencegate.t010.probe`).

가짜 전송(httpx.MockTransport)과 가짜 정책만 쓴다(Ollama·API 호출 없음). 009 Dev-A S0 기록은
임시 폴더에 가짜로 만들고, 탐침이 그 파일을 열기만 하는지(바이트·수정 시각 그대로) 확인한다.
"""

import ast
import hashlib
import io
import json
import os
import re
import shutil
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any

import httpx
import pytest

from evidencegate.bench_runner import BenchStopped, run_blocks
from evidencegate.harness import ACTION_TOOLS, OBSERVATION_TOOLS, HarnessText
from evidencegate.ids import content_id
from evidencegate.machine import Machine
from evidencegate.ollama_policy import OllamaPolicy, OllamaSettings
from evidencegate.policy import PolicyTrace, StopReason, TurnRecord, call_tool
from evidencegate.preflight import Check
from evidencegate.run_plan import Block
from evidencegate.runner import EpisodeKey, EpisodeRecord
from evidencegate.skills import Skill, load_skill
from evidencegate.store import EpisodeStore
from evidencegate.t009.splits import split_cases
from evidencegate.t010 import probe as pr
from opsworld.agent_env import TOOL_SPECS, AgentEnv

ROOT = Path(__file__).resolve().parents[2]
S0 = load_skill(ROOT / "skills" / "S0" / "SKILL.md")
DIGEST = "f38aa0c53da5f8c49d08c43a99df24ff53167fe68e24664a7777288e7656fdfe"
MACHINE = Machine("Mac15,3", "Apple M3")
DEV_A = split_cases("t009_dev_a")
# tests/test_preflight.py의 REFERENCE(S0 Dev-A 기록의 정책 설정)와 같은 값
REFERENCE: dict[str, Any] = {
    "type": "ollama",
    "model": "gpt-oss:20b",
    "model_digest": DIGEST,
    "quantization": "MXFP4",
    "ollama_version": "0.40.3",
    "instructions_sha256": "332b776ce67826fba8829790fc46090df243c2f9f38602e1c8c72faea4bfaed3",
    "options": {"temperature": 0.7, "num_ctx": 8192},
    "think": "low",
    "stop": {"max_tool_calls": 8, "max_consecutive_text_turns": 2},
    "format_error_policy": {"feedback": "error_as_tool_result", "max_consecutive_format_errors": 3},
    "seed_rule": "derive_llm_seed(family, incident_seed, repeat): sha256, first 8 bytes mod 2^31",
    "machine": {"hw_model": "Mac15,3", "chip": "Apple M3"},
    "parallel": {"client_concurrency": 2, "server_num_parallel": 2},
}
VARIANTS = {v.name: v for v in pr.VARIANTS}


# 가짜 Ollama ----------------------------------------------------------------------------


class Scripted:
    """상태 없이 답한다: 받은 도구 목록 순서로 관측 도구 `observe`개, 그다음 observe_only."""

    def __init__(self, observe: int = 3) -> None:
        self.observe = observe

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/version":
            return httpx.Response(200, json={"version": "0.40.3"})
        if path == "/api/tags":
            entry = {
                "name": "gpt-oss:20b",
                "model": "gpt-oss:20b",
                "digest": DIGEST,
                "details": {"quantization_level": "MXFP4"},
            }
            return httpx.Response(200, json={"models": [entry]})
        body = json.loads(request.content)
        done = sum(1 for m in body["messages"] if m["role"] == "assistant")
        listed = [t["function"]["name"] for t in body["tools"]]
        observation = [n for n in listed if n in OBSERVATION_TOOLS]
        name = observation[done] if done < self.observe else "observe_only"
        service = body["messages"][1]["content"].split("service ")[1].split(" ")[0]
        call = {"function": {"name": name, "arguments": {"service_id": service}}}
        message = {"role": "assistant", "content": "", "tool_calls": [call]}
        return httpx.Response(
            200,
            json={"model": "gpt-oss:20b", "message": message, "done": True,
                  "prompt_eval_count": 1, "eval_count": 1},
        )  # fmt: skip


def client(observe: int = 3) -> httpx.Client:
    return httpx.Client(
        transport=httpx.MockTransport(Scripted(observe).handler), base_url="http://o.test"
    )


def s0_policy(http: httpx.Client) -> OllamaPolicy:
    settings = OllamaSettings.for_model("gpt-oss:20b", concurrency=2, server_num_parallel=2)
    return OllamaPolicy.connect(settings, http, machine=MACHINE)


class Probe:
    parallel = True
    ratio = 1.3
    ratios = (1.3,)
    max_ratio = 1.68


def make_root(root: Path) -> Path:
    """S0 스킬만 있는 임시 저장소(결과 파일도 여기에 쓴다)."""
    shutil.copytree(ROOT / "skills" / "S0", root / "skills" / "S0")
    return root


def write_s0(runs: Path, policy: Any) -> Path:
    """009 Dev-A S0 기록 자리(`runs/t009_dev_a/…`)에 S0 60개를 만든다."""
    block = Block(0, "S0", "s0", S0.skill_id, "skills/S0/SKILL.md", "t009_dev_a", 0, tuple(DEV_A))
    run_blocks([block], policy, root=ROOT, runs_dir=runs, out=io.StringIO())
    path = runs / pr.S0_RECORDS
    assert path.exists() and len(path.read_text().splitlines()) == 60
    path.chmod(0o444)
    return path


def fingerprint(path: Path) -> tuple[str, int]:
    return hashlib.sha256(path.read_bytes()).hexdigest(), os.stat(path).st_mtime_ns


def main(root: Path, runs: Path, http: httpx.Client, *argv: str, **kw: Any) -> tuple[int, str]:
    out = io.StringIO()
    kw.setdefault("preflight", lambda policy: [Check("engine", True, "fake")])
    kw.setdefault("probe", lambda: Probe())
    kw.setdefault("power_check", lambda: None)
    kw.setdefault("ollama_version", lambda: "0.40.3")
    sha = hashlib.sha256((runs / pr.S0_RECORDS).read_bytes()).hexdigest()
    code = pr.main(
        [argv[0], "--runs-dir", str(runs), "--root", str(root), *argv[1:]],
        client=http,
        machine=MACHINE,
        s0_sha256=sha,
        out=out,
        err=out,
        **kw,
    )
    return code, out.getvalue()


# 고정 숫자와 판정 문장 -----------------------------------------------------------------------


def test_decision_numbers_are_the_approved_ones() -> None:
    assert pr.MOVED_MIN_EPISODES == 8  # k
    assert pr.MOVED_MIN_VARIANTS == 3  # m
    assert pr.MOVED_MIN_FIELDS == 2
    assert pr.P0_MIN_EPISODES == 50
    assert pr.EPISODES == 60
    assert pr.FORMAT_ANOMALY_BLOCKS == 3
    assert Fraction(1, 2) == pr.FORMAT_ANOMALY_RATE
    rule = pr.DECISION_RULE
    for phrase in ("8개 이상", "3개 이상", "두 종류 이상", "50개 미만", "판정 집계에 넣지 않는다"):
        assert phrase in rule


FIXED_NUMBERS = {
    "EPISODES": 60,
    "MOVED_MIN_EPISODES": 8,
    "MOVED_MIN_VARIANTS": 3,
    "MOVED_MIN_FIELDS": 2,
    "P0_MIN_EPISODES": 50,
    "FORMAT_ANOMALY_BLOCKS": 3,
    "S0_MAX_TOOL_CALLS": 8,
    "P0_MAX_TOOL_CALLS": 3,
}


def source_literals(path: Path) -> dict[str, Any]:
    """모듈 소스의 최상위 `이름 = 숫자 리터럴` 대입(바이트코드 캐시를 거치지 않는다)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out: dict[str, Any] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, int)
        ):
            out[node.targets[0].id] = node.value.value
    return out


def test_loaded_numbers_and_rule_match_the_source_and_the_approved_values() -> None:
    # 2026-11-10 탐침: 변이 검사 복구가 같은 초·같은 크기여서 .pyc(P0_MIN_EPISODES 49)가 유효로
    # 남았다. 불러온 값을 소스 리터럴·승인 값과 함께 비교해 낡은 바이트코드도 잡는다.
    source = source_literals(Path(pr.__file__))
    for name, approved in FIXED_NUMBERS.items():
        assert source[name] == approved, f"source {name}"
        assert getattr(pr, name) == approved, f"loaded {name} (stale bytecode?)"
    rule = pr.DECISION_RULE
    assert "50개 미만이면" in rule and "49개" not in rule
    assert re.findall(r"(\d+)개 미만", rule) == ["50"]
    assert re.findall(r"(\d+)개 이상", rule) == ["8", "3"]
    assert "max_tool_calls 3" in rule and "Dev-A 60개" in rule


def test_result_rule_carries_the_approved_numbers() -> None:
    rows = [record(c, "observe_only", [*S0_TOOLS, "observe_only"]) for c in DEV_A]
    configs = {v.name: pr.offline_policy(v, REFERENCE).config for v in pr.VARIANTS}
    s0 = [replace(r, policy_config=REFERENCE) for r in rows]
    records = {n: [replace(r, policy_config=c) for r in rows] for n, c in configs.items()}
    rule = pr.build_result(s0, records, "f" * 64)["rule"]
    assert (rule["k"], rule["m"], rule["min_fields"], rule["p0_min"], rule["episodes"]) == (
        8,
        3,
        2,
        50,
        60,
    )
    assert rule["text"] == pr.DECISION_RULE and "50개 미만" in rule["text"]


def test_s0_records_are_the_009_dev_a_file() -> None:
    assert (
        Path("t009_dev_a/ollama__gpt-oss__20b__7c66ee5bf13c/e9b4996f7529/bf65e569ae37.jsonl")
        == pr.S0_RECORDS
    )
    assert pr.S0_RECORDS_SHA256 == (
        "6d68923dd27ec65ffddb96be03f6a33b8331f569138f75b3c4565b586b3d1dd3"
    )
    assert pr.PROBE_SPLIT == "t010_probe"
    assert pr.SESSION_LOG == "ticket010/probe/sessions.jsonl"


# 변형 ----------------------------------------------------------------------------------------


def test_variants_are_p0_to_p6_with_the_approved_values() -> None:
    assert [v.name for v in pr.VARIANTS] == ["P0", "P1", "P2", "P3", "P4", "P5", "P6"]
    p = VARIANTS
    assert (p["P0"].field, p["P0"].judged, p["P0"].max_tool_calls, p["P0"].harness) == (
        "control",
        False,
        3,
        None,
    )
    assert all(v.judged and v.max_tool_calls == 8 for v in pr.VARIANTS[1:])
    assert p["P1"].harness == HarnessText(
        tool_order=(
            "get_traffic_metrics",
            "get_service_metrics",
            "get_recent_deployments",
            "get_dependency_health",
        )
    )
    assert p["P2"].harness == HarnessText(
        tool_order=(
            "get_service_metrics",
            "get_dependency_health",
            "get_recent_deployments",
            "get_traffic_metrics",
        )
    )
    assert p["P3"].harness == HarnessText(
        tool_descriptions=(
            (
                "get_traffic_metrics",
                "Get baseline and current request rate (requests per second) of the service. "
                "Useful to check whether demand changed.",
            ),
        )
    )
    assert p["P4"].harness == HarnessText(
        tool_descriptions=(
            (
                "get_dependency_health",
                "Get the health status of each dependency of the service and the minute any "
                "outage started (null if none). Useful to tell a problem in a dependency from a "
                "problem in the service itself.",
            ),
        )
    )
    assert p["P5"].harness == HarnessText(observation_renderer="key_value_lines")
    assert p["P6"].harness == HarnessText(observation_renderer="json_compact")
    fields = [v.field for v in pr.VARIANTS[1:]]
    assert fields == ["tool_order"] * 2 + ["tool_descriptions"] * 2 + ["observation_renderer"] * 2


def test_variant_configs_differ_from_s0_only_in_their_own_field() -> None:
    base = pr.offline_policy(None, REFERENCE).config
    assert base == REFERENCE and content_id(base) == "7c66ee5bf13c"
    ids = set()
    for v in pr.VARIANTS:
        config = pr.offline_policy(v, REFERENCE).config
        expected = {"stop.max_tool_calls"} if v.name == "P0" else {"harness"}
        assert pr.changed_paths(REFERENCE, config) == expected, v.name
        ids.add(content_id(config))
    assert len(ids) == 7 and "7c66ee5bf13c" not in ids


def test_description_variants_pass_the_section_1_checks() -> None:
    original = {s.name: s.description for s in TOOL_SPECS}
    for v in (VARIANTS["P3"], VARIANTS["P4"]):
        assert v.harness is not None
        ((tool, text),) = v.harness.tool_descriptions
        assert text.isascii()
        assert len(text) <= min(2 * len(original[tool]), 240)
        assert 1 <= len(re.findall(r"[.!?](?:\s|$)", text)) <= 2
        assert not re.search(r"\d", text)
        assert not any(name in text for name in original if name != tool)
        for word in (*ACTION_TOOLS, "roll back", "rollback", "scale", "capacity", "no action"):
            assert word not in text.lower()


def test_probe_blocks_are_one_dev_a_block_per_variant() -> None:
    blocks = pr.probe_blocks()
    assert [b.name for b in blocks] == [v.name for v in pr.VARIANTS]
    assert [b.index for b in blocks] == list(range(7))
    for b in blocks:
        assert (b.split, b.repeat, b.skill_id, b.skill_path) == (
            "t010_probe",
            0,
            S0.skill_id,
            "skills/S0/SKILL.md",
        )
        assert list(b.cases) == DEV_A
    assert [b.group for b in blocks] == ["control"] + ["probe"] * 6


# 판정 -----------------------------------------------------------------------------------------


def counts(p0: int, **others: int) -> dict[str, int]:
    return {"P0": p0, **{f"P{i}": 0 for i in range(1, 7)}, **others}


@pytest.mark.parametrize(
    ("different", "verdict", "moved"),
    [
        (counts(49, P1=60, P3=60, P5=60), "no_decision", []),
        (counts(50, P1=8, P3=8, P5=8), "proceed", ["P1", "P3", "P5"]),
        (counts(60, P1=8, P2=8, P3=7), "reconsider", ["P1", "P2"]),
        (counts(60, P1=8, P2=8, P3=8, P4=8, P5=8, P6=8), "proceed", [f"P{i}" for i in range(1, 7)]),
        (counts(60), "reconsider", []),
    ],
)
def test_judge_applies_the_fixed_rule(
    different: dict[str, int], verdict: str, moved: list[str]
) -> None:
    decision = pr.judge(different)
    assert decision.verdict == verdict
    assert list(decision.moved) == moved
    if verdict == "proceed":
        assert len(decision.fields) >= 2


def test_judge_needs_two_fields_even_with_three_moved() -> None:
    # 필드마다 변형이 둘이라 실제로는 3개가 움직이면 늘 두 종류 이상이다. 규칙 자체는 따로 확인한다.
    assert pr.enough_fields(["tool_order", "tool_order", "tool_order"]) is False
    assert pr.enough_fields(["tool_order", "observation_renderer"]) is True


def test_format_anomaly_needs_three_consecutive_blocks_over_one_half() -> None:
    half, over = Fraction(1, 2), Fraction(31, 60)
    assert not pr.format_anomaly([over, over])
    assert not pr.format_anomaly([over, over, half])
    assert not pr.format_anomaly([over, over, Fraction(0), over, over])
    assert pr.format_anomaly([Fraction(0), over, over, over])


# 짝 맞춤과 요약 -----------------------------------------------------------------------------


def record(case: tuple[str, str, int], action: str | None, tools: list[str], **kw: Any) -> Any:
    family, level, seed = case
    turns = tuple(TurnRecord(t, {}, "{}", t == "<unparsed>") for t in tools)
    trace = PolicyTrace(
        turns,
        kw.get("format_errors", sum(t == "<unparsed>" for t in tools)),
        kw.get("stop", StopReason.ACTION if action else StopReason.MAX_STEPS),
    )
    key = EpisodeKey("p", "c", "t", "s", family, seed, 0, level)
    return EpisodeRecord(
        key, {}, kw.get("llm_seed", seed), kw.get("cd", False), kw.get("es", False), action,
        (), (), trace, "2026-11-10T00:00:00+00:00", 1.0,
    )  # fmt: skip


S0_TOOLS = ["get_service_metrics", "get_recent_deployments", "get_dependency_health"]


def s0_records() -> dict[pr.PairKey, EpisodeRecord]:
    return {
        pr.pair_key(r): r
        for r in (record(c, "observe_only", [*S0_TOOLS, "observe_only"]) for c in DEV_A)
    }


def test_pairing_refuses_missing_extra_and_seed_mismatch() -> None:
    s0 = s0_records()
    variant = [record(c, "observe_only", [*S0_TOOLS, "observe_only"]) for c in DEV_A]
    assert len(pr.pair(s0, variant)) == 60
    with pytest.raises(ValueError, match="missing"):
        pr.pair(s0, variant[:-1])
    with pytest.raises(ValueError, match="llm_seed"):
        pr.pair(s0, [*variant[:-1], replace(variant[-1], llm_seed=1)])
    with pytest.raises(ValueError, match="not in"):
        pr.pair(s0, [*variant, record(("bad_deployment", "L1", 1), None, [])])


def test_summary_counts_dispositions_tool_sets_traffic_and_cells() -> None:
    s0 = s0_records()
    cell = [c for c in DEV_A if c[:2] == ("bad_deployment", "L1")]
    rows = []
    for c in DEV_A:
        if c == cell[0]:  # 처분 다름(조치 없음), 도구 집합 다름(조치 도구도 집합에 든다)
            rows.append(record(c, None, S0_TOOLS))
        elif c == cell[1]:  # 처분 다름, 트래픽 호출, 도구 집합 다름, CD 얻음
            rows.append(
                record(c, "rollback_deployment", ["get_traffic_metrics", "rollback_deployment"],
                       cd=True, es=True)
            )  # fmt: skip
        elif c == cell[2]:  # 처분 같음, 형식 오류 1회(형식 오류 턴은 도구 집합에 넣지 않음)
            rows.append(record(c, "observe_only", ["<unparsed>", *S0_TOOLS, "observe_only"]))
        else:
            rows.append(record(c, "observe_only", [*S0_TOOLS, "observe_only"]))
    summary = pr.summarize(VARIANTS["P1"], pr.pair(s0, rows))
    assert summary.episodes == 60
    assert summary.different_disposition == 2
    assert summary.different_tool_set == 2
    assert summary.traffic_episodes == 1
    assert summary.format_error_episodes == 1
    assert summary.stop_reasons == {"action": 59, "max_steps": 1}
    assert summary.cells["bad_deployment/L1"] == {"n": 10, "cd_diff": 1, "es_diff": 1}
    assert summary.cells["traffic_spike/L2"] == {"n": 10, "cd_diff": 0, "es_diff": 0}
    assert summary.mean_tool_calls == pytest.approx((57 * 4 + 3 + 2 + 5) / 60)


# 실행(가짜 전송, 실제 OllamaPolicy) ----------------------------------------------------------


@dataclass
class Done:
    root: Path
    runs: Path
    s0_path: Path
    before: tuple[str, int]
    code: int
    output: str


@pytest.fixture(scope="module")
def done(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Done]:
    root = make_root(tmp_path_factory.mktemp("root"))
    runs = root / "runs"
    s0_path = write_s0(runs, s0_policy(client()))
    before = fingerprint(s0_path)
    code, output = main(root, runs, client(), "run")
    yield Done(root, runs, s0_path, before, code, output)
    s0_path.chmod(0o644)


def stores(runs: Path) -> list[Path]:
    return sorted((runs / "t010_probe").glob("*/*/*.jsonl"))


def test_run_writes_one_t010_probe_store_per_variant(done: Done) -> None:
    assert done.code == 0, done.output
    files = stores(done.runs)
    assert len(files) == 7
    configs = {}
    for path in files:
        records = EpisodeStore(path, repair=False).records()
        assert len(records) == 60
        assert path.parent.name == "e9b4996f7529" and path.name == f"{S0.skill_id}.jsonl"
        configs[records[0].key.config_id] = records[0].policy_config
    renderers = {c.get("harness", {}).get("observation_renderer") for c in configs.values()}
    assert {"key_value_lines", "json_compact"} <= renderers
    assert sorted(c["stop"]["max_tool_calls"] for c in configs.values()) == [3] + [8] * 6


def test_run_only_reads_the_009_dev_a_s0_records(done: Done) -> None:
    assert fingerprint(done.s0_path) == done.before
    files = sorted(p for p in (done.runs / "t009_dev_a").rglob("*") if p.is_file())
    assert files == [done.s0_path]


def test_run_logs_to_the_probe_session_folder_without_cd_or_es(done: Done) -> None:
    log = done.runs / "ticket010" / "probe" / "sessions.jsonl"
    events = [json.loads(line) for line in log.read_text().splitlines()]
    start = next(e for e in events if e["event"] == "probe_start")
    assert set(start["variants"]) == {v.name for v in pr.VARIANTS}
    assert start["s0_records_sha256"] == done.before[0]
    assert start["ollama_version"] == "0.40.3"
    assert events[-1]["event"] == "probe_end" and events[-1]["reason"] == "plan_done"
    assert not re.search(r"correct|evidence|\bCD\b|\bES\b", done.output)


def test_rerun_skips_finished_variants(done: Done) -> None:
    code, output = main(done.root, done.runs, client(), "run")
    assert code == 0, output
    assert "to run" not in output
    assert all(len(p.read_text().splitlines()) == 60 for p in stores(done.runs))


def test_status_shows_progress_per_variant_without_cd_or_es(done: Done) -> None:
    code, output = main(done.root, done.runs, client(), "status")
    assert code == 0
    for v in pr.VARIANTS:
        assert re.search(rf"^{v.name} .* 60/60", output, re.M), output
    assert not re.search(r"correct|evidence|\bCD\b|\bES\b", output)


def test_report_writes_the_table_and_the_decision(done: Done) -> None:
    code, output = main(done.root, done.runs, client(), "report")
    assert code == 0, output
    result = json.loads((done.root / pr.RESULT_PATH).read_text())
    rows = {r["name"]: r for r in result["variants"]}
    # 가짜 모델은 처분을 늘 observe_only로 한다. P0만 조치 전에 멈춘다.
    assert rows["P0"]["different_disposition"] == 60
    assert all(rows[f"P{i}"]["different_disposition"] == 0 for i in range(1, 7))
    # P1은 트래픽을 맨 앞에 두어 가짜 모델이 트래픽·지표·배포를 부른다.
    assert rows["P1"]["traffic_episodes"] == 60 and rows["P1"]["different_tool_set"] == 60
    assert rows["P2"]["different_tool_set"] == 0  # 지표·의존성·배포: 집합은 S0와 같다
    assert result["s0"]["traffic_episodes"] == 0
    assert result["decision"]["verdict"] == "reconsider"
    assert result["rule"]["text"] == pr.DECISION_RULE
    assert result["rule"]["k"] == 8 and result["rule"]["m"] == 3
    assert result["rule"]["p0_min"] == 50
    assert result["s0_records_sha256"] == done.before[0]
    assert "reconsider" in output and "P0" in output
    assert fingerprint(done.s0_path) == done.before
    # 같은 기록이면 같은 바이트다. 다르면 덮어쓰지 않는다.
    assert main(done.root, done.runs, client(), "report")[0] == 0
    path = done.root / pr.RESULT_PATH
    original = path.read_bytes()
    path.write_text("{}\n")
    assert main(done.root, done.runs, client(), "report")[0] == 1
    path.write_bytes(original)


def test_report_refuses_s0_records_with_another_sha256(done: Done) -> None:
    out = io.StringIO()
    code = pr.main(
        ["report", "--runs-dir", str(done.runs), "--root", str(done.root)],
        s0_sha256="0" * 64,
        out=out,
        err=out,
    )
    assert code == 1 and "sha256" in out.getvalue()


def test_run_stops_after_p0_when_p0_does_not_move(tmp_path: Path) -> None:
    make_root(tmp_path)
    runs = tmp_path / "runs"
    # 관측 2회 뒤 조치하는 가짜 모델: max_tool_calls 3이 걸리지 않아 P0가 S0와 같다.
    write_s0(runs, s0_policy(client(observe=2)))
    code, output = main(tmp_path, runs, client(observe=2), "run")
    assert code == 2, output
    assert "P0" in output and "no probe decision" in output
    assert len(stores(runs)) == 1
    log = [json.loads(x) for x in (runs / pr.SESSION_LOG).read_text().splitlines()]
    assert "P0" in log[-1]["reason"]
    code, output = main(tmp_path, runs, client(observe=2), "report")
    assert code == 1 and not (tmp_path / pr.RESULT_PATH).exists()
    (runs / pr.S0_RECORDS).chmod(0o644)


def test_report_without_finished_variants_gives_no_decision(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    write_s0(runs, s0_policy(client()))
    code, output = main(tmp_path, runs, client(), "report")
    assert code == 1 and "incomplete" in output
    assert not (tmp_path / pr.RESULT_PATH).exists()
    (runs / pr.S0_RECORDS).chmod(0o644)


# 실행 규칙(가짜 정책) -------------------------------------------------------------------------


class Fake:
    policy_id = "ollama:gpt-oss:20b"

    def __init__(self, name: str, behave: Callable[[AgentEnv], PolicyTrace]) -> None:
        self.config: dict[str, Any] = {"fake": name}
        self.behave = behave

    def run(self, env: AgentEnv, skill: Skill, alert: str, llm_seed: int) -> PolicyTrace:
        return self.behave(env)


def act(env: AgentEnv) -> PolicyTrace:
    _, turn = call_tool(env, "observe_only", {"service_id": env.service_id})
    return PolicyTrace((turn,), 0, StopReason.ACTION)


def stall(env: AgentEnv) -> PolicyTrace:
    _, turn = call_tool(env, "get_service_metrics", {"service_id": env.service_id})
    return PolicyTrace((turn,), 0, StopReason.MAX_STEPS)


def garbled(env: AgentEnv) -> PolicyTrace:
    bad = TurnRecord("<unparsed>", {}, "{}", True)
    _, turn = call_tool(env, "observe_only", {"service_id": env.service_id})
    return PolicyTrace((bad, turn), 1, StopReason.ACTION)


def fakes(**behaviour: Callable[[AgentEnv], PolicyTrace]) -> dict[str, Any]:
    return {v.name: Fake(v.name, behaviour.get(v.name, act)) for v in pr.VARIANTS} | {
        "P0": Fake("P0", behaviour.get("P0", stall))
    }


def fake_s0(runs: Path) -> dict[pr.PairKey, EpisodeRecord]:
    block = Block(0, "S0", "s0", S0.skill_id, "skills/S0/SKILL.md", "t009_dev_a", 0, tuple(DEV_A))
    run_blocks([block], Fake("S0", act), root=ROOT, runs_dir=runs / "s0", out=io.StringIO())
    path = next((runs / "s0" / "t009_dev_a").glob("*/*/*.jsonl"))
    return {pr.pair_key(r): r for r in EpisodeStore(path, repair=False).records()}


def run_fakes(runs: Path, policies: Mapping[str, Any], **kw: Any) -> pr.ProbeRun:
    return pr.run_probe(
        policies, root=ROOT, runs_dir=runs, out=io.StringIO(), s0=fake_s0(runs), **kw
    )


def fake_counts(runs: Path) -> dict[str, int]:
    return {
        p.parent.parent.name: len(p.read_text().splitlines())
        for p in sorted((runs / "t010_probe").glob("*/*/*.jsonl"))
    }


def test_candidate_blocks_do_not_stop_on_format_errors(tmp_path: Path) -> None:
    # 모든 에피소드에 형식 오류가 있는 블록 둘(v1 규칙이면 20개째에 멈춤)
    result = run_fakes(tmp_path, fakes(P1=garbled, P2=garbled))
    assert result.blocks_run == 7 and result.stopped_by is None
    assert sorted(fake_counts(tmp_path).values()) == [60] * 7


def test_three_consecutive_format_error_blocks_stop_the_probe(tmp_path: Path) -> None:
    with pytest.raises(BenchStopped, match="format error"):
        run_fakes(tmp_path, fakes(P1=garbled, P2=garbled, P3=garbled))
    assert len(fake_counts(tmp_path)) == 4  # P0~P3


def test_stop_file_and_max_blocks_stop_at_a_variant_boundary(tmp_path: Path) -> None:
    result = run_fakes(tmp_path / "a", fakes(), max_blocks=2)
    assert result.blocks_run == 2 and result.stopped_by == "max_blocks"
    assert len(fake_counts(tmp_path / "a")) == 2
    stop = tmp_path / "STOP"
    stop.write_text("")
    result = run_fakes(tmp_path / "b", fakes(), stop_file=stop)
    assert result.blocks_run == 0 and result.stopped_by == "stop_file"


# 실행 전 점검 --------------------------------------------------------------------------------


@dataclass(frozen=True)
class Usage:
    free: int


def preflight(runs: Path, sha: str, config: Mapping[str, Any] = REFERENCE) -> dict[str, Check]:
    checks = pr.probe_preflight(
        root=runs.parent,
        runs_dir=runs,
        base_config=config,
        machine=MACHINE,
        s0_sha256=sha,
        power_run=lambda argv: "Now drawing from 'AC Power'",
        disk_usage=lambda path: Usage(10**12),
    )
    return {c.name: c for c in checks}


def test_preflight_checks_the_engine_s0_records_variants_and_paths(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    path = write_s0(runs, s0_policy(client()))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    checks = preflight(runs, sha)
    assert list(checks) == [
        "engine",
        "config_id",
        "machine",
        "parallel",
        "ollama_version",
        "power",
        "disk",
        "s0_dev_a",
        "variants",
        "probe_paths",
    ]
    for name in ("config_id", "machine", "parallel", "ollama_version", "s0_dev_a", "variants"):
        assert checks[name].ok, checks[name]
    assert checks["probe_paths"].ok
    assert not checks["engine"].ok  # 임시 폴더에는 v1 S0 Dev-A 기록(runs/dev_a)이 없다
    assert not preflight(runs, "0" * 64)["s0_dev_a"].ok
    other = REFERENCE | {"options": {"temperature": 0.5, "num_ctx": 8192}}
    assert not preflight(runs, sha, other)["s0_dev_a"].ok  # S0 기록의 설정과 다름
    path.chmod(0o644)


def test_preflight_refuses_s0_records_that_are_not_the_dev_a_split(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    path = runs / pr.S0_RECORDS
    write_s0(runs, s0_policy(client())).chmod(0o644)
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:-1]) + "\n")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    check = preflight(runs, sha)["s0_dev_a"]
    assert not check.ok and "59" in check.detail
