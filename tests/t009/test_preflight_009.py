"""Ticket 009 세션 시작 점검(티켓 원칙 2·9, §4.1).

engine 점검의 기준(v1 S0 Dev-A 기록의 정책 설정)은 실제 `runs/`가 아니라 임시 폴더의 가짜 기준 기록
한 줄에서 읽는다(Ticket 011 §5: 기록 없는 clone·CI에서도 점검 논리를 시험한다). 가짜 기록의 설정은
v1 기록과 같은 값(`REFERENCE`, content_id 7c66ee5bf13c)이다.
"""

import shutil
from pathlib import Path
from typing import Any

from evidencegate.eval_constants import CONFIG_ID, POLICY_ID, S0_SKILL_ID, TOOL_SPEC_ID
from evidencegate.ids import content_id
from evidencegate.machine import Machine
from evidencegate.policy import PolicyTrace, StopReason
from evidencegate.preflight import reference_policy_config
from evidencegate.run_plan import plan_bytes
from evidencegate.runner import EpisodeKey, EpisodeRecord
from evidencegate.store import EpisodeStore
from evidencegate.t009.preflight import EXPECTED_OLLAMA_VERSION, run_preflight_009
from evidencegate.t009.run_plan import build_calibration_plan

ROOT = Path(__file__).resolve().parents[2]
# tests/test_preflight.py의 REFERENCE(v1 S0 Dev-A 기록의 정책 설정)와 같은 값
REFERENCE: dict[str, Any] = {
    "type": "ollama",
    "model": "gpt-oss:20b",
    "model_digest": "f38aa0c53da5f8c49d08c43a99df24ff53167fe68e24664a7777288e7656fdfe",
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
GOOD_POWER = {
    "pmset -g batt": "Now drawing from 'AC Power'\n",
    "ioreg": '"AppleClamshellState" = No\n',
    "pmset -g assertions": (
        'pid 9(caffeinate): [0x1] 00:01:00 PreventUserIdleSystemSleep named: "c"\n'
    ),
}


def runner(outputs: dict[str, str]) -> Any:
    def run(argv: Any) -> str:
        cmd = " ".join(argv)
        return next(v for k, v in outputs.items() if cmd.startswith(k))

    return run


def fake_runs(base: Path) -> Path:
    """v1 S0 Dev-A 기록 자리(`runs/dev_a/…`)에 REFERENCE 설정의 기록 한 줄."""
    runs = base / "runs"
    store = EpisodeStore.for_run(
        runs,
        suite="dev_a",
        policy_id=POLICY_ID,
        config_id=CONFIG_ID,
        tool_spec_id=TOOL_SPEC_ID,
        skill_id=S0_SKILL_ID,
    )
    key = EpisodeKey(POLICY_ID, CONFIG_ID, TOOL_SPEC_ID, S0_SKILL_ID, "bad_deployment", 1, 0)
    trace = PolicyTrace((), 0, StopReason.ACTION)
    record = EpisodeRecord(
        key, REFERENCE, 0, True, True, "observe_only", (), (),
        trace, "2026-10-15T00:00:00+00:00", 1.0,
    )  # fmt: skip
    store.append(record)
    return runs


def test_the_fake_reference_is_the_v1_engine(tmp_path: Path) -> None:
    assert content_id(REFERENCE) == CONFIG_ID == "7c66ee5bf13c"
    assert reference_policy_config(fake_runs(tmp_path)) == REFERENCE


def setup(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "repo"
    shutil.copytree(ROOT / "preregistration", root / "preregistration")
    plan = root / "plans" / "ticket-009-calibration-plan.json"
    plan.parent.mkdir(parents=True)
    data = plan_bytes(build_calibration_plan(ROOT))
    plan.write_bytes(data)
    import hashlib

    (plan.parent / (plan.name + ".sha256")).write_text(
        f"{hashlib.sha256(data).hexdigest()}  {plan.name}\n"
    )
    return root, plan


def checks(tmp_path: Path, **kw: Any) -> dict[str, Any]:
    root, plan = setup(tmp_path)
    runs = fake_runs(tmp_path)
    config = kw.pop("config", None) or reference_policy_config(runs)
    result = run_preflight_009(
        root=root,
        code_root=ROOT,
        runs_dir=runs,
        current_config=config,
        machine=kw.pop("machine", Machine(hw_model="Mac15,3", chip="Apple M3")),
        phase="calibration",
        plan_path=kw.pop("plan_path", plan),
        power_run=runner(kw.pop("power", GOOD_POWER)),
        disk_usage=lambda p: type("U", (), {"free": 10**12})(),
    )
    return {c.name: c for c in result}


def test_all_checks_pass_with_the_v1_engine() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        result = checks(Path(d))
    assert set(result) == {
        "engine",
        "config_id",
        "machine",
        "parallel",
        "preregistration_009",
        "plan_009",
        "ollama_version",
        "power",
        "disk",
    }
    assert all(c.ok for c in result.values()), {n: c.detail for n, c in result.items()}


def test_a_different_ollama_version_fails(tmp_path: Path) -> None:
    config = dict(REFERENCE)
    config["ollama_version"] = "0.41.0"
    result = checks(tmp_path, config=config)
    assert not result["ollama_version"].ok and not result["engine"].ok
    assert EXPECTED_OLLAMA_VERSION == "0.40.3"


def test_power_failure_fails(tmp_path: Path) -> None:
    power = {**GOOD_POWER, "pmset -g batt": "Now drawing from 'Battery Power'\n"}
    assert not checks(tmp_path, power=power)["power"].ok


def test_plan_changes_and_missing_plans_fail(tmp_path: Path) -> None:
    _root, plan = setup(tmp_path / "a")
    plan.write_bytes(plan.read_bytes().replace(b'"repeat":0', b'"repeat":9', 1))
    assert not checks(tmp_path / "b", plan_path=plan)["plan_009"].ok
    assert not checks(tmp_path / "c", plan_path=tmp_path / "missing.json")["plan_009"].ok


def test_wrong_machine_fails(tmp_path: Path) -> None:
    assert not checks(tmp_path, machine=Machine(hw_model="Mac14,2", chip="x"))["machine"].ok


def test_a_plan_that_matches_its_own_sha256_but_not_a_rebuild_fails(tmp_path: Path) -> None:
    import hashlib

    _root, plan = setup(tmp_path / "a")
    changed = plan.read_bytes().replace(b'"repeat":0', b'"repeat":9', 1)
    plan.write_bytes(changed)
    (plan.parent / (plan.name + ".sha256")).write_text(
        f"{hashlib.sha256(changed).hexdigest()}  {plan.name}\n"
    )
    check = checks(tmp_path / "b", plan_path=plan)["plan_009"]
    assert not check.ok and "differs from a rebuild" in check.detail
