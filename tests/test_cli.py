"""cli: run(가짜 Ollama), summarize. 실제 Ollama를 부르지 않는다."""

import io
import json
import re
from pathlib import Path
from typing import Any

import pytest

from evidencegate.cli import main
from evidencegate.store import EpisodeStore
from fake_ollama import FakeOllama, reply_tools, tool_call

REPO_ROOT = Path(__file__).resolve().parents[1]
S0_PATH = REPO_ROOT / "skills" / "S0" / "SKILL.md"


def two_step_agent(body: dict[str, Any]) -> dict[str, Any]:
    """서비스 지표를 본 뒤 observe_only. 알림에서 서비스 ID를 읽는다."""
    alert = body["messages"][1]["content"]
    match = re.search(r"service (\S+) is degraded", alert)
    assert match is not None
    args = {"service_id": match.group(1)}
    if not any(m["role"] == "tool" for m in body["messages"]):
        return reply_tools(tool_call("get_service_metrics", args), cached=0)
    return reply_tools(tool_call("observe_only", args), cached=100)


def run_cli(argv: list[str], fake: FakeOllama | None = None) -> tuple[int, str]:
    out = io.StringIO()
    client = fake.client() if fake is not None else None
    code = main(argv, http_client=client, out=out)
    return code, out.getvalue()


def run_args(runs_dir: Path, n_per_family: int = 2, repeats: int = 1) -> list[str]:
    return [
        "run",
        "--policy",
        "ollama:gemma4:12b",
        "--skill",
        str(S0_PATH),
        "--suite",
        "calibration",
        "--n-per-family",
        str(n_per_family),
        "--repeats",
        str(repeats),
        "--runs-dir",
        str(runs_dir),
        "--machine",
        "test-mac",
        "--chip",
        "Apple M3",
    ]


def only_store_file(runs_dir: Path) -> Path:
    files = list(runs_dir.rglob("*.jsonl"))
    assert len(files) == 1
    return files[0]


def test_run_writes_store_at_keyed_path(tmp_path: Path) -> None:
    code, _ = run_cli(run_args(tmp_path), FakeOllama(two_step_agent))
    assert code == 0
    path = only_store_file(tmp_path)
    relative = path.relative_to(tmp_path).parts
    assert relative[0] == "calibration"
    assert re.fullmatch(r"ollama__gemma4__12b__[0-9a-f]{12}", relative[1])
    assert re.fullmatch(r"[0-9a-f]{12}", relative[2])
    assert re.fullmatch(r"[0-9a-f]{12}\.jsonl", relative[3])
    records = EpisodeStore(path).records()
    assert len(records) == 6
    assert {r.action for r in records} == {"observe_only"}
    assert records[0].trace.llm_config is not None
    assert records[0].trace.llm_config["machine"] == "test-mac"


def test_run_prints_progress_average_and_eta(tmp_path: Path) -> None:
    _, output = run_cli(run_args(tmp_path), FakeOllama(two_step_agent))
    assert "to run: 6, already stored: 0" in output
    for i in range(1, 7):
        assert f"[{i}/6]" in output
    assert "s/ep" in output and "ETA" in output
    assert "ran 6, skipped 0" in output


def test_run_again_skips_everything(tmp_path: Path) -> None:
    run_cli(run_args(tmp_path), FakeOllama(two_step_agent))
    fake = FakeOllama(two_step_agent)
    _, output = run_cli(run_args(tmp_path), fake)
    assert "to run: 0, already stored: 6" in output
    assert "ran 0, skipped 6" in output
    assert fake.chat_requests == []


def test_run_prints_summary_at_the_end(tmp_path: Path) -> None:
    _, output = run_cli(run_args(tmp_path), FakeOllama(two_step_agent))
    assert "correct_disposition" in output and "evidence_safe" in output
    assert "overall" in output and "dependency_failure" in output


def test_summarize_reports_rates_ci_cache_and_config(tmp_path: Path) -> None:
    run_cli(run_args(tmp_path), FakeOllama(two_step_agent))
    code, output = run_cli(["summarize", str(only_store_file(tmp_path))])
    assert code == 0
    # observe_only는 dependency_failure에서만 처분이 맞다: 2/6
    assert re.search(r"overall\s+6\s+0\.333 \[0\.\d{3}, 0\.\d{3}\]", output)
    assert re.search(r"dependency_failure\s+2\s+1\.000", output)
    # 에피소드마다 호출 2번: 첫 호출 cached 0, 둘째 호출 cached 100, prompt_eval은 각각 120
    assert "cache: cached=600 prompt_eval=1440 (41.7%)" in output
    assert "later calls with cached>0: 6/6" in output
    assert "gemma4:12b" in output and "Q4_K_M" in output and "0.33.3" in output
    assert "test-mac (Apple M3)" in output


def test_summarize_with_repeats_shows_each_repeat_and_noise_band(tmp_path: Path) -> None:
    run_cli(run_args(tmp_path, n_per_family=1, repeats=2), FakeOllama(two_step_agent))
    _, output = run_cli(["summarize", str(tmp_path)])
    assert "repeat 0" in output and "repeat 1" in output
    assert "noise band" in output
    assert "paired_abs_p95" in output and "case_mismatch_rate" in output


def test_summarize_single_repeat_has_no_noise_band(tmp_path: Path) -> None:
    run_cli(run_args(tmp_path), FakeOllama(two_step_agent))
    _, output = run_cli(["summarize", str(tmp_path)])
    assert "noise band" not in output


def test_unsupported_policy_is_rejected(tmp_path: Path) -> None:
    args = run_args(tmp_path)
    args[args.index("--policy") + 1] = "scripted:A"
    with pytest.raises(SystemExit) as excinfo:
        run_cli(args, FakeOllama(two_step_agent))
    assert excinfo.value.code == 2


def test_summarize_output_is_json_with_flag(tmp_path: Path) -> None:
    run_cli(run_args(tmp_path), FakeOllama(two_step_agent))
    _, output = run_cli(["summarize", "--json", str(only_store_file(tmp_path))])
    data = json.loads(output)
    assert data[0]["summary"]["overall"]["n"] == 6


def test_ollama_errors_exit_1_with_a_short_message(tmp_path: Path) -> None:
    fake = FakeOllama(two_step_agent, models=[])
    out = io.StringIO()
    err = io.StringIO()
    code = main(run_args(tmp_path), http_client=fake.client(), out=out, err=err)
    assert code == 1
    assert "error: model 'gemma4:12b' not found" in err.getvalue()
    assert "ollama pull gemma4:12b" in err.getvalue()
