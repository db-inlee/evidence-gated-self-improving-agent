"""Ticket 009 세션 시작 점검(티켓 원칙 2·9, §4.1)."""

import shutil
from pathlib import Path
from typing import Any

from evidencegate.machine import Machine
from evidencegate.preflight import reference_policy_config
from evidencegate.run_plan import plan_bytes
from evidencegate.t009.preflight import EXPECTED_OLLAMA_VERSION, run_preflight_009
from evidencegate.t009.run_plan import build_calibration_plan

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
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
    config = kw.pop("config", None) or reference_policy_config(RUNS)
    result = run_preflight_009(
        root=root,
        code_root=ROOT,
        runs_dir=RUNS,
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
    config = dict(reference_policy_config(RUNS))
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
