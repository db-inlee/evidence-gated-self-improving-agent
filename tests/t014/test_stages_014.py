"""Ticket 014 단계 결과(사전 등록 8절): 한 번 쓰기, 해시 사슬, 단계 1~5 마무리. LLM 없음."""

import json
import random
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path
from typing import Any

import pytest

from evidencegate.eval_constants import CONFIG_ID, POLICY_ID, TOOL_SPEC_ID
from evidencegate.eval_data import CaseId
from evidencegate.policy import PolicyTrace, StopReason
from evidencegate.run_plan import Block
from evidencegate.runner import EpisodeKey, EpisodeRecord
from evidencegate.store import EpisodeStore, record_to_json
from evidencegate.t014 import plans, stages

ROOT = Path(__file__).resolve().parents[2]
X = ("bad_deployment", "L1")


def record(skill: str, case: CaseId, repeat: int, ok: bool) -> EpisodeRecord:
    return EpisodeRecord(
        key=EpisodeKey(
            POLICY_ID, CONFIG_ID, TOOL_SPEC_ID, skill, case[0], case[2], repeat, case[1]
        ),
        policy_config={},
        llm_seed=0,
        correct_disposition=ok,
        evidence_safe=ok,
        action="observe_only",
        missing_predicates=(),
        failure_labels=(),
        trace=PolicyTrace((), 0, StopReason.ACTION),
        started_at="2026-11-27T00:00:00+00:00",
        duration_s=1.0,
    )


def fill(runs: Path, plan: tuple[Block, ...], outcome: Callable[[Block, CaseId], bool]) -> None:
    by_store: dict[Path, list[EpisodeRecord]] = {}
    for b in plan:
        path = stages.store_path(runs, b.split, b.skill_id)
        by_store.setdefault(path, []).extend(
            record(b.skill_id, c, b.repeat, outcome(b, c)) for c in b.cases
        )
    for path, records in by_store.items():  # fsync 없이 한 번에(EpisodeStore와 같은 줄 형식)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            json.dumps(record_to_json(r), ensure_ascii=False, separators=(",", ":"))
            for r in records
        ]
        with path.open("a", encoding="utf-8") as f:
            f.write("".join(x + "\n" for x in lines))
    assert all(EpisodeStore(p, repair=False).records() for p in by_store)


def setup_root(tmp: Path) -> Path:
    from evidencegate.skills import load_skill

    root = tmp / "repo"
    (root / "skills/S0").mkdir(parents=True)
    (root / "skills/S0/SKILL.md").write_bytes((ROOT / "skills/S0/SKILL.md").read_bytes())
    entries = []
    for i, name in enumerate(("cand-good", "cand-bad", "cand-none")):
        folder = root / "skills/candidates-014" / name
        folder.mkdir(parents=True)
        (folder / "SKILL.md").write_text(f"# {name}\n")
        entries.append(
            {
                "name": name,
                "group": "proposer",
                "skill_id": load_skill(folder / "SKILL.md").skill_id,
                "skill_path": f"skills/candidates-014/{name}/SKILL.md",
                "call_index": i,
            }
        )
    pool = root / "skills/candidates-014/POOL-014.json"
    pool.write_text(json.dumps({"candidates": entries}))
    return root


def s0_ok(case: CaseId, repeat: int = 0) -> bool:
    return (case[0], case[1]) == X or case[2] % 2 == 0


def test_write_result_is_write_once_and_chains_to_the_previous_stage(tmp_path: Path) -> None:
    sha0 = stages.write_result(tmp_path, 0, {"go": True})
    path0 = tmp_path / "results/ticket-014/stage0-pool.json"
    assert path0.with_name(path0.name + ".sha256").read_text().split()[0] == sha0
    with pytest.raises(FileExistsError, match="written once"):
        stages.write_result(tmp_path, 0, {"go": True})
    assert stages.check_previous(tmp_path, 1).ok
    stages.write_result(tmp_path, 1, {"x": Fraction(1, 3)})
    data = json.loads((tmp_path / "results/ticket-014/stage1-thresholds.json").read_text())
    assert data["previous"] == {"path": "results/ticket-014/stage0-pool.json", "sha256": sha0}
    assert data["x"] == {"exact": "1/3", "float": 1 / 3}
    path0.write_text(path0.read_text() + " ")
    check = stages.check_previous(tmp_path, 1)
    assert not check.ok and "sha256" in check.detail
    assert not stages.check_previous(tmp_path, 3).ok  # 단계 2 결과 없음
    assert stages.check_previous(tmp_path, 0).ok


def test_stage0_no_go_closes_the_chain(tmp_path: Path) -> None:
    stages.write_result(tmp_path, 0, {"go": False})
    check = stages.check_previous(tmp_path, 1)
    assert not check.ok and "No-Go" in check.detail


def test_store_path_is_the_episode_store_path(tmp_path: Path) -> None:
    expected = EpisodeStore.for_run(
        tmp_path,
        suite="t014_dev_b",
        policy_id=POLICY_ID,
        config_id=CONFIG_ID,
        tool_spec_id=TOOL_SPEC_ID,
        skill_id="abc",
    ).path
    assert stages.store_path(tmp_path, "t014_dev_b", "abc") == expected


def test_collect_records_refuses_an_incomplete_plan(tmp_path: Path) -> None:
    root = setup_root(tmp_path)
    plan = plans.build_plan(0, root)
    fill(tmp_path / "runs", plan[:5], lambda b, c: True)
    with pytest.raises(ValueError, match="incomplete"):
        stages.collect_records(tmp_path / "runs", plan)


def test_stages_1_to_5_end_to_end(tmp_path: Path) -> None:
    root, runs = setup_root(tmp_path), tmp_path / "runs"
    stages.write_result(root, 0, {"go": True})
    rng = random.Random(0)

    p1 = plans.build_plan(1, root)
    plans.write_plan(root, 1, p1)
    fill(runs, p1, lambda b, c: rng.random() < 0.5)
    r1 = stages.finalize(1, root=root, runs_dir=runs, n_boot=30)
    assert set(r1["thresholds"]) == {"correct_disposition", "evidence_safe"}
    assert {"h", "achieved", "delta"} == set(r1["thresholds"]["correct_disposition"])

    p2 = plans.build_plan(2, root)
    plans.write_plan(root, 2, p2)

    def dev_b(b: Block, c: CaseId) -> bool:
        if b.name == "cand-good":
            return True
        if b.name == "cand-bad":
            return (c[0], c[1]) != X
        return s0_ok(c)

    fill(runs, p2, dev_b)
    r2 = stages.finalize(2, root=root, runs_dir=runs, n_boot=30)
    assert r2["episodes"] == 4 * 900 and len(r2["stores"]) == 4
    assert "correct_disposition" not in json.dumps(r2)  # 단계 2 요약에 결과 값 없음

    with pytest.raises(ValueError, match="CONTROLS_IN_ORACLE"):
        stages.finalize(3, root=root, runs_dir=runs, n_boot=30, controls=None)
    r3 = stages.finalize(3, root=root, runs_dir=runs, n_boot=30, controls=False)
    assert r3["oracle_targets"] == {"proposer": ["cand-bad", "cand-good"], "controls": False}
    gates = {g["name"]: g for g in r3["gates"]["correct_disposition"]}
    assert gates["cand-good"]["c6"] and gates["cand-bad"]["c4"] and not gates["cand-bad"]["c6"]
    assert not gates["cand-none"]["c4"] and not gates["cand-none"]["sage"]

    p4 = plans.build_plan(4, root)
    plans.write_plan(root, 4, p4)

    def oracle(b: Block, c: CaseId) -> bool:
        return dev_b(b, c)

    fill(runs, p4, oracle)
    r4 = stages.finalize(4, root=root, runs_dir=runs, n_boot=30)
    assert r4["episodes"] == 3 * 600 * 3

    r5 = stages.finalize(5, root=root, runs_dir=runs, n_boot=30)
    labels = r5["metrics"]["correct_disposition"]["labels"]
    assert labels["cand-bad"]["m"] and not labels["cand-good"]["m"]
    assert labels["cand-good"]["clean_useful"]
    rq1 = r5["rq14_1"]
    assert (rq1["n"], rq1["f"], rq1["k"], rq1["a"]) == (2, 1, 1, 1)
    assert rq1["verdict"] == "direction_only"  # p = 1/2
    assert r5["rq14_2"]["u_4"] == 1 and r5["rq14_2"]["u"] == 0
    assert r5["rq14_3"]["C4"]["accepted"] == 2
    assert "no_oracle_label" in r5["metrics"]["correct_disposition"]
    assert r5["metrics"]["correct_disposition"]["no_oracle_label"] == ["cand-none"]
    with pytest.raises(FileExistsError):
        stages.finalize(5, root=root, runs_dir=runs, n_boot=30)


def test_a_changed_store_after_stage_2_is_refused_at_stage_3(tmp_path: Path) -> None:
    root, runs = setup_root(tmp_path), tmp_path / "runs"
    stages.write_result(root, 0, {"go": True})
    p1 = plans.build_plan(1, root)
    plans.write_plan(root, 1, p1)
    fill(runs, p1, lambda b, c: c[2] % 3 == 0)
    stages.finalize(1, root=root, runs_dir=runs, n_boot=10)
    p2 = plans.build_plan(2, root)
    plans.write_plan(root, 2, p2)
    fill(runs, p2, lambda b, c: s0_ok(c))
    stages.finalize(2, root=root, runs_dir=runs, n_boot=10)
    store = stages.store_path(runs, "t014_dev_b", p2[0].skill_id)
    store.write_text(store.read_text() + "\n")
    with pytest.raises(ValueError, match="changed since stage 2"):
        stages.finalize(3, root=root, runs_dir=runs, n_boot=10, controls=False)


def test_finalize_includes_the_stage_fingerprint(tmp_path: Path) -> None:
    root, runs = setup_root(tmp_path), tmp_path / "runs"
    stages.write_result(root, 0, {"go": True})
    p1 = plans.build_plan(1, root)
    plans.write_plan(root, 1, p1)
    fill(runs, p1, lambda b, c: c[2] % 2 == 0)
    fp: dict[str, Any] = {"stage": 1, "ollama_version": "0.40.3"}
    path = runs / "ticket014/stage1/fingerprint.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(fp))
    r1 = stages.finalize(1, root=root, runs_dir=runs, n_boot=10)
    assert r1["fingerprint"] == fp


def test_finalize_is_deterministic_for_the_same_records(tmp_path: Path) -> None:
    outs = []
    for name in ("a", "b"):
        root, runs = setup_root(tmp_path / name), tmp_path / name / "runs"
        stages.write_result(root, 0, {"go": True})
        p1 = plans.build_plan(1, root)
        plans.write_plan(root, 1, p1)
        fill(runs, p1, lambda b, c: (c[2] + b.repeat) % 3 == 0)
        stages.finalize(1, root=root, runs_dir=runs, n_boot=20)
        outs.append((root / "results/ticket-014/stage1-thresholds.json").read_bytes())
    assert outs[0] == outs[1]


def test_zero_oracle_targets_skip_stage_4_and_close_stage_5(tmp_path: Path) -> None:
    root, runs = setup_root(tmp_path), tmp_path / "runs"
    stages.write_result(root, 0, {"go": True})
    p1 = plans.build_plan(1, root)
    plans.write_plan(root, 1, p1)
    rng = random.Random(1)
    fill(runs, p1, lambda b, c: rng.random() < 0.5)
    stages.finalize(1, root=root, runs_dir=runs, n_boot=20)
    p2 = plans.build_plan(2, root)
    plans.write_plan(root, 2, p2)
    fill(runs, p2, lambda b, c: s0_ok(c))  # 모든 후보가 S0와 같다 -> C4·SAGE 채택 없음
    stages.finalize(2, root=root, runs_dir=runs, n_boot=20)
    r3 = stages.finalize(3, root=root, runs_dir=runs, n_boot=20, controls=False)
    assert r3["oracle_targets"]["proposer"] == []
    r4 = stages.finalize(4, root=root, runs_dir=runs, n_boot=20)
    assert r4["skipped"] is True and "no Oracle targets" in r4["reason"]
    assert not (root / "plans/ticket-014-stage4-plan.json").exists()
    r5 = stages.finalize(5, root=root, runs_dir=runs, n_boot=20)
    assert r5["rq14_1"]["verdict"] == "undecidable"
    assert r5["rq14_2"]["contains_prediction"] == "undefined"
    assert all(v["accepted"] == 0 for v in r5["rq14_3"].values())
    assert r5["stage4_skipped"] is True
