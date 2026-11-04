"""Ticket 009 보정 임계값 h(n_c)·δ(N) 기록(사전 등록 4.2·5절)."""

import json
import random
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from evidencegate.eval_constants import (
    CONFIG_ID,
    POLICY_ID,
    S0_ORACLE_REPEATS,
    S0_SKILL_ID,
    TOOL_SPEC_ID,
)
from evidencegate.exact import null_diff_quantile_exact
from evidencegate.policy import PolicyTrace, StopReason
from evidencegate.runner import EpisodeKey, EpisodeRecord
from evidencegate.store import EpisodeStore
from evidencegate.t009 import thresholds
from evidencegate.t009.rq9 import cell_threshold_h
from evidencegate.t009.splits import CALIBRATION_V1, calibration_all

CD = "correct_disposition"
ES = "evidence_safe"


def record(case: tuple[str, str, int], repeat: int, cd: bool, es: bool) -> EpisodeRecord:
    return EpisodeRecord(
        key=EpisodeKey(
            POLICY_ID, CONFIG_ID, TOOL_SPEC_ID, S0_SKILL_ID, case[0], case[2], repeat, case[1]
        ),
        policy_config={},
        llm_seed=0,
        correct_disposition=cd,
        evidence_safe=es,
        action="observe_only",
        missing_predicates=(),
        failure_labels=(),
        trace=PolicyTrace((), 0, StopReason.ACTION),
        started_at="2026-11-05T00:00:00+00:00",
        duration_s=1.0,
    )


def full_records(seed: int = 1) -> list[EpisodeRecord]:
    rng = random.Random(seed)
    out = []
    for case in calibration_all():
        for r in S0_ORACLE_REPEATS:
            cd = rng.random() < 0.6
            out.append(record(case, r, cd, cd and rng.random() < 0.5))
    return out


def test_h_and_delta_sizes_follow_the_preregistration() -> None:
    assert thresholds.H_NS == (70, 4, 8, 16, 34)
    assert thresholds.DELTA_NS == (420, 24, 48, 96, 204)


def test_full_calibration_passes_the_record_check() -> None:
    records = full_records()
    assert len(records) == 420 * 5
    thresholds.check_calibration_records(records)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda rs: rs[1:], "missing"),
        (lambda rs: [*rs, rs[0]], "duplicate"),
        (
            lambda rs: [replace(rs[0], key=replace(rs[0].key, config_id="x")), *rs[1:]],
            "run conditions",
        ),
        (lambda rs: [replace(rs[0], key=replace(rs[0].key, skill_id="x")), *rs[1:]], "skill"),
        (
            lambda rs: [*rs, record(("bad_deployment", "L1", 900_070), 0, True, True)],
            "not a calibration case",
        ),
        (lambda rs: [*rs, record(calibration_all()[0], 5, True, True)], "repeat"),
    ],
)
def test_record_check_rejects_bad_sets(mutate, message: str) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match=message):
        thresholds.check_calibration_records(mutate(full_records()))


def test_calibration_outcomes_use_all_70_cases_per_cell() -> None:
    records = full_records()
    table = thresholds.calibration_outcomes(records, ES)
    assert sorted(table) == list(S0_ORACLE_REPEATS)
    assert all(sorted(t) == calibration_all() for t in table.values())
    r = records[7]
    assert table[r.key.repeat][(r.key.family, r.key.level, r.key.incident_seed)] == int(
        r.evidence_safe
    )


def test_thresholds_match_the_preregistered_functions_per_metric() -> None:
    records = full_records()
    got = thresholds.compute_thresholds(records, n_boot=40)
    for metric in (CD, ES):
        table = thresholds.calibration_outcomes(records, metric)
        for n_c in thresholds.H_NS:
            h, rate = cell_threshold_h(table, n_c, n_boot=40)
            row = got["metrics"][metric]["h"][str(n_c)]
            assert row["h"] == str(h)
            assert row["h_float"] == float(h)
            assert row["achieved_false_rejection"] == str(rate)
            assert row["achieved_false_rejection_float"] == float(rate)
        for n in thresholds.DELTA_NS:
            delta = null_diff_quantile_exact(records, metric, n, n_boot=40)
            row = got["metrics"][metric]["delta"][str(n)]
            assert row == {"delta": str(delta), "delta_float": float(delta)}
    assert got["metrics"][CD] != got["metrics"][ES]
    assert got["settings"]["n_boot"] == 40


def test_threshold_bytes_are_deterministic() -> None:
    a = thresholds.threshold_bytes(thresholds.compute_thresholds(full_records(), n_boot=20))
    b = thresholds.threshold_bytes(thresholds.compute_thresholds(full_records(), n_boot=20))
    assert a == b
    assert a.endswith(b"\n")
    assert isinstance(json.loads(a), dict)


def write_stores(runs: Path, records: list[EpisodeRecord]) -> None:
    v1 = set(CALIBRATION_V1)
    for split in ("calibration", "t009_calibration"):
        store = EpisodeStore.for_run(
            runs,
            suite=split,
            policy_id=POLICY_ID,
            config_id=CONFIG_ID,
            tool_spec_id=TOOL_SPEC_ID,
            skill_id=S0_SKILL_ID,
        )
        for r in records:
            case = (r.key.family, r.key.level, r.key.incident_seed)
            if (case in v1) == (split == "calibration"):
                store.append(r)


def test_load_reads_the_v1_and_009_stores(tmp_path: Path) -> None:
    records = full_records()
    write_stores(tmp_path, records)
    loaded, inputs = thresholds.load_calibration_records(tmp_path)
    key = lambda r: r.key  # noqa: E731
    assert sorted(map(key, loaded), key=repr) == sorted(map(key, records), key=repr)
    assert [i["records"] for i in inputs] == [120 * 5, 300 * 5]
    assert all(len(i["sha256"]) == 64 for i in inputs)


def test_main_writes_json_and_sha_and_refuses_to_overwrite(tmp_path: Path) -> None:
    write_stores(tmp_path / "runs", full_records())
    out = tmp_path / "thresholds.json"
    args = ["--runs-dir", str(tmp_path / "runs"), "--out", str(out), "--n-boot", "20"]
    assert thresholds.main(args) == 0
    data = json.loads(out.read_text())
    assert Fraction(data["metrics"][CD]["h"]["70"]["h"]) >= 0
    assert len(data["inputs"]) == 2
    digest = out.with_name("thresholds.json.sha256").read_text().split()
    assert digest[1] == "thresholds.json" and len(digest[0]) == 64
    assert thresholds.main(args) == 0  # 같은 바이트면 그대로 둔다
    out.write_text("{}\n")
    assert thresholds.main(args) == 1


def test_main_rejects_an_incomplete_calibration(tmp_path: Path) -> None:
    write_stores(tmp_path / "runs", full_records()[:-1])
    out = tmp_path / "thresholds.json"
    assert thresholds.main(["--runs-dir", str(tmp_path / "runs"), "--out", str(out)]) == 1
    assert not out.exists()
