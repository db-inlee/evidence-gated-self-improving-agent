"""Ticket 014 계획: 대칭 Oracle 규모별 라벨 불확실성(`evidencegate.t014.oracle_scale`). LLM 없음."""

from fractions import Fraction as F
from pathlib import Path

import numpy as np
import pytest

from evidencegate.t013 import sim_core as sc
from evidencegate.t014 import oracle_scale as osc


def toy() -> sc.Baseline:
    cells = (
        (F(1), F(1)),
        (F(1), F(4, 5)),
        (F(1, 2), F(1, 2)),
        (F(1), F(1)),
        (F(0), F(1, 5)),
        (F(1, 5), F(2, 5)),
    )
    return sc.Baseline("toy", cells)


def test_aggregate_mean_and_majority_are_symmetric() -> None:
    cand = np.asarray(
        [[[1, 0, 1]], [[0, 0, 1]]], dtype=np.int64
    )  # (repeat 2? no) shape (R=2, cases=1, 3)
    s0 = np.asarray([[1], [0]], dtype=np.int64)
    d_mean, scale = osc.case_differences(cand, s0, aggregation="mean")
    assert scale == 2 and d_mean.tolist() == [[1 - 1, 0 - 1, 2 - 1]]
    d3 = osc.case_differences(
        np.asarray([[[1]], [[1]], [[0]]], dtype=np.int64),
        np.asarray([[0], [0], [1]], dtype=np.int64),
        aggregation="majority",
    )
    assert d3[1] == 1 and d3[0].tolist() == [[1]]  # 다수결 1 - 0
    with pytest.raises(ValueError, match="odd"):
        osc.case_differences(cand, s0, aggregation="majority")


def test_same_seed_pairs_share_the_uniforms_of_s0() -> None:
    rng = np.random.default_rng(0)
    u_s0, u_c = osc.draw_uniforms(rng, repeats=2, cases=5, candidates=3, pairing="same_seed")
    assert np.array_equal(u_c, np.broadcast_to(u_s0[:, :, None], u_c.shape))
    rng = np.random.default_rng(0)
    u_s0, u_c = osc.draw_uniforms(rng, repeats=2, cases=5, candidates=3, pairing="independent")
    assert not np.array_equal(u_c[:, :, 0], u_s0)


def test_labels_count_uncertain_useful_effect_and_severe() -> None:
    # 칸 6 x 케이스 2: 한 칸만 모두 -1이면 그 칸 상한 < -τ(Severe 확정), 전체 구간은 0 아래
    d = np.zeros((12, 2), dtype=np.int64)
    d[0:2, 0] = -1
    got = osc.labels(d, scale=1, n_o=2, n_boot=50)
    assert got["severe"].tolist() == [True, False]
    assert got["severe_uncertain"].tolist() == [False, False]
    assert got["useful_uncertain"].dtype == bool and got["effect_uncertain"].shape == (2,)


def test_evaluate_condition_is_deterministic_and_reports_rates() -> None:
    cond = osc.Condition(n_o=4, repeats=1, aggregation="mean", pairing="independent")
    one = osc.evaluate(toy(), cond, v1_effects=((F(0),) * 6,), pools=3, candidates=5, n_boot=30)
    two = osc.evaluate(toy(), cond, v1_effects=((F(0),) * 6,), pools=3, candidates=5, n_boot=30)
    assert one == two
    row = one["scenarios"]["useful_representative|-"]
    assert set(row) == {"useful_uncertain", "effect_uncertain", "severe_uncertain", "severe"}
    assert 0 <= row["useful_uncertain"]["estimate"] <= 1
    assert one["oracle_episodes"] == 48 * 6 * 4 * 1


def test_oracle_episodes_and_time_use_the_009b_throughput() -> None:
    assert osc.oracle_episodes(n_o=50, repeats=3) == 48 * 6 * 50 * 3
    hours = osc.hours(55860)
    assert hours["active"] == pytest.approx(55860 * osc.ACTIVE_S_PER_EPISODE / 3600)
    assert hours["wall"] > hours["active"]


def test_throughput_constants_are_the_committed_run_summary() -> None:
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    run = json.loads((root / "writeup/inputs/run_summary_009b.json").read_text())["run"]["t009b"][
        "main"
    ]
    assert run["records"]["episodes"] == osc.EPISODES_009B
    assert run["records"]["wall_s"] == osc.WALL_S_009B
    assert run["idle"]["active_s"] == osc.ACTIVE_S_009B


def test_parts_by_n_o_merge_into_one_table(tmp_path: Path) -> None:
    import json

    parts = []
    for n_o in (50, 100):
        part = {
            "rows": [{"condition": f"n{n_o}_r1_mean_independent", "oracle_episodes": 1}],
            "pools": 3,
        }
        path = tmp_path / f"p{n_o}.json"
        path.write_text(json.dumps(part | {"n_o_filter": n_o}))
        parts.append(path)
    merged = osc.merge_parts(parts)
    assert [r["condition"] for r in merged["rows"]] == [
        "n50_r1_mean_independent",
        "n100_r1_mean_independent",
    ]
    assert "n_o_filter" not in merged and merged["pools"] == 3
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"rows": [], "pools": 4, "n_o_filter": 150}))
    with pytest.raises(ValueError, match="settings"):
        osc.merge_parts([*parts, bad])
