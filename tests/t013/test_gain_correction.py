"""Ticket 013 이득 지표 보정(보고서 2026-11-26-01 §9, 결정 (가)).

`sim.json`은 고치지 않고, 체크포인트에서 `gain_per_proposal`만 다시 계산해 별도 파일로 한 번 쓴다.
"""

import hashlib
import json
from pathlib import Path

import pytest
from test_sim_run import tiny_inputs, tiny_settings

from evidencegate.t013 import sim


@pytest.fixture()
def tiny_run(tmp_path: Path) -> tuple[Path, Path, str]:
    blocks, final = tmp_path / "blocks", tmp_path / "sim.json"
    sim.run_simulation(tiny_inputs(), tiny_settings(), blocks_dir=blocks, final=final)
    return blocks, final, hashlib.sha256(final.read_bytes()).hexdigest()


def test_the_original_result_is_pinned() -> None:
    assert sim.SIM_SHA256.startswith("02b95b09")
    assert Path("results/ticket-013/sim-gain-correction.json") == sim.GAIN_CORRECTION


def test_correction_recomputes_gain_from_the_checkpoints(tiny_run: tuple[Path, Path, str]) -> None:
    blocks, final, digest = tiny_run
    out = sim.gain_correction(
        tiny_settings(), ("a", "b"), blocks_dir=blocks, sim_path=final, sim_sha=digest
    )
    assert out["corrects"] == {"path": str(final), "sha256": digest}
    assert "int()" in out["bug"] and "not used by the Go rule" in out["bug"]
    data = json.loads((blocks / "a_4_none.json").read_text())
    key = next(iter(data["results"][0]["gates"]))
    gains = [p["gates"][key]["gain"] for p in data["results"]]
    n = [p["gates"][key]["n"] for p in data["results"]]
    row = out["gain_per_proposal"]["a_4_none"][key]
    assert row["estimate"] == pytest.approx(sum(gains) / sum(n))
    original = json.loads(final.read_text())["metrics"]["a_4_none"][key]["gain_per_proposal"][
        "estimate"
    ]
    assert row["original_estimate"] == original


def test_correction_refuses_another_result_file(tiny_run: tuple[Path, Path, str]) -> None:
    blocks, final, _digest = tiny_run
    with pytest.raises(ValueError, match="sha256"):
        sim.gain_correction(
            tiny_settings(), ("a", "b"), blocks_dir=blocks, sim_path=final, sim_sha="0" * 64
        )


def test_correction_is_written_once(tiny_run: tuple[Path, Path, str], tmp_path: Path) -> None:
    blocks, final, digest = tiny_run
    out = tmp_path / "sim-gain-correction.json"
    kw = dict(blocks_dir=blocks, sim_path=final, sim_sha=digest, out=out)
    sim.write_gain_correction(tiny_settings(), ("a", "b"), **kw)  # type: ignore[arg-type]
    recorded = out.with_name(out.name + ".sha256").read_text().split()
    assert recorded == [hashlib.sha256(out.read_bytes()).hexdigest(), out.name]
    before = final.read_bytes()
    with pytest.raises(FileExistsError):
        sim.write_gain_correction(tiny_settings(), ("a", "b"), **kw)  # type: ignore[arg-type]
    assert final.read_bytes() == before
