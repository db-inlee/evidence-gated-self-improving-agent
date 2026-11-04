"""Ticket 009 보정 임계값 h(n_c)·δ(N)의 계산과 기록(사전 등록 4.2·5절).

    uv run python -m evidencegate.t009.thresholds

- 입력은 보정 S0 기록 칸당 70 케이스 x 반복 0~4(2,100개)다. v1 보정 기록(`runs/calibration/…`,
  900,000~900,019)과 009 추가 기록(`runs/t009_calibration/…`, 900,020~900,069)을 합친다. 실행
  조건(정책, config, 도구 명세, S0)이 하나여야 하고, 빠지거나 겹치거나 범위 밖인 기록이 있으면
  거부한다.
- h(n_c)는 `rq9.cell_threshold_h`, δ(N)은 `exact.null_diff_quantile_exact`를 metric(CD, ES)마다
  부른다.
  값은 유리수 문자열과 float를 함께 적는다. 판정에는 유리수를 쓴다.
- 결과는 `results/ticket-009/thresholds.json`이고 `<결과>.sha256`에 "sha256  이름"으로 고정한다.
  이미 있는 파일과 바이트가 다르면 거부한다.
"""

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from evidencegate.analysis import REPO_ROOT
from evidencegate.eval_constants import (
    CONFIG_ID,
    N_BOOT,
    NOISE_Q,
    POLICY_ID,
    S0_ORACLE_REPEATS,
    S0_SKILL_ID,
    SEED,
    TOOL_SPEC_ID,
)
from evidencegate.eval_data import CaseId, case_of, check_run_conditions, outcomes
from evidencegate.exact import null_diff_quantile_exact
from evidencegate.metrics import METRICS
from evidencegate.runner import EpisodeRecord
from evidencegate.store import EpisodeStore
from evidencegate.t009.preflight import check_preregistration_009
from evidencegate.t009.rq9 import H_Q, cell_threshold_h
from evidencegate.t009.splits import calibration_all

H_NS = (70, 4, 8, 16, 34)  # 4.2: 70은 게이트, 나머지는 Budget Curve
DELTA_NS = (420, 24, 48, 96, 204)  # 5절
CALIBRATION_SPLITS = ("calibration", "t009_calibration")  # v1 기록, 009 추가 기록
OUT_PATH = Path("results/ticket-009/thresholds.json")


def check_calibration_records(records: Sequence[EpisodeRecord]) -> None:
    """보정 케이스 x 반복 0~4가 빠짐없이 한 번씩, 한 실행 조건의 S0 기록인지."""
    check_run_conditions(
        records, policy_id=POLICY_ID, config_id=CONFIG_ID, tool_spec_id=TOOL_SPEC_ID
    )
    skills = {r.key.skill_id for r in records}
    if skills != {S0_SKILL_ID}:
        raise ValueError(f"calibration needs S0 {S0_SKILL_ID} only, got skill {sorted(skills)}")
    wanted = set(calibration_all())
    for r in records:
        if case_of(r) not in wanted:
            raise ValueError(f"{case_of(r)} is not a calibration case")
        if r.key.repeat not in S0_ORACLE_REPEATS:
            raise ValueError(f"repeat {r.key.repeat} is not one of {S0_ORACLE_REPEATS}")
    counts = Counter((case_of(r), r.key.repeat) for r in records)
    duplicates = [k for k, n in counts.items() if n > 1]
    if duplicates:
        raise ValueError(f"{len(duplicates)} duplicate records, e.g. {duplicates[0]}")
    missing = len(wanted) * len(S0_ORACLE_REPEATS) - len(counts)
    if missing:
        raise ValueError(f"{missing} missing calibration records")


def calibration_outcomes(
    records: Sequence[EpisodeRecord], metric: str
) -> dict[int, dict[CaseId, int]]:
    """반복 → 보정 케이스(칸당 70) → 0/1."""
    cases = calibration_all()
    return {
        r: outcomes(
            records,
            skill_id=S0_SKILL_ID,
            split="calibration",
            metric=metric,
            repeat=r,
            cases=cases,
        )
        for r in S0_ORACLE_REPEATS
    }


def _value(name: str, x: Fraction) -> dict[str, Any]:
    return {name: str(x), f"{name}_float": float(x)}


def compute_thresholds(
    records: Sequence[EpisodeRecord], n_boot: int = N_BOOT, seed: int = SEED
) -> dict[str, Any]:
    check_calibration_records(records)
    metrics: dict[str, Any] = {}
    for metric in METRICS:
        table = calibration_outcomes(records, metric)
        h_rows = {}
        for n_c in H_NS:
            h, rate = cell_threshold_h(table, n_c, n_boot=n_boot, seed=seed)
            h_rows[str(n_c)] = _value("h", h) | _value("achieved_false_rejection", rate)
        delta_rows = {
            str(n): _value(
                "delta", null_diff_quantile_exact(records, metric, n, n_boot=n_boot, seed=seed)
            )
            for n in DELTA_NS
        }
        metrics[metric] = {"h": h_rows, "delta": delta_rows}
    return {
        "metrics": metrics,
        "settings": {
            "n_boot": n_boot,
            "seed": seed,
            "h_q": str(H_Q),
            "delta_q": str(NOISE_Q),
            "cases_per_cell": len(calibration_all()) // 6,
            "repeats": list(S0_ORACLE_REPEATS),
        },
    }


def threshold_bytes(data: dict[str, Any]) -> bytes:
    return (json.dumps(data, indent=1, sort_keys=True, ensure_ascii=False) + "\n").encode()


def load_calibration_records(
    runs_dir: Path,
) -> tuple[list[EpisodeRecord], list[dict[str, Any]]]:
    """두 보정 기록 파일을 읽는다. 파일마다 (경로, sha256, 기록 수)를 함께 돌려준다."""
    records: list[EpisodeRecord] = []
    inputs: list[dict[str, Any]] = []
    for split in CALIBRATION_SPLITS:
        path = EpisodeStore.for_run(
            runs_dir,
            suite=split,
            policy_id=POLICY_ID,
            config_id=CONFIG_ID,
            tool_spec_id=TOOL_SPEC_ID,
            skill_id=S0_SKILL_ID,
        ).path
        part = EpisodeStore(path, repair=False).records()
        records += part
        inputs.append(
            {
                "path": str(path.relative_to(runs_dir.parent)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "records": len(part),
            }
        )
    return records, inputs


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evidencegate-t009-thresholds")
    parser.add_argument("--runs-dir", type=Path, default=REPO_ROOT / "runs")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / OUT_PATH)
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    prereg = check_preregistration_009(args.root)
    if not prereg.ok:
        print(f"error: {prereg.detail}", file=sys.stderr)
        return 1
    try:
        records, inputs = load_calibration_records(args.runs_dir)
        data = compute_thresholds(records, n_boot=args.n_boot)
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    data["inputs"] = inputs
    data["preregistration"] = prereg.detail
    out: Path = args.out
    payload = threshold_bytes(data)
    digest = hashlib.sha256(payload).hexdigest()
    if out.exists():
        if out.read_bytes() != payload:
            print(f"error: {out} exists and differs from the values computed now", file=sys.stderr)
            return 1
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(payload)
    out.with_name(out.name + ".sha256").write_text(f"{digest}  {out.name}\n")
    print(f"thresholds {out}: sha256 {digest}")
    for metric in METRICS:
        row = data["metrics"][metric]
        print(
            f"  {metric}: h(70) {row['h']['70']['h_float']:.4f}"
            f" (rate {row['h']['70']['achieved_false_rejection_float']:.4f}),"
            f" delta(420) {row['delta']['420']['delta_float']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
