"""Ticket 013 1단계 감사 요약. 정의: `writeup/inputs/audit_013_definition.txt`.

    uv run python -m evidencegate.t013.audit --runs-dir runs --out writeup/inputs/audit_013.json

- 정의 파일(sha256 `DEFINITION_SHA256`)의 A~H절을 그대로 따른다. 커밋된 결과를 다시 판정하지 않고,
  실행 기록과 009b 결과 파일에서 세기만 한다.
- 입력을 **읽기만** 한다. 기록 파일 sha256은 이미 커밋된 파일(`run_summary_009b.json`,
  `thresholds.json`)과 `T012_RECORD_PINS`에 고정한 값과 같아야 한다. 다르면 아무것도 쓰지 않는다.
- 출력 `{"audit": …, "provenance": …, "notes": […]}`. 정렬된 키, 생성 시각 없음. 같은 입력이면 같은
  바이트다. 이미 있는 출력과 바이트가 다르면 거부한다.
"""

import argparse
import hashlib
import inspect
import json
import sys
from collections.abc import Mapping, Sequence
from fractions import Fraction
from itertools import combinations
from pathlib import Path
from typing import Any

from evidencegate.analysis import REPO_ROOT
from evidencegate.eval_constants import (
    CONFIG_ID,
    POLICY_ID,
    S0_ORACLE_REPEATS,
    S0_SKILL_ID,
    TOOL_SPEC_ID,
)
from evidencegate.eval_data import CELLS, CaseId
from evidencegate.runner import EpisodeRecord
from evidencegate.seeds import derive_llm_seed
from evidencegate.store import EpisodeStore
from evidencegate.t009.splits import calibration_all, split_cases
from evidencegate.t009b.settings import POOL_PATH, THRESHOLDS_PATH

DEFINITION_PATH = Path("writeup/inputs/audit_013_definition.txt")
DEFINITION_SHA256 = "1bf214ae99731ae8b483fcdada894dd4eff692ca6f1c2707c02db0f487fdc151"
DEFAULT_OUT = Path("writeup/inputs/audit_013.json")
T009B_ANALYSIS = Path("results/ticket-009b/analysis.json")
T009B_ANALYSIS_SHA256 = "21c87f2798fa032e3ce627029e8b01c7629cb9b73215aeac1b3742f36d3cdafe"
RUN_SUMMARY_009B = Path("writeup/inputs/run_summary_009b.json")
S0_PRIME_SKILL_ID = "3665eab231d6"
T012_SPLITS = ("t012_stage0", "t012_stage0_rerun1", "t012_stage0_rerun2")
T012_RECORD_PINS = {
    "runs/t012_stage0/ollama__gpt-oss__20b__7c66ee5bf13c/e9b4996f7529/3665eab231d6.jsonl": (
        "32517a7e14c9f91b4b9595845af00d8dd60f4096e993d1dc80cbb0131227b06b"
    ),
    "runs/t012_stage0_rerun1/ollama__gpt-oss__20b__7c66ee5bf13c/e9b4996f7529/3665eab231d6.jsonl": (
        "750feffd883b3d810d69494f53b18b5083564f3ad5d12b2d14fb202a029d6e74"
    ),
    "runs/t012_stage0_rerun2/ollama__gpt-oss__20b__7c66ee5bf13c/e9b4996f7529/3665eab231d6.jsonl": (
        "737becc08b482bac9282d0907a8750970ccaeb87ba0de67294c757afde90b44d"
    ),
}
METRICS = {"cd": "correct_disposition", "es": "evidence_safe"}
NEAR = Fraction(1, 10)  # B절: near_floor <= 1/10, near_ceiling >= 9/10
NOTES = (
    "After-the-fact audit; definition in writeup/inputs/audit_013_definition.txt.",
    "Every value was first computed in the review report of 2026-11-24, before the definition.",
    "No committed result is re-judged; outcomes of candidates are read only from analysis.json.",
)

Table = dict[int, dict[CaseId, dict[str, int]]]  # 반복 -> 케이스 -> metric -> 0/1


def cell_name(cell: tuple[str, str]) -> str:
    return f"{cell[0]}/{cell[1]}"


def case_key(r: EpisodeRecord) -> CaseId:
    return r.key.family, r.key.level, r.key.incident_seed


def outcome_table(records: Sequence[EpisodeRecord]) -> Table:
    table: Table = {}
    for r in records:
        row = table.setdefault(r.key.repeat, {})
        case = case_key(r)
        if case in row:
            raise ValueError(f"duplicate record for repeat {r.key.repeat}: {case}")
        row[case] = {"cd": int(r.correct_disposition), "es": int(r.evidence_safe)}
    return table


def _cells_of(cases: Sequence[CaseId]) -> list[tuple[str, str]]:
    present = {c[:2] for c in cases}
    return [c for c in CELLS if c in present]


def cell_rates(row: Mapping[CaseId, Mapping[str, int]], metric: str) -> dict[str, dict[str, int]]:
    """B절: 칸마다 correct와 cases."""
    out = {}
    for cell in _cells_of(list(row)):
        cases = [c for c in row if c[:2] == cell]
        out[cell_name(cell)] = {"correct": sum(row[c][metric] for c in cases), "cases": len(cases)}
    return out


def majority_table(table: Table) -> dict[CaseId, dict[str, int]]:
    if len(table) % 2 == 0:
        raise ValueError(f"majority needs an odd number of repeats, got {len(table)}")
    first = next(iter(table.values()))
    if any(set(row) != set(first) for row in table.values()):
        raise ValueError("majority needs the same cases in every repeat")
    n = len(table)
    return {
        c: {m: int(2 * sum(row[c][m] for row in table.values()) > n) for m in METRICS}
        for c in first
    }


def majority_offsets(table: Table, metric: str) -> dict[str, dict[str, Any]]:
    """A절: 단일 실행 평균 - 다수결, 반복 0 - 다수결, 갈린 케이스 수."""
    maj = majority_table(table)
    reps = sorted(table)
    out: dict[str, dict[str, Any]] = {}
    for cell in _cells_of(list(maj)):
        cases = [c for c in maj if c[:2] == cell]
        single = sum(
            (Fraction(sum(table[r][c][metric] for r in reps), len(reps)) - maj[c][metric]
             for c in cases),
            Fraction(0),
        )  # fmt: skip
        r0 = sum(table[reps[0]][c][metric] - maj[c][metric] for c in cases)
        mixed = sum(1 for c in cases if len({table[r][c][metric] for r in reps}) > 1)
        out[cell_name(cell)] = {
            "single_minus_majority": single / len(cases),
            "r0_minus_majority": Fraction(r0, len(cases)),
            "mixed_cases": mixed,
            "cases": len(cases),
        }
    return out


def repeat_discordance(table: Table, metric: str) -> dict[str, dict[str, int]]:
    """D절: 반복 쌍마다 처분(metric) 불일치."""
    reps = sorted(table)
    cases = list(table[reps[0]])
    out: dict[str, dict[str, int]] = {}
    total = {"differ": 0, "pairs": 0}
    for cell in _cells_of(cases):
        in_cell = [c for c in cases if c[:2] == cell]
        differ = sum(
            1 for a, b in combinations(reps, 2) for c in in_cell
            if table[a][c][metric] != table[b][c][metric]
        )  # fmt: skip
        pairs = len(in_cell) * len(list(combinations(reps, 2)))
        out[cell_name(cell)] = {"differ": differ, "pairs": pairs}
        total = {"differ": total["differ"] + differ, "pairs": total["pairs"] + pairs}
    out["all"] = total
    return out


def edge_cells(cells: Mapping[str, Mapping[str, int]]) -> dict[str, int]:
    """B절 edges: floor, ceiling, near_floor(<= 1/10), near_ceiling(>= 9/10), cells."""
    rates = [Fraction(v["correct"], v["cases"]) for v in cells.values()]
    return {
        "floor": sum(1 for p in rates if p == 0),
        "ceiling": sum(1 for p in rates if p == 1),
        "near_floor": sum(1 for p in rates if p <= NEAR),
        "near_ceiling": sum(1 for p in rates if p >= 1 - NEAR),
        "cells": len(rates),
    }


def seed_sharing(cases: Sequence[CaseId]) -> dict[str, int]:
    """C절: (family, incident_seed) 묶음과 L1·L2가 함께 있는 묶음."""
    groups: dict[tuple[str, int], set[str]] = {}
    for family, level, seed in cases:
        groups.setdefault((family, seed), set()).add(level)
    return {
        "cases": len(cases),
        "seed_groups": len(groups),
        "shared_by_l1_l2": sum(1 for levels in groups.values() if {"L1", "L2"} <= levels),
    }


def level_free_seed_rule() -> bool:
    """C절: seed 규칙이 level을 받지 않으면 같은 묶음의 L1·L2는 같은 LLM seed다."""
    return "level" not in inspect.signature(derive_llm_seed).parameters


def same_seed_pairs(
    candidates: Mapping[str, Sequence[EpisodeRecord]], s0: Sequence[EpisodeRecord]
) -> dict[str, int]:
    """E절: 후보 기록마다 같은 케이스 S0 반복 0의 llm_seed와 같은가. 결과(CD·ES)는 보지 않는다."""
    base = {case_key(r): r.llm_seed for r in s0 if r.key.repeat == 0}
    pairs = same = 0
    for name, records in sorted(candidates.items()):
        cases = {case_key(r) for r in records}
        if cases != set(base):
            raise ValueError(f"{name}: missing or extra cases against S0 repeat 0")
        for r in records:
            pairs += 1
            same += int(r.llm_seed == base[case_key(r)])
    return {"candidates": len(candidates), "pairs": pairs, "same_seed": same}


def label_cells(analysis: Mapping[str, Any]) -> dict[str, Any]:
    """F절: 커밋된 009b 결과(CD)에서 Severe와 Local Harm(Severe 아님)의 -τ 아래 칸."""
    tau = {c: Fraction(v["exact"]) for c, v in analysis["tau"]["correct_disposition"].items()}
    names = sorted(tau)
    worst: list[Fraction] = []
    with_cell = dict.fromkeys(names, 0)
    only_cell = dict.fromkeys(names, 0)
    severe = local_only = 0
    for entry in analysis["candidates"].values():
        x = entry["correct_disposition"]
        cells = {c: Fraction(v["exact"]) for c, v in x["oracle_cells"].items()}
        below = [c for c in names if cells[c] < -tau[c]]
        if x["severe"]:
            severe += 1
            worst.append(min(cells.values()))
        elif x["local_harm"]:
            local_only += 1
            for c in below:
                with_cell[c] += 1
            if len(below) == 1:
                only_cell[below[0]] += 1
    return {
        "severe": {
            "candidates": severe,
            "worst_cell": {"closest_to_zero": max(worst), "farthest": min(worst)} if worst else {},
        },
        "local_harm_only": {
            "candidates": local_only,
            "with_cell": with_cell,
            "this_cell_only": only_cell,
        },
    }


def _by_case(records: Sequence[EpisodeRecord]) -> dict[CaseId, EpisodeRecord]:
    out: dict[CaseId, EpisodeRecord] = {}
    for r in records:
        if case_key(r) in out:
            raise ValueError(f"duplicate record: {case_key(r)}")
        out[case_key(r)] = r
    return out


def _three(runs: Sequence[Sequence[EpisodeRecord]]) -> list[dict[CaseId, EpisodeRecord]]:
    if len(runs) != 3:
        raise ValueError(f"need three runs, got {len(runs)}")
    by = [_by_case(r) for r in runs]
    if any(set(b) != set(by[0]) for b in by):
        raise ValueError("the three runs hold different cases")
    for b in by[1:]:
        for case, r in b.items():
            if r.llm_seed != by[0][case].llm_seed:
                raise ValueError(f"{case}: llm_seed {r.llm_seed} != {by[0][case].llm_seed}")
    return by


def t012_pairs(runs: Sequence[Sequence[EpisodeRecord]]) -> dict[str, int]:
    """G절: 0단계 쌍(1회째 대 재실행 1, 1회째 대 재실행 2)의 공유 구조."""
    first, one, two = _three(runs)
    both = single = none = 0
    for case, r in first.items():
        d = (one[case].action != r.action) + (two[case].action != r.action)
        both += d == 2
        single += d == 1
        none += d == 0
    return {
        "cases": len(first),
        "llm_seeds": len({r.llm_seed for r in first.values()}),
        "pairs": 2 * len(first),
        "differing": 2 * both + single,
        "cases_both_pairs_differ": both,
        "cases_one_pair_differs": single,
        "cases_no_pair_differs": none,
        "reruns_differing": sum(1 for c in first if one[c].action != two[c].action),
    }


def s0_prime_cells(runs: Sequence[Sequence[EpisodeRecord]]) -> dict[str, Any]:
    """H절: 칸별 실행마다 맞힌 수, 양방향 관측 가능성, G절 쌍의 칸별 불일치."""
    by = _three(runs)
    cases = list(by[0])
    out: dict[str, Any] = {}
    counts: dict[str, int] = {}
    for short, attr in METRICS.items():
        cells: dict[str, Any] = {}
        for cell in _cells_of(cases):
            in_cell = [c for c in cases if c[:2] == cell]
            per_run = [sum(int(getattr(b[c], attr)) for c in in_cell) for b in by]
            pooled, pooled_of = sum(per_run), 3 * len(in_cell)
            cells[cell_name(cell)] = {
                "runs": per_run,
                "cases": len(in_cell),
                "pooled": pooled,
                "pooled_of": pooled_of,
                "both_directions": 0 < pooled < pooled_of,
            }
        out[short] = cells
        counts[short] = sum(1 for v in cells.values() if v["both_directions"])
    out["both_directions_cells"] = counts
    first = by[0]
    out["discordance"] = {
        cell_name(cell): {
            "differ": sum(
                int(b[c].action != first[c].action)
                for b in by[1:] for c in cases if c[:2] == cell
            ),
            "pairs": 2 * sum(1 for c in cases if c[:2] == cell),
        }
        for cell in _cells_of(cases)
    }  # fmt: skip
    return out


# 입력과 조립 -----------------------------------------------------------------------------------


def _store(runs_dir: Path, split: str, skill: str) -> Path:
    return EpisodeStore.for_run(
        runs_dir,
        suite=split,
        policy_id=POLICY_ID,
        config_id=CONFIG_ID,
        tool_spec_id=TOOL_SPEC_ID,
        skill_id=skill,
    ).path


def record_pins(root: Path) -> dict[str, str]:
    """기록 파일 경로(저장소 기준) -> 고정 sha256. 새로 믿을 값은 012 세 파일뿐이다."""
    summary = json.loads((root / RUN_SUMMARY_009B).read_text(encoding="utf-8"))
    pins = {
        path: digest
        for path, digest in summary["provenance"]["inputs"].items()
        if path.startswith(("runs/t009_dev_b/", "runs/t009_oracle/"))
    }
    thresholds = json.loads((root / THRESHOLDS_PATH).read_text(encoding="utf-8"))
    pins |= {i["path"]: i["sha256"] for i in thresholds["inputs"]}
    return pins | T012_RECORD_PINS


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(
    path: Path, root: Path, pins: Mapping[str, str], used: dict[str, str]
) -> list[EpisodeRecord]:
    rel = str(path.relative_to(root))
    if rel not in pins:
        raise ValueError(f"{rel} has no pinned sha256")
    actual = _sha(path)
    if actual != pins[rel]:
        raise ValueError(f"{rel} sha256 {actual} does not match the pin {pins[rel]}")
    used[rel] = actual
    return EpisodeStore(path, repair=False).records()


def build_audit(root: Path, runs_dir: Path | None = None) -> dict[str, Any]:
    runs_dir = runs_dir or root / "runs"
    if _sha(root / DEFINITION_PATH) != DEFINITION_SHA256:
        raise ValueError(f"{DEFINITION_PATH} differs from its pinned sha256")
    if _sha(root / T009B_ANALYSIS) != T009B_ANALYSIS_SHA256:
        raise ValueError(f"{T009B_ANALYSIS} differs from its pinned sha256")
    pins = record_pins(root)
    used: dict[str, str] = {}
    oracle = outcome_table(_read(_store(runs_dir, "t009_oracle", S0_SKILL_ID), root, pins, used))
    if sorted(oracle) != list(S0_ORACLE_REPEATS):
        raise ValueError(f"S0 Oracle repeats {sorted(oracle)}")
    dev_s0 = _read(_store(runs_dir, "t009_dev_b", S0_SKILL_ID), root, pins, used)
    dev = outcome_table(dev_s0)
    calibration: list[EpisodeRecord] = []
    for split in ("calibration", "t009_calibration"):
        calibration += _read(_store(runs_dir, split, S0_SKILL_ID), root, pins, used)
    pool = json.loads((root / POOL_PATH).read_text(encoding="utf-8"))["candidates"]
    proposers = sorted(c["skill_id"] for c in pool if c["group"] == "proposer")
    candidates = {
        skill: _read(_store(runs_dir, "t009_dev_b", skill), root, pins, used) for skill in proposers
    }
    stage0 = [
        _read(_store(runs_dir, split, S0_PRIME_SKILL_ID), root, pins, used) for split in T012_SPLITS
    ]
    majority = majority_table(oracle)
    rates: dict[str, Any] = {
        "dev_b": {m: cell_rates(dev[0], m) for m in METRICS},
        "oracle_majority": {m: cell_rates(majority, m) for m in METRICS},
        "oracle_repeats": {
            m: {str(r): cell_rates(oracle[r], m) for r in sorted(oracle)} for m in METRICS
        },
    }
    rates["edges"] = {m: edge_cells(rates["oracle_majority"][m]) for m in METRICS}
    analysis = json.loads((root / T009B_ANALYSIS).read_text(encoding="utf-8"))
    audit = {
        "oracle_asymmetry": {m: majority_offsets(oracle, m) for m in METRICS},
        "s0_rates": rates,
        "seed_sharing": {
            "t009_dev_b": seed_sharing(split_cases("t009_dev_b")),
            "t009_oracle": seed_sharing(split_cases("t009_oracle")),
            "calibration": seed_sharing(calibration_all()),
            "level_free_rule": level_free_seed_rule(),
        },
        "calibration_discordance": {"cd": repeat_discordance(outcome_table(calibration), "cd")},
        "dev_b_pairs": same_seed_pairs(candidates, dev_s0),
        "labels": label_cells(analysis),
        "t012_pairs": t012_pairs(stage0),
        "s0_prime": s0_prime_cells(stage0),
    }
    provenance = {
        "definition_sha256": DEFINITION_SHA256,
        "t009b_analysis_sha256": T009B_ANALYSIS_SHA256,
        "inputs": dict(sorted(used.items())),
    }
    return {"audit": audit, "provenance": provenance, "notes": list(NOTES)}


def _plain(obj: Any) -> Any:
    if isinstance(obj, Fraction):
        return {"exact": str(obj), "float": float(obj)}
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_plain(v) for v in obj]
    return obj


def audit_bytes(data: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(_plain(dict(data)), indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evidencegate-t013-audit")
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--runs-dir", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    out: Path = args.out or args.root / DEFAULT_OUT
    try:
        payload = audit_bytes(build_audit(args.root, args.runs_dir))
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    if out.exists() and out.read_bytes() != payload:
        print(f"error: {out} exists and differs from the values computed now", file=sys.stderr)
        return 1
    out.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    out.with_name(out.name + ".sha256").write_text(f"{digest}  {out.name}\n")
    print(f"audit {out}: sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
