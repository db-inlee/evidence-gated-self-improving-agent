"""Ticket 014 단계 결과(사전 등록 8절, 티켓 §11.2): 한 번 쓰기와 해시 사슬, 단계 1~5 마무리.

    uv run python -m evidencegate.t014.run finalize --stage N

- 결과 파일 `results/ticket-014/<이름>`과 `.sha256`. 이미 있으면 거부한다. 같은 입력이면 같은
  바이트다(시각을 넣지 않음). 단계 1부터는 `previous`에 앞 단계 결과의 경로와 sha256을 넣는다.
- `check_previous(N)`: 앞 단계 결과가 있고 `.sha256`과 같아야 한다. 단계 0이 No-Go면 닫힌다.
- 마무리는 그 단계 계획의 모든 블록 기록이 있을 때만 한다(`collect_records`).
  - 1: 보정 h(150)·δ(900)(CD, ES). 2·4: 기록 파일 sha256과 수(결과 값 없음).
  - 3: 단계 2 기록이 그대로인지 확인하고, 게이트(CD, ES)와 Oracle 대상 목록(CD 기준 C4 또는 SAGE λ1,
    통제 후보 포함 여부)을 고정한다.
  - 4: Oracle 대상이 0개이면 기록 없이 `skipped: true`를 쓴다(사전 등록 8절).
  - 5: 단계 2·4 기록이 그대로인지 확인하고, Oracle 라벨과 RQ14-1~3(CD), ES 기술표를 낸다. 단계 4를
    건너뛰었으면 라벨 없이 같은 정의로 계산한다(RQ14-1 undecidable, RQ14-2 정의 안 됨, RQ14-3 0).
- 각 결과에는 그 단계 지문 파일(`runs/ticket014/stage<N>/fingerprint.json`)의 내용을 넣는다.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
from typing import Any

from evidencegate.eval_constants import CONFIG_ID, N_BOOT, POLICY_ID, S0_SKILL_ID, TOOL_SPEC_ID
from evidencegate.eval_data import CaseId, outcomes
from evidencegate.metrics import METRICS
from evidencegate.preflight import Check
from evidencegate.run_plan import Block, load_plan
from evidencegate.runner import EpisodeRecord
from evidencegate.store import EpisodeStore
from evidencegate.t014 import analysis as an
from evidencegate.t014.plans import load_pool, load_targets, plan_path
from evidencegate.t014.settings import (
    CALIBRATION,
    CONTROLS_IN_ORACLE,
    DEV_B,
    GATE_REPEAT,
    LOG_DIR,
    ORACLE,
    ORACLE_REPEATS,
    RESULT_NAMES,
    RESULTS_DIR,
)
from evidencegate.t014.splits import split_cases

TICKET = "014"


def result_path(root: Path, stage: int) -> Path:
    return root / RESULTS_DIR / RESULT_NAMES[stage]


def fingerprint_path(runs_dir: Path, stage: int) -> Path:
    return runs_dir / LOG_DIR / f"stage{stage}" / "fingerprint.json"


def _plain(x: Any) -> Any:
    if isinstance(x, Fraction):
        return {"exact": str(x), "float": float(x)}
    if isinstance(x, Mapping):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_plain(v) for v in x]
    return x


def result_bytes(payload: Mapping[str, Any]) -> bytes:
    text = json.dumps(_plain(payload), indent=1, sort_keys=True, ensure_ascii=False)
    return (text + "\n").encode("utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_previous(root: Path, stage: int) -> Check:
    if stage == 0:
        return Check("previous_stage", True, "stage 0 has no previous stage")
    path = result_path(root, stage - 1)
    sha_file = path.with_name(path.name + ".sha256")
    if not path.exists() or not sha_file.exists():
        return Check("previous_stage", False, f"missing {path} or its .sha256")
    actual, recorded = _sha(path), sha_file.read_text().split()[0]
    if actual != recorded:
        return Check("previous_stage", False, f"{path.name} sha256 {actual} != {recorded}")
    if stage == 1 and json.loads(path.read_text(encoding="utf-8")).get("go") is not True:
        return Check("previous_stage", False, "stage 0 is No-Go; Ticket 014 closes at stage 0")
    return Check("previous_stage", True, f"{path.name} sha256 {actual}")


def write_result(root: Path, stage: int, payload: Mapping[str, Any]) -> str:
    out = result_path(root, stage)
    if out.exists():
        raise FileExistsError(f"{out} already exists; a stage result is written once")
    body: dict[str, Any] = {"ticket": TICKET, "stage": stage, **payload}
    if stage > 0:
        prev = check_previous(root, stage)
        if not prev.ok:
            raise ValueError(prev.detail)
        path = result_path(root, stage - 1)
        body["previous"] = {"path": path.relative_to(root).as_posix(), "sha256": _sha(path)}
    data = result_bytes(body)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    out.with_name(out.name + ".sha256").write_text(f"{digest}  {out.name}\n")
    return digest


def load_result(root: Path, stage: int) -> dict[str, Any]:
    check = check_previous(root, stage + 1) if stage < 5 else None
    if check is not None and not check.ok and "No-Go" not in check.detail:
        raise ValueError(check.detail)
    data: dict[str, Any] = json.loads(result_path(root, stage).read_text(encoding="utf-8"))
    return data


# 기록 ---------------------------------------------------------------------------------


def store_path(runs_dir: Path, split: str, skill_id: str) -> Path:
    """`EpisodeStore.for_run`과 같은 경로(그 생성자는 파일을 읽으므로 경로만 계산한다)."""
    safe_policy_id = POLICY_ID.replace(":", "__")
    return runs_dir / split / f"{safe_policy_id}__{CONFIG_ID}" / TOOL_SPEC_ID / f"{skill_id}.jsonl"


def _stores(runs_dir: Path, plan: Sequence[Block]) -> list[Path]:
    return sorted(store_path(runs_dir, s, k) for s, k in {(b.split, b.skill_id) for b in plan})


def collect_records(runs_dir: Path, plan: Sequence[Block]) -> list[EpisodeRecord]:
    """계획의 기록(각 블록의 (케이스, 반복)이 빠짐없이). 빠지면 ValueError."""
    records: list[EpisodeRecord] = []
    for path in _stores(runs_dir, plan):
        if path.exists():
            records += EpisodeStore(path, repair=False).records()
    have = {
        (r.key.skill_id, r.key.repeat, (r.key.family, r.key.level, r.key.incident_seed))
        for r in records
    }
    missing = [(b.index, c) for b in plan for c in b.cases if (b.skill_id, b.repeat, c) not in have]
    if missing:
        raise ValueError(
            f"plan incomplete: {len(missing)} episodes missing, e.g. block {missing[0][0]} "
            f"{missing[0][1]}"
        )
    return records


def store_hashes(root: Path, runs_dir: Path, plan: Sequence[Block]) -> dict[str, str]:
    return {_rel(root, runs_dir, p): _sha(p) for p in _stores(runs_dir, plan)}


def _rel(root: Path, runs_dir: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return "runs/" + path.relative_to(runs_dir).as_posix()


def _check_stores(root: Path, runs_dir: Path, stage: int, plan: Sequence[Block]) -> None:
    recorded = load_result(root, stage)["stores"]
    now = store_hashes(root, runs_dir, plan)
    if now != recorded:
        changed = sorted(k for k in set(now) | set(recorded) if now.get(k) != recorded.get(k))
        raise ValueError(f"records changed since stage {stage}: {changed[:3]}")


def _plan(root: Path, stage: int) -> tuple[tuple[Block, ...], str]:
    path = root / plan_path(stage)
    return load_plan(path), _sha(path)


def _fingerprint(runs_dir: Path, stage: int) -> dict[str, Any] | None:
    path = fingerprint_path(runs_dir, stage)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _run_summary(
    root: Path, runs_dir: Path, stage: int, plan: Sequence[Block], sha: str
) -> dict[str, Any]:
    records = collect_records(runs_dir, plan)
    wanted = {(b.skill_id, b.repeat, c) for b in plan for c in b.cases}
    used = [
        r
        for r in records
        if (r.key.skill_id, r.key.repeat, (r.key.family, r.key.level, r.key.incident_seed))
        in wanted
    ]
    return {
        "plan": {"path": plan_path(stage).as_posix(), "sha256": sha, "blocks": len(plan)},
        "episodes": len(used),
        "format_error_episodes": sum(1 for r in used if r.trace.format_errors > 0),
        "stores": store_hashes(root, runs_dir, plan),
        "fingerprint": _fingerprint(runs_dir, stage),
    }


# 단계별 마무리 ------------------------------------------------------------------------


def _by_repeat(
    records: Sequence[EpisodeRecord], skill: str, split: str, metric: str, repeats: Sequence[int]
) -> dict[int, dict[CaseId, int]]:
    cases = split_cases(split)
    mine = [r for r in records if r.key.skill_id == skill]
    return {
        r: outcomes(mine, skill_id=skill, split=split, metric=metric, repeat=r, cases=cases)
        for r in repeats
    }


def _stage1(root: Path, runs_dir: Path, n_boot: int) -> dict[str, Any]:
    plan, sha = _plan(root, 1)
    summary = _run_summary(root, runs_dir, 1, plan, sha)
    records = collect_records(runs_dir, plan)
    values = an.thresholds(records, split_cases(CALIBRATION[0]), n_boot=n_boot)
    return {**summary, "thresholds": values, "settings": {"n_boot": n_boot}}


def _thresholds(root: Path) -> dict[str, dict[str, Fraction]]:
    data = load_result(root, 1)["thresholds"]
    return {m: {k: Fraction(v["exact"]) for k, v in row.items()} for m, row in data.items()}


def _gate_rows(
    root: Path, runs_dir: Path, n_boot: int
) -> tuple[dict[str, list[an.GateRow]], tuple[Block, ...]]:
    plan, _ = _plan(root, 2)
    _check_stores(root, runs_dir, 2, plan)
    records = collect_records(runs_dir, plan)
    th = _thresholds(root)
    rows: dict[str, list[an.GateRow]] = {}
    for metric in METRICS:
        s0 = _by_repeat(records, S0_SKILL_ID, DEV_B[0], metric, (GATE_REPEAT,))[GATE_REPEAT]
        rows[metric] = []
        for c in load_pool(root):
            cand = _by_repeat(records, c["skill_id"], DEV_B[0], metric, (GATE_REPEAT,))
            rows[metric].append(
                an.gate_row(
                    c["name"],
                    metric,
                    cand[GATE_REPEAT],
                    s0,
                    h=th[metric]["h"],
                    delta=th[metric]["delta"],
                    n_boot=n_boot,
                )
            )
    return rows, plan


def _stage3(root: Path, runs_dir: Path, n_boot: int, controls: bool | None) -> dict[str, Any]:
    if controls is None:
        raise ValueError(
            "CONTROLS_IN_ORACLE is not fixed; decide preregistration 13.1 before stage 3"
        )
    rows, _ = _gate_rows(root, runs_dir, n_boot)
    targets = {"proposer": an.oracle_targets(rows[an.CD]), "controls": controls}
    target_bytes = json.dumps(targets, sort_keys=True, separators=(",", ":")).encode()
    return {
        "gates": {m: [asdict(r) for r in rs] for m, rs in rows.items()},
        "oracle_targets": targets,
        "oracle_targets_sha256": hashlib.sha256(target_bytes).hexdigest(),
        "settings": {"n_boot": n_boot},
    }


def _label_dict(label: an.OracleLabel) -> dict[str, Any]:
    return asdict(label)


def _stage5(root: Path, runs_dir: Path, n_boot: int) -> dict[str, Any]:
    rows, _ = _gate_rows(root, runs_dir, n_boot)
    targets = load_targets(root)
    skipped = bool(load_result(root, 4).get("skipped"))
    if skipped:
        plan4: tuple[Block, ...] = ()
        records: list[EpisodeRecord] = []
    else:
        plan4, _ = _plan(root, 4)
        _check_stores(root, runs_dir, 4, plan4)
        records = collect_records(runs_dir, plan4)
    skills = {b.name: (b.skill_id, b.group) for b in plan4}
    metrics: dict[str, Any] = {}
    labels_by_metric: dict[str, dict[str, an.OracleLabel]] = {}
    for metric in METRICS:
        s0 = _by_repeat(records, S0_SKILL_ID, ORACLE[0], metric, ORACLE_REPEATS) if plan4 else {}
        labels = {
            name: an.oracle_label(
                _by_repeat(records, skill_id, ORACLE[0], metric, ORACLE_REPEATS), s0, n_boot=n_boot
            )
            for name, (skill_id, group) in sorted(skills.items())
            if group != "s0"
        }
        labels_by_metric[metric] = labels
        proposer = {n: x for n, x in labels.items() if skills[n][1] == "proposer"}
        labeled = [r for r in rows[metric] if r.name in proposer]
        metrics[metric] = {
            "labels": {n: _label_dict(x) for n, x in labels.items()},
            "no_oracle_label": sorted(r.name for r in rows[metric] if r.name not in proposer),
            "counts": {
                "m": sum(1 for x in proposer.values() if x.m),
                "m_point": sum(1 for x in proposer.values() if x.m_point),
                "m_uncertain": sum(1 for x in proposer.values() if x.m_uncertain),
                "useful": sum(1 for x in proposer.values() if x.useful),
                "useful_uncertain": sum(1 for x in proposer.values() if x.useful_uncertain),
                "clean_useful": sum(1 for x in proposer.values() if x.clean_useful),
            },
            "gates_on_labeled": an.rq14_3(labeled, proposer),
            "unlabeled_accepted": {
                g: sorted(r.name for r in rows[metric] if r.accepted(g) and r.name not in proposer)
                for g in an.GATES
            },
        }
    cd = {n: x for n, x in labels_by_metric[an.CD].items() if skills[n][1] == "proposer"}
    return {
        "oracle_targets": {"proposer": list(targets.proposer), "controls": targets.controls},
        "stage4_skipped": skipped,
        "rq14_1": an.rq14_1(rows[an.CD], cd),
        "rq14_2": an.rq14_2(rows[an.CD], cd),
        "rq14_3": an.rq14_3(rows[an.CD], cd),
        "metrics": metrics,
        "settings": {"n_boot": n_boot},
        "fingerprints": {str(s): _fingerprint(runs_dir, s) for s in (0, 1, 2, 4)},
    }


def _no_targets(root: Path) -> bool:
    targets = load_targets(root)
    return not targets.proposer and not targets.controls


def finalize(
    stage: int,
    *,
    root: Path,
    runs_dir: Path,
    n_boot: int = N_BOOT,
    controls: bool | None = CONTROLS_IN_ORACLE,
) -> dict[str, Any]:
    """단계 1~5의 결과를 만들어 한 번 쓰고, 쓴 내용을 돌려준다. 단계 0은 `t014.pool`."""
    if result_path(root, stage).exists():
        raise FileExistsError(f"{result_path(root, stage)} already exists")
    prev = check_previous(root, stage)
    if not prev.ok:
        raise ValueError(prev.detail)
    if stage == 1:
        payload = _stage1(root, runs_dir, n_boot)
    elif stage == 4 and _no_targets(root):
        payload = {"skipped": True, "reason": "no Oracle targets at stage 3 (preregistration 8)"}
    elif stage in (2, 4):
        plan, sha = _plan(root, stage)
        payload = _run_summary(root, runs_dir, stage, plan, sha)
    elif stage == 3:
        payload = _stage3(root, runs_dir, n_boot, controls)
    elif stage == 5:
        payload = _stage5(root, runs_dir, n_boot)
    else:
        raise ValueError(f"stage {stage} is finalized by evidencegate.t014.pool")
    write_result(root, stage, payload)
    data: dict[str, Any] = json.loads(result_path(root, stage).read_text(encoding="utf-8"))
    return data
