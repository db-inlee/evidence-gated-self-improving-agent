"""Ticket 009 Dev-A 실패 궤적 내보내기(사전 등록 2·2.1절). v1 `proposer_inputs`는 고치지 않는다.

    uv run python -m evidencegate.t009.proposer_inputs

- v1 `export_dev_a`는 v1 Dev-A 케이스만 받는다. 그래서 009 Dev-A(`t009_dev_a`, 칸당 10,
  500,000~500,009)용으로 케이스 확인만 바꾼다. 나머지는 v1과 같다: 실패(CD 거짓 또는 ES 거짓)만,
  (family, seed, level) 정렬, v1 `_to_case`(숨은 값·UNNECESSARY_INTERVENTION 제외, 내용 해시 ID),
  v1 섞기 seed.
- 확인: 한 실행 조건, 009 Dev-A 케이스만, 반복 0만, 60 케이스가 빠짐·중복 없이.
- 층 확인(2.1절): evidence-only(CD 참, ES 거짓)와 wrong(CD 거짓)이 각각 `min_per_stratum` 이상이고,
  호출 0~79의 추출(v1 `proposer.sample_cases`)이 모두 되는지 본다. 안 되면 종료 코드 1이다.
- 결과는 `runs/proposer_inputs-009/…json`(git 무시)이고 `<결과>.sha256`에 "sha256  이름"으로
  고정한다. 이미 있는 파일과 바이트가 다르면 거부한다.
"""

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from evidencegate.analysis import REPO_ROOT
from evidencegate.eval_constants import CONFIG_ID, POLICY_ID, S0_SKILL_ID, TOOL_SPEC_ID
from evidencegate.proposer import ProposerConfig, sample_cases
from evidencegate.proposer_inputs import (
    EXPORT_SHUFFLE_SEED,
    ProposerCase,
    _to_case,
)
from evidencegate.runner import Case, EpisodeRecord
from evidencegate.store import EpisodeStore
from evidencegate.suites import calibration
from evidencegate.t009.splits import DEV_A, LEVELS

MAX_CALLS = 80  # 2.1절: 호출 상한
DEV_A_SPLIT = DEV_A[0]
OUT_PATH = Path(
    "runs/proposer_inputs-009/dev_a__ollama__gpt-oss__20b__7c66ee5bf13c__bf65e569ae37.json"
)
# v1 proposer_config 기본값(2.1절). 모델·단가는 추출에 쓰이지 않는다.
_SAMPLING = ProposerConfig(
    model="-", max_completion_tokens=1, input_usd_per_mtok=0, output_usd_per_mtok=0
)
CASES_PER_CALL = _SAMPLING.cases_per_call
MIN_PER_STRATUM = _SAMPLING.min_per_stratum


def dev_a_cases() -> list[Case]:
    _, offset, per_cell = DEV_A
    return calibration(per_cell, seed_offset=offset, levels=LEVELS)


def _check(records: Sequence[EpisodeRecord]) -> None:
    runs = {(r.key.policy_id, r.key.config_id, r.key.tool_spec_id, r.key.skill_id) for r in records}
    if len(runs) > 1:
        raise ValueError(f"records from mixed runs: {sorted(runs)}")
    allowed = {(c.family.value, c.incident_seed, c.level.value) for c in dev_a_cases()}
    for r in records:
        if (r.key.family, r.key.incident_seed, r.key.level) not in allowed:
            raise ValueError(f"record is not a Ticket 009 Dev-A case: {r.key}")
        if r.key.repeat != 0:
            raise ValueError(f"Dev-A export uses repeat 0 only, got {r.key}")
    counts = Counter((r.key.family, r.key.incident_seed, r.key.level) for r in records)
    duplicates = [k for k, n in counts.items() if n > 1]
    if duplicates:
        raise ValueError(f"{len(duplicates)} duplicate Dev-A records, e.g. {duplicates[0]}")
    missing = len(allowed) - len(counts)
    if missing:
        raise ValueError(f"{missing} missing Dev-A records")


def export_dev_a_009(
    records: Sequence[EpisodeRecord],
    *,
    shuffle: bool = True,
    shuffle_seed: int = EXPORT_SHUFFLE_SEED,
) -> list[ProposerCase]:
    _check(records)
    failures = [r for r in records if not (r.correct_disposition and r.evidence_safe)]
    ordered = sorted(failures, key=lambda r: (r.key.family, r.key.incident_seed, r.key.level))
    cases = [_to_case(r) for r in ordered]
    if shuffle:
        random.Random(shuffle_seed).shuffle(cases)
    return cases


def strata(cases: Sequence[ProposerCase]) -> dict[str, int]:
    return {
        "failures": len(cases),
        "evidence_only": sum(1 for c in cases if c.correct_disposition and not c.evidence_safe),
        "wrong": sum(1 for c in cases if not c.correct_disposition),
    }


@dataclass(frozen=True)
class SamplingCheck:
    ok: bool
    calls: int
    distinct_cases_shown: int
    detail: str


def check_sampling(cases: Sequence[ProposerCase], calls: int = MAX_CALLS) -> SamplingCheck:
    """호출 0..calls-1의 추출을 LLM 없이 미리 해 본다(v1 `sample_cases`, 같은 seed 규칙)."""
    shown: set[str] = set()
    for call_index in range(calls):
        try:
            picked = sample_cases(cases, call_index, CASES_PER_CALL, MIN_PER_STRATUM)
        except ValueError as error:
            return SamplingCheck(False, call_index, len(shown), f"call {call_index}: {error}")
        shown.update(c.export_id for c in picked)
    return SamplingCheck(True, calls, len(shown), f"calls 0-{calls - 1} sample without error")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evidencegate-t009-proposer-inputs")
    parser.add_argument("--runs-dir", type=Path, default=REPO_ROOT / "runs")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / OUT_PATH)
    args = parser.parse_args(argv)
    path = EpisodeStore.for_run(
        args.runs_dir,
        suite=DEV_A_SPLIT,
        policy_id=POLICY_ID,
        config_id=CONFIG_ID,
        tool_spec_id=TOOL_SPEC_ID,
        skill_id=S0_SKILL_ID,
    ).path
    try:
        cases = export_dev_a_009(EpisodeStore(path, repair=False).records())
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    counts = strata(cases)
    sampling = check_sampling(cases)
    print(
        f"strata: failures {counts['failures']}, evidence-only {counts['evidence_only']}, "
        f"wrong {counts['wrong']}; sampling: {sampling.detail}, "
        f"distinct cases shown {sampling.distinct_cases_shown}"
    )
    if not sampling.ok:
        print("error: strata too thin for the proposer; do not call it", file=sys.stderr)
        return 1
    out: Path = args.out
    payload = (
        json.dumps([c.to_json() for c in cases], indent=2, sort_keys=True, ensure_ascii=False)
        + "\n"
    ).encode("utf-8")  # v1 `save_proposer_cases`와 같은 바이트
    digest = hashlib.sha256(payload).hexdigest()
    if out.exists():
        if out.read_bytes() != payload:
            print(f"error: {out} exists and differs from the export built now", file=sys.stderr)
            return 1
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(payload)
    out.with_name(out.name + ".sha256").write_text(f"{digest}  {out.name}\n")
    print(f"export {out}: {len(cases)} cases, sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
