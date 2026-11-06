"""Ticket 009b 실행 전 점검(티켓 §5의 4, 사전 등록 2.1·4.2).

009 점검(`t009.preflight`)은 고치지 않는다.

proposer 본 호출 전(`proposer_preflight`):

| 점검 | 기준 |
| preregistration_009b | `ticket-009b.txt` 파일 전체의 sha256이 `ticket-009b.sha256` 기록과 같음 |
| prereg_tag | 태그 `t009b-prereg`가 HEAD의 조상(`git merge-base --is-ancestor`) |
| thresholds | `results/ticket-009/thresholds.json` sha256이 사전 등록 값(`d030b173…3520`) |
| export | 009 Dev-A 내보내기가 `.sha256`과 같고 사전 등록 값(`4f9afa27…4c6d`)과 같음 |
| strata | 실패 층과 호출 0~119의 사례 추출(`t009.proposer_inputs.check_sampling`) |
| prompt | 프롬프트 sha256 `58ffd9f0…e2c6` |
| paths | 원장·출력 폴더가 v1·009 경로가 아님 |
| outputs | 출력 폴더에 결과가 없거나, 있으면 같은 고정 설정(`run_options.json`)의 이어 부르기 |
| power | 전원 AC, 덮개 열림, caffeinate(`evidencegate.power`) |

본 실행(Dev-B·Oracle) 전에는 009의 9개 점검(`t009.preflight.run_preflight_009`, 계획 점검은 009b
계획)에 `main_run_extra_checks`의 셋(preregistration_009b, prereg_tag, thresholds)을 더한다.
"""

import hashlib
import json
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

from evidencegate.power import Run, power_problems, system_run
from evidencegate.preflight import Check
from evidencegate.proposer import prompt_sha256
from evidencegate.proposer_inputs import load_proposer_cases
from evidencegate.t009.preflight import recorded_sha256
from evidencegate.t009.propose import PROMPT_SHA256, Paths
from evidencegate.t009.proposer_inputs import check_sampling
from evidencegate.t009b.settings import (
    EXPORT_SHA256,
    FIXED_RUN_OPTIONS,
    MAX_CALLS,
    PREREG_FILE,
    PREREG_SHA_FILE,
    PREREG_TAG,
    REJECTED_PATHS,
    RUN_OPTIONS_FILE,
    THRESHOLDS_PATH,
    THRESHOLDS_SHA256,
)

Git = Callable[[Sequence[str]], int]


def system_git(argv: Sequence[str]) -> int:
    return subprocess.run(list(argv), capture_output=True, check=False).returncode


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_preregistration_009b(root: Path) -> Check:
    text, sha_file = root / PREREG_FILE, root / PREREG_SHA_FILE
    if not text.exists() or not sha_file.exists():
        return Check("preregistration_009b", False, f"missing {text} or {sha_file}")
    recorded, actual = recorded_sha256(sha_file), _sha256(text)
    if actual != recorded:
        return Check("preregistration_009b", False, f"sha256 {actual} does not match {recorded}")
    return Check("preregistration_009b", True, f"009b sha256 {actual}")


def check_prereg_tag(root: Path, git: Git = system_git) -> Check:
    code = git(["git", "-C", str(root), "merge-base", "--is-ancestor", PREREG_TAG, "HEAD"])
    if code == 0:
        return Check("prereg_tag", True, f"{PREREG_TAG} is an ancestor of HEAD")
    if code == 1:
        return Check("prereg_tag", False, f"{PREREG_TAG} is not an ancestor of HEAD")
    return Check("prereg_tag", False, f"cannot check {PREREG_TAG} (git exit {code})")


def check_thresholds(root: Path) -> Check:
    path = root / THRESHOLDS_PATH
    if not path.exists():
        return Check("thresholds", False, f"missing {path}")
    actual = _sha256(path)
    return Check("thresholds", actual == THRESHOLDS_SHA256, f"thresholds sha256 {actual}")


def check_export(paths: Paths, expected: str = EXPORT_SHA256) -> Check:
    sha_file = paths.inputs.with_name(paths.inputs.name + ".sha256")
    if not paths.inputs.exists() or not sha_file.exists():
        return Check("export", False, f"missing {paths.inputs} or its .sha256")
    actual = _sha256(paths.inputs)
    recorded = sha_file.read_text().split()[0]
    ok = actual == recorded == expected
    return Check("export", ok, f"export sha256 {actual} (recorded {recorded}, expected {expected})")


def check_strata(paths: Paths) -> Check:
    if not paths.inputs.exists():
        return Check("strata", False, f"missing {paths.inputs}")
    sampling = check_sampling(load_proposer_cases(paths.inputs), calls=MAX_CALLS)
    return Check("strata", sampling.ok, sampling.detail)


def check_prompt() -> Check:
    actual = prompt_sha256()
    return Check("prompt", actual == PROMPT_SHA256, f"prompt sha256 {actual}")


def check_paths(root: Path, paths: Paths) -> Check:
    rejected = {(root / p).resolve() for p in REJECTED_PATHS}
    bad = [str(p) for p in (paths.ledger, paths.out_dir) if p.resolve() in rejected]
    if bad:
        return Check("paths", False, f"v1 or 009 path: {', '.join(bad)}")
    return Check("paths", True, f"ledger {paths.ledger}, out {paths.out_dir}")


def check_outputs(paths: Paths, fixed: dict[str, object] = FIXED_RUN_OPTIONS) -> Check:
    calls = sorted(paths.out_dir.glob("call-*.json")) if paths.out_dir.exists() else []
    if not calls:
        return Check("outputs", True, f"no saved calls in {paths.out_dir} (new run)")
    options_path = paths.out_dir / RUN_OPTIONS_FILE
    if not options_path.exists():
        return Check("outputs", False, f"{len(calls)} calls without {RUN_OPTIONS_FILE}")
    recorded = json.loads(options_path.read_text(encoding="utf-8"))
    if recorded != fixed:
        return Check("outputs", False, f"{RUN_OPTIONS_FILE} {recorded} is not the 009b setting")
    noun = "call" if len(calls) == 1 else "calls"
    return Check("outputs", True, f"resume: {len(calls)} {noun} saved with the 009b setting")


def check_power_now(run: Run = system_run) -> Check:
    problems = power_problems(run)
    return Check("power", not problems, "; ".join(problems) or "AC power, lid open, caffeinate")


def proposer_preflight(
    root: Path,
    paths: Paths,
    *,
    power_run: Run = system_run,
    git: Git = system_git,
    expected_export_sha256: str = EXPORT_SHA256,
) -> list[Check]:
    return [
        check_preregistration_009b(root),
        check_prereg_tag(root, git),
        check_thresholds(root),
        check_export(paths, expected_export_sha256),
        check_strata(paths),
        check_prompt(),
        check_paths(root, paths),
        check_outputs(paths),
        check_power_now(power_run),
    ]


def main_run_extra_checks(root: Path, *, git: Git = system_git) -> list[Check]:
    """본 실행 점검에서 009의 9개에 더하는 셋."""
    return [check_preregistration_009b(root), check_prereg_tag(root, git), check_thresholds(root)]
