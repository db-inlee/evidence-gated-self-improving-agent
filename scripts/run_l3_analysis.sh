#!/usr/bin/env bash
# Ticket 006 L3 분석을 한 번 실행한다(§6 순서 6, §10 해석 3). L3 분석 코드 고정 커밋을 직접 받는다.
#
#   bash scripts/run_l3_analysis.sh <고정 커밋>                 # 확인 뒤 분석 실행(결과: results/ticket-006/)
#   bash scripts/run_l3_analysis.sh --check-only <고정 커밋>    # 확인만
#   bash scripts/run_l3_analysis.sh <고정 커밋> --out-dir DIR   # 출력 폴더를 바꿀 때
#
# 실행 전에 확인하고, 하나라도 어긋나면 아무것도 실행하지 않는다(종료 1).
# - 받은 것이 커밋이고, HEAD의 조상(또는 HEAD)이다.
# - 그 커밋 뒤로 분석 모듈(src/evidencegate/, src/opsworld/)이 커밋으로도, 작업 트리에서도 바뀌지 않았다.
# - L3 계획 파일이 자기 sha256 기록과 같다.
# - Ticket 005 결과 파일이 자기 sha256 기록과 같다(분석은 그 결과를 같은 레코드로 다시 계산해 바이트까지
#   같은지도 확인한다).
# - 출력 폴더에 analysis.json이 없다(분석은 한 번만).
# 분석에는 받은 커밋의 짧은 해시와 L3 계획 sha256을 기록한다.
set -euo pipefail
cd "$(dirname "$0")/.."

ANALYSIS_PATHS=(src/evidencegate src/opsworld)
NOISE=runs/calibration/ollama__gpt-oss__20b__7c66ee5bf13c/e9b4996f7529/bf65e569ae37.jsonl
PLAN=plans/ticket-006-l3-plan.json
T005=results/ticket-005/analysis.json
out_dir=results/ticket-006
check_only=0

usage() {
  sed -n '4,6p' "$0" >&2
  exit 64
}

fail() {
  echo "error: $*" >&2
  exit 1
}

commit_arg=""
while [ $# -gt 0 ]; do
  case "$1" in
    --check-only) check_only=1 ;;
    --out-dir) shift; [ $# -gt 0 ] || usage; out_dir=$1 ;;
    -*) usage ;;
    *) [ -z "$commit_arg" ] || usage; commit_arg=$1 ;;
  esac
  shift
done
[ -n "$commit_arg" ] || usage

commit=$(git rev-parse --verify --quiet "${commit_arg}^{commit}") || fail "not a commit: $commit_arg"
git merge-base --is-ancestor "$commit" HEAD || fail "$commit_arg is not an ancestor of HEAD"

changed=$(git diff --name-only "$commit" HEAD -- "${ANALYSIS_PATHS[@]}")
[ -z "$changed" ] || fail "analysis modules changed since $commit_arg:"$'\n'"$changed"
dirty=$(git status --porcelain --untracked-files=all -- "${ANALYSIS_PATHS[@]}")
[ -z "$dirty" ] || fail "uncommitted changes in the analysis modules:"$'\n'"$dirty"

[ -f "$PLAN" ] && [ -f "$PLAN.sha256" ] || fail "missing $PLAN or $PLAN.sha256"
plan_sha=$(cut -d' ' -f1 "$PLAN.sha256")
actual_sha=$(shasum -a 256 "$PLAN" | cut -d' ' -f1)
[ "$plan_sha" = "$actual_sha" ] || fail "$PLAN sha256 $actual_sha does not match the record $plan_sha"

t005_sha=$(cut -d' ' -f1 "$T005.sha256")
t005_actual=$(shasum -a 256 "$T005" | cut -d' ' -f1)
[ "$t005_sha" = "$t005_actual" ] || fail "Ticket 005 result $T005 sha256 $t005_actual does not match the record $t005_sha"
[ ! -e "$out_dir/analysis.json" ] || fail "$out_dir/analysis.json exists; the analysis runs once"

short=$(git rev-parse --short "$commit")
echo "ok: $short is an ancestor of HEAD; ${ANALYSIS_PATHS[*]} unchanged since; L3 plan sha256 $plan_sha; Ticket 005 result sha256 $t005_sha"
[ "$check_only" -eq 0 ] || exit 0

uv run python -m evidencegate.l3_analysis --runs-dir runs --noise "$NOISE" --out-dir "$out_dir" \
  --analysis-commit "$short" --plan-sha256 "$plan_sha"
