#!/usr/bin/env bash
# proposer 실측과 풀 생성 진입점 (Ticket 004 §4).
#
#   bash scripts/proposer_probe.sh models
#   bash scripts/proposer_probe.sh probe <model> <input_usd_per_mtok> <output_usd_per_mtok>
#   bash scripts/proposer_probe.sh pool  <model> <input_usd_per_mtok> <output_usd_per_mtok> <calls>
#
# - API 키는 환경 변수 OPENAI_API_KEY에서만 읽는다. 이 스크립트는 키를 출력하지 않는다.
# - probe: 호출 1회, 결과는 runs/proposer/probe/<prompt_sha256 앞 8자>/<model>/call-NNN.json
#   (풀에 넣지 않음). 프롬프트가 바뀌면 폴더가 갈린다.
# - pool: 호출 <calls>회, 결과는 runs/proposer/pool/call-NNN.json. 다시 돌리면 이어서 부른다.
#   티켓 §5·§7: 폴더 전체 호출은 최대 18회(목표 12 + 추가 6), 풀(유효·비중복 후보)이 12개가 되면
#   더 부르지 않는다. 끝에 요약 줄(풀 크기, 중복, 형식 오류, 무효 사유별 개수)을 출력한다.
# - 원장 runs/proposer/ledger.jsonl은 probe와 pool이 함께 쓴다(누적 비용 $1.5 상한이 이어짐).
# - 중단(예산·할당량·인증·서버 오류·형식 오류율)은 종료 코드 2. 재개는 보고서에 결정을 남긴 뒤.
# 추가 옵션(--max-completion-tokens, --temperature, --reasoning-effort)은 끝에 붙이면 그대로 전달된다.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${OPENAI_API_KEY:?OPENAI_API_KEY is not set (export it first)}"

LEDGER=runs/proposer/ledger.jsonl
CLI=(uv run python -m evidencegate.propose_cli)

show_last_ledger_entry() {
  [ -f "$LEDGER" ] || return 0
  tail -n 1 "$LEDGER" | uv run python -c '
import json, sys
e = json.loads(sys.stdin.read())
keys = ("call_index", "model", "status", "error_kind", "input_tokens", "cached_input_tokens",
        "output_tokens", "reasoning_tokens", "cost_usd", "cumulative_usd", "duration_s")
print("ledger:", json.dumps({k: e.get(k) for k in keys}))'
}

usage() {
  sed -n '4,6p' "$0" >&2
  exit 64
}

[ $# -ge 1 ] || usage
command=$1
shift
case "$command" in
  models)
    "${CLI[@]}" models
    ;;
  probe)
    [ $# -ge 3 ] || usage
    model=$1 in_usd=$2 out_usd=$3
    shift 3
    prompt8=$(uv run python -c 'from evidencegate.proposer import prompt_sha256; print(prompt_sha256()[:8])')
    out_dir="runs/proposer/probe/$prompt8/$model"
    echo "out-dir: $out_dir"
    status=0
    "${CLI[@]}" propose --model "$model" \
      --input-usd-per-mtok "$in_usd" --output-usd-per-mtok "$out_usd" \
      --out-dir "$out_dir" --calls 1 --ledger "$LEDGER" "$@" || status=$?
    show_last_ledger_entry
    exit "$status"
    ;;
  pool)
    [ $# -ge 4 ] || usage
    model=$1 in_usd=$2 out_usd=$3 calls=$4
    shift 4
    status=0
    "${CLI[@]}" propose --model "$model" \
      --input-usd-per-mtok "$in_usd" --output-usd-per-mtok "$out_usd" \
      --out-dir runs/proposer/pool --calls "$calls" --ledger "$LEDGER" \
      --max-total-calls 18 --stop-when-pool 12 "$@" || status=$?
    show_last_ledger_entry
    exit "$status"
    ;;
  *)
    usage
    ;;
esac
