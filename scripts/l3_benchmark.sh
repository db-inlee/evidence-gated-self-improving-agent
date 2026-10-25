#!/usr/bin/env bash
# Ticket 006 실행 (§3, §6). 사용자가 직접 실행한다.
#
#   bash scripts/l3_benchmark.sh preflight   # 실행 전 점검만(Ollama에 버전·모델 정보만 묻고, 에피소드는 돌지 않음)
#   bash scripts/l3_benchmark.sh check       # L3-check: S0 x L3 30개(반복 0). 운영 지표만 출력
#   bash scripts/l3_benchmark.sh summary     # 저장된 L3-check의 운영 지표(CD·ES는 내지 않음)
#
# - 실행 전 점검(Ticket 006 사전 등록 해시 포함)과 병렬 탐침이 하나라도 실패하면 아무것도 돌지 않는다(종료 1).
# - 중단 규칙은 Ticket 005와 같다. 걸리면 이유와 마지막 레코드를 출력하고 종료 2.
# - 이어서 실행: 같은 명령을 다시 치면 저장된 에피소드는 건너뛴다.
# - 기록: runs/l3_check/…, 세션 로그 runs/ticket006/sessions.jsonl
# - check는 caffeinate로 잠자기를 막는다.
set -euo pipefail
cd "$(dirname "$0")/.."
CLI=(uv run python -m evidencegate.l3_bench)

case "${1:-}" in
  preflight) "${CLI[@]}" preflight ;;
  check) caffeinate -dims "${CLI[@]}" check ;;
  summary) "${CLI[@]}" summary ;;
  *)
    sed -n '4,6p' "$0" >&2
    exit 64
    ;;
esac
