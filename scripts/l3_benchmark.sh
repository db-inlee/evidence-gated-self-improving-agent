#!/usr/bin/env bash
# Ticket 006 실행 (§3, §6). 사용자가 직접 실행한다.
#
#   bash scripts/l3_benchmark.sh preflight   # 본 실행 점검만(L3 계획 파일 점검 포함, 에피소드는 돌지 않음)
#   bash scripts/l3_benchmark.sh run         # 본 실행 7,200개(블록 24개), 이어서 실행; `run N`은 블록 N개 뒤 멈춤
#   bash scripts/l3_benchmark.sh status      # 본 실행 진행 확인 한 줄(CD·ES는 내지 않음)
#   bash scripts/l3_benchmark.sh check       # L3-check: S0 x L3 30개(반복 0). 운영 지표만 출력
#   bash scripts/l3_benchmark.sh summary     # 저장된 L3-check의 운영 지표(CD·ES는 내지 않음)
#
# - 실행 전 점검(Ticket 006 사전 등록 해시, 본 실행은 L3 계획 해시 포함)과 병렬 탐침이 하나라도
#   실패하면 아무것도 돌지 않는다(종료 1).
# - 중단 규칙은 Ticket 005와 같다. 걸리면 이유와 마지막 레코드를 출력하고 종료 2.
# - 이어서 실행: 같은 명령을 다시 치면 저장된 에피소드는 건너뛴다.
# - 기록: runs/l3_oracle/…, runs/l3_check/…, 세션 로그 runs/ticket006/sessions.jsonl
# - run·check는 caffeinate로 잠자기를 막는다(-d 화면, -i 유휴, -m 디스크, -s 전원 연결 시 시스템).
set -euo pipefail
cd "$(dirname "$0")/.."
CLI=(uv run python -m evidencegate.l3_bench)

usage() {
  sed -n '4,8p' "$0" >&2
  exit 64
}

case "${1:-}" in
  preflight) "${CLI[@]}" preflight ;;
  run)
    if [ $# -ge 2 ]; then
      caffeinate -dims "${CLI[@]}" run --max-blocks "$2"
    else
      caffeinate -dims "${CLI[@]}" run
    fi
    ;;
  status) "${CLI[@]}" status ;;
  check) caffeinate -dims "${CLI[@]}" check ;;
  summary) "${CLI[@]}" summary ;;
  *) usage ;;
esac
