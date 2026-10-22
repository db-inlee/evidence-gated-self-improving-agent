#!/usr/bin/env bash
# Ticket 005 Gate Benchmark 실행 (§2·§4·§5). 사용자가 직접 실행한다.
#
#   bash scripts/gate_benchmark.sh preflight   # 실행 전 점검만(Ollama에 버전·모델 정보만 묻고, 에피소드는 돌지 않음)
#   bash scripts/gate_benchmark.sh pilot       # 파일럿: 계획의 첫 두 블록(cand-000·mixed-scale-on-traffic의 Dev-A, 120개)만
#   bash scripts/gate_benchmark.sh run         # 본 실행: 남은 블록을 끝까지(이어서 실행)
#   bash scripts/gate_benchmark.sh run N       # 남은 블록 중 N개만 돌고 멈춤(나눠서 실행)
#   bash scripts/gate_benchmark.sh status      # 진행 확인 한 줄(CD·ES는 내지 않음)
#
# - 세션마다 실행 전 점검과 병렬 탐침을 하고, 하나라도 실패하면 아무것도 돌지 않는다(종료 1).
# - 중단 규칙에 걸리면 이유와 마지막 레코드를 출력하고 종료 2. 재개는 결정과 이유를 보고서에 남긴 뒤.
# - 이어서 실행: 같은 명령을 다시 치면 저장된 에피소드는 건너뛴다.
# - pilot·run은 caffeinate로 잠자기를 막는다(-d 화면, -i 유휴, -m 디스크, -s 전원 연결 시 시스템).
set -euo pipefail
cd "$(dirname "$0")/.."
CLI=(uv run python -m evidencegate.bench_runner)

usage() {
  sed -n '4,8p' "$0" >&2
  exit 64
}

case "${1:-}" in
  preflight)
    "${CLI[@]}" preflight
    ;;
  pilot)
    caffeinate -dims "${CLI[@]}" run --pilot
    ;;
  run)
    if [ $# -ge 2 ]; then
      caffeinate -dims "${CLI[@]}" run --max-blocks "$2"
    else
      caffeinate -dims "${CLI[@]}" run
    fi
    ;;
  status)
    "${CLI[@]}" status
    ;;
  *)
    usage
    ;;
esac
