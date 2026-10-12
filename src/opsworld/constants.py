"""OpsWorld v0 임계값과 생성기 수치 범위.

티켓 표와 1:1로 맞춘다. 값을 바꾸려면 티켓 문서를 먼저 고치고 이유를 남긴다.
시간은 모두 분 단위 정수, 오류율은 0~1 비율, CPU는 0~100 퍼센트다.
"""

from typing import Final

# ---------------------------------------------------------------------------
# 술어 임계값 (Ticket 001 §5)
# ---------------------------------------------------------------------------

# incident_confirmed: 현재 p95 ≥ 배수 x baseline 또는 현재 오류율 ≥ baseline + 증가분
INCIDENT_P95_MULTIPLIER: Final = 2.0
INCIDENT_ERROR_RATE_DELTA: Final = 0.02

# recent_deployment_found: |deployed_at - 저하 시작| ≤ 창 (이전·이후 모두)
RECENT_DEPLOYMENT_WINDOW_MIN: Final = 30

# deploy_before_regression: 저하 시작 이전 0~창 분 사이의 배포
DEPLOY_BEFORE_REGRESSION_WINDOW_MIN: Final = 15

# traffic_ruled_out: current_rps < 배수 x baseline_rps
TRAFFIC_RULED_OUT_MULTIPLIER: Final = 1.5

# traffic_spike_confirmed: current_rps ≥ 배수 x baseline_rps
TRAFFIC_SPIKE_MULTIPLIER: Final = 3.0

# capacity_saturated: CPU ≥ 임계값 (%)
CPU_SATURATION_PERCENT: Final = 85.0

# 수치 비교 허용 오차. 도구 출력은 반올림된 십진 값이라, 경계에 놓인 값이 이진 부동소수점
# 오차로 반대쪽에 떨어지지 않도록 차이가 이 값 이내면 같은 값으로 본다. 임계값은 바꾸지 않는다.
FLOAT_TOLERANCE: Final = 1e-9

# ---------------------------------------------------------------------------
# 생성기 수치 범위 (L1). 술어 임계값과 여유를 두고 떨어뜨려 둔다.
# ---------------------------------------------------------------------------

SERVICE_IDS: Final = ("checkout-api", "orders-api", "search-api", "profile-api", "catalog-api")
DEPENDENCY_IDS: Final = (
    "payment-provider",
    "auth-provider",
    "geo-lookup",
    "email-gateway",
    "fraud-scoring",
)
DEPENDENCY_COUNT_RANGE: Final = (2, 3)

# 저하 시작 시각 (분)
REGRESSION_START_RANGE: Final = (120, 600)

# 서비스 baseline
BASELINE_P95_MS_RANGE: Final = (80.0, 250.0)
BASELINE_ERROR_RATE_RANGE: Final = (0.001, 0.01)
REPLICAS_RANGE: Final = (3, 8)

# 장애 중 지연·오류 (모든 family 공통, incident_confirmed가 항상 성립하도록)
DEGRADED_P95_MULTIPLIER_RANGE: Final = (2.5, 6.0)
DEGRADED_ERROR_RATE_DELTA_RANGE: Final = (0.03, 0.15)

# CPU (%)
NORMAL_CPU_RANGE: Final = (25.0, 65.0)
SATURATED_CPU_RANGE: Final = (88.0, 99.0)

# 트래픽
BASELINE_RPS_RANGE: Final = (200.0, 2000.0)
NORMAL_TRAFFIC_MULTIPLIER_RANGE: Final = (0.85, 1.25)
SPIKE_TRAFFIC_MULTIPLIER_RANGE: Final = (3.3, 6.0)

# bad_deployment: 버그 있는 배포는 저하 시작 이전 이 범위(분)에 있다
BUGGY_DEPLOY_LEAD_MIN_RANGE: Final = (1, 12)

# dependency_failure: 의존성 장애 시작은 저하 시작보다 이 범위(분)만큼 앞선다 (0 = 동시)
OUTAGE_LEAD_MIN_RANGE: Final = (0, 10)

# 방해 배포(버그 없음): recent_deployment_found는 성립하되 deploy_before_regression은
# 성립하지 않는 시각. 저하 시작 이후(+) 또는 이전 창 밖(-)의 오프셋(분).
DISTRACTOR_DEPLOY_AFTER_MIN_RANGE: Final = (1, 25)
DISTRACTOR_DEPLOY_BEFORE_MIN_RANGE: Final = (18, 28)
DISTRACTOR_DEPLOY_AFTER_PROBABILITY: Final = 0.5

# 배포 버전 문자열 (표시용)
VERSION_MAJOR_RANGE: Final = (1, 4)
VERSION_MINOR_RANGE: Final = (0, 30)
VERSION_PATCH_RANGE: Final = (0, 20)
