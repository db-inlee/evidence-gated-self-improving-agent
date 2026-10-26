"""Ticket 005 분석 상수 (설계 문서 7.8절, 티켓 §8 해석). 임계값과 seed는 여기에만 둔다."""

from fractions import Fraction
from pathlib import Path

SEED = 0  # 7.8: 모든 bootstrap·재표본의 seed(§8-1: 호출마다 새로 시작)
N_BOOT = 2000  # 7.8 A·B·D: δ(N), C4, Oracle 구간의 bootstrap 횟수
NOISE_Q = Fraction(19, 20)  # 7.8 A: δ(N)의 q = 0.95
CI_ALPHA = Fraction(1, 20)  # 7.8 D: Clopper-Pearson 95% 양측(§8-21)

GATE_REPEAT = 0  # 7.8 B: 게이트는 후보·S0 모두 반복 0(§8-5: 후보는 모든 split에서 반복 0)
S0_ORACLE_REPEATS = (0, 1, 2, 3, 4)  # 7.8 A: Oracle S0 반복 5회 다수결
C4_Q = Fraction(1, 20)  # 7.8 B: C4는 bootstrap 평균 Δ의 5번째 백분위수(단측 95% 하한)

EPSILON = Fraction(1, 20)  # 7.8 D: 실질 효과 임계값 ε = 5%p
ORACLE_CI_QS = (Fraction(1, 40), Fraction(39, 40))  # 7.8 D: Oracle Δ 95% 양측(2.5·97.5 백분위수)
TAU_Q = Fraction(19, 20)  # 7.8 D: 칸 잡음은 leave-one-out 다섯 값의 95번째 백분위수
FITNESS_SIZE = Fraction(3, 10)  # 7.8 F: 크기 기준, 같은 family 두 칸의 Δ_cell 차이 >= 30%p
BUDGET_NS = (24, 48, 96, 204)  # 7.8 C: Budget Curve의 N(칸당 4, 8, 16, 34)
N_RESAMPLE = 1000  # 7.8 C: Budget Curve와 RQ2 민감도의 재표본 수
# 7.8 C: RQ2 민감도 재표본의 칸 구성(CELLS 순서: bad L1·L2, dep L1·L2, traffic L1·L2)
SENSITIVITY_TAKE = (18, 18, 3, 3, 9, 9)
# §8-10: 재표본 분포의 2.5·97.5 백분위수(기술용)
BUDGET_SPREAD_QS = (Fraction(1, 40), Fraction(39, 40))

# 7.8 A: 실행 조건(Ticket 005 끝까지 고정)
POLICY_ID = "ollama:gpt-oss:20b"
CONFIG_ID = "7c66ee5bf13c"
TOOL_SPEC_ID = "e9b4996f7529"
S0_SKILL_ID = "bf65e569ae37"
SECTION8_VERSION = "2026-10-21 (Ticket 005 §8 items 1-33)"  # 결과 파일에 기록하는 해석 버전

# Ticket 005 §2(2026-10-22): 분석 칸·split은 명시적 목록이다. Ticket 006이 Level.L3나 L3 split을
# 더해도 고정된 Ticket 005 분석 동작이 바뀌지 않게 한다.
EVAL_LEVELS = ("L1", "L2")
EVAL_SPLITS = ("dev_a", "dev_b", "oracle")

# Ticket 005 §2·§4: 실행 계획. 파일럿 블록 두 개를 맨 앞에 두고, 나머지 블록 순서는 Random(0)으로
# 섞는다. 파일럿은 candidate_id가 가장 작은 proposer와 이름순 첫째 통제 후보의 Dev-A다.
PILOT_BLOCKS = (("cand-000", "dev_a"), ("mixed-scale-on-traffic", "dev_a"))
PLAN_ORDER_SEED = 0
S0_RUN_SPLITS = (("dev_b", (0,)), ("oracle", S0_ORACLE_REPEATS))  # S0 Dev-A는 Ticket 004에서 완료

# Ticket 005 §2: 실행 전 점검
EXPECTED_HW_MODEL = "Mac15,3"
EXPECTED_PARALLEL = {"client_concurrency": 2, "server_num_parallel": 2}
MIN_FREE_DISK_BYTES = 1_000_000_000  # 12,540개 기록은 약 50MB(Dev-A 기록 기준 3.6KB/개). 넉넉히 1GB

# Ticket 005 §2: 실행기 중단 규칙
FORMAT_WINDOW = 200  # 최근 에피소드 창
FORMAT_MIN_EPISODES = 20  # 이 수부터 판정
FORMAT_MAX_RATE = Fraction(1, 20)  # 형식 오류가 1회 이상 있었던 에피소드 비율이 이를 넘으면 멈춤
EPISODE_RETRIES = 3  # 서버 오류: 같은 에피소드 재시도 횟수
MAX_CONSECUTIVE_FAILED_EPISODES = 3  # 재시도 뒤에도 실패한 에피소드가 연속 이만큼이면 멈춤
RETRY_WAIT_S = 5.0
RUN_WORKERS = 2  # config 7c66ee5bf13c의 동시 실행 수
SESSION_LOG = "ticket005/sessions.jsonl"  # runs/ 아래
STATUS_RECENT = 50  # 진행 확인의 최근 처리 속도를 내는 에피소드 수

# ---------------------------------------------------------------------------
# Ticket 006 (L3 compositional shift). Ticket 005 분석은 위의 EVAL_LEVELS·EVAL_SPLITS만 쓰므로
# 여기 값이 Ticket 005 분석 동작을 바꾸지 않는다(Ticket 006 고정 결정 6).
# ---------------------------------------------------------------------------

L3_LEVEL = "L3"
# §3 표: split 이름 → (첫 시드, family당 개수). 칸 = family 3개.
L3_SPLIT_SEEDS = {"l3_oracle": (400_000, 100), "l3_check": (410_000, 10)}
L3_SPLITS = ("l3_oracle",)  # 분석에 쓰는 L3 split. l3_check는 운영 확인용이라 분석에 쓰지 않는다
L3_CHECK_SPLIT = "l3_check"
L3_ORACLE_SPLIT = "l3_oracle"
L3_MAIN_EPISODES = 7_200  # §3: 후보 19 x 300(반복 0) + S0 x 300(반복 0~4)
L3_SESSION_LOG = "ticket006/sessions.jsonl"  # runs/ 아래. Ticket 005 로그와 나눈다
# §3: 본 실행 계획(후보 19 x l3_oracle 반복 0 + S0 x l3_oracle 반복 0~4). 블록 순서는
# Ticket 005 §2와 같이 정해진 순서를 Random(PLAN_ORDER_SEED)로 섞는다. L3-check를 마쳤으므로
# 파일럿 블록은 없다.
L3_PLAN_PATH = Path("plans/ticket-006-l3-plan.json")
# Ticket 006 고정 결정 3: 게이트 결정은 이 Ticket 005 결과(results/ticket-005/analysis.json)
# 그대로다. L3 분석은 같은 레코드로 Ticket 005 분석을 다시 계산해 이 파일과 바이트까지 같은지
# 확인한다.
TICKET005_ANALYSIS_SHA256 = "51d51c9b01e5471e72c7927133d9f9c5732a46036d7867ba82359601e89fc982"

# Ticket 007 §1: 실행 요약(run logs, not part of the analysis). 에피소드가 하나도 돌지 않은 구간을
# 정지 구간으로 셀 최소 길이(초).
IDLE_GAP_S = 60.0
# Ticket 007 §1: 원고의 결과 출처. Ticket 006 결과 파일과 E1 정의 파일(커밋 185d69c)의 sha256.
TICKET006_ANALYSIS_SHA256 = "1de4e7d3cab607a68c7e8c213e342a010ccf9d2956635b50e4fbd896d14eaa5f"
E1_DEFINITION_SHA256 = "ef70f01a73260ff814993a8cf3d52ac078c0ec8f22c10baa3975d4f7bd3b5fbd"
