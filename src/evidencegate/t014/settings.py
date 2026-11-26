"""Ticket 014 고정 값(사전 등록 `preregistration/ticket-014.txt` 1~8절). 한곳에 모은다."""

from fractions import Fraction
from pathlib import Path
from typing import Any

from evidencegate.t009b import settings as s009b

PREREG_FILE = Path("preregistration/ticket-014.txt")
PREREG_SHA_FILE = Path("preregistration/ticket-014.sha256")
PREREG_TAG = "t014-prereg"

# 2절 분할: (split 이름, 첫 incident seed, 칸당 수). 이름은 `t014_`로 시작한다(기록 폴더가 split
# 이름으로 갈리므로 지금까지의 기록에 이어 쓰지 않는다).
DEV_A = ("t014_dev_a", 1_100_000, 10)
CALIBRATION = ("t014_calibration", 1_200_000, 150)
DEV_B = ("t014_dev_b", 1_300_000, 150)
ORACLE = ("t014_oracle", 1_400_000, 100)
GATE_REPEAT = 0
CALIBRATION_REPEATS = (0, 1, 2, 3, 4)
ORACLE_REPEATS = (0, 1, 2)
PLAN_ORDER_SEED = 0  # 블록 순서 섞기(009b와 같은 규칙)

# 4절 게이트
N_C = DEV_B[2]  # Dev-B 칸당 150 = h(n_c)의 n_c
N_DEV_B = 6 * DEV_B[2]  # 900 = δ(N)의 N
MARGIN = Fraction(1, 20)  # m
C6_ALPHA = Fraction(1, 20)  # Tango 한쪽 수준
TANGO_TIE = 1e-12  # 하한이 -m에서 이 안이면 통과로 보지 않는다
SAGE_LAMBDA = 1
SAGE_ALPHA = Fraction(1, 20)
SAGE_TAU = Fraction(1, 20)  # rho = l / n_S의 상한

# 6절 Oracle 라벨
USEFUL_EPS = Fraction(1, 20)  # Useful: 전체 하한 > ε

# 7.2절 모의 예측(`results/ticket-013/sim.json` decision.a, C6_IUT, n_c 150, m_c 1/20)
SIM_POWER_HITS = 59_352
SIM_POWER_POOLS = 80_000
PREDICTED_LOSS = Fraction(SIM_POWER_POOLS - SIM_POWER_HITS, SIM_POWER_POOLS)  # 20,648 / 80,000

# 3절 후보 생성(009b 2.1절과 같은 값)
DEDUPE_THRESHOLD = s009b.DEDUPE_THRESHOLD
POOL_TARGET = s009b.POOL_TARGET
STOP_AFTER_IDLE = s009b.STOP_AFTER_IDLE
MAX_CALLS = s009b.MAX_CALLS
BUDGET_USD = s009b.BUDGET_USD
QUOTA_MESSAGE = s009b.QUOTA_MESSAGE
N_GO = s009b.N_GO
EXCLUDE_DIR = Path("skills/candidates-009b")  # 중복 판정에만 쓴다(피드백에 넣지 않음)
EXCLUDE_COUNT = 40
CONTROLS_DIR = Path("skills/controls")

# 3절(David 결정, 2026-11-27): 통제 후보는 이번 연구에서 Dev-B·Oracle 모두 쓰지 않는다.
CONTROLS_IN_ORACLE: bool | None = False

# 경로
RUN_DIR = Path("runs/proposer-014")
EXPORT_PATH = Path(
    "runs/proposer_inputs-014/dev_a__ollama__gpt-oss__20b__7c66ee5bf13c__bf65e569ae37.json"
)
CANDIDATES_DIR = Path("skills/candidates-014")
POOL_PATH = CANDIDATES_DIR / "POOL-014.json"
POOL_GROUPS_PATH = Path("plans/ticket-014-pool-groups.json")
RESULTS_DIR = Path("results/ticket-014")
LOG_DIR = "ticket014"  # runs/ticket014/stage<N>/
REJECTED_PATHS = (
    *s009b.REJECTED_PATHS,
    s009b.RUN_DIR / "ledger.jsonl",
    s009b.RUN_DIR / "pool",
)
RUN_OPTIONS_FILE = s009b.RUN_OPTIONS_FILE


def fixed_run_options(root: Path) -> dict[str, Any]:
    """`run_options.json`에 남는 고정 설정(009b 설정 + 제외 폴더)."""
    return {**s009b.FIXED_RUN_OPTIONS, "exclude_dir": str(root / EXCLUDE_DIR)}


# 단계(8절)
STAGES = (0, 1, 2, 3, 4, 5)
EPISODE_STAGES = (0, 1, 2, 4)
STOP_STAGES = (0, 3)  # 결과를 쓴 뒤 David 확인
RESULT_NAMES = {
    0: "stage0-pool.json",
    1: "stage1-thresholds.json",
    2: "stage2-devb-summary.json",
    3: "stage3-gates.json",
    4: "stage4-oracle-summary.json",
    5: "analysis.json",
}
