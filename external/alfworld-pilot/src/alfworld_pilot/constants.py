"""P2 이후 파일럿 상수(정책 설정, 실행기, 보조 지표). 흩어진 숫자를 두지 않는다.

엔진 설정은 v1과 같다(Ticket 008 원칙 3): gpt-oss:20b, think "low", temperature 0.7, keep_alive 60m,
seed = sha256 앞 8바이트 mod 2^31. 정책 설정(걸음 상한, 이력, num_ctx, 형식 오류)은 파일럿 고유다.
"""

# 엔진 (v1과 같음)
MODEL = "gpt-oss:20b"
THINK = "low"
TEMPERATURE = 0.7
KEEP_ALIVE = "60m"
REQUEST_TIMEOUT_S = 600.0
SEED_MODULUS = 2**31

# 정책 설정 (파일럿 고유, P2 고정)
MAX_STEPS = 50  # ALFWorld 평가 상한(base_config max_nb_steps_per_episode)
NUM_CTX = 8192  # v1과 같은 값. 근거는 P2 보고서(이력 길이 추정)
RESERVE_TOKENS = 2048  # 생성(생각 + 도구 호출)과 채팅 틀 몫
CHARS_PER_TOKEN = 3  # 영어 관측의 보수적 환산(실제는 약 4). P3에서 prompt_eval_count로 확인
MAX_CONSECUTIVE_FORMAT_ERRORS = 3  # v1과 같음
MAX_CONSECUTIVE_TEXT_TURNS = 2  # v1과 같음
MAX_LLM_CALLS = 2 * MAX_STEPS  # 텍스트·형식 오류가 번갈아 나와도 끝나게

# 실행기
BLOCK_SIZE = 6  # 원칙 7 점검 간격(에피소드). 시작할 때도 한 번
CONCURRENCY = 2  # 실행 설정(결정 5). 정책 설정이 아니다
SERVER_NUM_PARALLEL = 2  # 서버를 OLLAMA_NUM_PARALLEL=2로 띄운다는 선언
OLLAMA_URL = "http://127.0.0.1:11434"
REQUIRED_HASH_SEED = "0"  # 결정 0: 모든 환경 실행은 PYTHONHASHSEED=0

# 보조 지표
E_C_K = 2  # E-c: 성공 걸음 수 ≤ k x 전문가 계획 길이

# 진단 기록 (2026-11-03 amendment)
TEXT_CONTENT_LIMIT = 500  # 텍스트 턴 assistant 내용을 남기는 최대 글자 수

# P5 국소 해로움 탐침의 표적 칸 규칙 (2026-11-03 amendment, P4 전에 기록)
P5_TARGET_RULE = "P4 반복 3회 평균 성공률이 가장 높은 칸, 동률이면 SIX_TYPES 순서에서 앞"
