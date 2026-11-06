"""Ticket 009b 고정 값(사전 등록 `preregistration/ticket-009b.txt` 2.1·4.2·8절). 한곳에 모은다."""

from pathlib import Path
from typing import Any

PREREG_FILE = Path("preregistration/ticket-009b.txt")
PREREG_SHA_FILE = Path("preregistration/ticket-009b.sha256")
PREREG_TAG = "t009b-prereg"
THRESHOLDS_PATH = Path("results/ticket-009/thresholds.json")
THRESHOLDS_SHA256 = "d030b1734811d8a0acc8fdfd198d3d2f0b8ce87c1e81faf322e1f6b1805c3520"
EXPORT_SHA256 = "4f9afa27360f01cfbf6a680bbc4ae2fc0a9f52cbe30f5d14f7bba11b54504c6d"

# 2.1 후보 생성
DEDUPE_THRESHOLD = 0.55
POOL_TARGET = 40
STOP_AFTER_IDLE = 25
MAX_CALLS = 120
BUDGET_USD = 5.0
QUOTA_MESSAGE = "no credits remaining"
FIXED_RUN_OPTIONS: dict[str, Any] = {
    "previous": "valid",
    "dedupe": "word",
    "dedupe_threshold": DEDUPE_THRESHOLD,
    "stop_after_idle": STOP_AFTER_IDLE,
}
N_GO = 25  # 8절 진행 기준

# 경로
RUN_DIR = Path("runs/proposer-009b")
STOP_FILE = RUN_DIR / "STOP"
SESSIONS_FILE = RUN_DIR / "sessions.jsonl"
CANDIDATES_DIR = Path("skills/candidates-009b")
POOL_PATH = CANDIDATES_DIR / "POOL-009b.json"
POOL_GROUPS_PATH = Path("plans/ticket-009b-pool-groups.json")
REJECTED_PATHS = (
    Path("runs/proposer/ledger.jsonl"),
    Path("runs/proposer/pool"),
    Path("runs/proposer-009/ledger.jsonl"),
    Path("runs/proposer-009/pool"),
)
RUN_OPTIONS_FILE = "run_options.json"
