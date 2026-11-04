"""Ticket 009 실행 전 점검 (티켓 원칙 2, §4.1). v1 `preflight`는 고치지 않고 그 `Check` 형식을 쓴다.

| 점검 | 기준 |
| preregistration_009 | `preregistration/ticket-009.txt` 파일 전체(UTF-8 바이트)의 sha256이
|                     | `preregistration/ticket-009.sha256`의 기록과 같음 |

009 세션 점검 전체(엔진, config, 장비, 009 풀·계획, 전원)는 §4의 3에서 이 모듈에 더한다.
"""

import hashlib
from pathlib import Path

from evidencegate.preflight import Check

PREREG_FILE = Path("preregistration/ticket-009.txt")
PREREG_SHA_FILE = Path("preregistration/ticket-009.sha256")


def recorded_sha256(sha_file: Path) -> str | None:
    for line in sha_file.read_text().splitlines():
        if line.startswith("sha256: "):
            return line.removeprefix("sha256: ").strip()
    return None


def check_preregistration_009(root: Path) -> Check:
    text, sha_file = root / PREREG_FILE, root / PREREG_SHA_FILE
    if not text.exists() or not sha_file.exists():
        return Check("preregistration_009", False, f"missing {text} or {sha_file}")
    recorded = recorded_sha256(sha_file)
    actual = hashlib.sha256(text.read_bytes()).hexdigest()
    if actual != recorded:
        return Check("preregistration_009", False, f"009 sha256 {actual} does not match {recorded}")
    return Check("preregistration_009", True, f"009 sha256 {actual}")
