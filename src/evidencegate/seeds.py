"""LLM 샘플링 seed 규칙.

후보와 S0는 같은 (family, incident_seed, repeat)에서 같은 seed를 쓴다(paired). sha256 기반이라
프로세스·PYTHONHASHSEED와 무관하다. 이 정의를 바꾸면 이전 결과와 paired 비교가 깨진다.
"""

import hashlib

_SEED_MODULUS = 2**31


def derive_llm_seed(family: str, incident_seed: int, repeat: int) -> int:
    key = f"{family}:{incident_seed}:{repeat}".encode()
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], "big") % _SEED_MODULUS
