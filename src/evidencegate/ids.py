"""실행 키에 들어가는 내용 ID.

결과에 영향을 주는 입력(도구 명세, 정책 설정)은 내용의 해시로 식별한다. 하나라도 바뀌면 다른
키가 되어 재개 로직이 이전 결과를 재사용하지 않는다. skill_id는 파일 바이트 해시라
`skills.py`에 있다.
"""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

from evidencegate.policy import Policy
from opsworld.agent_env import ToolSpec

_ID_LENGTH = 12


def content_id(obj: Any) -> str:
    """정렬된 키의 compact JSON을 sha256으로 해시한 앞 12자."""
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:_ID_LENGTH]


def tool_spec_id(specs: Sequence[ToolSpec]) -> str:
    return content_id([asdict(spec) for spec in specs])


def config_id(policy: Policy) -> str:
    return content_id(policy.config)
