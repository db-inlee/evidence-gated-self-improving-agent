"""SKILL.md 로드. skill_id는 파일 바이트의 sha256 앞 12자라서 문구가 바뀌면 새 ID가 된다."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

_SKILL_ID_LENGTH = 12


@dataclass(frozen=True)
class Skill:
    skill_id: str
    text: str

    @classmethod
    def from_text(cls, text: str) -> "Skill":
        return cls._from_bytes(text.encode("utf-8"))

    @classmethod
    def _from_bytes(cls, data: bytes) -> "Skill":
        skill_id = hashlib.sha256(data).hexdigest()[:_SKILL_ID_LENGTH]
        return cls(skill_id=skill_id, text=data.decode("utf-8"))


def load_skill(path: Path) -> Skill:
    # read_text는 줄바꿈을 바꾸므로 바이트 그대로 읽어 ID를 만든다.
    return Skill._from_bytes(path.read_bytes())
