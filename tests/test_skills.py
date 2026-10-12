"""SKILL.md 로드와 skill_id."""

import hashlib
from pathlib import Path

from evidencegate.skills import Skill, load_skill

REPO_ROOT = Path(__file__).resolve().parents[1]
S0_PATH = REPO_ROOT / "skills" / "S0" / "SKILL.md"

S0_TEXT = """# Incident Response Skill

1. Confirm the reported service degradation using service metrics.
2. Check for recent deployments.
3. If a recent deployment appears related, consider rolling it back.
4. Otherwise inspect resource usage and traffic.
5. Apply the least disruptive action likely to restore service.
"""


def test_skill_id_is_first_12_hex_of_sha256_of_file_bytes(tmp_path: Path) -> None:
    path = tmp_path / "SKILL.md"
    path.write_bytes(b"# Skill\n\n1. Do a thing.\n")
    skill = load_skill(path)
    assert skill.skill_id == hashlib.sha256(b"# Skill\n\n1. Do a thing.\n").hexdigest()[:12]
    assert skill.text == "# Skill\n\n1. Do a thing.\n"


def test_any_byte_change_gives_new_skill_id(tmp_path: Path) -> None:
    a = tmp_path / "a.md"
    b = tmp_path / "b.md"
    a.write_bytes(b"1. Check deployments.\n")
    b.write_bytes(b"1. Check deployments.\r\n")
    assert load_skill(a).skill_id != load_skill(b).skill_id


def test_skill_from_text_matches_file(tmp_path: Path) -> None:
    path = tmp_path / "SKILL.md"
    path.write_text(S0_TEXT, encoding="utf-8")
    assert load_skill(path) == Skill.from_text(S0_TEXT)


def test_s0_matches_design_doc_text() -> None:
    assert S0_PATH.read_text(encoding="utf-8") == S0_TEXT


def test_s0_is_english() -> None:
    assert load_skill(S0_PATH).text.isascii()
