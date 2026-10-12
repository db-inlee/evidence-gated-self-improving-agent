"""SKILL.md 로드와 skill_id."""

import hashlib
from pathlib import Path

from evidencegate.skills import Skill, load_skill

REPO_ROOT = Path(__file__).resolve().parents[1]
S0_PATH = REPO_ROOT / "skills" / "S0" / "SKILL.md"
S0_V0_PATH = REPO_ROOT / "skills" / "S0-v0" / "SKILL.md"

# 설계 문서 5.7절의 확정본과 같은 문구. 설계 문서는 git에 올리지 않으므로 여기에 둔다.
S0_TEXT = """# Incident Response Skill

1. Confirm the reported service degradation using service metrics.
2. Check for recent deployments.
3. Check dependency health.
4. If a recent deployment exists and all dependencies are healthy, roll the deployment back.
5. If a dependency is degraded, do not change the service; observe and keep monitoring.
6. Otherwise inspect resource usage and traffic.
"""
# Ticket 002 B3 보정 결과(runs/)는 이 skill_id로 연결된다.
S0_SKILL_ID = "bf65e569ae37"

# 첫 초안. 보정 전 결과와 연결된다.
S0_V0_TEXT = """# Incident Response Skill

1. Confirm the reported service degradation using service metrics.
2. Check for recent deployments.
3. If a recent deployment appears related, consider rolling it back.
4. Otherwise inspect resource usage and traffic.
5. Apply the least disruptive action likely to restore service.
"""
S0_V0_SKILL_ID = "d32174164c2c"


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


def test_s0_matches_design_doc_final_text() -> None:
    assert S0_PATH.read_text(encoding="utf-8") == S0_TEXT


def test_s0_keeps_the_calibrated_skill_id() -> None:
    assert load_skill(S0_PATH).skill_id == S0_SKILL_ID


def test_s0_v0_is_the_first_draft_with_its_original_skill_id() -> None:
    assert S0_V0_PATH.read_text(encoding="utf-8") == S0_V0_TEXT
    assert load_skill(S0_V0_PATH).skill_id == S0_V0_SKILL_ID


def test_s0_is_english() -> None:
    assert load_skill(S0_PATH).text.isascii()
