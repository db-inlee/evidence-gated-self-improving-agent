"""동결 테스트(Ticket 004 §6): POOL.json이 있으면 후보·통제 후보 파일이 동결 때와 같아야 한다."""

import json
from pathlib import Path

import pytest

from evidencegate.pool_freeze import verify_frozen_pool
from evidencegate.skills import load_skill

REPO_ROOT = Path(__file__).resolve().parents[1]
POOL_PATH = REPO_ROOT / "skills" / "candidates" / "POOL.json"

pytestmark = pytest.mark.skipif(not POOL_PATH.exists(), reason="candidate pool is not frozen yet")


def test_frozen_files_are_unchanged() -> None:
    assert verify_frozen_pool(REPO_ROOT) == []


def test_frozen_skill_ids_match_the_files() -> None:
    pool = json.loads(POOL_PATH.read_text(encoding="utf-8"))
    for c in pool["proposer_candidates"]:
        path = REPO_ROOT / "skills" / "candidates" / c["candidate_id"] / "SKILL.md"
        assert load_skill(path).skill_id == c["skill_id"]
    for c in pool["controls"]:
        path = REPO_ROOT / "skills" / "controls" / c["name"] / "SKILL.md"
        assert load_skill(path).skill_id == c["skill_id"]
