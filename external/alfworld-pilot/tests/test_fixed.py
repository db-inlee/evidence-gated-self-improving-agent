"""고정 커밋 파일(Ticket 008 §4의 4).

- 정의(기준, 식, 지표, 설정, P3 목록)는 코드와 출처 파일에서 다시 만든 값과 같아야 한다.
- 출처 파일의 sha256은 고정 값이거나, `fixed/amendments.json`에 기록된 최신 값이어야 한다.
- 티켓은 저장소 안의 사본(`fixed/ticket-008.snapshot.md`)만 본다.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from alfworld_pilot.fixed import (
    AMENDMENTS_JSON,
    FIXED_JSON,
    FIXED_SHA,
    TICKET_SNAPSHOT,
    AmendmentError,
    build_fixed,
    expected_source_shas,
    validate_amendments,
)

ROOT = Path(__file__).parents[1]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def committed() -> dict[str, Any]:
    data: dict[str, Any] = json.loads((ROOT / FIXED_JSON).read_text())
    return data


def amendments() -> list[dict[str, Any]]:
    data: list[dict[str, Any]] = json.loads((ROOT / AMENDMENTS_JSON).read_text())
    return data


def test_committed_definitions_match_a_fresh_build() -> None:
    fresh = build_fixed(ROOT)
    assert {k: v for k, v in committed().items() if k != "sources"} == {
        k: v for k, v in fresh.items() if k != "sources"
    }


def test_sha256_file_matches_the_fixed_file() -> None:
    line = (ROOT / FIXED_SHA).read_text()
    assert line == f"{sha(ROOT / FIXED_JSON)}  {FIXED_JSON.name}\n"


def test_sources_match_the_fixed_sha_or_the_latest_amendment() -> None:
    expected = expected_source_shas(committed(), amendments())
    assert set(expected) == {e["path"] for e in committed()["sources"].values()}
    for path, digest in expected.items():
        assert sha(ROOT / path) == digest, path


def test_ticket_source_is_the_in_repo_snapshot() -> None:
    ticket = committed()["sources"]["ticket"]
    assert ticket["path"] == TICKET_SNAPSHOT.as_posix()
    assert not ticket["path"].startswith("..")
    assert all(not e["path"].startswith("..") for e in committed()["sources"].values())


def test_fixed_values_follow_the_approved_decisions() -> None:
    f = build_fixed(ROOT)
    assert f["criteria"]["G3"]["constants"] == {
        "z": 1.96,
        "h": 0.2,
        "d_floor": 0.1,
        "n_B_ratio": "35/50",
        "n_A": 10,
    }
    assert f["criteria"]["G4"]["constants"] == {"range": [0.15, 0.85], "min_cells": 4, "cells": 6}
    assert f["criteria"]["G6"]["constants"]["resamples"] == 2000
    assert f["evidence"]["E-c"]["k"] == 2
    assert f["evidence"]["conclusion_rule"] == {
        "base_rate_range": [0.15, 0.85],
        "min_share_of_wins_failing_e_a": 0.05,
    }
    assert f["run"]["concurrency"] == 2 and f["run"]["server_num_parallel"] == 2
    assert f["run"]["pythonhashseed"] == "0"
    assert f["policy"]["command_list_in_prompt"] is False
    assert f["s0"]["sha256"] == "0e59f723b8c0de107e5f58ca841ddaae6d9b98f8558f9f90a6524775abdda2b9"
    assert len(f["p3"]["episodes"]) == 12


# --- 고정 뒤 변경 기록(amendments.json) -----------------------------------------------------


def test_committed_amendments_are_valid() -> None:
    validate_amendments(amendments())


def good(**overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "date": "2026-11-03",
        "what": "fix a crash in runner resume",
        "why": "resume failed on an empty file",
        "files": [{"path": "src/alfworld_pilot/runner.py", "sha256": "a" * 64}],
        "affects_episode_results": {"answer": "no", "reason": "only the resume path changed"},
        "handling": None,
        "changes_criteria_or_definitions": False,
    }
    entry.update(overrides)
    return entry


def test_a_complete_amendment_passes() -> None:
    validate_amendments([good()])
    validate_amendments(
        [
            good(
                affects_episode_results={"answer": "yes", "reason": "prompt text changed"},
                handling="rerun",
            )
        ]
    )


@pytest.mark.parametrize(
    "field",
    [
        "date",
        "what",
        "why",
        "files",
        "affects_episode_results",
        "handling",
        "changes_criteria_or_definitions",
    ],
)
def test_a_missing_field_fails(field: str) -> None:
    entry = good()
    del entry[field]
    with pytest.raises(AmendmentError, match=field):
        validate_amendments([entry])


@pytest.mark.parametrize(
    "entry",
    [
        good(what=""),
        good(why=" "),
        good(files=[]),
        good(files=[{"path": "x.py", "sha256": "short"}]),
        good(files=[{"path": "x.py"}]),
        good(affects_episode_results={"answer": "maybe", "reason": "r"}),
        good(affects_episode_results={"answer": "no", "reason": ""}),
        good(affects_episode_results={"answer": "yes", "reason": "r"}, handling=None),
        good(affects_episode_results={"answer": "yes", "reason": "r"}, handling="ignore"),
        good(handling="rerun"),  # "아니오"인데 처리 방법을 적음
        good(changes_criteria_or_definitions=True),  # 원칙 6: 기준·정의는 바꾸지 않는다
    ],
)
def test_an_incomplete_or_forbidden_amendment_fails(entry: dict[str, Any]) -> None:
    with pytest.raises(AmendmentError):
        validate_amendments([entry])


def test_the_latest_amendment_wins_for_a_path() -> None:
    fixed = {"sources": {"runner_code": {"path": "r.py", "sha256": "0" * 64}}}
    first = good(files=[{"path": "r.py", "sha256": "1" * 64}])
    second = good(files=[{"path": "r.py", "sha256": "2" * 64}])
    assert expected_source_shas(fixed, []) == {"r.py": "0" * 64}
    assert expected_source_shas(fixed, [first, second]) == {"r.py": "2" * 64}
    with pytest.raises(AmendmentError, match="not a fixed source"):
        expected_source_shas(fixed, [good(files=[{"path": "other.py", "sha256": "3" * 64}])])
