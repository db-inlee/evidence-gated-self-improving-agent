"""Ticket 009b §2: 009 후보 80개의 아이디어 묶음 파일(성질 계산 전에 고정)."""

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
GROUPS = ROOT / "plans" / "ticket-009b-dedupe-groups.json"
MANIFEST = ROOT / "results" / "ticket-009" / "proposer-calls-manifest.json"


def load() -> dict[str, Any]:
    data: dict[str, Any] = json.loads(GROUPS.read_text(encoding="utf-8"))
    return data


def test_groups_partition_the_80_calls() -> None:
    groups = load()["groups"]
    calls = sorted(i for g in groups.values() for i in g["calls"])
    assert calls == list(range(80))
    assert len(groups) == 17
    assert all(len(g["calls"]) == 1 for name, g in groups.items() if name.startswith("O"))


def test_labels_are_bound_to_the_recorded_call_files() -> None:
    data = load()
    assert data["source_manifest"]["sha256"] == hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    files = {e["path"]: e["sha256"] for e in json.loads(MANIFEST.read_text())["files"]}
    assert data["call_sha256"] == {
        str(i): files[f"runs/proposer-009/pool/call-{i:03d}.json"] for i in range(80)
    }
