"""009b·010 탐침 기록 압축본 (Ticket 011 §4). v1 `release`는 바꾸지 않고 그 압축 함수를 쓴다.

    uv run python -m evidencegate.release_009b dry-run t009b         # 파일 목록과 크기만(쓰지 않음)
    EVIDENCEGATE_SECRET_NAMES='…' uv run python -m evidencegate.release_009b build t009b
    EVIDENCEGATE_SECRET_NAMES='…' uv run python -m evidencegate.release_009b build t010_probe

- 압축본마다 넣는 것(`included`, `runs/` 기준)과 빼는 것(`excluded`)을 티켓 §4 그대로 둔다. 넣는
  폴더 안의 파일이라도 빼는 목록에 있으면 넣지 않는다. 넣는 항목이 없으면 거부한다.
- 비밀 검사: v1 `SECRET_PATTERNS`에 더해, 환경 변수 `EVIDENCEGATE_SECRET_NAMES`(`|`로 구분, 대소문자
  무시)의 이름을 파일 내용과 경로에서 찾는다. 변수가 없거나 비면 아무것도 만들지 않는다. 걸린 곳은
  파일과 줄 번호(경로에서 걸리면 0)만 내고, 걸린 이름·패턴은 출력·매니페스트 어디에도 남기지 않는다.
  하나라도 걸리면 압축하지 않는다.
- 압축은 v1과 같은 결정적 압축이다(`release._archive_bytes`). 매니페스트는 v1과 같은 형식이고
  압축본마다 하나다(`MANIFESTS`). 같은 기록이면 같은 바이트이고, 다른 매니페스트가 있으면 덮어쓰지
  않는다. 업로드는 사용자가 한다.
- `dry-run`은 비밀 검사 없이 파일 목록, 원본 크기, 압축 크기만 낸다. 아무것도 쓰지 않는다.
"""

import argparse
import hashlib
import json
import os
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evidencegate.analysis import REPO_ROOT
from evidencegate.release import _COMPILED, DEFAULT_OUT, SECRET_PATTERNS, _archive_bytes

NAMES_ENV = "EVIDENCEGATE_SECRET_NAMES"
NAMES_LABEL = "secret_names"


@dataclass(frozen=True)
class Archive:
    name: str
    included: tuple[str, ...]
    excluded: tuple[str, ...]


ARCHIVES: dict[str, Archive] = {
    "t009b": Archive(
        "evidencegate-records-009b.tar.gz",
        included=(
            "t009_calibration",
            "t009_dev_a",
            "t009_dev_b",
            "t009_oracle",
            "ticket009/calibration/sessions.jsonl",
            "ticket009/dev_a/sessions.jsonl",
            "ticket009b/main/sessions.jsonl",
            "proposer_inputs-009",
            "proposer-009b/pool",
            "proposer-009b/response_models.jsonl",
            "proposer-009/pool",
        ),
        excluded=(
            "proposer-009/ledger.jsonl",
            "proposer-009b/ledger.jsonl",
            "proposer-009b/sessions.jsonl",  # 집 절대 경로(사용자 결정)
            "ticket009/calibration/run.log",
            "ticket009/calibration/run_wrapper.sh",
            "ticket009/calibration/wrapper_status.txt",
            "ticket009/dev_a/run.log",
        ),
    ),
    "t010_probe": Archive(
        "evidencegate-records-010-probe.tar.gz",
        included=("t010_probe", "ticket010/probe/sessions.jsonl"),
        excluded=(),
    ),
}
MANIFESTS: dict[str, Path] = {
    "t009b": Path("writeup/inputs/release_manifest_009b.json"),
    "t010_probe": Path("writeup/inputs/release_manifest_010_probe.json"),
}


def archive_files(spec: Archive, runs_dir: Path) -> list[Path]:
    excluded = set(spec.excluded)
    out: list[Path] = []
    for part in spec.included:
        path = runs_dir / part
        if path.is_file():
            out.append(path)
        elif path.is_dir():
            out += [p for p in path.rglob("*") if p.is_file()]
        else:
            raise ValueError(f"included part {part} is missing under {runs_dir}")
    kept = [p for p in out if p.relative_to(runs_dir).as_posix() not in excluded]
    return sorted(set(kept), key=lambda p: p.relative_to(runs_dir).as_posix())


def secret_names() -> tuple[str, ...]:
    """환경 변수의 이름 목록. 없거나 비면 ValueError(이름은 메시지에 넣지 않는다)."""
    raw = os.environ.get(NAMES_ENV)
    names = tuple(n.strip() for n in (raw or "").split("|") if n.strip())
    if not names:
        raise ValueError(f"set {NAMES_ENV} (names separated by |) before building an archive")
    return names


def scan(files: Sequence[Path], runs_dir: Path, names: Sequence[str]) -> list[dict[str, Any]]:
    """걸린 파일과 줄 번호만(경로에서 걸리면 줄 0). 무엇이 걸렸는지는 담지 않는다."""
    named = re.compile("|".join(re.escape(n) for n in names), re.IGNORECASE)
    hits: list[dict[str, Any]] = []
    for path in files:
        rel = path.relative_to(runs_dir).as_posix()
        if named.search(rel):
            hits.append({"path": rel, "line": 0})
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if named.search(line) or any(p.search(line) for p in _COMPILED.values()):
                hits.append({"path": rel, "line": number})
    return hits


def manifest(spec: Archive, files: Sequence[Path], runs_dir: Path, data: bytes) -> dict[str, Any]:
    return {
        "archive": spec.name,
        "archive_sha256": hashlib.sha256(data).hexdigest(),
        "archive_bytes": len(data),
        "files": [
            {
                "path": p.relative_to(runs_dir).as_posix(),
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                "bytes": p.stat().st_size,
            }
            for p in files
        ],
        "included": list(spec.included),
        "excluded": list(spec.excluded),
        "secret_scan": {
            "patterns": len(SECRET_PATTERNS) + 1,
            "hits": 0,
            "checked": [*sorted(SECRET_PATTERNS), NAMES_LABEL],
        },
    }


def pack(key: str, runs_dir: Path, *, names: Sequence[str]) -> tuple[bytes, dict[str, Any]]:
    """검사에 걸린 곳이 없을 때만 압축본 바이트와 매니페스트를 만든다(쓰지 않음)."""
    spec = ARCHIVES[key]
    files = archive_files(spec, runs_dir)
    hits = scan(files, runs_dir, names)
    if hits:
        where = ", ".join(f"{h['path']}:{h['line']}" for h in hits[:5])
        raise ValueError(f"{len(hits)} hit(s) in the secret scan; nothing was archived: {where}")
    data = _archive_bytes(files, runs_dir)
    return data, manifest(spec, files, runs_dir, data)


def write_archive(key: str, out_dir: Path, data: bytes, digest: str) -> None:
    name = ARCHIVES[key].name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / name).write_bytes(data)
    (out_dir / f"{name}.sha256").write_text(f"{digest}  {name}\n")


def build_archive(
    key: str, runs_dir: Path, out_dir: Path, *, names: Sequence[str]
) -> dict[str, Any]:
    """검사에 걸린 곳이 없을 때만 압축본과 .sha256을 쓰고 매니페스트를 돌려준다."""
    data, out = pack(key, runs_dir, names=names)
    write_archive(key, out_dir, data, out["archive_sha256"])
    return out


def manifest_bytes(data: dict[str, Any]) -> bytes:
    return (json.dumps(data, indent=2, sort_keys=True) + "\n").encode()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evidencegate-release-009b")
    parser.add_argument("command", choices=("dry-run", "build"))
    parser.add_argument("archive", choices=sorted(ARCHIVES))
    parser.add_argument("--runs-dir", type=Path, default=REPO_ROOT / "runs")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / DEFAULT_OUT)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    args = parser.parse_args(argv)
    spec = ARCHIVES[args.archive]
    try:
        files = archive_files(spec, args.runs_dir)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    if args.command == "dry-run":
        for p in files:
            print(f"  {p.relative_to(args.runs_dir).as_posix()}")
        raw = sum(p.stat().st_size for p in files)
        packed = len(_archive_bytes(files, args.runs_dir))
        print(
            f"{spec.name}: {len(files)} files, {raw} bytes raw, about {packed} bytes compressed "
            "(secret scan skipped; nothing written)"
        )
        return 0

    try:
        names = secret_names()
        files_hits = scan(files, args.runs_dir, names)
    except ValueError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"secret scan: {len(files)} files, {len(SECRET_PATTERNS)} patterns and a name list")
    if files_hits:
        for hit in files_hits:
            print(f"  {hit['path']}:{hit['line']}", file=sys.stderr)
        print(f"error: {len(files_hits)} hit(s); nothing was archived", file=sys.stderr)
        return 1
    path = args.root / MANIFESTS[args.archive]
    archive_data, data = pack(args.archive, args.runs_dir, names=names)
    blob = manifest_bytes(data)
    if path.exists() and path.read_bytes() != blob:
        print(f"error: {path} exists and differs; nothing was written", file=sys.stderr)
        return 1
    write_archive(args.archive, args.out_dir, archive_data, data["archive_sha256"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(blob)
    print(
        f"archive {args.out_dir / spec.name}: {data['archive_bytes']} bytes, "
        f"sha256 {data['archive_sha256']}; manifest {MANIFESTS[args.archive]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
