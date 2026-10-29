"""재현용 기록 공개 (Ticket 007 §3): 비밀 값 확인 → 결정적 압축 → 매니페스트.

    uv run python -m evidencegate.release

압축본은 dist/에, 매니페스트는 writeup/inputs/release_manifest.json에 쓴다.

- 공개 대상(`INCLUDED`, `runs/` 기준): 분석 입력 기록(dev_a, dev_b, oracle, l3_oracle,
  calibration)과, 실행 요약(`run_summary.json`)을 다시 만들 수 있게 L3-check 기록과 세션 기록 두 개.
  proposer 장부와 그 밖의 시험 폴더는 공개하지 않는다(`EXCLUDED`는 명시적으로 뺀 것).
- 비밀 값 확인을 먼저 한다. 패턴 하나라도 걸리면 압축하지 않고 종료 1이다(파일·줄·패턴 출력,
  걸린 값 자체는 출력하지 않음).
- 압축본은 결정적이다: 이름순, mtime 0, uid/gid 0, 빈 사용자 이름, 0644, gzip mtime 0.
- 매니페스트(git에 올림)는 압축본 이름·sha256·크기, 파일마다 경로·sha256·바이트, 뺀 것, 확인한
  패턴 목록과 걸린 수를 담는다. Release 업로드는 사용자가 한다.
"""

import argparse
import gzip
import hashlib
import io
import json
import re
import sys
import tarfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from evidencegate.analysis import REPO_ROOT

INCLUDED = (
    "dev_a",
    "dev_b",
    "oracle",
    "l3_oracle",
    "calibration",
    "l3_check",
    "ticket005/sessions.jsonl",
    "ticket006/sessions.jsonl",
)
EXCLUDED = ("proposer/ledger.jsonl", "proposer/pool", "proposer/probe", "proposer_inputs")
ARCHIVE_NAME = "evidencegate-records.tar.gz"
DEFAULT_OUT = Path("dist")
DEFAULT_MANIFEST = Path("writeup/inputs/release_manifest.json")
SECRET_PATTERNS = {
    "openai_key": r"sk-(?:proj-)?[A-Za-z0-9_-]{20,}",
    "openai_env": r"OPENAI_API_KEY",
    "bearer_token": r"Bearer\s+[A-Za-z0-9._-]{10,}",
    "aws_key": r"AKIA[0-9A-Z]{16}",
    "private_key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "github_token": r"gh[pousr]_[A-Za-z0-9]{36}",
    "slack_token": r"xox[abprs]-[A-Za-z0-9-]{10,}",
    "email": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    "home_path": r"/(?:Users|home)/[^/\s\"']+",
    "api_key_field": r"api[_-]?key",
}
_COMPILED = {
    name: re.compile(p, re.IGNORECASE if name == "api_key_field" else 0)
    for name, p in SECRET_PATTERNS.items()
}


def release_files(runs_dir: Path) -> list[Path]:
    out: list[Path] = []
    for part in INCLUDED:
        path = runs_dir / part
        if path.is_file():
            out.append(path)
        elif path.is_dir():
            out += [p for p in path.rglob("*") if p.is_file()]
    return sorted(out, key=lambda p: p.relative_to(runs_dir).as_posix())


def scan_secrets(files: Sequence[Path], runs_dir: Path) -> list[dict[str, Any]]:
    hits = []
    for path in files:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for name, pattern in _COMPILED.items():
                if pattern.search(line):
                    hits.append(
                        {
                            "path": path.relative_to(runs_dir).as_posix(),
                            "line": number,
                            "pattern": name,
                        }
                    )
    return hits


def _archive_bytes(files: Sequence[Path], runs_dir: Path) -> bytes:
    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for path in files:
            data = path.read_bytes()
            info = tarfile.TarInfo(name=f"runs/{path.relative_to(runs_dir).as_posix()}")
            info.size = len(data)
            info.mtime = 0
            info.mode = 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            tar.addfile(info, io.BytesIO(data))
    out = io.BytesIO()
    with gzip.GzipFile(fileobj=out, mode="wb", mtime=0, compresslevel=9) as gz:
        gz.write(raw.getvalue())
    return out.getvalue()


def build_release(runs_dir: Path, out_dir: Path) -> dict[str, Any]:
    """비밀 값이 없을 때만 압축본을 쓰고 매니페스트를 돌려준다. 있으면 ValueError."""
    files = release_files(runs_dir)
    hits = scan_secrets(files, runs_dir)
    if hits:
        raise ValueError(f"{len(hits)} secret pattern hits; nothing was archived: {hits[:5]}")
    data = _archive_bytes(files, runs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / ARCHIVE_NAME).write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    (out_dir / f"{ARCHIVE_NAME}.sha256").write_text(f"{digest}  {ARCHIVE_NAME}\n")
    return {
        "archive": ARCHIVE_NAME,
        "archive_sha256": digest,
        "archive_bytes": len(data),
        "files": [
            {
                "path": p.relative_to(runs_dir).as_posix(),
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                "bytes": p.stat().st_size,
            }
            for p in files
        ],
        "included": list(INCLUDED),
        "excluded": list(EXCLUDED),
        "secret_scan": {
            "patterns": len(SECRET_PATTERNS),
            "hits": 0,
            "checked": sorted(SECRET_PATTERNS),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evidencegate-release")
    parser.add_argument("--runs-dir", type=Path, default=REPO_ROOT / "runs")
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / DEFAULT_OUT)
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / DEFAULT_MANIFEST)
    args = parser.parse_args(argv)
    files = release_files(args.runs_dir)
    hits = scan_secrets(files, args.runs_dir)
    print(f"secret scan: {len(files)} files, {len(SECRET_PATTERNS)} patterns, {len(hits)} hits")
    for hit in hits:
        print(f"  {hit['path']}:{hit['line']}: {hit['pattern']}", file=sys.stderr)
    if hits:
        print("error: secret patterns found; nothing was archived", file=sys.stderr)
        return 1
    manifest = build_release(args.runs_dir, args.out_dir)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        f"archive {args.out_dir / ARCHIVE_NAME}: {manifest['archive_bytes']} bytes, "
        f"sha256 {manifest['archive_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
