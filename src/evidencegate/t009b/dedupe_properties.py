"""Ticket 009b §2: 새 중복 판정의 성질 표와 t 선택(009 후보 80개, 기술용 근거).

    uv run python -m evidencegate.t009b.dedupe_properties

- 입력: 009 call 파일 80개(`runs/proposer-009/pool`, git 밖)와 묶음 파일
  (`plans/ticket-009b-dedupe-groups.json`, 성질 계산 전에 커밋). call 파일의 sha256이 묶음 파일의
  기록과 다르면 거부한다.
- 정의: D1(`candidate_dedupe.word_change_set`, 주), D2(D1에서 기능어 `FUNCTION_WORDS`를 뺀 것,
  기술용), v1(`candidate_dedupe.change_set`, 기술용). 셋 다 정규화 전문이 같으면 중복이다.
- t마다 성질:
  - 묶음 안 흡수율 = 같은 묶음 쌍 중 J ≥ t인 비율
  - 묶음 분리율 = 서로 다른 묶음 쌍 중 J < t인 비율
  - 순차 풀(호출 순서, 앞선 풀 후보와 중복이면 뺌) 크기, 풀에 대표가 남은 묶음 수, 묶음별 풀 수
- t 선택(티켓 §2의 3, D1만): 분리율 ≥ 0.95인 t 중 흡수율이 가장 큰 t(같으면 작은 t). 그런 t가
  없으면 분리율이 가장 큰 t(같으면 큰 t)이고 "fallback"으로 표시한다.
- 출력: `plans/ticket-009b-dedupe-properties.json`(+ `.sha256`). 이 80개는 009b 풀에 넣지 않는다.
"""

import argparse
import hashlib
import itertools
import json
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from evidencegate.analysis import REPO_ROOT
from evidencegate.candidate_checks import normalize_lines
from evidencegate.candidate_dedupe import change_set, jaccard, word_change_set

T_GRID = (0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
MIN_SEPARATION = 0.95
# D2에서 빼는 기능어(009b 검토 보고서 2026-11-05-08에서 쓴 목록). "not"은 뜻을 바꾸므로 넣지 않는다.
_FUNCTION_WORDS_TEXT = (
    "a an the and or is are be to of for if then it its that this with when before after on in"
)
FUNCTION_WORDS = frozenset(_FUNCTION_WORDS_TEXT.split(" "))
GROUPS_PATH = Path("plans/ticket-009b-dedupe-groups.json")
POOL_DIR = Path("runs/proposer-009/pool")
OUT_PATH = Path("plans/ticket-009b-dedupe-properties.json")
S0_PATH = Path("skills/S0/SKILL.md")


def without_function_words(changes: frozenset[str]) -> frozenset[str]:
    return frozenset(c for c in changes if c[1:] not in FUNCTION_WORDS)


def pair_rates(sets: Sequence[frozenset[str]], groups: Sequence[str], t: float) -> dict[str, Any]:
    within = between = absorbed = separated = 0
    for i, j in itertools.combinations(range(len(sets)), 2):
        j_value = jaccard(sets[i], sets[j])
        if groups[i] == groups[j]:
            within += 1
            absorbed += j_value >= t
        else:
            between += 1
            separated += j_value < t
    return {
        "absorption": absorbed / within if within else None,
        "separation": separated / between if between else None,
        "within_pairs": within,
        "between_pairs": between,
    }


def sequential_pool(items: Sequence[str], duplicate: Callable[[str, str], bool]) -> list[int]:
    """호출 순서로 앞선 풀 후보와 비교해 중복이면 뺀다(먼저 것을 남김, propose_cli와 같음)."""
    pool: list[int] = []
    for i, item in enumerate(items):
        if not any(duplicate(item, items[k]) for k in pool):
            pool.append(i)
    return pool


def choose_t(rows: Sequence[Mapping[str, Any]]) -> tuple[float, str]:
    passing = [r for r in rows if r["separation"] >= MIN_SEPARATION]
    if passing:
        best = max(passing, key=lambda r: (r["absorption"], -r["t"]))
        return best["t"], "rule"
    best = max(rows, key=lambda r: (r["separation"], r["t"]))
    return best["t"], "fallback"


def properties(texts: Sequence[str], groups: Sequence[str], s0_text: str) -> list[dict[str, Any]]:
    d1 = [word_change_set(t, s0_text) for t in texts]
    definitions = {
        "D1": d1,
        "D2": [without_function_words(s) for s in d1],
        "v1": [change_set(t, s0_text) for t in texts],
    }
    normalized = [normalize_lines(t) for t in texts]
    rows = []
    for name, sets in definitions.items():
        for t in T_GRID:

            def duplicate(
                a: str, b: str, sets: Sequence[frozenset[str]] = sets, t: float = t
            ) -> bool:
                i, k = int(a), int(b)
                return normalized[i] == normalized[k] or jaccard(sets[i], sets[k]) >= t

            pool = sequential_pool([str(i) for i in range(len(texts))], duplicate)
            by_group = Counter(groups[i] for i in pool)
            rows.append(
                {
                    "definition": name,
                    "t": t,
                    **pair_rates(sets, groups, t),
                    "pool": len(pool),
                    "groups_in_pool": len(by_group),
                    "pool_by_group": dict(sorted(by_group.items())),
                    "pool_calls": pool,
                }
            )
    return rows


def call_groups(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    labels: dict[int, str] = {}
    for name, group in data["groups"].items():
        for i in group["calls"]:
            labels[i] = name
    return [labels[i] for i in range(len(labels))]


def load_texts(pool_dir: Path, call_sha256: Mapping[str, str]) -> list[str]:
    texts = []
    for i in range(len(call_sha256)):
        path = pool_dir / f"call-{i:03d}.json"
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != call_sha256[str(i)]:
            raise ValueError(f"{path} sha256 does not match the groups file")
        texts.append(str(json.loads(data)["candidate"]["skill_md"]))
    return texts


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evidencegate-t009b-dedupe-properties")
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    root: Path = args.root
    groups_path = root / GROUPS_PATH
    groups_data = json.loads(groups_path.read_text(encoding="utf-8"))
    try:
        texts = load_texts(root / POOL_DIR, groups_data["call_sha256"])
    except (FileNotFoundError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    rows = properties(texts, call_groups(groups_path), (root / S0_PATH).read_text())
    t, how = choose_t([r for r in rows if r["definition"] == "D1"])
    data = {
        "groups_file": {
            "path": GROUPS_PATH.as_posix(),
            "sha256": hashlib.sha256(groups_path.read_bytes()).hexdigest(),
        },
        "t_grid": list(T_GRID),
        "min_separation": MIN_SEPARATION,
        "function_words": sorted(FUNCTION_WORDS),
        "chosen": {"definition": "D1", "t": t, "how": how},
        "rows": rows,
    }
    out: Path = args.out or root / OUT_PATH
    payload = (json.dumps(data, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
    out.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    out.with_name(out.name + ".sha256").write_text(f"{digest}  {out.name}\n")
    print(f"chosen t = {t} ({how}); {out} sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
