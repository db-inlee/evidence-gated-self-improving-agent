"""고정 커밋 파일 (Ticket 008 §4의 4, 원칙 6). P3 전에 커밋하고 그 뒤에는 바꾸지 않는다.

    PYTHONHASHSEED=0 uv run python -m alfworld_pilot.fixed

`fixed/p3_fixed.json`과 `fixed/p3_fixed.sha256`을 쓴다. 값은 손으로 옮기지 않고 코드 상수와 출처
파일에서 만든다. 테스트는 정의가 다시 만든 값과 같은지, 출처 sha256이 고정 값이거나 변경 기록의 최신
값인지 본다.

- 티켓은 git 밖(`tickets/`)이라 저장소 안 사본 `fixed/ticket-008.snapshot.md`를 출처로 쓴다.
- 고정 뒤 변경(2026-11-02 결정): 고정 파일은 바꾸지 않고 `fixed/amendments.json`에 항목을 더한다.
  항목마다 무엇을·왜, 새 sha256, 에피소드 결과 영향(예/아니오와 근거), "예"일 때 처리(keep / rerun /
  exclude)를 적는다. 판정 기준·정의 자체는 바꾸지 않는다(원칙 6, `changes_criteria_or_definitions`는
  항상 false).

담는 것: §3 기준과 G3·G5·G6 식(값이 아니라 식)과 상수, E-a·E-b·E-c 정의와 판정 패턴, 보조 지표
결론 규칙, S0 해시, 정책 설정, 실행 설정(동시 2), P3 에피소드 목록, 출처 파일의 sha256.
엔진의 서버 값(모델 digest, 양자화, Ollama 버전)은 P3 시작 때 서버에서 읽어 각 기록에 남긴다.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

from alfworld_pilot import constants as C
from alfworld_pilot import evidence
from alfworld_pilot.agent import PolicySettings, instructions_sha256
from alfworld_pilot.runner import first_per_type, plan_episodes

FIXED_JSON = Path("fixed/p3_fixed.json")
FIXED_SHA = Path("fixed/p3_fixed.sha256")
AMENDMENTS_JSON = Path("fixed/amendments.json")
TICKET_SNAPSHOT = Path("fixed/ticket-008.snapshot.md")
P3_PER_TYPE = 2
P3_REPEATS = 1

SOURCES = {
    "ticket": TICKET_SNAPSHOT.as_posix(),
    "s0": "skills/S0/SKILL.md",
    "evidence_code": "src/alfworld_pilot/evidence.py",
    "agent_code": "src/alfworld_pilot/agent.py",
    "constants_code": "src/alfworld_pilot/constants.py",
    "runner_code": "src/alfworld_pilot/runner.py",
    "power_code": "src/alfworld_pilot/power.py",
    "p1_expert": "records/p1_expert.json",
    "p1_pilot_pool": "records/p1_pilot_pool.json",
    "p1_counts": "records/p1_counts.json",
}

CRITERIA: dict[str, Any] = {
    "G1": {
        "rule": "잠금 파일로 새 환경을 만들고 텍스트 게임 하나를 끝까지 돈다. "
        "conda가 필요하면 조건부",
    },
    "G2": {
        "rule": "풀에서 유형마다 3~4개, 모두 20개, 리셋 3회: 첫 관측과 가능한 명령 목록이 순서까지 "
        "같고, 같은 행동 열이면 같은 관측 열",
    },
    "G3": {
        "formula": [
            "d_c = 칸 c의 S0 반복 쌍 불일치율(3쌍 평균)",
            "d_hat_c = max(d_c, d_floor)",
            "n_O,c = ceil(z^2 * d_hat_c / (h/2)^2)",
            "n_B,c = ceil(n_O,c * n_B_ratio)",
            "n_min = max_c (n_A + n_B,c + n_O,c)",
            "통과: 모든 유형에서 (train - 파일럿 게임 수) >= n_min",
        ],
        "constants": {"z": 1.96, "h": 0.2, "d_floor": 0.1, "n_B_ratio": "35/50", "n_A": 10},
    },
    "G4": {
        "rule": "S0 성공률(반복 평균의 점추정)이 range 안인 유형이 min_cells개 이상. 칸별 "
        "Clopper-Pearson 구간을 함께 보고",
        "constants": {"range": [0.15, 0.85], "min_cells": 4, "cells": 6},
    },
    "G5": {
        "formula": [
            "T_009 = E_009 * s_bar * margin",
            "s_bar = P3·P4의 평균 처리량(초/에피소드, 실제 동시 수, 실제 벽시계)",
            "E_009 = 후보 수 * (Dev-A + Dev-B + Oracle 칸 합) + S0 반복",
            "통과: T_009 <= B_009",
        ],
        "constants": {"margin": 1.1, "candidates": 40, "B_009_days": 7},
    },
    "G6": {
        "formula": [
            "Delta_t = 표적 칸 게임 집합 G에서 (탐침 1회 성공 - S0 반복 3회 다수결)의 게임 평균",
            "잡음: G를 복원 추출(|G|개)하고 매번 S0 반복 하나 r을 골라 (S0_r - 다수결)의 게임 평균",
            "q_lo, q_hi = 잡음 분포의 percentiles",
            "통과: 표적 칸 Delta_t < q_lo 그리고 비표적 칸마다 Delta가 [q_lo, q_hi] 안",
            "중립·전체 도움 탐침은 기술만 하고 판정에 넣지 않는다",
        ],
        "constants": {
            "resamples": 2000,
            "seed": 0,
            "percentiles": [2.5, 97.5],
            "resample_index": "evidencegate.exact 층화 재표본 색인",
        },
    },
    "overall": [
        "G1~G6 모두 통과: Go",
        "G3나 G5만 미달이고 축소 설계로 B_009 안: 조건부 Go",
        "G1이 조건부(conda)이고 나머지 통과: 조건부 Go",
        "그 밖: No-Go",
    ],
    "budget": {"B_009_days": 7, "pilot_hours": 12},
}


AMENDMENT_FIELDS = (
    "date",
    "what",
    "why",
    "files",
    "affects_episode_results",
    "handling",
    "changes_criteria_or_definitions",
)
HANDLING = ("keep", "rerun", "exclude")
_HEX = set("0123456789abcdef")


class AmendmentError(ValueError):
    pass


def _text(entry: dict[str, Any], field: str) -> None:
    value = entry[field]
    if not isinstance(value, str) or not value.strip():
        raise AmendmentError(f"{field} must be a non-empty string")


def validate_amendments(entries: list[dict[str, Any]]) -> None:
    for n, entry in enumerate(entries):
        missing = [f for f in AMENDMENT_FIELDS if f not in entry]
        if missing:
            raise AmendmentError(f"amendment {n}: missing {', '.join(missing)}")
        for field in ("date", "what", "why"):
            _text(entry, field)
        files = entry["files"]
        if not isinstance(files, list) or not files:
            raise AmendmentError(f"amendment {n}: files must list the changed files")
        for f in files:
            digest = f.get("sha256") if isinstance(f, dict) else None
            if not (isinstance(f, dict) and isinstance(f.get("path"), str) and f["path"]):
                raise AmendmentError(f"amendment {n}: each file needs a path")
            if not (isinstance(digest, str) and len(digest) == 64 and set(digest) <= _HEX):
                raise AmendmentError(f"amendment {n}: {f.get('path')} needs a new sha256")
        affects = entry["affects_episode_results"]
        if not isinstance(affects, dict) or affects.get("answer") not in ("yes", "no"):
            raise AmendmentError(f"amendment {n}: affects_episode_results.answer is yes or no")
        _text(affects, "reason")
        if affects["answer"] == "yes" and entry["handling"] not in HANDLING:
            raise AmendmentError(f"amendment {n}: handling must be one of {HANDLING}")
        if affects["answer"] == "no" and entry["handling"] is not None:
            raise AmendmentError(f"amendment {n}: handling is only for results that may change")
        if entry["changes_criteria_or_definitions"] is not False:
            raise AmendmentError(
                f"amendment {n}: criteria and definitions are not changed (principle 6)"
            )


def expected_source_shas(fixed: dict[str, Any], entries: list[dict[str, Any]]) -> dict[str, str]:
    """출처 경로마다 지금 있어야 할 sha256: 고정 값, 변경 기록이 있으면 그 최신 값."""
    expected = {e["path"]: e["sha256"] for e in fixed["sources"].values()}
    for entry in entries:
        for f in entry["files"]:
            if f["path"] not in expected:
                raise AmendmentError(f"{f['path']} is not a fixed source")
            expected[f["path"]] = f["sha256"]
    return expected


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_fixed(root: Path) -> dict[str, Any]:
    pool = json.loads((root / SOURCES["p1_pilot_pool"]).read_text())["games"]
    p3 = plan_episodes(first_per_type(pool, P3_PER_TYPE), P3_REPEATS)
    settings = PolicySettings()
    return {
        "ticket": "Ticket 008",
        "fixed_on": "2026-11-01",
        "criteria": CRITERIA,
        "evidence": {
            "E-a": {
                "step": "성공한 `take X from R`마다, 그 걸음 이전 관측(첫 관측 포함) 중에 "
                "R 안·위에 X가 보이는 문장이 있었는가",
                "episode": "won 그리고 성공한 take가 하나 이상 그리고 모두 걸음 판정 충족",
                "patterns": {
                    "on": evidence._ON.pattern,
                    "in_it": evidence._IN.pattern,
                    "take_command": evidence._TAKE.pattern,
                    "item": evidence._ITEM.pattern,
                    "take_success": "You pick up the {X} from the {R}.",
                    "not_contents": ["you see nothing.", "Looking quickly around you, you see ..."],
                },
            },
            "E-b": {
                "rule": "실행된 걸음 중 행동이 직전 가능한 명령 목록에 있던 비율. 형식 오류 턴은 "
                "세지 않고 형식 오류 수는 따로 기록. 보조 기록, Evidence-Safe 대응 아님",
            },
            "E-c": {
                "rule": "won 그리고 걸음 수 <= k * L_expert. 비율(걸음 수 / L_expert) 분포도 보고. "
                "성공에만 정의되는 한계를 적는다",
                "k": C.E_C_K,
                "L_expert": "records/p1_expert.json의 성공 걸음 수"
                "(전문가 시드 0, PYTHONHASHSEED=0)",
                "undefined_when": "전문가가 그 게임을 150걸음 안에 못 풀었을 때",
            },
            "conclusion_rule": {
                "base_rate_range": [0.15, 0.85],
                "min_share_of_wins_failing_e_a": 0.05,
            },
            "conclusion_text": "S0 기준률이 base_rate_range 밖이면 바닥·천장. 성공한 에피소드 중 "
            "E-a 불성립 비율이 min_share 미만이면 성공과 다른 정보 없음. 둘 다 아니면 E-a를 쓴다. "
            "아니면 대응 지표 없음, 주 지표만",
        },
        "s0": {"path": SOURCES["s0"], "sha256": _sha(root / SOURCES["s0"])},
        "policy": settings.config(),
        "instructions_sha256": instructions_sha256(),
        "engine": {
            "same_as_v1": True,
            "model": C.MODEL,
            "think": C.THINK,
            "temperature": C.TEMPERATURE,
            "keep_alive": C.KEEP_ALIVE,
            "server_values": "모델 digest, 양자화, Ollama 버전은 P3 시작 때 읽어 "
            "기록의 config.server에 남긴다",
        },
        "run": {
            "concurrency": C.CONCURRENCY,
            "server_num_parallel": C.SERVER_NUM_PARALLEL,
            "pythonhashseed": C.REQUIRED_HASH_SEED,
            "power_check": "시작할 때와 블록 끝마다: AC 전원, 덮개 열림, caffeinate 잠자기 방지",
            "block_size": C.BLOCK_SIZE,
        },
        "principle_9": "티켓 원칙 9(2026-11-01 개정)를 따른다. 티켓 sha256은 sources.ticket",
        "amendments": {
            "file": AMENDMENTS_JSON.as_posix(),
            "rule": "고정 파일은 바꾸지 않는다. 고정된 출처를 고치면 항목을 더한다. 판정 기준·정의 "
            "자체는 바꾸지 않는다(원칙 6)",
            "required_fields": list(AMENDMENT_FIELDS),
            "handling_if_results_may_change": list(HANDLING),
        },
        "p3": {
            "rule": "풀에서 과제 유형마다 경로순 앞 2개, 반복 1회",
            "episodes": [{"key": e.key, "llm_seed": e.llm_seed} for e in p3],
        },
        "sources": {
            name: {"path": path, "sha256": _sha(root / path)} for name, path in SOURCES.items()
        },
    }


def main() -> int:
    root = Path(".")
    data = build_fixed(root)
    (root / FIXED_JSON).parent.mkdir(parents=True, exist_ok=True)
    (root / FIXED_JSON).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    (root / FIXED_SHA).write_text(f"{_sha(root / FIXED_JSON)}  {FIXED_JSON.name}\n")
    if not (root / AMENDMENTS_JSON).exists():
        (root / AMENDMENTS_JSON).write_text("[]\n")
    print(_sha(root / FIXED_JSON))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
