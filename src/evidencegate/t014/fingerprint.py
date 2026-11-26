"""Ticket 014 환경 지문(사전 등록 1절, 티켓 §11.3).

- 단계마다 파일 하나(`runs/ticket014/stage<N>/fingerprint.json`)다. 그 단계의 첫 실행에 쓰고 다시
  고치지 않는다. 재개할 때마다 다시 재서 대조하고, 다르면 실행기가 종료 코드 2로 끝낸다.
- 항목:
  - 환경(모든 단계가 같아야 함): Ollama 버전, 모델 digest, config_id, tool_spec_id, S0 skill_id,
    코드 트리(`git rev-parse HEAD:src`), 사전 등록 sha256.
  - 단계: 단계 번호, 실행 계획 sha256, 계획의 skill_id 목록.
  - 기록만(대조하지 않음): 코드 커밋(`git rev-parse HEAD`). 단계 사이에 David가 결과 파일(풀 등)을
    커밋하면 커밋 해시는 바뀌지만 코드(`src/`)는 같아야 하므로, 대조는 코드 트리로 한다.
- `src/`에 커밋하지 않은 변경이 있으면 지문을 만들지 않는다(FingerprintError).
"""

import json
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from evidencegate.eval_constants import S0_SKILL_ID
from evidencegate.ids import content_id, tool_spec_id
from evidencegate.run_plan import Block
from opsworld import agent_env

Git = Callable[[Sequence[str]], tuple[int, str]]

ENVIRONMENT_FIELDS = (
    "ollama_version",
    "model_digest",
    "config_id",
    "tool_spec_id",
    "s0_skill_id",
    "code_tree",
    "prereg_sha256",
)
STAGE_FIELDS = ("stage", "plan_sha256", "skill_ids")
NOT_COMPARED = ("code_commit",)


class FingerprintError(RuntimeError):
    pass


def system_git(argv: Sequence[str]) -> tuple[int, str]:
    done = subprocess.run(list(argv), capture_output=True, text=True, check=False)
    return done.returncode, done.stdout


def _git(git: Git, root: Path, *args: str) -> str:
    code, out = git(["git", "-C", str(root), *args])
    if code != 0:
        raise FingerprintError(f"git {' '.join(args)} failed (exit {code})")
    return out.strip()


def collect(
    *,
    stage: int,
    config: Mapping[str, Any],
    plan: Sequence[Block],
    plan_sha256: str,
    prereg_sha256: str,
    root: Path,
    git: Git = system_git,
) -> dict[str, Any]:
    if _git(git, root, "status", "--porcelain", "--", "src"):
        raise FingerprintError("src/ has uncommitted changes; commit the code before a run")
    return {
        "stage": stage,
        "ollama_version": config.get("ollama_version"),
        "model_digest": config.get("model_digest"),
        "config_id": content_id(dict(config)),
        "tool_spec_id": tool_spec_id(agent_env.TOOL_SPECS),
        "s0_skill_id": S0_SKILL_ID,
        "code_tree": _git(git, root, "rev-parse", "HEAD:src"),
        "code_commit": _git(git, root, "rev-parse", "HEAD"),
        "prereg_sha256": prereg_sha256,
        "plan_sha256": plan_sha256,
        "skill_ids": sorted({b.skill_id for b in plan}),
    }


def _dump(data: Mapping[str, Any]) -> str:
    return json.dumps(data, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def _differences(
    recorded: Mapping[str, Any], now: Mapping[str, Any], fields: Sequence[str]
) -> list[str]:
    return [
        f"{k}: recorded {recorded.get(k)}, now {now.get(k)}"
        for k in fields
        if recorded.get(k) != now.get(k)
    ]


def ensure(path: Path, now: Mapping[str, Any]) -> list[str]:
    """없으면 쓰고 [], 있으면 대조해 다른 항목들(파일은 고치지 않는다)."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_dump(now), encoding="utf-8")
        return []
    recorded = json.loads(path.read_text(encoding="utf-8"))
    return _differences(recorded, now, (*ENVIRONMENT_FIELDS, *STAGE_FIELDS))


def compare_environment(earlier: Sequence[Path], now: Mapping[str, Any]) -> list[str]:
    """앞 단계 지문 파일들과 환경 항목을 대조한다."""
    problems = []
    for path in earlier:
        recorded = json.loads(path.read_text(encoding="utf-8"))
        problems += [
            f"{k}: stage {recorded.get('stage')} recorded {recorded.get(k)}, now {now.get(k)}"
            for k in ENVIRONMENT_FIELDS
            if recorded.get(k) != now.get(k)
        ]
    return problems
