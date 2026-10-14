"""Dev-A 실패 궤적 내보내기 (Ticket 004 §3). proposer는 이 내보내기만 본다.

- 내보내는 것: 초기 알림, 도구 호출 순서와 각 도구의 출력(관측 그대로), 조치, Correct
  Disposition·Evidence-Safe 결과, 실패 라벨, 빠진 술어.
- 내보내지 않는 것: root_cause, has_bug, family, level, l2_subtype, incident_id, 시드, LLM seed.
  실패 라벨 중 UNNECESSARY_INTERVENTION은 뺀다(정답 처분이 observe_only라는 정보이므로).
- 사례 ID는 내용의 해시(불투명)이고, 순서는 고정 seed로 섞는다. 위치·ID로 family나 시드를
  알 수 없다.
- 실패 사례(CD 거짓 또는 ES 거짓)만, Dev-A의 반복 0만, 한 실행 조건의 레코드만 받는다.
"""

import hashlib
import json
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evidencegate.runner import EpisodeRecord, make_alert
from evidencegate.splits import dev_a
from opsworld.generator import generate_incident
from opsworld.state import Family, Level

EXPORT_SHUFFLE_SEED = 0
HIDDEN_LABELS = frozenset({"UNNECESSARY_INTERVENTION"})
_ID_LENGTH = 12


@dataclass(frozen=True)
class ProposerCase:
    export_id: str
    alert: str
    steps: tuple[dict[str, Any], ...]  # {"tool", "args", "output"}
    action: str | None
    correct_disposition: bool
    evidence_safe: bool
    failure_labels: tuple[str, ...]
    missing_predicates: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "export_id": self.export_id,
            "alert": self.alert,
            "steps": list(self.steps),
            "action": self.action,
            "correct_disposition": self.correct_disposition,
            "evidence_safe": self.evidence_safe,
            "failure_labels": list(self.failure_labels),
            "missing_predicates": list(self.missing_predicates),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "ProposerCase":
        return cls(
            export_id=data["export_id"],
            alert=data["alert"],
            steps=tuple(data["steps"]),
            action=data["action"],
            correct_disposition=data["correct_disposition"],
            evidence_safe=data["evidence_safe"],
            failure_labels=tuple(data["failure_labels"]),
            missing_predicates=tuple(data["missing_predicates"]),
        )


def export_dev_a(
    records: Sequence[EpisodeRecord],
    *,
    shuffle: bool = True,
    shuffle_seed: int = EXPORT_SHUFFLE_SEED,
) -> list[ProposerCase]:
    _check_dev_a(records)
    failures = [r for r in records if not (r.correct_disposition and r.evidence_safe)]
    ordered = sorted(failures, key=lambda r: (r.key.family, r.key.incident_seed, r.key.level))
    cases = [_to_case(r) for r in ordered]
    if shuffle:
        random.Random(shuffle_seed).shuffle(cases)
    return cases


def save_proposer_cases(cases: Sequence[ProposerCase], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps([c.to_json() for c in cases], indent=2, sort_keys=True, ensure_ascii=False)
    path.write_text(text + "\n", encoding="utf-8")


def load_proposer_cases(path: Path) -> list[ProposerCase]:
    return [ProposerCase.from_json(d) for d in json.loads(path.read_text(encoding="utf-8"))]


def _check_dev_a(records: Sequence[EpisodeRecord]) -> None:
    runs = {(r.key.policy_id, r.key.config_id, r.key.tool_spec_id, r.key.skill_id) for r in records}
    if len(runs) > 1:
        raise ValueError(f"records from mixed runs: {sorted(runs)}")
    allowed = {(c.family.value, c.incident_seed, c.level.value) for c in dev_a()}
    for r in records:
        if (r.key.family, r.key.incident_seed, r.key.level) not in allowed:
            raise ValueError(f"record is not a Dev-A case: {r.key}")
        if r.key.repeat != 0:
            raise ValueError(f"Dev-A export uses repeat 0 only, got {r.key}")


def _to_case(record: EpisodeRecord) -> ProposerCase:
    key = record.key
    incident = generate_incident(Family(key.family), key.incident_seed, Level(key.level))
    steps = []
    for turn in record.trace.turns:
        try:
            output = json.loads(turn.result_summary)
        except json.JSONDecodeError as error:
            raise ValueError(f"tool output is truncated or invalid in {key}") from error
        steps.append({"tool": turn.tool, "args": dict(turn.args), "output": output})
    content: dict[str, Any] = {
        "alert": make_alert(incident.service.service_id),
        "steps": steps,
        "action": record.action,
        "correct_disposition": record.correct_disposition,
        "evidence_safe": record.evidence_safe,
        "failure_labels": [x for x in record.failure_labels if x not in HIDDEN_LABELS],
        "missing_predicates": list(record.missing_predicates),
    }
    digest = hashlib.sha256(json.dumps(content, sort_keys=True).encode("utf-8")).hexdigest()
    return ProposerCase(
        export_id=f"case-{digest[:_ID_LENGTH]}",
        alert=content["alert"],
        steps=tuple(steps),
        action=record.action,
        correct_disposition=record.correct_disposition,
        evidence_safe=record.evidence_safe,
        failure_labels=tuple(content["failure_labels"]),
        missing_predicates=tuple(content["missing_predicates"]),
    )
