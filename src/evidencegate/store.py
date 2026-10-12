"""에피소드 결과 JSONL 저장소.

경로는 `runs/<suite>/<safe_policy_id>__<config_id>/<tool_spec_id>/<skill_id>.jsonl`, 한 줄이
한 에피소드다(`safe_policy_id`는 `policy_id`의 `:`를 `__`로 바꾼 값). 매 에피소드마다
append 후 flush·fsync한다. 실행이 중단돼 마지막 줄이 깨졌으면 그 줄만 버리고 경고하며, 다음
append가 깨진 줄에 이어 붙지 않도록 파일을 마지막 정상 줄까지 잘라 둔다. 중간 줄이 깨졌으면
중단 때문이 아니므로 `StoreCorruptedError`를 낸다.

결과 파일에는 숨은 값이 들어갈 수 있다(채점 분석용). 이 파일을 정책 입력으로 다시 쓰지 않는다.
"""

import json
import os
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any

from evidencegate.policy import PolicyTrace, StopReason, TurnRecord
from evidencegate.runner import EpisodeKey, EpisodeRecord


class StoreCorruptedError(RuntimeError):
    """마지막 줄이 아닌 곳이 깨진 결과 파일."""


class EpisodeStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._keys: set[EpisodeKey] = {r.key for r in self._load_and_repair()}

    @classmethod
    def for_run(
        cls,
        root: Path,
        *,
        suite: str,
        policy_id: str,
        config_id: str,
        tool_spec_id: str,
        skill_id: str,
    ) -> "EpisodeStore":
        for part in (suite, policy_id, config_id, tool_spec_id, skill_id):
            if not part or "/" in part or part in {".", ".."}:
                raise ValueError(f"invalid path component: {part!r}")
        safe_policy_id = policy_id.replace(":", "__")
        run_dir = root / suite / f"{safe_policy_id}__{config_id}" / tool_spec_id
        return cls(run_dir / f"{skill_id}.jsonl")

    def __contains__(self, key: EpisodeKey) -> bool:
        return key in self._keys

    def keys(self) -> set[EpisodeKey]:
        return set(self._keys)

    def records(self) -> list[EpisodeRecord]:
        return self._load_and_repair()

    def append(self, record: EpisodeRecord) -> None:
        if record.key in self._keys:
            raise ValueError(f"key already stored: {record.key}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record_to_json(record), ensure_ascii=False, separators=(",", ":"))
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._keys.add(record.key)

    def _load_and_repair(self) -> list[EpisodeRecord]:
        if not self.path.exists():
            return []
        data = self.path.read_bytes()
        lines = data.split(b"\n")
        # 정상 파일은 b"\n"으로 끝나므로 마지막 조각은 빈 바이트다.
        complete, tail = lines[:-1], lines[-1]
        records: list[EpisodeRecord] = []
        good_bytes = 0
        for index, raw in enumerate(complete):
            try:
                records.append(record_from_json(json.loads(raw)))
            except (ValueError, KeyError, TypeError) as error:
                if index == len(complete) - 1 and not tail:
                    tail = raw  # 줄바꿈까지 쓰였지만 내용이 깨진 마지막 줄
                    break
                raise StoreCorruptedError(f"{self.path}: line {index + 1} is corrupted") from error
            good_bytes += len(raw) + 1
        if tail and good_bytes == len(data) - len(tail):
            # 내용은 다 쓰이고 줄바꿈만 빠진 줄이면 살리고 줄바꿈을 보충한다.
            try:
                record = record_from_json(json.loads(tail))
            except (ValueError, KeyError, TypeError):
                pass
            else:
                with self.path.open("ab") as f:
                    f.write(b"\n")
                return [*records, record]
        if tail:
            warnings.warn(
                f"{self.path}: dropped truncated last line ({len(tail)} bytes)",
                UserWarning,
                stacklevel=3,
            )
            with self.path.open("r+b") as f:
                f.truncate(good_bytes)
        return records


def record_to_json(record: EpisodeRecord) -> dict[str, Any]:
    return asdict(record)


def record_from_json(data: dict[str, Any]) -> EpisodeRecord:
    trace = data["trace"]
    return EpisodeRecord(
        key=EpisodeKey(**data["key"]),
        policy_config=data["policy_config"],
        llm_seed=data["llm_seed"],
        correct_disposition=data["correct_disposition"],
        evidence_safe=data["evidence_safe"],
        action=data["action"],
        missing_predicates=tuple(data["missing_predicates"]),
        failure_labels=tuple(data["failure_labels"]),
        trace=PolicyTrace(
            turns=tuple(TurnRecord(**t) for t in trace["turns"]),
            format_errors=trace["format_errors"],
            stop_reason=StopReason(trace["stop_reason"]),
            llm_config=trace["llm_config"],
            usage=trace["usage"],
        ),
        started_at=data["started_at"],
        duration_s=data["duration_s"],
        schema_version=data["schema_version"],
    )
