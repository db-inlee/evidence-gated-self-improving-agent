"""실행 진입점.

    uv run python -m evidencegate.cli run --policy ollama:gpt-oss:20b --skill skills/S0/SKILL.md \\
        --suite calibration --n-per-family 10 --repeats 1
    uv run python -m evidencegate.cli summarize runs/calibration/...

장비(`sysctl` hw.model, CPU brand string)는 자동으로 감지해 정책 설정에 넣는다. 장비가 바뀌면
config_id가 바뀌어 결과 경로가 갈린다.

`run`은 결과를 `runs/<suite>/<safe_policy_id>__<config_id>/<tool_spec_id>/<skill_id>.jsonl`에
쌓고, 이미 있는 키는 건너뛴다(같은 명령으로 재개). 진행률, 에피소드당 평균 시간, 남은 시간을
출력한다. `summarize`는 파일이나 폴더 안의 결과 파일마다 집계를 출력한다. 반복이 2회 이상이면
반복별 집계와 노이즈 밴드를 함께 낸다.
"""

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, TextIO

import httpx

from evidencegate.ids import config_id, tool_spec_id
from evidencegate.machine import Machine, MachineDetectionError, detect_machine
from evidencegate.metrics import METRICS, GroupSummary, Summary, noise_band, summarize
from evidencegate.ollama_policy import (
    OllamaError,
    OllamaPolicy,
    OllamaSettings,
    probe_parallelism,
)
from evidencegate.runner import Case, EpisodeRecord, Progress, episode_key, run_suite
from evidencegate.skills import Skill, load_skill
from evidencegate.store import EpisodeStore
from evidencegate.suites import calibration
from opsworld import agent_env
from opsworld.state import Level

DEFAULT_OLLAMA_URL = "http://localhost:11434"
SUITES = ("calibration",)


def main(
    argv: Sequence[str] | None = None,
    *,
    http_client: httpx.Client | None = None,
    out: TextIO = sys.stdout,
    err: TextIO = sys.stderr,
    machine: Machine | None = None,
) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        if not args.policy.startswith("ollama:"):
            parser.error(f"unsupported policy {args.policy!r}; expected ollama:<model>")
        if args.concurrency < 1 or args.concurrency > args.ollama_num_parallel:
            parser.error(
                "--concurrency must be >= 1 and <= --ollama-num-parallel "
                "(the server's OLLAMA_NUM_PARALLEL)"
            )
        try:
            return _run(args, http_client, machine or detect_machine(), out)
        except MachineDetectionError as error:
            print(f"error: {error}", file=err)
            return 1
        except OllamaError as error:
            # 이미 끝난 에피소드는 저장돼 있으므로 같은 명령으로 이어서 돌 수 있다.
            print(f"error: {error}", file=err)
            if "not found" in str(error):
                model = args.policy.removeprefix("ollama:")
                print(f"hint: ollama pull {model}", file=err)
            return 1
    return _summarize(args, out)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="evidencegate")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a policy on a suite and store results")
    run.add_argument("--policy", required=True, help="ollama:<model>, e.g. ollama:gemma4:12b")
    run.add_argument("--skill", required=True, type=Path)
    run.add_argument("--suite", required=True, choices=SUITES)
    run.add_argument("--n-per-family", required=True, type=int)
    run.add_argument("--repeats", required=True, type=int)
    run.add_argument(
        "--levels",
        type=_parse_levels,
        default=(Level.L1,),
        help="comma-separated incident levels, e.g. L1 or L1,L2 (default: L1)",
    )
    run.add_argument("--runs-dir", type=Path, default=Path("runs"))
    run.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    run.add_argument(
        "--concurrency", type=int, default=1, help="episodes run at the same time (default: 1)"
    )
    run.add_argument(
        "--ollama-num-parallel",
        type=int,
        default=1,
        help="declared OLLAMA_NUM_PARALLEL of the server; recorded in the policy config",
    )

    summ = sub.add_parser("summarize", help="summarize stored results")
    summ.add_argument("paths", nargs="+", type=Path, help="result .jsonl files or folders")
    summ.add_argument("--json", action="store_true", help="print JSON instead of text")
    return parser


def _parse_levels(text: str) -> tuple[Level, ...]:
    try:
        return tuple(Level(part.strip()) for part in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"unknown level in {text!r}") from error


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def _run(
    args: argparse.Namespace, http_client: httpx.Client | None, machine: Machine, out: TextIO
) -> int:
    client = http_client or httpx.Client(base_url=args.ollama_url)
    model = args.policy.removeprefix("ollama:")
    settings = OllamaSettings.for_model(
        model, concurrency=args.concurrency, server_num_parallel=args.ollama_num_parallel
    )
    if settings.concurrency > 1:
        # 서버가 실제로 병렬 처리하지 않으면 조용히 순차가 되므로 시작 전에 확인한다.
        probe = probe_parallelism(client, settings)
        verdict = "ok" if probe.parallel else "FAILED"
        print(
            f"parallel probe: single {probe.single_s:.2f}s, {settings.concurrency} at once "
            f"{probe.concurrent_s:.2f}s, ratio {probe.ratio:.2f} -> {verdict}",
            file=out,
        )
        if not probe.parallel:
            raise OllamaError(
                f"server is not processing requests in parallel (ratio {probe.ratio:.2f}); "
                f"restart Ollama with OLLAMA_NUM_PARALLEL={settings.server_num_parallel}"
            )
    policy = OllamaPolicy.connect(settings, client, machine=machine)
    skill = load_skill(args.skill)
    cases = calibration(args.n_per_family, levels=args.levels)
    store = EpisodeStore.for_run(
        args.runs_dir,
        suite=args.suite,
        policy_id=policy.policy_id,
        config_id=config_id(policy),
        tool_spec_id=tool_spec_id(agent_env.TOOL_SPECS),
        skill_id=skill.skill_id,
    )

    total = len(cases) * args.repeats
    stored = total - _count_missing(store, policy, skill, cases, args.repeats)
    print(f"store: {store.path}", file=out)
    print(
        f"policy: {policy.policy_id}  config_id={config_id(policy)}  "
        f"tool_spec_id={tool_spec_id(agent_env.TOOL_SPECS)}  skill_id={skill.skill_id}",
        file=out,
    )
    print(
        f"cases: {len(cases)} x repeats {args.repeats} = {total} episodes "
        f"(to run: {total - stored}, already stored: {stored})",
        file=out,
    )

    def report(p: Progress) -> None:
        key = p.record.key
        print(
            f"[{p.done}/{p.total}] {key.family} {key.incident_seed} r{key.repeat}  "
            f"{p.record.duration_s:.1f}s  avg {p.mean_s:.1f}s/ep  ETA {_duration(p.eta_s)}",
            file=out,
            flush=True,
        )

    result = run_suite(
        policy, skill, cases, args.repeats, store, on_progress=report, workers=settings.concurrency
    )
    print(f"done: ran {result.executed}, skipped {result.skipped}", file=out)
    _print_file_summary(store.path, store.records(), out)
    return 0


def _count_missing(
    store: EpisodeStore, policy: OllamaPolicy, skill: Skill, cases: Sequence[Case], repeats: int
) -> int:
    return sum(
        episode_key(policy, skill, c.family, c.incident_seed, r, c.level) not in store
        for r in range(repeats)
        for c in cases
    )


def _duration(seconds: float) -> str:
    seconds = round(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------


def _summarize(args: argparse.Namespace, out: TextIO) -> int:
    # 실행 중인 파일도 집계할 수 있도록 읽기 전용으로 연다(파일을 고치지 않음).
    files = _result_files(args.paths)
    if args.json:
        payload = [_file_json(path, EpisodeStore(path, repair=False).records()) for path in files]
        json.dump(payload, out, indent=2, default=str)
        out.write("\n")
        return 0
    for path in files:
        _print_file_summary(path, EpisodeStore(path, repair=False).records(), out)
    return 0


def _result_files(paths: Sequence[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        files.extend(sorted(path.rglob("*.jsonl")) if path.is_dir() else [path])
    return files


def _print_file_summary(path: Path, records: list[EpisodeRecord], out: TextIO) -> None:
    print(f"\n== {path}", file=out)
    if not records:
        print("(no records)", file=out)
        return
    key = records[0].key
    repeats = sorted({r.key.repeat for r in records})
    print(
        f"policy {key.policy_id}  config_id {key.config_id}  tool_spec_id {key.tool_spec_id}  "
        f"skill_id {key.skill_id}  records {len(records)}  repeats {repeats}",
        file=out,
    )
    llm = records[0].trace.llm_config
    if llm:
        print(
            f"model {llm['model']} ({llm.get('quantization')}, {llm.get('model_digest')})  "
            f"ollama {llm.get('ollama_version')}  on {llm.get('hw_model')} ({llm.get('chip')})",
            file=out,
        )
    cache = _cache_stats(records)
    if cache is not None:
        print(cache, file=out)

    if len(repeats) == 1:
        _print_summary(f"repeat {repeats[0]}", summarize(records), out)
        return
    for repeat in repeats:
        subset = [r for r in records if r.key.repeat == repeat]
        _print_summary(f"repeat {repeat}", summarize(subset), out)
    print(f"-- noise band (repeats {repeats}, {len(records) // len(repeats)} cases)", file=out)
    for metric in METRICS:
        try:
            band = noise_band(records, metric)
        except ValueError as error:
            print(f"{metric}: unavailable ({error})", file=out)
            continue
        print(
            f"{metric}: repeat_std={band.repeat_std:.3f}  "
            f"paired_abs_p95={band.paired_abs_p95:.3f}  "
            f"case_mismatch_rate={band.case_mismatch_rate:.3f}",
            file=out,
        )


def _print_summary(title: str, summary: Summary, out: TextIO) -> None:
    print(f"-- {title} (n={summary.overall.n})", file=out)
    print(
        f"{'group':<20}{'n':>4}  {'correct_disposition':<22}  {'evidence_safe':<22}  "
        f"{'correct_but_unsafe':<22}  {'fmt_err':>7}  {'calls':>5}  {'sec':>6}",
        file=out,
    )
    rows = [("overall", summary.overall), *summary.by_family.items()]
    if len(summary.by_level) > 1:
        rows += [*summary.by_level.items(), *summary.by_family_level.items()]
    for name, group in rows:
        print(_row(name, group), file=out)
    labels = ", ".join(f"{k}={v}" for k, v in sorted(summary.overall.failure_labels.items()))
    print(f"labels: {labels or '-'}", file=out)


def _row(name: str, g: GroupSummary) -> str:
    def rate(p: Any) -> str:
        return f"{p.rate:.3f} [{p.ci_low:.3f}, {p.ci_high:.3f}]"

    return (
        f"{name:<20}{g.n:>4}  {rate(g.correct_disposition):<22}  {rate(g.evidence_safe):<22}  "
        f"{rate(g.correct_but_unsafe):<22}  {g.format_error_rate:>7.3f}  "
        f"{g.mean_tool_calls:>5.2f}  {g.mean_duration_s:>6.1f}"
    )


def _cache_stats(records: Sequence[EpisodeRecord]) -> str | None:
    """prompt 캐시 지표. 비율 = cached / prompt_eval (정의는 B3 연기 테스트에서 실측으로 확인)."""
    cached = prompt = later = later_hits = 0
    seen = False
    for record in records:
        usage = record.trace.usage
        if not usage:
            continue
        for index, call in enumerate(usage["calls"]):
            if call.get("prompt_eval_cached_count") is None:
                continue
            seen = True
            cached += call["prompt_eval_cached_count"]
            prompt += call.get("prompt_eval_count") or 0
            if index >= 1:
                later += 1
                later_hits += call["prompt_eval_cached_count"] > 0
    if not seen:
        return None
    share = f"{cached / prompt:.1%}" if prompt else "n/a"
    return (
        f"cache: cached={cached} prompt_eval={prompt} ({share}); "
        f"later calls with cached>0: {later_hits}/{later}"
    )


def _file_json(path: Path, records: list[EpisodeRecord]) -> dict[str, Any]:
    repeats = sorted({r.key.repeat for r in records})
    data: dict[str, Any] = {"path": str(path), "records": len(records), "repeats": repeats}
    if records:
        data["summary"] = asdict(summarize(records))
        data["by_repeat"] = {
            r: asdict(summarize([x for x in records if x.key.repeat == r])) for r in repeats
        }
        if len(repeats) > 1:
            data["noise_band"] = {
                m: {
                    k: v
                    for k, v in asdict(noise_band(records, m)).items()
                    if k not in {"pair_mean_diffs", "pair_mismatch_rates"}
                }
                for m in METRICS
            }
    return data


if __name__ == "__main__":
    raise SystemExit(main())
