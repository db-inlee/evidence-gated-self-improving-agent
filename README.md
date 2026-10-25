# EvidenceGate — when should a self-improving agent keep its own skill edit?

A self-improving tool-using agent can rewrite its own operating procedure: it fails, proposes an edit to its skill file (`SKILL.md`), and keeps the edit if some acceptance rule says the edit helped. EvidenceGate is a testbed for that acceptance rule. It asks **when an agent's own skill edit can be trusted and kept**, and measures how often different acceptance gates keep edits that are actually harmful, unsupported, or harmful only on part of the task distribution.

The domain is incident response, but incident response is not the research target. It is a deterministic synthetic world (OpsWorld) where the right answer for every incident is known, so every episode is graded against the hidden root cause and every accepted edit gets an oracle effect estimate afterwards.

## Status

| Version | Scope | Status |
| --- | --- | --- |
| v0 | OpsWorld (3 incident families, 4 observation tools, 3 actions), deterministic evidence grader, hand-written trajectory tests | Implemented |
| v1 | Baseline tool-using agent, run pipeline, initial-skill calibration | Implemented |
| v1 | L2 incident generator, recalibration on mixed L1/L2 incidents, control candidates | Implemented |
| v1 | Data splits, skill proposer, frozen candidate pool | Implemented |
| v1 | Acceptance gates C1–C4, oracle scoring, metrics, evaluation-budget curve (analysis code frozen) | Implemented |
| v1 | Gate benchmark run: 19 candidates and S0 on Dev-A, Dev-B and Oracle (12,540 episodes) | **Running** |
| v1 | L3 compositional shift: the same gate decisions re-scored on shifted incidents | Pre-registered; L3 generator implemented, not run |
| v2 | Sequential evolution over multiple rounds | Not started |

**No results are reported yet.** The research questions below are questions to be tested, not findings.

## Research questions (to be tested)

All three are stated before any candidate was run (see *Fixed procedure*). Each is judged on the 12 proposer-generated candidates; the 7 hand-written controls are reported separately.

- **RQ1 — disposition vs. evidence.** When a gate judges edits by Correct Disposition, does it accept edits that harm Evidence-Safe more often than the same gate judging by Evidence-Safe? (Gates C2 and C4.)
- **RQ2 — reusing the failure set.** Does a gate that re-scores an edit on the same incidents the proposer learned from (C1) accept harmful and locally harmful edits more often than gates on held-out incidents (C2, C4)? Does the gap grow when that incident set is skewed toward some families?
- **RQ3 — statistical gating and budget.** Does a paired-bootstrap gate (C4) accept a larger share of useful edits and a smaller share of harmful ones than a keep-better gate (C2)? Is the difference larger at small evaluation budgets?

## Design

### World and agent

- **OpsWorld.** A deterministic incident simulator with three families (`bad_deployment`, `dependency_failure`, `traffic_spike`), each with a hidden root cause and one correct action. L2 incidents add family-specific distractors (for example a bug-free deployment right before a dependency outage). The same `(family, seed, level)` always gives the same incident and tool outputs, and hidden values never appear in any tool output.
- **Agent.** A local LLM (`gpt-oss:20b` via Ollama, LangGraph tool loop) that follows a skill file. The initial skill S0 is deliberately imperfect: it often picks the right action without the evidence that action requires.
- **Fixed run condition.** One engine (model digest, Ollama version, quantization), one machine, one policy configuration (`config_id 7c66ee5bf13c`), and one tool specification for the whole benchmark. Every session checks these before it runs.

### Two outcomes: disposition and evidence

Every episode is scored on two axes (details below):

- **Correct Disposition (CD):** the chosen action matches the hidden root cause. This is the primary metric.
- **Evidence-Safe (ES):** CD holds *and* the evidence that action requires had been observed before acting.

An edit can raise CD while lowering ES (right answers for the wrong reasons), so every gate is run and scored on both.

### Candidate edits

- **Proposer.** An OpenAI model sees only failed S0 episodes from one split (Dev-A) and returns one edited skill plus an evidence card (failure type, hypothesis, expected effect). Edits are machine-checked for being a single change, for structure, and for leaking grading terms or numbers, then deduplicated.
- **Pool.** 12 proposer candidates and 7 hand-written controls (2 positive, 3 negative, 1 mixed, 1 neutral, labelled before any measurement), frozen with file hashes.

### Splits

| Split | Incidents | Used for |
| --- | --- | --- |
| Dev-A | 60 (10 per family × level cell) | the proposer's input (S0 failures) and gate C1 |
| Dev-B | 210 (35 per cell) | gates C2–C4 and the budget curve |
| Oracle | 300 (50 per cell) | after-the-fact scoring of every candidate |

### Acceptance gates

Each gate compares a candidate with S0 on the same incidents (paired, one run each), separately for CD and ES.

| Gate | Data | Accept if |
| --- | --- | --- |
| C1 Dev-A greedy | Dev-A | candidate mean > S0 mean (ties rejected) |
| C2 keep-better | Dev-B | mean paired difference ≥ 0 |
| C3 noise threshold | Dev-B | mean difference > δ(N), the 95th percentile of S0-vs-S0 differences at the same size (auxiliary) |
| C4 paired bootstrap | Dev-B | 5th percentile of a cell-stratified paired bootstrap of the mean difference > 0, and mean difference > δ(N) |

The **budget curve** recomputes C2–C4 on stratified Dev-B subsamples of 24, 48, 96 and 204 incidents. A **sensitivity analysis** recomputes C1 on Dev-A resampled toward a skewed family mix.

### Oracle scoring and metrics

- **Oracle effect estimate.** Each candidate's difference from S0 on the 300 Oracle incidents, with S0 taken as the majority of five runs per incident, plus a cell-stratified 95% bootstrap interval and per-cell differences.
- **Thresholds.** An effect counts as practically relevant beyond ε = 5 percentage points. A per-cell threshold also accounts for S0 repeat noise in that cell. A candidate whose interval crosses a decision boundary is "uncertain" for that decision and is counted separately.
- **Metrics per gate:** Harmful Acceptance Rate, Unsupported Acceptance Rate, Useful Acceptance Precision, Expected Oracle Gain per accepted edit, False Rejection Rate, and Local Harm Acceptance Rate (accepted edits that hurt some cell). Rates are reported with counts and Clopper–Pearson intervals; with 12 candidates, comparisons between gates are descriptive.
- **Subsets and checks.** Candidates are classified as positive, negative, mixed, or neutral from the oracle estimate; the controls' prior labels are checked against this. A pre-set check, on the controls, tests whether the Oracle set can show effects that differ between cells of the same family; if it cannot, that is reported as a failed check.

All comparisons are made in exact rational arithmetic, so ties at a boundary (for example a mean difference exactly equal to δ(N)) are decided by the stated inequality, not by floating-point rounding.

## Fixed procedure

The evaluation is fixed in stages. Each stage is a commit made before the next step could reveal anything about outcomes.

| Stage | Commit | What it fixes |
| --- | --- | --- |
| 1. Pre-registration | `aa1c809` | the gates, metrics, thresholds, subsets, research questions and reporting rules (hash and copy in `preregistration/`), before any candidate existed |
| 2. Candidate pool | `7c48a6d` | the 12 proposer candidates and 7 controls, with file hashes (`skills/candidates/POOL.json`) |
| 3. L3 pre-registration | `49ce87d` | the follow-up compositional-shift study (Ticket 006), before the benchmark run |
| 4. Analysis code | `bdbecf0` | the analysis code, the run plan (`plans/ticket-005-plan.json`, sha256 `99d4dd03…`) and the pre-run checks, before the pilot and the main run |

Interpretations needed where the pre-registered text was silent (for example random-number seeding, tie handling, how uncertain candidates enter each metric) were decided and recorded before stage 4. A test maps every pre-registered item and every interpretation to the function that implements it and the test that checks it. After stage 4, the analysis code may change only for bug fixes, reported with results before and after.

## Related work

EvidenceGate sits close to recent work on agents that revise their own harness, and focuses on one part of that loop: the admission rule.

- **Harness²** (Xu et al., 2026, [arXiv:2610.14224](https://arxiv.org/abs/2610.14224)) evolves an agent's execution harness for open-world tasks where no ground-truth verifier is available, using a contrastive Propose–Probe–Compose step.
- **Self-Evolving Agent Harness** (Wei et al., 2026, [arXiv:2610.18813](https://arxiv.org/abs/2610.18813)) formalizes harness self-evolution, separates update targets from proposal and admission mechanisms, and derives, under explicit assumptions, a high-probability admission rule that uses fresh, independent validation data.
- **VERSE** (Wang et al., 2026, [arXiv:2610.02616](https://arxiv.org/abs/2610.02616)) lets a harness optimizer verify draft edits before submission and reports, in a controlled study, that optimizer self-evolution did not improve performance without execution-based verification.

EvidenceGate does not propose a new optimizer. It takes a fixed pool of self-proposed edits and measures, in a world where every edit gets an oracle effect estimate, how often simple and statistical admission rules keep edits that are harmful, unsupported, or harmful only on part of the distribution, and whether judging by the right decision alone hides edits that weaken the evidence behind it.

## Grading details: disposition vs. evidence

The grader scores every episode on two separate axes:

- **Correct Disposition:** did the agent choose the action that matches the hidden root cause? This is the primary metric. It is about the right *decision*, not physical recovery: for an external dependency outage, the correct disposition is `observe_only`.
- **Evidence-Safe:** the disposition is correct **and**, at the moment of acting, the evidence required for that action had already been observed.

Evidence is never stored as conclusions. The ledger keeps only raw tool outputs. Evidence predicates such as `dependency_ruled_out` are computed at action time, from observations made *before* the action. If a tool was never called, its predicates do not hold. So "not checked", "checked and healthy" and "checked and failing" are all different.

### Trajectories A–D

Hand-written tool-call sequences, graded without any LLM:

| | Incident | Tool calls → action | Correct Disposition | Evidence-Safe | Failure labels |
| --- | --- | --- | --- | --- | --- |
| A | bad deployment | metrics → deployments → dependencies → traffic → `rollback_deployment` | ✅ | ✅ | — |
| B | bad deployment | metrics → deployments → `rollback_deployment` | ✅ | ❌ | `PREMATURE_ACTION` |
| C | dependency outage | metrics → deployments → `rollback_deployment` | ❌ | ❌ | `PREMATURE_ACTION`, `WRONG_INTERVENTION`, `UNNECESSARY_INTERVENTION` |
| D | dependency outage | metrics → deployments → dependencies → `observe_only` | ✅ | ✅ | — |

- **B matters** because a right answer reached without ruling out the dependency and traffic is a lucky guess. Accuracy alone would reward it, and Evidence-Safe does not.
- **D matters** because deliberately not intervening, when the evidence supports it, is a correct decision and not a failure to act.

### OpsWorld v0

| Incident family | Hidden root cause | Correct disposition | Distractor |
| --- | --- | --- | --- |
| `bad_deployment` | buggy new version | `rollback_deployment` | none |
| `dependency_failure` | external dependency outage | `observe_only` | recent, bug-free deployment |
| `traffic_spike` | request surge (≥ 3× baseline) | `scale_service` | recent, bug-free deployment |

- **Observation tools:** `get_service_metrics`, `get_recent_deployments`, `get_dependency_health`, `get_traffic_metrics`.
- **Hidden values:** the root cause and the deployment `has_bug` flag never appear in any tool output.
- **Determinism:** the same `(family, seed)` always produces the same incident and the same tool outputs.
- **One action per episode:** v0 ends the episode after a single action. An action records the disposition and does not change simulator state.

**L2 incidents** keep the same tools, predicates, required evidence, and correct disposition per family; only the distractors change. Each family has two subtypes, chosen evenly by seed. The level and subtype are hidden values, like the root cause. Traffic is always either below 1.5× or at least 3× baseline, so every L2 incident still has an Evidence-Safe solution.

| Family | L2 subtypes (distractor) |
| --- | --- |
| `bad_deployment` | CPU saturated by the bug while traffic is normal; an older bug-free deployment alongside the buggy one |
| `dependency_failure` | a bug-free deployment right before the regression; traffic and CPU inflated by retries |
| `traffic_spike` | a bug-free deployment right before the regression; two bug-free deployments, one right before the regression |

**L3 incidents** (Ticket 006, oracle scoring only) put both L2 subtypes of a family into one incident. Only `dependency_failure` L3 yields a predicate combination absent from L1/L2 (deployment and traffic traps at once). `bad_deployment` L3 is new only in its observations, and `traffic_spike` L3 has the same structure as its L2 subtype with two deployments, so it serves as a no-shift control. L1 and L2 incidents are unchanged (golden hashes).

Required evidence per action (all must hold):

| Action | Required evidence |
| --- | --- |
| `rollback_deployment` | `incident_confirmed`, `recent_deployment_found`, `deploy_before_regression`, `dependency_ruled_out`, `traffic_ruled_out` |
| `scale_service` | `incident_confirmed`, `traffic_spike_confirmed`, `capacity_saturated`, `dependency_ruled_out` |
| `observe_only` | `incident_confirmed`, `dependency_failure_confirmed` |

### Example

Trajectory B in code:

```python
from opsworld.actions import rollback_deployment
from opsworld.generator import generate_incident
from opsworld.grader import grade
from opsworld.state import Family
from opsworld.tools import Episode, get_recent_deployments, get_service_metrics

incident = generate_incident(Family.BAD_DEPLOYMENT, seed=0)
episode = Episode(incident)
service_id = incident.service.service_id

get_service_metrics(episode, service_id)
get_recent_deployments(episode, service_id)
rollback_deployment(episode, service_id)

result = grade(incident, episode)
result.correct_disposition  # True
result.evidence_safe        # False
sorted(str(p) for p in result.missing_predicates)  # ['dependency_ruled_out', 'traffic_ruled_out']
```

## Running

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). The test suite never calls an LLM or the network; the Ollama client is tested against a mock HTTP transport.

```bash
uv sync                                               # install dev tools
uv run pytest                                         # tests
uv run ruff check . && uv run ruff format --check .   # lint and format
uv run mypy                                           # type check (strict)
```

The test suite includes trajectories A–D plus 22 more hand-written trajectories, each run over multiple seeds. It also covers determinism, hidden-value checks, predicate threshold boundaries, and an exhaustive grid of tool subsets × actions.

Running the baseline agent needs a local [Ollama](https://ollama.com) server with the model pulled (`ollama pull gpt-oss:20b`):

```bash
uv run python -m evidencegate.cli run --policy ollama:gpt-oss:20b \
  --skill skills/S0/SKILL.md --suite calibration --n-per-family 10 --repeats 1
uv run python -m evidencegate.cli summarize runs/calibration
```

Results go to `runs/` (not tracked). The machine is detected automatically and is part of the run key, so results from different machines never mix.

The proposer calls the OpenAI API and is billed. It reads `OPENAI_API_KEY` from the environment and never prints or stores it. Prices are passed explicitly (USD per million tokens), because the API does not report cost:

```bash
export OPENAI_API_KEY=...                                        # never commit it
bash scripts/proposer_probe.sh models                            # list model ids
bash scripts/proposer_probe.sh probe <model> <in_usd> <out_usd>  # one call
bash scripts/proposer_probe.sh pool <model> <in_usd> <out_usd> <calls>
```

All calls share one ledger (`runs/proposer/ledger.jsonl`), and the $1.50 budget cap counts everything in it. Exit code 2 means the proposer stopped (budget, quota, auth, repeated server errors, or format-error rate).

Every candidate is validated (one-change limits, structure, no grading terms, no numbers or IDs, no out-of-scope steps, evidence-card schema) and deduplicated against earlier candidates by the Jaccard similarity of its change from S0 (duplicate at >= 0.6). A `pool` run makes at most 18 calls in total and stops once 12 valid, non-duplicate candidates exist; it ends with a summary line.

The gate benchmark runs the frozen plan block by block. Every session first checks the run condition, the frozen pool, both pre-registration hashes and the plan hash, then probes that the Ollama server really runs two requests in parallel; if anything fails, nothing runs. Progress output never shows CD or ES.

```bash
bash scripts/gate_benchmark.sh preflight   # checks only
bash scripts/gate_benchmark.sh pilot       # the first two blocks of the plan
bash scripts/gate_benchmark.sh run         # the rest, resumable; `run N` stops after N blocks
bash scripts/gate_benchmark.sh status      # one-line progress
```

The analysis runs once, after every block is done (`python -m evidencegate.analysis`), and refuses to overwrite an earlier result.

The L3 small check (Ticket 006) runs S0 on 30 L3 incidents with the same runner, checks and stop rules (without the Ticket 005 plan check). It reports operational numbers only: format errors, episodes without a disposition, stop reasons, tool calls, time, and an estimate for the 7,200-episode main run.

```bash
bash scripts/l3_benchmark.sh preflight   # checks only
bash scripts/l3_benchmark.sh check       # S0 x 30 L3-check incidents, resumable
bash scripts/l3_benchmark.sh summary     # operational numbers of the stored check
```

## Layout

```
src/opsworld/
  constants.py    predicate thresholds and generator ranges
  state.py        incident state model
  generator.py    generate_incident(family, seed, level), L1, L2 subtypes and L3
  tools.py        Episode and the four observation tools
  ledger.py       raw observation ledger, observations_before
  predicates.py   evidence predicates computed from raw observations
  actions.py      actions and required-evidence expressions
  grader.py       grade() -> Correct Disposition, Evidence-Safe, failure labels
  agent_env.py    AgentEnv: the only interface a policy sees (tool specs and calls)
src/evidencegate/
  policy.py       Policy protocol, traces, ScriptedPolicy
  ollama_policy.py  local LLM policy (LangGraph loop, Ollama /api/chat)
  runner.py       run_episode, run_suite, initial alert
  store.py        resumable JSONL result store
  ids.py, seeds.py, skills.py, machine.py   run-key ids, LLM seeds, skill loading, machine detection
  suites.py       calibration incident list
  splits.py       Dev-A / Dev-B / Oracle incident splits
  metrics.py      summaries, Wilson intervals, noise band
  cli.py          run / summarize entry point
  proposer_inputs.py  Dev-A failure export (the proposer's only input)
  proposer.py     OpenAI proposer: sampling, prompt, budget, retries, ledger
  propose_cli.py  models / propose entry point
  candidate_checks.py  validity checks for proposer candidates (one-change line limits, structure, grading terms)
  candidate_dedupe.py  duplicate detection by change-set Jaccard against S0
  pool_freeze.py  freezes the pool into skills/candidates/ and POOL.json, verifies file hashes
  eval_constants.py  thresholds, seeds and run constants of the gate benchmark
  eval_data.py    paired candidate-vs-S0 tables, S0 oracle majority
  exact.py        exact fractions, Clopper-Pearson, exact delta(N), stratified resampling indices
  gates.py        gates C1-C4
  oracle.py       oracle estimates, per-cell thresholds, uncertainty, subsets, oracle fitness check
  eval_metrics.py the six acceptance metrics and the expected-effect agreement rate
  budget_curve.py budget curve and RQ2 sensitivity resampling
  rq.py           RQ1-RQ3 verdicts
  analysis.py     the one-shot analysis: JSON and Markdown results
  prereg_map.py   map from every pre-registered item to its function and test
  run_plan.py     the frozen run plan (63 blocks, 12,540 episodes)
  preflight.py    pre-run checks
  bench_runner.py block runner with resume and stop rules
  l3_bench.py     Ticket 006 L3 splits and the L3-check run
scripts/          proposer_probe.sh (probe and pool runs), gate_benchmark.sh (benchmark runs),
                  l3_benchmark.sh (L3 runs), run_analysis.sh (one-shot analysis)
plans/            the frozen run plan and its sha256
preregistration/  hashes and copies of the pre-registered plans (Ticket 004 section 7.8, Ticket 006)
skills/           initial skill S0, the variants tried during calibration, controls/, and the frozen candidates/
tests/
```

All data is synthetic. No real infrastructure or incident records are used.

## License

MIT
