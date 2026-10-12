# EvidenceGate — statistically gated self-improvement for tool-using agents

EvidenceGate is a testbed for one question: **when can a self-improving agent's own procedure change be trusted and kept?** The planned loop is: a tool-using agent fails, proposes an edit to its operating skill (`SKILL.md`), and the edit is kept only if it survives evidence-aware, held-out, paired statistical validation. The research target is the *acceptance rule* for skill edits, not incident response itself. Incident response is just the domain the loop is meant to improve, inside a deterministic synthetic incident simulator where the right answer is always known.

## Status / Roadmap

Only v0 is implemented so far. This repository currently contains **OpsWorld**, a deterministic incident simulator, and an **evidence grader**. There is no LLM agent, skill proposer, or acceptance gate yet, and no experimental results.

| Version | Scope | Status |
| --- | --- | --- |
| v0 | OpsWorld (3 incident families, 4 observation tools, 3 actions), deterministic evidence grader, hand-written trajectory tests | Implemented |
| v1 | Baseline tool-using agent, skill proposer, candidate edit pool, acceptance gates (same-set greedy, keep-better, paired-bootstrap gate), evaluation-budget curve | Not started |
| v2 | Sequential evolution over multiple rounds | Not started |
| v3 | More incident types, out-of-distribution evaluation | Not started |

## Core idea: disposition vs. evidence

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

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). There are no runtime dependencies and nothing calls an LLM or the network.

```bash
uv sync                                               # install dev tools
uv run pytest                                         # tests
uv run ruff check . && uv run ruff format --check .   # lint and format
uv run mypy                                           # type check (strict)
```

The test suite includes trajectories A–D plus 22 more hand-written trajectories, each run over multiple seeds. It also covers determinism, hidden-value checks, predicate threshold boundaries, and an exhaustive grid of tool subsets × actions.

## Layout

```
src/opsworld/
  constants.py    predicate thresholds and generator ranges
  state.py        incident state model
  generator.py    generate_incident(family, seed)
  tools.py        Episode and the four observation tools
  ledger.py       raw observation ledger, observations_before
  predicates.py   evidence predicates computed from raw observations
  actions.py      actions and required-evidence expressions
  grader.py       grade() -> Correct Disposition, Evidence-Safe, failure labels
tests/
```

All data is synthetic. No real infrastructure or incident records are used.

## License

MIT
