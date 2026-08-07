# Evaluation Status

Thorn has reproducible evaluation infrastructure, but it does not yet publish
a model-backed performance result. The material under [`evals/harbor/`](../evals/harbor/)
should be read as adapter, provenance, task-admission, and experiment-control
evidence—not as a benchmark score for Thorn.

## Evidence available today

### Provider-inert product checks

The normal [test and installation workflows](../.github/workflows/ci.yml)
exercise Thorn without contacting a model provider. They cover executable
behavior, packaging, optional gateway dependencies, and fresh installation
from both a built wheel and the public repository. These checks establish that
the product can be installed and that its tested behavior works; they do not
measure agent quality.

### Harbor adapter and experiment records

The [Harbor adapter](../evals/harbor/README.md) installs a pinned Thorn wheel in
a Harbor task container, runs the local `thorn run` path, and retains Thorn's
trace, structured outcome, installation record, and input provenance. The
[evaluation protocol](../evals/harbor/protocol.md) and run-manifest schema
define additional identities and outcome records for controlled trials.

This establishes a reproducible path for conducting an evaluation. The
adapter does not currently emit Harbor ATIF trajectories, and its provenance
file is not yet a complete run manifest.

### Task and environment admission

The committed [development-set declaration](../evals/harbor/task-selection.md),
[validation-v1 oracle admission](../evals/harbor/task-selection-validation-v1.md),
and [validation-v2 package and canary declaration](../evals/harbor/task-selection-validation-v2.md)
record two kinds of non-performance evidence:

- deterministic oracle runs show that the selected task packages and
  verifiers accept their bundled reference solutions; and
- provider-inert canaries show that the pinned task images start, preserve the
  expected workspace identities, and report the expected unsolved outcome
  without invoking a model.

An oracle reward belongs to the reference solution, not to Thorn. A
provider-inert canary deliberately does not attempt the task. Neither is a
measurement of agent performance.

### Declared model-backed gate

The [paired validation-v2 declaration](../evals/harbor/task-selection-validation-v2.md)
fixes two tasks, selected-Thorn and exact-Codex arms, ordering, admission
requirements, and no-retry stop rules. It explicitly does not authorize a
provider call. Its exact Thorn candidate, authorization, and arm-specific
launcher inputs remain pending, and no model-backed result from this gate is
published in the repository.

## Claims the repository does not support

The current evidence does not establish a benchmark score, population-level
success rate, comparison with another harness, improvement from any individual
policy, or a performance release gate. The small development and validation
sets are useful for integration and catastrophic-failure checks, not broad
quality estimates.

A defensible performance claim would first require a frozen Thorn revision and
configuration, provider and model identity, predeclared task selection and
metrics, completed model-backed trials including failures, and privacy-safe
publication of the resulting measurements and limitations. Repeated trials or
an untouched external gate would be needed for claims broader than the exact
runs performed.

## Calibration-only agency role

Gateway experiments that intentionally reduce prompt and tool-surface overhead
can bootstrap an agency with the non-default `LeanProjectCoordinator` role:

```console
$ thorn serve bootstrap \
    --agent-id thorn-agent \
    --project-name tasknote \
    --project-url https://gitlab.com/group/tasknote \
    --agent-class LeanProjectCoordinator \
    --agency-home ./agency \
    --agency-workspace ./workspace
```

For an existing agency, the corresponding persisted agent setting is
`"agent_class": "LeanProjectCoordinator"`. This role is an evaluation aid,
not the recommended general-purpose coordinator. It is also separate from the
current Harbor adapter, which selects the local CLI agent profile.
