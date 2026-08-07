# Thorn under Harbor

This directory contains Thorn's initial custom installed-agent adapter for
[Harbor](https://github.com/laude-institute/harbor). It runs the existing
local/direct `thorn run` path inside Harbor's task container. Thorn uses its
subprocess tool executor, so the Harbor task container remains the isolation
boundary; no broker, OCI sidecar, or nested container runtime is involved.

See [Evaluation status](../../docs/evaluation.md) for what this infrastructure
does and does not establish. The adapter and its admission receipts are not a
published Thorn performance result.

The integration is pinned to Harbor commit
`071281b3d931aafd6a5375fa7d5933e23054d784`. Harbor is an external evaluation
tool, not a Thorn package dependency. The adapter installs a caller-supplied
Thorn wheel and frozen dependency constraints with uv `0.7.13` and uv-managed
CPython `3.11`.

## Build the inputs

Use a detached worktree so that the wheel corresponds exactly to the revision
recorded in every trial:

```bash
THORN_CHECKOUT=/path/to/thorn
THORN_REVISION="$(git -C "$THORN_CHECKOUT" rev-parse HEAD)"
THORN_BUILD_CHECKOUT=/tmp/thorn-harbor-build
THORN_WHEEL_DIR=/tmp/thorn-harbor-wheel
THORN_CONSTRAINTS=/tmp/thorn-harbor-constraints.txt
THORN_HARBOR_CHECKOUT=/tmp/harbor-thorn-evals

git -C "$THORN_CHECKOUT" worktree add --detach \
  "$THORN_BUILD_CHECKOUT" "$THORN_REVISION"
uv build --wheel --out-dir "$THORN_WHEEL_DIR" "$THORN_BUILD_CHECKOUT"
uv export --project "$THORN_BUILD_CHECKOUT" --frozen --no-dev \
  --no-emit-project --no-hashes --output-file "$THORN_CONSTRAINTS"
THORN_WHEEL="$THORN_WHEEL_DIR/thorn-0.2.0-py3-none-any.whl"

git clone https://github.com/laude-institute/harbor.git \
  "$THORN_HARBOR_CHECKOUT"
git -C "$THORN_HARBOR_CHECKOUT" checkout --detach \
  071281b3d931aafd6a5375fa7d5933e23054d784
git -C "$THORN_HARBOR_CHECKOUT" apply --unidiff-zero \
  "$THORN_CHECKOUT/evals/harbor/patches/0001-swegym-pytest-output-parsing.patch"
git -C "$THORN_HARBOR_CHECKOUT" apply --unidiff-zero \
  "$THORN_CHECKOUT/evals/harbor/patches/0002-pin-swegym-lite-dataset-revision.patch"
uv sync --directory "$THORN_HARBOR_CHECKOUT"
```

The pinned SWE-Gym patch strips ANSI escapes, recognizes pytest status before
compact/progress suffixes, and uses aggregate counts only when pytest/xdist
emits no individual test names, with zero reported failures and enough passes
for all declared FAIL_TO_PASS and PASS_TO_PASS tests. This is a temporary
compatibility patch and should be removed once the pinned Harbor revision
includes an equivalent upstream fix.

## Generate the local catastrophic-failure set

The second Harbor patch pins `SWE-Gym/SWE-Gym-Lite` to Hugging Face revision
`f70b1a29ab120eb0a0ee7a1deb029825e735b2b0`. Generate the predeclared five-task
development set locally; the pinned Harbor registry does not contain a
`swegym-lite` dataset entry:

```bash
SWEGYM_TASKS="$THORN_HARBOR_CHECKOUT/datasets/thorn-swegym-catastrophic-v1"

for SWEGYM_TASK_ID in \
  dask__dask-8820 \
  iterative__dvc-5148 \
  getmoto__moto-5502 \
  python__mypy-15413 \
  Project-MONAI__MONAI-3547
do
  uv run --project "$THORN_HARBOR_CHECKOUT" --extra huggingface \
    --with swebench==4.1.0 \
    python "$THORN_HARBOR_CHECKOUT/adapters/swegym/run_adapter.py" \
    --dataset lite \
    --dataset-revision f70b1a29ab120eb0a0ee7a1deb029825e735b2b0 \
    --instance-id "$SWEGYM_TASK_ID" \
    --task-dir "$SWEGYM_TASKS"
done
```

The generated MONAI directory is lowercase:
`$SWEGYM_TASKS/project-monai__monai-3547`.

Change `THORN_REVISION` and the wheel filename when evaluating another build.
The adapter records the declared revision and both input SHA-256 digests; it
cannot prove that arbitrary supplied artifacts were built from that revision.

## Use the paired validation-v2 packages

The two validation packages are committed under
[`tasks/validation-v2`](tasks/validation-v2) with digest-pinned source images.
Their paired schedule and stop rules are declared in
[`task-selection-validation-v2.md`](task-selection-validation-v2.md). Neither
the packages nor the declaration authorize a provider-backed run.

The package manifest records Harbor task checksums plus every file hash and
mode. The normal test suite verifies those identities, Docker source digests,
provider-inert admission facts, and schedule hashes:

```bash
uv run pytest evals/harbor/tests/test_validation_v2_tasks.py
```

To independently reproduce Harbor's task checksums in the pinned checkout:

```bash
export VALIDATION_V2_TASKS="$THORN_CHECKOUT/evals/harbor/tasks/validation-v2"
PYTHONPATH="$THORN_HARBOR_CHECKOUT/src" \
  uv run --project "$THORN_HARBOR_CHECKOUT" python -c \
  'from harbor.models.task.task import Task; from pathlib import Path; import os; root = Path(os.environ["VALIDATION_V2_TASKS"]); print(*(f"{path.name} {Task(path).checksum}" for path in sorted(root.iterdir()) if path.is_dir()), sep="\n")'
```

The packages include raw benchmark instructions, tests, and oracle solutions.
Keep repository and resulting artifact access restricted, and publish only the
content-free evidence permitted by the validation declaration.

## Enforce the validation-v2 live-gate state

[`validation_gate.py`](validation_gate.py) is the provider-agnostic state
controller for the later four-attempt gate. It validates documents against the
contract in [`validation-gate.schema.json`](validation-gate.schema.json), but
it does not launch Harbor, read credentials, or contact a model provider.

The final candidate freeze must supply a gate-definition document with the
exact repository revision; declaration and task-manifest identities; four
per-slot, non-secret launcher-configuration hashes; task checksums and source
image digests; required content-free artifact IDs; and the already-declared
schedule. The final definition is intentionally not committed yet because the
selected Thorn candidate and its per-arm configurations are still pending.

Provider-inert validation and status inspection use the exact externally
reviewed definition hash:

```bash
uv run python -m evals.harbor.validation_gate validate \
  --definition /secure/gate/validation-gate-definition.json \
  --expected-definition-sha256 "$VALIDATION_GATE_DEFINITION_SHA256" \
  --repository-root "$THORN_CHECKOUT" \
  --state-root /secure/gate/state

uv run python -m evals.harbor.validation_gate status \
  --definition /secure/gate/validation-gate-definition.json \
  --expected-definition-sha256 "$VALIDATION_GATE_DEFINITION_SHA256" \
  --repository-root "$THORN_CHECKOUT" \
  --state-root /secure/gate/state
```

The arm-specific launcher must perform these additional steps after its own
provider-inert admission and while holding the global evaluation lock:

1. Verify a separately created mode-0600 authorization document. It must bind
   the definition hash, repository revision, all four slot IDs,
   `provider-backed-validation-v2` scope, and `single-no-retry` policy.
2. Call `consume` for the exact next slot immediately before starting Harbor.
   The confirmation is
   `CONSUME-<gate-id>-<slot-id>-NO-RETRY`.
3. From the same broad success/failure boundary, write a schema-valid,
   content-free terminal observation and call `record-terminal`. Its
   confirmation is `RECORD-<gate-id>-<slot-id>-TERMINAL`.

Consumption creates the slot receipt with `O_EXCL`; terminal recording creates
the audit the same way. Successful state files become mode 0400. A partial
write still leaves the path present and permanently blocks replacement. There
is deliberately no reset, retry, delete, or overwrite command. Keep the
mode-0700 state root outside the repository and preserve it with the private
raw Harbor artifacts.

Progress is derived only from the immutable schedule prefix. Codex outcome or
reward does not skip the matching Thorn arm. Infrastructure or integrity
failure stops immediately. Conan remains closed until the complete Bokeh pair
has terminal audits, selected Thorn is a clean reward-one usable completion,
and the pair has at most 1,200,000 submitted tokens. After Conan starts, its
pair is completed unless infrastructure or integrity failure stops the gate.

Terminal observations cannot contain prompts, responses, commands, endpoints,
credentials, tracebacks, or raw error messages: the schema rejects additional
fields. Audits retain only classifications, counts, frozen identities, receipt
links, and hashes of the required artifacts; artifact paths and contents are
not serialized.

The controller does not determine reward, usability, provider/model admission,
secret-scan success, or launcher-specific integrity. It also does not acquire
the global lock or make the operating-system account append-only. Those remain
responsibilities of the reviewed arm-specific launchers and host access
controls. The controller fails closed if their reported integrity, exact
identities, retries, artifacts, or schedule do not satisfy the frozen gate.

## Check installation on one SWE-Gym image

Run this from `THORN_CHECKOUT` so Python can import the `evals` package. The
install-only run does not contact a model provider:

```bash
PYTHONPATH="$THORN_CHECKOUT" \
  uv run --project "$THORN_HARBOR_CHECKOUT" harbor run \
  --path "$SWEGYM_TASKS/python__mypy-15413" \
  --agent evals.harbor.thorn_harbor_agent:ThornHarborAgent \
  --agent-kwarg "thorn_wheel_path=$THORN_WHEEL" \
  --agent-kwarg "thorn_constraints_path=$THORN_CONSTRAINTS" \
  --agent-kwarg "thorn_revision=$THORN_REVISION" \
  --install-only \
  --n-concurrent 1 \
  --max-retries 0
```

This is a compatibility check, not an installation cache. Harbor installs the
agent separately in each selected task environment.

## Run one SWE-Gym task

Export the provider settings expected by `thorn run` on the host. Pass literal
Harbor environment templates, as below, so secrets are resolved by Harbor and
do not become part of the Thorn shell command:

```bash
export OPENAI_API_URL=https://provider.example/v1
export OPENAI_API_KEY=replace-me
export OPENAI_API_MODEL_NAME=model-deployment-name
THORN_HARBOR_MODEL_LABEL=provider/model-deployment-name

PYTHONPATH="$THORN_CHECKOUT" \
  uv run --project "$THORN_HARBOR_CHECKOUT" harbor run \
  --path "$SWEGYM_TASKS/python__mypy-15413" \
  --agent evals.harbor.thorn_harbor_agent:ThornHarborAgent \
  --model "$THORN_HARBOR_MODEL_LABEL" \
  --agent-kwarg "thorn_wheel_path=$THORN_WHEEL" \
  --agent-kwarg "thorn_constraints_path=$THORN_CONSTRAINTS" \
  --agent-kwarg "thorn_revision=$THORN_REVISION" \
  --agent-kwarg "action_policy=baseline" \
  --agent-kwarg "history_policy=baseline" \
  --agent-kwarg "validation_convergence_policy=baseline" \
  --agent-kwarg "read_reuse_policy=baseline" \
  --agent-kwarg "prompt_trace_capture=redacted" \
  --agent-kwarg "task_shell_environment=conda-testbed" \
  --agent-env 'OPENAI_API_URL=${OPENAI_API_URL}' \
  --agent-env 'OPENAI_API_KEY=${OPENAI_API_KEY}' \
  --agent-env 'OPENAI_API_MODEL_NAME=${OPENAI_API_MODEL_NAME}' \
  --n-concurrent 1 \
  --max-retries 0
```

`--model` labels Harbor's results; `OPENAI_API_MODEL_NAME` configures Thorn's
current legacy provider loader. Use `--agent-env` for other Thorn controls such
as `THORN_MAX_TOOL_ROUNDS_WITHOUT_PROGRESS` as well. This adapter passes
`--agent-profile local` and records `direct` prompt delivery and the selected
action policy as separate axes. `baseline` preserves the existing direct-run
prompt. `bounded-action-v1` is the original top-level-call batching,
bounded-inspection, reuse, and edit-test-finish contract. `semantic-work-v2`
preserves those bounds while defining batches as
meaningful semantic work that may use one composite call or several top-level
calls. Changing the action policy creates a new logical run rather than an
exact retry. The independent `history_policy` axis defaults to `baseline`,
which preserves the stored history presentation. `bounded-history-v1` applies
the request-local history projection and fixed/relative budget policy.
`bounded-history-v2` additionally replaces a single-file
search when one newer current read contains every exact returned line/content
pair. The
separate `read_reuse_policy` axis defaults to `baseline`. Both arms retain the
same bounded, content-epoch-aware hashed observations and emit telemetry schema
v2 for every successful native read. Only `session-ledger-v1` exposes the
resulting session-reuse advisory to the model; `baseline` preserves the
existing prompt and tool-result bytes. The advisory never suppresses a read.
The independent `validation_convergence_policy` axis also defaults to
`baseline`, preserving the existing substring-based progress heuristic.
`baseline` and `action-epoch-v1` retain their schema-v2 action-epoch contract.
The separately versioned `workspace-content-observe-v2` control and
`workspace-content-v2` treatment collect the same bounded task-content facts;
only the treatment applies them so a repeated equivalent check on known-equal
content does not reset the no-progress guard. Changing any policy creates a
new logical run.
SWE-Gym verifier commands activate the image's `testbed` conda environment, so
the run command selects `task_shell_environment=conda-testbed` to give
`run_shell` the same interpreter and installed dependencies. The adapter's
default is `inherit` for non-conda Harbor workloads, and the selected policy is
recorded in provenance.

Prompt sidecars are redacted by default. For an explicitly secured audit run
on non-sensitive benchmark content, set `prompt_trace_capture=raw`; the adapter
then passes `--trace-raw-prompts` and records that choice in provenance. Raw
sidecars may contain credentials disclosed in prompts, source code, and tool
output, so do not publish or attach them to issues without review.

Each trial preserves these files in its `agent/` result directory:

- `thorn.jsonl`: Thorn's native JSONL event trace;
- `thorn-result.json`: Thorn's structured outcome and token summary;
- `thorn-output.txt`: combined standard output and error;
- `thorn-harbor-provenance.json`: pinned toolchain and input provenance;
- `thorn-install.txt`: setup output and installed versions.

The native trace records validation telemetry schema v2 in both validation
action-epoch policies. Each recognized check emits `validation_convergence` with its tool
call/render join IDs, action epoch, normalized identity fingerprint, outcome,
decision, counterfactual progress effect, and `policy_effect_applied`. The
baseline policy always records `policy_effect_applied=false` and retains the
legacy no-progress behavior; only `action-epoch-v1` applies the recorded
effect. Separate `validation_action_epoch` events identify every successful
native mutation or opaque shell call that advanced the tracker, including the
prior/current epoch and a typed reason. Neither event retains raw command or
output text. This makes zero equivalent-repeat opportunities measurable rather
than conflating them with disabled instrumentation.

Workspace-content policies emit schema-v3 `validation_convergence` records.
Before each recognized validator, Thorn hashes tracked and non-ignored
untracked task content twice, with fixed 20,000-path, 128-MiB-per-pass, and
three-second total limits. A mismatch or collection failure is recorded as
unknown and cannot cause a content-based stop. The current CLI toolhost log is
excluded only when the active runtime proves its exact framework-owned path;
there is no exclusion glob. Schema-v3 records contain current/prior digests,
content epoch and transition, bounded aggregate counts, validation identity
and outcome, and the counterfactual progress effect without raw task data.

The adapter deliberately reports `SUPPORTS_ATIF = False`. Native traces are
preserved without pretending they are Harbor ATIF trajectories; conversion is
a separate evaluation milestone. The provenance file is likewise not yet the
full run-manifest schema; that assembly remains a separate milestone.

## Adapter tests

The normal Thorn test suite skips these tests when Harbor is absent. To run
them against the pinned checkout without adding Harbor to Thorn's dependencies:

```bash
PYTHONPATH="$THORN_CHECKOUT" \
  uv run --project "$THORN_HARBOR_CHECKOUT" pytest \
  "$THORN_CHECKOUT/evals/harbor/tests"
```
