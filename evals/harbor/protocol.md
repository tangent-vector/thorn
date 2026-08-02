# Thorn Harbor Evaluation Protocol

Protocol version: **1.7.0**

This protocol defines the first reproducible contract for running Thorn under
Harbor. It covers both predeclared run configuration and final result metadata.
The machine-readable form is
[`run-manifest.schema.json`](run-manifest.schema.json), with a planned-run
example in [`run-manifest.example.json`](run-manifest.example.json).
The example is synthetic and intentionally uses placeholder hashes, provider
labels, and unresolved image metadata; it is not authorization to launch its
pending task.

The initial task selection is recorded in
[`task-selection.md`](task-selection.md). Changing this protocol, the schema,
or a predeclared task set requires a committed version change; an operator must
not rewrite an earlier run manifest in place to make a later result look
comparable.

## Experimental Questions

The evaluation program separates three questions:

1. How capable and efficient is Thorn's local/core agent loop?
2. What changes when the same task uses the lean or full coordinator profile?
3. What changes when the same profile receives work through Thorn's inbox and
   scheduler instead of direct one-shot prompt delivery?

Agent profile and prompt delivery are independent variables. The first Harbor
milestone uses direct delivery. A later local mini-gateway may exercise the real
inbox path without adding forge polling, brokers, or nested containers.

## Manifest Lifecycle

Every Harbor trial has a manifest with these identities:

- `experiment_id` names the declared experiment.
- `logical_run_id` names one task/configuration/repetition tuple.
- `attempt_id` names one execution of that logical run.
- `repetition_index` distinguishes intentional repeated measurements.
- `configuration_sha256` fingerprints the canonical non-secret run
  configuration.
- planned, started, and completed timestamps expose provider/time drift and
  make scheduled-order claims auditable.

A `planned` manifest is written before agent execution. It contains `null` for
`outcome` and `metrics`. On termination, the attempt gets a `completed`
manifest with typed outcome and metric objects. A failed or interrupted attempt
is still completed metadata; it is never deleted merely because a later retry
succeeds.

The following inputs must be pinned or recorded:

- exact Thorn and Harbor revisions and Thorn working-tree state;
- adapter revision;
- dataset revision, task ID, task image reference and digest when resolved,
  verifier identity, and task-selection file digest;
- agent profile and prompt-delivery mode;
- named action policy (`baseline`, `bounded-action-v1`, or `semantic-work-v2`);
- named history policy (`baseline`, `bounded-history-v1`, or
  `bounded-history-v2`);
- named validation-convergence policy (`baseline`, `action-epoch-v1`,
  `workspace-content-observe-v2`, or `workspace-content-v2`);
- named read-reuse policy (`baseline` or `session-ledger-v1`);
- read-reuse telemetry schema version (the agent manifest field
  `read_reuse_telemetry_schema_version`, fixed at v2 for arm-independent
  observations);
- task-shell environment policy inherited by subprocess tools;
- model/provider identity and all supported non-secret model options;
- context-window assumption, provider/tool/run timeouts, tool-round and
  no-progress limits;
- CPU, memory, disk, network, isolation, and workspace settings;
- scheduled order and repetition index.

The final record also names the result, raw Thorn trace, verifier artifacts,
and the ATIF trajectory when conversion is supported. The ATIF path is `null`
until that capability exists; an adapter must not relabel Thorn's native trace
as ATIF. Paths point to artifacts; their contents are not embedded in the
manifest.

Validation-convergence analysis uses schema-v2 `validation_convergence` and
`validation_action_epoch` records from the native trace. Both baseline and
action-epoch-v1 trials emit the same normalized observations and epoch
advances. The former is observation-only, while the latter alone sets
`policy_effect_applied=true` and applies `progress_effect` to no-progress
accounting. Analyses must report the eligible `equivalent_repeat` count; zero
eligible repeats is not evidence that the treatment was effective.

Content-aware trials are a separate schema-v3 family and must not be pooled
with action-epoch-v1. `workspace-content-observe-v2` and
`workspace-content-v2` both double-sample a bounded Git task-content identity
immediately before recognized validation. The identity covers tracked and
non-ignored untracked content. Known equal content can expose an equivalent
repeat across opaque shell actions; known changed content starts a fresh
content epoch; unsupported, raced, or limit-exhausted collection is `unknown`
and defers to conservative behavior. Only `workspace-content-v2` applies the
recorded progress effect. Telemetry contains digests and aggregate bounds, not
raw paths, commands, output, or file content.

## Outcome Taxonomy

Outcomes form three explicit classes. The schema restricts the valid code for
each class.

### Infrastructure failure

The experiment did not produce a valid agent trial or verifier result:

- `installation_failure`
- `startup_failure`
- `provider_authentication_failure`
- `provider_transport_failure`
- `run_timeout`
- `missing_artifact`
- `verifier_execution_failure`

These outcomes block the catastrophic-failure integration milestone until they
are fixed or explained by a concrete external blocker.

### Agent-behavior failure

The harness worked, but Thorn failed in a severe, diagnostic way:

- `no_material_action`
- `no_progress_eviction`
- `tool_round_limit`
- `invalid_patch`
- `agent_error`

These are valid measurements, not adapter failures. They must remain visible
even when a later Thorn change succeeds.

### Completed

The run and verifier completed normally:

- `resolved`
- `unresolved`

`resolved` is the primary quality signal. Earlier edits, fewer reads, or a more
orderly-looking trace are not success when the verifier remains unresolved.

## Required Metrics

Completed manifests record:

- wall-clock duration;
- provider attempts, successes, failures, retries, and latency distribution;
- verifier reward and provider cost when available;
- aggregate prompt, completion, cache, and total tokens when reported;
- first, final, and maximum prompt tokens;
- round/time to first material action and first validation;
- total tool calls and counts by tool name;
- structured read/search calls, classified shell inspection, and observed or
  hinted redundant inspection;
- working-set transitions and no-progress warnings/evictions.

Unavailable values are represented as `null`, not guessed. Analyses report
quality first and efficiency second. Efficiency should also be conditioned on
successful outcomes, including tokens, cost, and time per resolved task.

## Retries and Iteration

An exact retry preserves the logical run and configuration digest, increments
`attempt_number`, assigns a new `attempt_id`, and points
`retry_of_attempt_id` to the prior attempt.

Any change to Thorn/Harbor revision, task image, prompt, agent profile, prompt
delivery, action policy, history policy, validation-convergence policy,
read-reuse policy, model options, resource limits, timeout/guard settings,
or adapter behavior creates a new logical run and configuration digest. A
before/after integration repair may reference the earlier attempt through
`supersedes_attempt_id`, but it is not an exact retry.

Retries are never used to select the best-looking attempt silently. Reports
include every attempt and state the aggregation rule before execution.

## Corpus and Held-Out Methodology

SWE-Gym Lite is the development/calibration corpus. Tasks are assigned one of
four roles before execution:

- `development`: visible during adapter and behavior iteration;
- `validation`: consulted only at declared promotion points;
- `holdout`: untouched until a predeclared final evaluation;
- `external_gate`: a separately maintained corpus such as SWE-bench Verified.

Repository-level separation is preferred when the corpus permits it. Task IDs,
selection rationale, split, and selection-file digest are committed before the
first corresponding agent run. Gold patches and verifier internals for
validation, holdout, and external-gate cases must not be used to tune Thorn.

The initial research produced three development-task candidates. Only
candidates that pass image and oracle validation enter the active
catastrophic-failure set; failed attempts remain recorded as adaptation
evidence. The initial set expanded to five only after the exact verifier
adaptation and all oracle results were recorded, as specified in
`task-selection.md`. Future task IDs must be committed before Thorn runs on
them and must not be chosen based on Thorn's outcomes on the active set.
The original one-shot promotion set was declared in
[`task-selection-validation-v1.md`](task-selection-validation-v1.md). Its
task selection is unchanged, while the paired execution policy and
digest-pinned packages are superseded by
[`task-selection-validation-v2.md`](task-selection-validation-v2.md). These
tasks are validation data, not a pristine final holdout.

## Comparison Modes

`standalone` runs establish basic operation and baseline measurements.

`same_model_scaffold` comparisons hold task, model/provider, model options,
resources, limits, and run ordering policy constant while changing the agent
scaffold/profile. These comparisons are the appropriate evidence for causal
claims about Thorn's prompts, tools, history, or workflow.

`best_supported_product` comparisons run production agents with their supported
configurations. They measure practical competitiveness, but model and scaffold
effects are confounded and must not be presented as causal ablations.

Run order is predeclared or randomized with a recorded index. Small calibration
sets use repeated attempts because provider and agent outcomes vary. Held-out
cases are not repeatedly queried while tuning.

## Secret and Sensitive-Data Exclusions

The manifest must never contain:

- API keys, bearer tokens, cookies, passwords, or forge credentials;
- authorization headers or a complete environment dump;
- endpoint URLs containing credentials or secret query parameters;
- raw prompts, tool outputs, source-file contents, diffs, or verifier logs;
- private-repository authentication material.

Provider endpoints are represented by a non-secret operator-defined label.
Credential values are supplied through Harbor's secret/environment mechanism
and are not represented in this schema. Artifact paths and capture policy may
be recorded; sensitive artifact contents remain governed by their own access
controls. Raw prompt capture is disabled by default and must be an explicit,
separately secured choice.

The JSON Schema uses `additionalProperties: false` throughout and allowlists
model options so accidental environment or credential fields fail validation.
