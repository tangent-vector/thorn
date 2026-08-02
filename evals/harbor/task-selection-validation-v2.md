# SWE-Gym Paired Validation Gate v2

Declaration version: **2.0.0**

Corpus: **SWE-Gym Lite**

Dataset revision: **`f70b1a29ab120eb0a0ee7a1deb029825e735b2b0`**

Role: **one-shot paired validation gate**

This declaration does not authorize a model or provider call. It fixes the
task packages, paired design, order, and stop rules that must be reviewed
alongside a later exact candidate freeze before any attempt is consumed.

## Relationship to Validation v1

The task identities and deterministic order selected in
[`task-selection-validation-v1.md`](task-selection-validation-v1.md) remain
unchanged. Validation v1 admitted the source-tag packages with oracle runs and
called for one model-backed evaluation per task. This v2 declaration
supersedes only that execution policy: every opened task receives one selected
Thorn attempt and one exact-Codex reference attempt.

These cases are held out from the five development repository families, but
they are not a pristine final holdout and must not be relabeled as one after
the gate opens. The four attempts are a usability and practical-reference
check, not an estimate of a treatment effect or population performance.

## Digest-Pinned Task Packages

The exact packages are committed under
[`tasks/validation-v2`](tasks/validation-v2). The machine-readable
[`manifest.json`](tasks/validation-v2/manifest.json) records every file hash
and mode, the Harbor task checksums, source-image digests, sanitizer identity,
provider-inert canary facts, and schedule hashes. Its SHA-256 is
`43af8d4b6476552595a5a2f55378b8c5396d8612f1f7ccb108426112340a6169`.

Each package is byte-for-byte identical to its oracle-admitted validation-v1
package except that the sole Docker `FROM` line now appends the source digest
already declared in validation v1. No instruction, verifier, test
configuration, oracle patch, or task metadata changed.

| Order | Task ID | Package path | Harbor task checksum | Pinned source digest |
| ---: | --- | --- | --- | --- |
| 1 | `bokeh__bokeh-13289` | `tasks/validation-v2/bokeh__bokeh-13289` | `eec893cc542db1f29c2e399afedebb7baf41999eabbc310700b3ff637ec475a5` | `sha256:ef51e22313c72834be97f6beaa3f822accb037cdf29832a8ab0fae1e89b0f504` |
| 2 | `conan-io__conan-11560` | `tasks/validation-v2/conan-io__conan-11560` | `23b3afbcee2d69bd33736a7b3fccee90685f3aa70a64633e73512ac0b1636fa3` | `sha256:2ab97e51d7c77d280b6e818043472af58b388043171bfb6a725deda81583b44e` |

## Arms

### Selected Thorn

Freeze one named candidate using a committed revision, wheel and constraints,
Harbor adapter, analyzer, policy identities, and content-free artifact hashes.
Candidate selection and all effect analysis must finish on development tasks.
The candidate may include only correctness or treatment versions selected and
named before this gate; an existing policy must not change behavior silently.

Fixed execution settings are local/direct CLI delivery, a fresh
agency/session/container, `conda-testbed`, secured raw-prompt capture, GPT-5.6
Sol at `xhigh`, 50 provider rounds, the current declared no-progress limit,
`fresh-repository-v1`, task-environment normalization `off`, no provider
retry, and no trial retry. Read-reuse and validation-convergence policies stay
at baseline unless the frozen candidate explicitly names a newer predeclared
mechanism.

### Exact Codex reference

Reuse the admitted #145 V3 surface: Codex 0.144.6, exact GPT-5.6 Sol at
`xhigh`, reasoning summary `none`, web disabled, Responses Lite plus Code
Mode, provider request and stream retries zero, startup updates disabled,
exact model-metadata admission, and `fresh-repository-v1`. Use the corrected
endpoint canonicalization and the same provider endpoint and credential
transport as Thorn.

The scaffolds intentionally retain different tool surfaces. This arm is a
same-model practical reference, not a causal scaffold ablation.

## Common Freeze and Admission

Before separate authorization, record and verify:

- the exact candidate and Codex identities described above;
- Harbor revision `071281b3d931aafd6a5375fa7d5933e23054d784` and its
  retained patch identities: verifier patch SHA-256
  `7afa6d4790bd8fdf0154974346403f3632593049631556c4795e78a59b5493a4`,
  dataset-revision patch SHA-256
  `06e54a343cf8929c12dba7bd550315fb63c503f1e7d62baa2dbce9cb1fbf18e3`,
  and resulting verifier script SHA-256
  `28ee869bc50392efc20d16705e5dd382ffd5bb6a376849d527bc8d9bbad78665`;
- the committed task-package paths and checksums in this declaration;
- sanitizer v2 SHA-256
  `49fb8771e32cba650c1a26915462335648d7b04fb79bcbf3893e0fa98b236a31`;
- task-environment normalization `off`, equal timeout/resources/network
  policy, one concurrent trial, and the global sequential lock;
- exclusive per-slot receipts, content-free terminal audits, and private raw
  artifact retention with no overwrite path.

Candidate-specific Thorn installation/admission and exact-Codex
factory/model-surface admission must run provider-inertly against both task
images after the final revisions are frozen.

The generic immutable receipt/audit state machine is implemented in
[`validation_gate.py`](validation_gate.py), with its document contract in
[`validation-gate.schema.json`](validation-gate.schema.json). It neither
authorizes nor launches a provider-backed run. The later exact candidate freeze
must supply the definition, authorization, arm-specific admission, global-lock,
and launcher inputs described by that contract.

## Fixed Schedule

Maximum provider-backed count: **four attempts**, with no retry or replacement.

Seed literal: `thorn-issue-146-heldout-paired-gate-v1`

Seed SHA-256:
`fbc08a41d890c77835001ce07f3dd25fdca15cafc28b7b1d4f15093d990127ac`

Task-block order is inherited from validation v1 and remains fixed: Bokeh,
then Conan. Within each task block only, ascending SHA-256 of
`seed|task|arm` fixes arm order. The hash arm literals are `codex-v3` and
`thorn-candidate`. The selection hashes are not sorted across task blocks:

| Order | Task | Arm | Hash literal | Selection SHA-256 |
| ---: | --- | --- | --- | --- |
| 1 | Bokeh | exact Codex | `codex-v3` | `82a0b5e456c89563582346e8fd9e0a1e0beda4d8d6925eceab10e0a9c18c12e9` |
| 2 | Bokeh | selected Thorn | `thorn-candidate` | `df4e96fffd8d99ccea89edd0ee34d040f932b7b255bb76f52a541e626e5d73cb` |
| 3 | Conan | exact Codex | `codex-v3` | `bb38b74d25729020c12d983e86c61715fe9f636ae357a1bd5334463d44f1b5d7` |
| 4 | Conan | selected Thorn | `thorn-candidate` | `c4abb0dfae205ea4c3e816da24881e6ce625423eebdf4d4ecdccbd2c449a391b` |

Complete both arms of a started task block regardless of reward or apparent
efficiency.

## Sequential Stop Rules

Pause before consuming a slot on any source, wheel, configuration, model,
effort, provider, task, image, sanitation, verifier, network, credential,
artifact, process, or global-lock mismatch.

If an external or infrastructure failure occurs after a slot is consumed,
retain the attempt and stop the entire gate without retry.

After the complete Bokeh pair, do not open Conan if any of the following is
true:

- selected Thorn has reward zero or `usable_resolution=false`;
- selected Thorn ends with an agent exception, loop/no-progress stop, or
  timeout;
- either arm lacks required integrity or terminal-audit evidence;
- combined submitted tokens for the pair exceed 1,200,000.

The token rule is a between-block budget stop, not a within-arm kill. Codex
reward does not control continuation. Proceed only when the Bokeh Thorn result
is reward one and usable, both arms pass integrity checks, and the complete
pair remains within budget. Stop unconditionally after Conan. No validation
diagnosis may trigger an additional provider call.

## Provider-Inert Environment Admission

The temporary admission canary installed Codex 0.144.6, ignored the task
instruction, invoked no model or provider, applied `fresh-repository-v1`,
collected workspace identity, and allowed the stock verifier to evaluate the
unsolved workspace. The digest-pinned packages completed 2/2 with zero
exceptions, zero retries, reward zero as expected, and null token/cache/output
and cost fields.

- canary source SHA-256:
  `13ad79589828f8e4383f9b4b55e6adf8ea2fb4f926111203ce3373159b8becb3`;
- canary configuration SHA-256:
  `73d0289bcb8420db9a2c41f23a0155e3e15ac83c0a35a653290d6671aa67a1d5`;
- aggregate result SHA-256:
  `26d3778a910e2e9c95cf59ba1c146a8077b9ab54e585705fdeca89e3c5100083`.

Bokeh sanitation retained source/input/result tree
`8eb01513aa8dd7bc9ee4a65f8bafd81687e8b590`, source HEAD
`d5ce3ec088f06bfaeac1e553bb42b4b88c069f58`, synthetic HEAD
`cff438c2c4ea968f38db2e510f573378164f65cd`, 5,173 tracked paths, and a clean
overlay. Conan retained tree `e34214d86fd1cd12e56ad04b71583a9d2c0ed6a2`,
source HEAD `345be91a038e1bda707e07a19889953412d358dc`, synthetic HEAD
`ad1534d71768148f646b3953e8fcd46c9a3f2373`, 1,243 tracked paths, and a clean
overlay.

This admits the generic task, sanitation, and verifier environment only. It
does not replace the exact-candidate preflights required above.

## Outcome Rule and Data Handling

Promotion requires selected Thorn to finish cleanly with reward one on both
tasks, zero retries, and no integrity failure. One failure rejects or defers
the candidate; do not tune from the trajectory and rerun. Report Codex beside
the matching task as a practical reference without requiring Thorn to beat it
or claiming a population effect.

The committed packages contain raw benchmark instructions, tests, and oracle
solutions and therefore inherit repository access controls. Do not reproduce
those contents in public notes or aggregate manifests. Published evidence is
limited to content-free hashes, counts, timing, terminal state, and manual
audit classifications; private raw trajectories remain separately secured.
