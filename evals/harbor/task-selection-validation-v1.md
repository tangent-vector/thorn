# SWE-Gym Validation Gate v1

Selection version: **1.0.0**

Corpus: **SWE-Gym Lite**

Dataset revision: **`f70b1a29ab120eb0a0ee7a1deb029825e735b2b0`**

Role: **one-shot validation gate**

This is a promotion check for changes developed against the five-task
catastrophic-failure set. It is not a pristine final holdout and must not be
relabeled as one after use.

## Predeclared Selection Rule

The rule was fixed before candidate instructions or gold patches were viewed:

1. Load the 230 records in the pinned SWE-Gym Lite `train` split.
2. Exclude records whose exact `repo` is one of the five development families:
   `Project-MONAI/MONAI`, `dask/dask`, `getmoto/moto`, `iterative/dvc`, or
   `python/mypy`.
3. Sort the remaining records by the UTF-8 byte order of `instance_id`.
4. Select the first record, then the first later record whose `repo` differs
   from every already selected repository. Stop after two records.

The filter leaves 54 records across six repository families. The rule does not
consider issue text, difficulty, patch size, verifier complexity, prior agent
outcomes, or expected success.

## Selected Tasks

| Order | Task ID | Repository | Version | Base commit | Source image |
| ---: | --- | --- | --- | --- | --- |
| 1 | `bokeh__bokeh-13289` | `bokeh/bokeh` | `3.3` | `d5ce3ec088f06bfaeac1e553bb42b4b88c069f58` | `xingyaoww/sweb.eval.x86_64.bokeh_s_bokeh-13289@sha256:ef51e22313c72834be97f6beaa3f822accb037cdf29832a8ab0fae1e89b0f504` |
| 2 | `conan-io__conan-11560` | `conan-io/conan` | `1.51` | `345be91a038e1bda707e07a19889953412d358dc` | `xingyaoww/sweb.eval.x86_64.conan-io_s_conan-11560@sha256:2ab97e51d7c77d280b6e818043472af58b388043171bfb6a725deda81583b44e` |

## Non-Agent Validation

Validation used Harbor revision
`071281b3d931aafd6a5375fa7d5933e23054d784` and the pinned verifier template
with SHA-256
`28ee869bc50392efc20d16705e5dd382ffd5bb6a376849d527bc8d9bbad78665`.
Both task packages had a non-empty instruction, executable verifier, non-empty
test patch, non-empty oracle patch, and resolvable source image. The images
started successfully under Harbor.

Harbor's deterministic oracle adapter was used only to apply each bundled
reference patch and execute the verifier. No Thorn, Codex, or other
model-backed agent has received either task instruction or run on either task.

| Task ID | Harbor task checksum | Oracle trial | Reward | Verifier evidence |
| --- | --- | --- | ---: | --- |
| `bokeh__bokeh-13289` | `3a1f854154a29bec803954d72c8308d8d03135bf6067da3dafc0778dd98898a0` | `bokeh__bokeh-13289__BPViU9x` | 1 | 22 passed, 0 failed; FAIL_TO_PASS and PASS_TO_PASS satisfied |
| `conan-io__conan-11560` | `041d542cd4a5bab906f8a6e44f8819d146ac691992f33855fa890f9f81b3bad7` | `conan-io__conan-11560__3LNRUWS` | 1 | 7 passed, 0 failed; FAIL_TO_PASS and PASS_TO_PASS satisfied |

Neither trial raised an exception or retried. The two-task oracle job completed
2/2 trials with mean reward `1.0`.

## Artifact Identities

The preflight evidence was retained in isolated Harbor state. The ephemeral
filesystem root is intentionally not part of the public identity; the retained
content identities are:

- selection manifest `selection-manifest.json`, SHA-256
  `92eeb2dccd4e8607087f659b344956942fc87b9a013da67d8347f2c2c2db19b5`;
- oracle job: `jobs/swegym-validation-selection-oracle-20260719`;
- oracle job `result.json` SHA-256
  `edd2833c99ee5940d078f88a25f35dd52a9ccc67e839bf7dcfe9319b1db9d445`.

These paths are local evidence locations, not portable storage. The hashes
identify the exact manifest and aggregate result used for this declaration.
The manifest records per-task instruction, verifier, oracle-patch, image, and
task checksums without embedding instruction or gold-patch content here.

No adapter failures occurred. The Hugging Face client emitted non-fatal 404s
while probing optional legacy metadata paths before loading the pinned dataset,
and Harbor skipped a pre-build OS inspection of each derived image; both
environments subsequently started and passed their verifiers.

## Use Policy

The task IDs and this declaration must be committed before the first
model-backed run. Evaluate both tasks exactly once at a declared promotion
point and report both results. Do not substitute another task based on an
outcome. After the gate is opened, treat these cases as exposed validation
data: they may diagnose a rejected promotion, but must not be queried
repeatedly during tuning. A final performance claim requires a separately
predeclared, untouched holdout or external gate.
