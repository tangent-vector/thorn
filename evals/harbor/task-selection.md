# Initial SWE-Gym Catastrophic-Failure Task Selection

Selection version: **1.0.0**

Corpus: **SWE-Gym Lite**

Split: **development**

This is an integration and catastrophic-failure set, not a statistically
meaningful quality sample. Its first purpose is to prove that Harbor can install
and launch pinned Thorn, enforce limits, invoke the verifier, and preserve a
complete manifest and trajectory.

## Validation Environment

The active set was validated against:

- Harbor revision `071281b3d931aafd6a5375fa7d5933e23054d784`;
- a pinned local SWE-Gym verifier adaptation whose
  `adapters/swegym/template/tests/test.sh` SHA-256 was
  `28ee869bc50392efc20d16705e5dd382ffd5bb6a376849d527bc8d9bbad78665`;
- oracle job `thorn-swegym-oracle-final-20260719`.

The verifier adaptation keeps named `FAIL_TO_PASS`/`PASS_TO_PASS` status
matching when available. For compact or xdist-style output that omits names,
it uses a count fallback only when there are zero failures and the number of
passing tests covers all declared tests. This adaptation must be committed and
pinned before agent trials; the recorded job identity is evidence, not a
durable distribution mechanism.

Before any Thorn attempt, each task must pass a non-agent validation rehearsal:

1. The task exists in the pinned SWE-Gym Lite revision.
2. Its image resolves and starts under the declared Harbor resources.
3. The task instruction and verifier load successfully.
4. The oracle patch applies and passes the verifier, or the upstream adapter
   provides equivalent recorded oracle evidence.
5. Harbor preserves verifier and task artifacts for the rehearsal.

If a task fails these checks, retain its failed validation record and amend this
selection in a new committed version. Do not silently substitute a task or
label it as predeclared for agent execution.

## Adaptation Attempts Before Selection

The first research pass proposed Dask, DVC, and Moto. Dask produced reward `1`
after the first parser correction. DVC and Moto initially produced reward `0`
despite successful test summaries:

- `iterative__dvc-5148` reported `50/50` tests passing but
  `FAIL_TO_PASS passed: False`;
- `getmoto__moto-5502` reported `79` tests passing while both expected
  `FAIL_TO_PASS` IDs were missing or unrecognized.

Those were adapter/oracle failures, not Thorn results. They were retained as
excluded candidates until the safer count fallback was implemented. The final
exact-digest oracle rerun then returned reward `1` for all five selected tasks.
This history must not be rewritten as five first-try successes.

## Active Catastrophic-Failure Set

The expansion gate is complete. All five images started and all five oracle
runs returned reward `1` in the same final job:

| Active order | Task ID | Repository family | Image validation | Oracle reward |
| ---: | --- | --- | --- | ---: |
| 1 | `dask__dask-8820` | Dask | passed | 1 |
| 2 | `iterative__dvc-5148` | DVC | passed | 1 |
| 3 | `getmoto__moto-5502` | Moto | passed | 1 |
| 4 | `python__mypy-15413` | mypy | passed | 1 |
| 5 | `Project-MONAI__MONAI-3547` | MONAI | passed | 1 |

Harbor generated the MONAI task directory as
`project-monai__monai-3547`; manifests and reports retain the corpus task ID
`Project-MONAI__MONAI-3547` and may additionally record the generated path.

The last two task IDs were added only after image/oracle validation; they were
not invented to fill the table and were not selected from Thorn outcomes.
Before agent execution, commit/pin the verifier adaptation and update each run
manifest with the durable Harbor revision or patch digest, resolved image
digest, and this file's SHA-256.

All five tasks remain development cases. Validation and holdout task lists will
be declared separately before controlled behavior changes or autoresearch begin.
