# Thorn Documentation

This index distinguishes documentation of the system that exists today from
operator guidance, future designs, and research ideas. A document's location is
part of its status: aspirational material does not override current reference
or executable behavior. Completed implementation plans live in Git history
rather than the current documentation tree.

## Start here

- [Repository README](../README.md) — project purpose, agency vocabulary,
  lightweight CLI trial, implemented scope, and limitations.
- [Current architecture](architecture.md) — the implemented agency/runtime
  model, CLI and gateway interaction modes, component boundaries, persistence,
  and source organization.
- [Agency operations](operations.md) — installation, configuration,
  credentials, gateway startup, status, and recovery.
- [Roadmap](../ROADMAP.md) — implemented foundation, near-term reliability
  priorities, and explicitly exploratory directions.

## Current reference and operator guidance

- [Gateway startup and recovery](startup_flow.md) — detailed `thorn serve`
  lifecycle, bundled-broker startup, shutdown, notification acknowledgement,
  and recovery behavior.
- [Threat model](threat-model.md) — peer-trigger trust, content boundaries,
  sandbox guarantees, non-goals, and operating guidance.
- [Per-prompt context gathering](context-gathering.md) — current context and
  tool-discovery pipeline. Its final “future work” section is prospective; the
  earlier sections describe implemented behavior.
- [Agency policy templates](policy-templates/README.md) — reusable starting
  policy for agencies that work through GitHub or GitLab.
- [`examples/gateway/`](../examples/gateway/) — complete configuration
  skeletons for supported forge, broker, sandbox, and peer shapes. The lean
  coordinator calibration example is an evaluation aid, not a recommended
  production architecture.

## Deployment and evaluation

- [NVIDIA Brev deployment](../deploy/brev/README.md) — optional
  provider-specific deployment recipe. It is not Thorn's primary deployment
  model and still needs an outside-NVIDIA public-account rehearsal.
- [Harbor adapter](../evals/harbor/README.md) — reproducible adapter and bounded
  experiment machinery. Its presence is infrastructure evidence, not a claim
  of benchmark performance.

## Supporting Python API

- [Python library guide](library.md) — the current supporting library surface.
  The CLI and gateway are Thorn's products; this API is not a stable
  compatibility contract and should not be treated as the primary entry point.

## Aspirational designs

Documents in [`docs/aspirational/`](aspirational/) describe possible target
designs. They are retained to communicate intent and constraints, but they may
not match the current implementation:

- [System vision](aspirational/architecture.md)
- [Agency configuration and operator UX](aspirational/agency-configuration.md)
- [Agency control plane](aspirational/agency-control-plane.md)
- [Multi-session coordination](aspirational/coordination.md)
- [Scoped agent memory](aspirational/memory.md)

## Research ideas

Documents in [`docs/ideas/`](ideas/) explore concepts without committing to an
implementation or roadmap position:

- [Entity templates](ideas/entity-templates.md)
- [Hierarchical context management](ideas/hierarchical-context-management.md)
