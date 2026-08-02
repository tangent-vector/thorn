# Thorn Roadmap

Thorn is an experimental end-user application. The CLI and the long-lived
gateway are the product surfaces; the Python library API is supporting
infrastructure and is not yet a stable compatibility contract.

## Implemented foundation

- persistent CLI and gateway sessions with durable inbox state;
- concurrent per-agent and per-session scheduling;
- GitHub and GitLab event sources and forge tools;
- peer-aware trigger authorization inside one agency trust domain;
- container and subprocess tool execution;
- bundled credential brokering and operator preflight/status commands; and
- provider-inert tests plus a Harbor evaluation adapter.

## Reliability and operator experience

Near-term work prioritizes a trustworthy first-run and recovery path:

- settle external-notification acknowledgement and local recovery semantics;
- make multi-project routing explicit instead of relying on fallback agents;
- add bounded-work, interruption, and provider-spend controls;
- improve lifecycle cleanup, status, diagnostics, and operator approvals;
- validate installation artifacts and deployment from fresh public machines;
  and
- keep documentation and CI aligned with executable behavior.

## Research directions

Longer-term exploration includes:

- runtime-enforced memory scopes with ownership and provenance;
- explicit cross-session and multi-agent coordination primitives;
- reproducible, privacy-aware agent evaluation and release rehearsals; and
- additional provider-neutral deployment recipes.

These are directions, not claims about implemented behavior. Public GitHub
issues will be kept deliberately small and outcome-oriented as work is selected.
