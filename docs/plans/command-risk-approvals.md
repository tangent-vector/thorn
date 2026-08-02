# Command Risk Classification And Approval Prompts

## Status

The current mitigation warns whenever Thorn uses host-subprocess shell
execution. This plan describes the next step: classify risky model-proposed
commands and ask an operator to approve them before execution.

## Goals

- Classify `run_shell` commands before they reach the toolhost.
- Use both static policy and LLM-assisted judgement, with static policy
  taking precedence for clear allows and denies.
- Surface the reason a command needs approval in CLI and gateway
  operator channels.
- Preserve unattended safe automation by allowing low-risk commands
  without prompting.

## Risk Model

Classify each shell command into an explicit command-risk class:

- `low`: read-only inspection, ordinary build/test commands, and other
  commands allowed by operator policy.
- `needs_approval`: filesystem writes outside the agent workspace,
  network access outside an expected forge or package registry,
  privilege-sensitive commands, package installation, process control,
  credential helper access, or commands whose intent is ambiguous.
- `deny`: commands that attempt to read host secrets, disable security
  controls, mutate system directories, escape the configured sandbox,
  or otherwise violate operator policy.

Static policy should evaluate structured features first: executable,
arguments, working directory, environment changes, redirections,
pipeline members, and whether the command came from external content.
LLM-assisted classification can provide a second opinion for ambiguous
cases, but it must not downgrade a static `deny`.

## Approval UX

For `thorn run` and `thorn chat`, approval prompts should be interactive
terminal prompts showing the command, working directory, risk class,
and short reason. Operators can approve once, deny once, or approve a
bounded pattern for the current session.

For `thorn serve`, approvals need an operator channel rather than a
blocking stdin prompt. The gateway should park the inbox item with a
clear pending-approval status, expose it through `thorn status` and
`thorn inbox show`, and provide a command such as `thorn inbox approve
<item-id>` that records the decision before the scheduler resumes.

Approval records should be durable session metadata with timestamp,
operator identity when available, command digest, risk class, and the
policy/LLM reason. A later audit should be able to tell whether the
command was auto-allowed, denied, or human-approved.

## Open Design Points

- Exact command parser: shell syntax needs structured parsing rather
  than splitting on spaces. A POSIX shell parser is likely enough for
  the first pass, but Windows shell support needs a separate shape.
- Policy source: decide whether operator policy lives in `gateway.json`,
  per-agent config, or a dedicated policy file.
- Pattern approvals: define a typed approval pattern language so
  "approve npm test in this workspace" cannot accidentally become
  "approve any npm command anywhere".
- Non-interactive CLI mode: decide whether `thorn run --yes` exists, or
  whether non-interactive approval must come only from explicit policy.
