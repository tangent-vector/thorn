# Thorn Policy Templates

This directory contains reusable policy text for Thorn agencies. The templates
are meant to be copied or adapted into operator-authored instructions such as
an agent-home `AGENTS.md`, not loaded as hidden Thorn defaults.

Use these templates when an agency needs a clear policy for common operating
workflows but the behavior should remain reviewable as ordinary text.

## Templates

- [Git forge workflow](git-forge-workflow.md): GitHub/GitLab contribution
  policy covering issue ownership, branch and change-request scope,
  self-review, reviewer handoff, closing references, CI, and follow-up work.

## Setup Skills

Repository skills that help apply these templates live in `.agents/skills/`:

- [`thorn-git-forge-policy-setup`](../../.agents/skills/thorn-git-forge-policy-setup/SKILL.md):
  adapts the Git forge workflow template for a particular agent or repository
  policy file.
- [`thorn-agency-policy-bootstrap`](../../.agents/skills/thorn-agency-policy-bootstrap/SKILL.md):
  inventories an agency and chooses where copied policy guidance should live.

An operator can copy those skill directories into an agency or project
`.agents/skills/` directory if they want an agent to use the same setup flow
outside this repository.

## Applying A Template

1. Pick the agent or scope that should receive the policy. For a gateway agent,
   operator-owned policy usually lives at `<agency-home>/agents/<agent-id>/AGENTS.md`.
   Use `<agency-home>/agents/<agent-id>/home/AGENTS.md` only when the guidance is
   intentionally agent-visible and agent-editable.
2. Copy only the sections that match the agency's job. Prefer shorter,
   specific policy over a large generic rulebook.
3. Replace placeholders with local names, forge hosts, reviewer conventions,
   and test commands.
4. Keep secrets out of policy files. Refer to environment variable names or
   credential references instead of literal tokens.
5. If agency configuration changed, run:

   ```console
   $ uv run thorn agency check --agency <agency-home>
   ```

6. Before consuming live forge notifications, run the gateway preflight from
   the same checkout and environment:

   ```console
   $ uv run thorn serve --agency <agency-home> preflight
   ```
