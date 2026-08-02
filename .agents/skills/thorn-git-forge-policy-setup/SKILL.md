---
name: thorn-git-forge-policy-setup
description: Adapt Thorn's GitHub/GitLab workflow policy template into an agency or repository policy file.
---

# Thorn Git Forge Policy Setup

Use this skill when a user asks to add or refresh GitHub/GitLab contribution
policy for a Thorn agent, agency, or repository.

## Source Material

Read these files before editing policy:

- `docs/policy-templates/README.md`
- `docs/policy-templates/git-forge-workflow.md`
- any existing target `AGENTS.md`, `MEMORY.md`, agency config, and repository
  workflow instructions that already apply

## Choose The Target

Prefer the narrowest policy location that will do the job:

- For one gateway agent, use `<agency-home>/agents/<agent-id>/home/AGENTS.md`.
- For one repository, use that repository's `AGENTS.md`.
- For a scoped session or project area, use the closest applicable
  session/project instruction file.

If the target agent, repository, forge, or reviewer convention is unclear and
cannot be inferred from files already present, ask the user before editing.

## Adapt The Template

Copy only the sections that match the agency's work. Replace placeholders with
the actual agent ID, forge name, reviewer convention, and local validation
commands.

Preserve stricter existing instructions. If existing policy conflicts with the
template, keep the local policy and explain the conflict in your summary unless
the user explicitly asks you to change it.

Do not add literal credentials, tokens, private key material, or one-off secret
values to policy files. Mention environment variable names or credential
references instead.

## Validate

If you changed only policy text, review the rendered Markdown and summarize the
effective behavior changes.

If you also changed agency configuration, run:

```console
$ uv run thorn agency check --agency <agency-home>
```

Before an operator starts live notification polling, recommend the same
checkout and environment run:

```console
$ uv run thorn serve --agency <agency-home> preflight
```
