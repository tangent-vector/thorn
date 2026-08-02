---
name: thorn-agency-policy-bootstrap
description: Inventory a Thorn agency and install appropriate reusable policy guidance.
---

# Thorn Agency Policy Bootstrap

Use this skill when a user wants an agent to help set up agency operating
policy from Thorn's built-in templates.

## Inventory First

Read the agency layout before proposing edits:

- agency config: `agency.yaml`, `agency.json`, `gateway.yaml`, or
  `gateway.json`
- configured agents under `agents/`
- existing agent-home `AGENTS.md` and `MEMORY.md`
- configured forges, projects, peers, and credential references
- repository-local `AGENTS.md` files for projects the agency will work on

Do not infer missing credentials or write secrets into repository or agency
files.

## Select Policy

Use `docs/policy-templates/README.md` as the index of available templates.
For GitHub/GitLab contribution work, apply
`docs/policy-templates/git-forge-workflow.md` through the
`thorn-git-forge-policy-setup` skill.

Prefer a small, targeted policy file over copying every template wholesale.
If the agency has multiple agents with different authority, install separate
policy text for each agent instead of giving every agent the same broad
instructions.

## Finish With A Review

After editing, report:

- which policy files changed;
- which templates or sections were used;
- which local conventions were filled in;
- any unresolved operator decisions;
- which validation commands ran.

Run `uv run thorn agency check --agency <agency-home>` when agency
configuration changed. For policy-only edits, inspect the changed Markdown and
state that no config validation was needed.
