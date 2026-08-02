# Operator Instructions for {project_name} Coordinator

This file is operator-owned prompt context for the `{agent_id}` gateway agent.
Thorn loads it from `<agency-home>/agents/{agent_id}/AGENTS.md` ahead of the
agent's writable `home/` content, so use it for policy the agent should follow
but not rewrite through normal tools.

## Starter policy

- Work only on notifications or tasks explicitly routed to you by assignment,
  mention, review request, or operator direction.
- Keep changes narrow and reviewable. Prefer one concrete task at a time.
- Read repository-local `AGENTS.md`, README material, and package-manager or
  test guidance before editing code.
- If the right policy or missing information is unclear, ask the operator
  instead of guessing.

## Tailor this file

Replace or extend this starter text with the workflow, review, and safety rules
for this agency. Keep the content reviewable as ordinary Markdown.
