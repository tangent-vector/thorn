# Git Forge Workflow Policy Template

This template is starter policy for Thorn agents that work on GitHub or
GitLab issues and change requests. Copy and adapt the sections that match the
agency's job into the relevant agent or repository `AGENTS.md`.

Replace placeholders such as `<agent-id>`, `<primary-forge>`, and
`<reviewer-convention>` before using the text.

## Scope

The agent may work on issues, merge requests, pull requests, comments, review
threads, and CI results on `<primary-forge>` when the work is explicitly routed
to the agent by assignment, mention, review request, or an operator command.

The agent should keep work tied to the triggering issue or change request. It
should not start unrelated follow-on work after a merge, close, reopen, label
change, or CI notification unless an operator or reviewer explicitly asks for
that work.

## Repository Preparation

Before editing a repository, the agent should read the repository's local
instructions and workflow files, including `AGENTS.md`, README material,
package-manager files, and test guidance. Project-local instructions override
this template when they are more specific.

The agent should use the repository's declared package manager and test
workflow. If the repository requires `uv`, npm, cargo, make, or another
specific command surface, the agent should use that surface rather than
ad-hoc substitutes.

## Change Scope

Prefer one cohesive change request for one coherent task. Split work only when
the issue, operator, reviewer, or repository policy asks for separate changes,
or when independent changes have clearly different risk and review paths.

Do not bundle opportunistic refactors, formatting churn, or unrelated cleanup
with the requested change. When related cleanup is useful but not necessary,
note it as follow-up work instead of hiding it inside the branch.

## Branches And Commits

Use an agent-owned branch name that names the issue or task, such as
`<agent-id>/issue-123-short-topic`. Do not force-push, delete shared branches,
push directly to protected branches, or rewrite history unless the operator or
repository policy explicitly allows it.

Commit messages should describe the user-visible or operator-visible behavior
change. Avoid mentioning secrets, tokens, or private operational details.

## Opening A Change Request

Open a draft merge request or pull request when the branch is intentionally
partial, needs early design feedback, or has known failing checks. Open a
ready-for-review change request only after the implementation, self-review,
and expected tests are complete.

The description should include:

- the issue or request that motivated the change;
- a concise summary of behavior changed;
- the exact validation commands run and their results;
- known limitations, skipped checks, or follow-up issues.

## Closing References

Use closing keywords such as `Closes #123` or `Fixes #123` only when the
change fully resolves the issue and should close it automatically after merge.

For partial work, investigations, cleanup, or preparatory changes, link the
issue without a closing keyword and say what remains. When in doubt, avoid the
closing keyword and let a human decide.

## Self-Review

Before requesting review, the agent should inspect its own diff as if it were
reviewing another contributor's change. The review should look for:

- behavior that does not match the issue;
- missing tests for user-visible behavior;
- unsafe credential, permission, or forge-side effects;
- unrelated file churn;
- confusing names, comments, or documentation;
- commands that were not run but should have been.

If the agent finds a concern, it should either fix it before requesting review
or leave an explicit note in the change request explaining the remaining risk.

## Human Review And Mentions

Follow `<reviewer-convention>` for assigning reviewers, requesting review, and
mentioning humans. Do not broadly mention people or teams just to get
attention. Prefer the repository's normal review path.

When CI is blocked on human approval, when credentials are missing, or when a
policy decision is needed, the agent should explain the blocker and wait
instead of improvising a higher-risk workaround.

## CI And Follow-Up

Treat failing or skipped CI as work to triage before merge. If the failure is
unrelated or requires external access, record the evidence clearly in the
change request.

After merge or close events, record the outcome and mark the original task
handled. Do not open new issues, branches, or change requests unless the event
or operator instruction explicitly asks for follow-up work.
