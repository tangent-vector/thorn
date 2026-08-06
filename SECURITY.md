# Security Policy

Thorn is experimental research software that can execute tools, interact with
forge accounts, and call external model providers. Review the documented
[threat model](docs/threat-model.md), use dedicated least-privilege credentials,
and supervise deployments until their behavior and boundaries fit your
environment.

## Supported versions

Thorn does not currently publish a supported release series. Security fixes are
evaluated against the latest commit on the protected `main` branch; older
commits and private forks are not maintained as supported versions.

## Report a vulnerability privately

Do not open a public issue for a suspected vulnerability. Use GitHub's
[private vulnerability reporting](https://github.com/tangent-vector/thorn/security/advisories/new)
to share the report with the repository owner without publishing it.

Include, when available:

- the affected component and commit;
- the expected and observed security boundary;
- reproduction steps or a minimal proof of concept;
- the potential impact; and
- any suggested mitigation.

Remove live credentials, private repository content, personal data, and
provider request payloads from the report. If a secret may have been exposed,
revoke or rotate it rather than sending it through the report.

This is an independently maintained project and does not offer a response or
remediation SLA. Reports will be assessed against Thorn's stated security
boundaries and current implementation.
