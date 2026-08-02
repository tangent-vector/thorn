"""Unified forge abstraction layer.

Provides a forge-neutral interface (``ForgeClient``) with GitLab and
GitHub adapter implementations, plus the service types that represent
forge connections and project definitions in a Thorn agency.

The ``FORGE_TOOLS`` toolset exposes twelve ``@tool`` functions that
resolve the backing ``ForgeClient`` through the ambient
``ExecutionContext.runtime`` -- agents never need to know which forge
backend is in use.

Architecture:

    ForgeClient  (Protocol)
        GitLabForgeClient  (wraps thorn.tools.gitlab.GitLabClient)
        GitHubForgeClient  (wraps thorn.tools.github.GitHubClient)

    ForgeHostService (Service)  -- GitLabForgeService or GitHubForgeService
    ProjectService (Service)  -- represents a project on a forge

In-process execution and untrusted-data hygiene
-----------------------------------------------

``FORGE_TOOLS`` are venue-tagged ``IN_PROCESS`` (see
``thorn.tools._catalog``) because they need brain-side runtime
state -- the service registry and the agency's credential broker --
that the toolhost daemon's ``ExecutionContext`` does not have.  This
means agent-supplied arguments reach this module's Python code
directly, rather than being marshaled through the daemon's RPC
boundary first.

Argument-handling discipline reviewed against this venue choice:

- No ``subprocess`` / ``os.system`` / ``shell=True`` usage; all
  network access goes through ``pygithub`` / ``python-gitlab``,
  which URL-quote their own parameters.
- No ``eval`` / ``exec`` / ``compile``.
- No local filesystem writes, and no local reads driven by
  agent-supplied paths.  ``forge_read_file(file_path)`` sends
  ``file_path`` to the *remote* forge API; it is a key into the
  remote repository, not a local path.
- No logging of agent-supplied values, so no risk of leaking
  credentials or PII via log capture.
- The ``project`` argument is resolved through
  ``runtime.get_service(...)``, which raises if the operator has
  not declared that name; agents cannot point a forge tool at an
  arbitrary URL.

Anything added to this module that does perform shell, filesystem,
or eval-shaped work must either re-validate its inputs explicitly or
convince itself that it can keep this discipline; the venue tag is
not a substitute for the per-tool review.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Literal, Protocol

from pydantic import BaseModel, Field

from thorn.core._account import (
    AccountConfig,
    require_credential,
)
from thorn.core._brokering import (
    BrokerableService,
    BrokerCredentialPlan,
    HeaderInjection,
)
from thorn.core._context import get_context
from thorn.core._credentials import ServiceCredential
from thorn.core._executor import ToolResultContent, ToolVenue
from thorn.core._external_content import ExternalContentPeerStatus
from thorn.core._func import tool
from thorn.core._service import Service
from thorn.tools._credential_scopes import CredentialScopeInspection
from thorn.tools._github_connection import (
    GitHubConnectionConfig,
    GitHubPatAuth,
)
from thorn.tools._github_notification_ids import GitHubNotificationThreadID

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ForgeClient protocol
# ---------------------------------------------------------------------------

CommentTargetKind = Literal["Issue", "ChangeRequest"]
"""Forge-neutral target type for comments."""

ChangeRequestState = Literal["open", "closed", "merged", "all"]
"""Normalized state vocabulary for change requests."""

IssueState = Literal["open", "closed", "all"]
"""Normalized state vocabulary for issues."""

AuthoredForgeTextKind = Literal[
    "comment",
    "issue_body",
    "issue_title",
    "pr_body",
    "pr_title",
]
"""External-content kind values emitted by forge tool result wrappers."""


class ForgeUserIdentity(BaseModel):
    """Immutable account identity resolved from a forge handle."""

    account_id: str = Field(
        min_length=1,
        description=(
            "Platform-immutable user ID, stringified for JSON "
            "compatibility.  GitHub/GitLab expose numeric IDs."
        ),
    )
    display_handle: str = Field(
        default="",
        description="Current user-facing handle, used only for display.",
    )
    display_name: str = Field(
        default="",
        description="Current human display name, used only for display.",
    )


class ForgeClient(Protocol):
    """Unified interface for forge operations.

    Implementations translate between normalized names/states and the
    native forge API.  ``native_project_id`` is the forge-native
    identifier (stringified int for GitLab, ``owner/repo`` for GitHub).
    """

    def get_issue(
        self, native_project_id: str, issue_id: int,
    ) -> dict[str, Any]: ...

    def post_comment(
        self, native_project_id: str, target_type: str,
        target_id: int, body: str,
    ) -> None: ...

    def create_change_request(
        self, native_project_id: str, source_branch: str,
        title: str, target_branch: str, description: str,
    ) -> dict[str, Any]: ...

    def get_change_request(
        self, native_project_id: str, cr_id: int,
    ) -> dict[str, Any]: ...

    def list_change_requests(
        self, native_project_id: str, state: str,
    ) -> list[dict[str, Any]]: ...

    def list_comments(
        self, native_project_id: str, target_type: str, target_id: int,
    ) -> list[dict[str, Any]]: ...

    def get_project_info(
        self, native_project_id: str,
    ) -> dict[str, Any]: ...

    def inspect_credential_scopes(self) -> CredentialScopeInspection:
        """Return observable credential-scope warnings for preflight."""
        ...

    def read_file(
        self, native_project_id: str, file_path: str, ref: str,
    ) -> dict[str, Any]: ...

    def mark_notification_done(
        self, notification_id: str,
    ) -> None: ...

    def create_issue(
        self, native_project_id: str, title: str, description: str,
        labels: list[str], assignees: list[str],
    ) -> dict[str, Any]: ...

    def list_issues(
        self, native_project_id: str, state: str,
        labels: list[str] | None,
    ) -> list[dict[str, Any]]: ...

    def update_issue(
        self, native_project_id: str, issue_id: int,
        title: str | None, description: str | None,
        state: str | None, labels: list[str] | None,
        assignees: list[str] | None,
    ) -> dict[str, Any]: ...

    def resolve_account_handle(self, handle: str) -> ForgeUserIdentity:
        """Resolve a mutable forge handle to an immutable user identity."""
        ...


# ---------------------------------------------------------------------------
# GitLab adapter
# ---------------------------------------------------------------------------


class GitLabForgeClient:
    """``ForgeClient`` adapter wrapping ``GitLabClient``.

    Translates terminology (MR -> change request, note -> comment),
    maps ``native_project_id`` from string to int, and normalizes
    state vocabulary (``"opened"`` <-> ``"open"``).
    """

    def __init__(self, gl_client: Any) -> None:
        self._gl = gl_client

    @staticmethod
    def _pid(native_project_id: str) -> int | str:
        """Coerce a config-supplied native ID for ``GitLabClient``.

        Numeric strings become ``int`` for parity with the legacy
        config shape.  Human-readable ``namespace/project`` paths are
        passed through; the lower-level GitLab client resolves them to
        numeric IDs when a self-hosted instance rejects path-based API
        project lookups.
        """
        if native_project_id.isdigit():
            return int(native_project_id)
        return native_project_id

    @staticmethod
    def _target_type(target_type: str) -> str:
        if target_type == "ChangeRequest":
            return "MergeRequest"
        return target_type

    @staticmethod
    def _to_normalized_state(gl_state: str) -> str:
        if gl_state == "opened":
            return "open"
        return gl_state

    @staticmethod
    def _from_normalized_state(state: str) -> str:
        if state == "open":
            return "opened"
        return state

    def get_issue(
        self, native_project_id: str, issue_id: int,
    ) -> dict[str, Any]:
        raw = self._gl.get_issue(self._pid(native_project_id), issue_id)
        return {
            "id": raw["iid"],
            "title": raw["title"],
            "state": self._to_normalized_state(raw["state"]),
            "url": raw["web_url"],
            "description": raw.get("description", ""),
            "labels": raw.get("labels", []),
            "assignees": raw.get("assignees", []),
            "author": raw.get("author"),
            "created_at": raw.get("created_at"),
        }

    def post_comment(
        self, native_project_id: str, target_type: str,
        target_id: int, body: str,
    ) -> None:
        self._gl.post_note(
            self._pid(native_project_id),
            self._target_type(target_type),
            target_id,
            body,
        )

    def create_change_request(
        self, native_project_id: str, source_branch: str,
        title: str, target_branch: str, description: str,
    ) -> dict[str, Any]:
        raw = self._gl.create_merge_request(
            self._pid(native_project_id),
            source_branch=source_branch,
            title=title,
            target_branch=target_branch,
            description=description,
        )
        return {
            "id": raw["iid"],
            "title": raw["title"],
            "state": self._to_normalized_state(raw["state"]),
            "url": raw["web_url"],
            "source_branch": raw["source_branch"],
            "target_branch": raw["target_branch"],
        }

    def get_change_request(
        self, native_project_id: str, cr_id: int,
    ) -> dict[str, Any]:
        raw = self._gl.get_merge_request(
            self._pid(native_project_id), cr_id,
        )
        return {
            "id": raw["iid"],
            "title": raw["title"],
            "state": self._to_normalized_state(raw["state"]),
            "url": raw["web_url"],
            "description": raw.get("description", ""),
            "source_branch": raw["source_branch"],
            "target_branch": raw["target_branch"],
            "merge_status": raw.get("merge_status", ""),
            "author": raw.get("author"),
            "created_at": raw.get("created_at"),
        }

    def list_change_requests(
        self, native_project_id: str, state: str,
    ) -> list[dict[str, Any]]:
        gl_state = self._from_normalized_state(state)
        raw_list = self._gl.list_merge_requests(
            self._pid(native_project_id), gl_state,
        )
        return [
            {
                "id": mr["iid"],
                "title": mr["title"],
                "state": self._to_normalized_state(mr["state"]),
                "url": mr["web_url"],
                "source_branch": mr["source_branch"],
                "author": mr.get("author"),
                "author_user": mr.get("author_user"),
            }
            for mr in raw_list
        ]

    def list_comments(
        self, native_project_id: str, target_type: str, target_id: int,
    ) -> list[dict[str, Any]]:
        raw_notes = self._gl.list_notes(
            self._pid(native_project_id),
            self._target_type(target_type),
            target_id,
        )
        return [
            {
                "author": n["author"],
                "author_user": n.get("author_user"),
                "created_at": n["created_at"],
                "body": n["body"],
                "is_system": n.get("system", False),
            }
            for n in raw_notes
        ]

    def get_project_info(
        self, native_project_id: str,
    ) -> dict[str, Any]:
        raw = self._gl.get_project_info(self._pid(native_project_id))
        return {
            "name": raw.get("name_with_namespace", raw.get("name", "")),
            "path": raw.get("path_with_namespace", ""),
            "clone_url": raw.get("http_url_to_repo", ""),
            "ssh_url": raw.get("ssh_url_to_repo", ""),
            "default_branch": raw.get("default_branch", ""),
            "url": raw.get("web_url", ""),
            "description": raw.get("description", ""),
        }

    def inspect_credential_scopes(self) -> CredentialScopeInspection:
        return self._gl.inspect_credential_scopes()

    def read_file(
        self, native_project_id: str, file_path: str, ref: str,
    ) -> dict[str, Any]:
        return self._gl.read_file(
            self._pid(native_project_id), file_path, ref,
        )

    def mark_notification_done(self, notification_id: str) -> None:
        self._gl.mark_todo_done(int(notification_id))

    def resolve_account_handle(self, handle: str) -> ForgeUserIdentity:
        raw = self._gl.get_user_by_username(handle)
        return ForgeUserIdentity(
            account_id=str(raw["id"]),
            display_handle=str(raw.get("username") or handle),
            display_name=str(raw.get("name") or ""),
        )

    def create_issue(
        self, native_project_id: str, title: str, description: str,
        labels: list[str], assignees: list[str],
    ) -> dict[str, Any]:
        raw = self._gl.create_issue(
            self._pid(native_project_id),
            title=title,
            description=description,
            labels=labels or None,
            assignees=assignees or None,
        )
        return {
            "id": raw["iid"],
            "title": raw["title"],
            "state": self._to_normalized_state(raw["state"]),
            "url": raw["web_url"],
            "description": raw.get("description", ""),
            "labels": raw.get("labels", []),
            "assignees": raw.get("assignees", []),
        }

    def list_issues(
        self, native_project_id: str, state: str,
        labels: list[str] | None,
    ) -> list[dict[str, Any]]:
        gl_state = self._from_normalized_state(state)
        raw_list = self._gl.list_issues(
            self._pid(native_project_id), gl_state, labels,
        )
        return [
            {
                "id": issue["iid"],
                "title": issue["title"],
                "state": self._to_normalized_state(issue["state"]),
                "url": issue["web_url"],
                "labels": issue.get("labels", []),
                "assignees": issue.get("assignees", []),
                "author": issue.get("author"),
                "author_user": issue.get("author_user"),
            }
            for issue in raw_list
        ]

    def update_issue(
        self, native_project_id: str, issue_id: int,
        title: str | None, description: str | None,
        state: str | None, labels: list[str] | None,
        assignees: list[str] | None,
    ) -> dict[str, Any]:
        gl_state = None
        if state is not None:
            gl_state = {"open": "reopen", "closed": "close"}.get(state, state)
        raw = self._gl.update_issue(
            self._pid(native_project_id), issue_id,
            title=title, description=description,
            state=gl_state, labels=labels, assignees=assignees,
        )
        return {
            "id": raw["iid"],
            "title": raw["title"],
            "state": self._to_normalized_state(raw["state"]),
            "url": raw["web_url"],
            "description": raw.get("description", ""),
            "labels": raw.get("labels", []),
            "assignees": raw.get("assignees", []),
            "author": raw.get("author"),
            "created_at": raw.get("created_at"),
        }


# ---------------------------------------------------------------------------
# GitHub adapter
# ---------------------------------------------------------------------------


class GitHubForgeClient:
    """``ForgeClient`` adapter wrapping ``GitHubClient``.

    Translates terminology (PR -> change request) and synthesizes the
    ``"merged"`` state from ``state == "closed" and merged == True``.
    """

    def __init__(self, gh_client: Any) -> None:
        self._gh = gh_client

    @staticmethod
    def _normalize_cr_state(raw: dict[str, Any]) -> str:
        """Derive normalized state from GitHub's state + merged flag."""
        if raw.get("merged"):
            return "merged"
        gh_state = raw.get("state", "open")
        return gh_state

    def get_issue(
        self, native_project_id: str, issue_id: int,
    ) -> dict[str, Any]:
        raw = self._gh.get_issue(native_project_id, issue_id)
        return {
            "id": raw["number"],
            "title": raw["title"],
            "state": raw["state"],
            "url": raw["html_url"],
            "description": raw.get("body", ""),
            "labels": raw.get("labels", []),
            "assignees": raw.get("assignees", []),
            "author": raw.get("author"),
            "created_at": raw.get("created_at"),
        }

    def post_comment(
        self, native_project_id: str, target_type: str,
        target_id: int, body: str,
    ) -> None:
        self._gh.post_comment(native_project_id, target_id, body)

    def create_change_request(
        self, native_project_id: str, source_branch: str,
        title: str, target_branch: str, description: str,
    ) -> dict[str, Any]:
        raw = self._gh.create_pull_request(
            native_project_id,
            head=source_branch,
            title=title,
            base=target_branch,
            body=description,
        )
        return {
            "id": raw["number"],
            "title": raw["title"],
            "state": raw["state"],
            "url": raw["html_url"],
            "source_branch": raw["head"],
            "target_branch": raw["base"],
        }

    def get_change_request(
        self, native_project_id: str, cr_id: int,
    ) -> dict[str, Any]:
        raw = self._gh.get_pull_request(native_project_id, cr_id)
        return {
            "id": raw["number"],
            "title": raw["title"],
            "state": self._normalize_cr_state(raw),
            "url": raw["html_url"],
            "description": raw.get("body", ""),
            "source_branch": raw["head"],
            "target_branch": raw["base"],
            "mergeable": raw.get("mergeable"),
            "mergeable_state": raw.get("mergeable_state", ""),
            "author": raw.get("author"),
            "created_at": raw.get("created_at"),
        }

    def list_change_requests(
        self, native_project_id: str, state: str,
    ) -> list[dict[str, Any]]:
        if state == "merged":
            raw_list = self._gh.list_pull_requests(
                native_project_id, "closed",
            )
            raw_list = [pr for pr in raw_list if pr.get("merged")]
        elif state == "open":
            raw_list = self._gh.list_pull_requests(
                native_project_id, "open",
            )
        else:
            raw_list = self._gh.list_pull_requests(
                native_project_id, state,
            )

        return [
            {
                "id": pr["number"],
                "title": pr["title"],
                "state": self._normalize_cr_state(pr),
                "url": pr["html_url"],
                "source_branch": pr.get("head", ""),
                "author": pr.get("author"),
                "author_user": pr.get("author_user"),
            }
            for pr in raw_list
        ]

    def list_comments(
        self, native_project_id: str, target_type: str, target_id: int,
    ) -> list[dict[str, Any]]:
        raw_comments = self._gh.list_comments(
            native_project_id, target_id,
        )
        return [
            {
                "author": c["author"],
                "author_user": c.get("author_user"),
                "created_at": c["created_at"],
                "body": c["body"],
                "is_system": c.get("is_bot", False),
            }
            for c in raw_comments
        ]

    def get_project_info(
        self, native_project_id: str,
    ) -> dict[str, Any]:
        raw = self._gh.get_repo_info(native_project_id)
        return {
            "name": raw.get("full_name", raw.get("name", "")),
            "path": raw.get("full_name", ""),
            "clone_url": raw.get("clone_url", ""),
            "ssh_url": raw.get("ssh_url", ""),
            "default_branch": raw.get("default_branch", ""),
            "url": raw.get("html_url", ""),
            "description": raw.get("description", ""),
        }

    def inspect_credential_scopes(self) -> CredentialScopeInspection:
        return self._gh.inspect_credential_scopes()

    def read_file(
        self, native_project_id: str, file_path: str, ref: str,
    ) -> dict[str, Any]:
        return self._gh.read_file(native_project_id, file_path, ref)

    def mark_notification_done(self, notification_id: str) -> None:
        thread_id = GitHubNotificationThreadID.parse(notification_id)
        self._gh.mark_notification_done(thread_id)

    def resolve_account_handle(self, handle: str) -> ForgeUserIdentity:
        raw = self._gh.get_user_by_login(handle)
        return ForgeUserIdentity(
            account_id=str(raw["id"]),
            display_handle=str(raw.get("login") or handle),
            display_name=str(raw.get("name") or ""),
        )

    def create_issue(
        self, native_project_id: str, title: str, description: str,
        labels: list[str], assignees: list[str],
    ) -> dict[str, Any]:
        raw = self._gh.create_issue(
            native_project_id,
            title=title,
            body=description,
            labels=labels or None,
            assignees=assignees or None,
        )
        return {
            "id": raw["number"],
            "title": raw["title"],
            "state": raw["state"],
            "url": raw["html_url"],
            "description": raw.get("body", ""),
            "labels": raw.get("labels", []),
            "assignees": raw.get("assignees", []),
        }

    def list_issues(
        self, native_project_id: str, state: str,
        labels: list[str] | None,
    ) -> list[dict[str, Any]]:
        raw_list = self._gh.list_issues(native_project_id, state, labels)
        return [
            {
                "id": issue["number"],
                "title": issue["title"],
                "state": issue["state"],
                "url": issue["html_url"],
                "labels": issue.get("labels", []),
                "assignees": issue.get("assignees", []),
                "author": issue.get("author"),
                "author_user": issue.get("author_user"),
            }
            for issue in raw_list
        ]

    def update_issue(
        self, native_project_id: str, issue_id: int,
        title: str | None, description: str | None,
        state: str | None, labels: list[str] | None,
        assignees: list[str] | None,
    ) -> dict[str, Any]:
        raw = self._gh.update_issue(
            native_project_id, issue_id,
            title=title, body=description,
            state=state, labels=labels, assignees=assignees,
        )
        return {
            "id": raw["number"],
            "title": raw["title"],
            "state": raw["state"],
            "url": raw["html_url"],
            "description": raw.get("body", ""),
            "labels": raw.get("labels", []),
            "assignees": raw.get("assignees", []),
            "author": raw.get("author"),
            "created_at": raw.get("created_at"),
        }


# ---------------------------------------------------------------------------
# Forge host services (GitLab vs GitHub)
# ---------------------------------------------------------------------------


class GitLabForgeServiceConfig(BaseModel):
    """Configuration for a ``gitlab`` forge service."""

    url: str = Field(description="GitLab API base URL")
    # Empty by default: per-agent credentials live on the agent's
    # :class:`GitLabAccountConfig` and are injected via the broker
    # at agent-load time.  The field is retained as an empty
    # placeholder so older code paths that read ``config.token``
    # continue to compile; new code paths use the account-side
    # credential exclusively.
    token: str = Field(default="", description="Deprecated; left empty")


# ---------------------------------------------------------------------------
# Per-agent forge account configs
# ---------------------------------------------------------------------------


class _ForgeAccountBase(AccountConfig):
    """Common fields for accounts on a forge service.

    Both GitHub and GitLab accounts carry the same git-author
    identity fields (``git_user_name`` / ``git_user_email``); we
    factor them onto an internal base to keep the per-forge classes
    free of repetition.  Operators don't see this class -- the
    discrimination at parse time is by the ``service`` discriminator
    on :class:`UntypedAccountConfig`, which the gateway's validation
    pass routes to the right concrete subclass.
    """

    git_user_name: str = Field(
        default="",
        description=(
            "Git author/committer name when this account drives "
            "git operations on its forge."
        ),
    )
    git_user_email: str = Field(
        default="",
        description=(
            "Git author/committer email when this account drives "
            "git operations on its forge."
        ),
    )


class GitHubAccountConfig(_ForgeAccountBase):
    """An agent's account on a GitHub (or GitHub Enterprise) forge.

    The credentials list typically holds a single ``"pat"`` entry
    referencing the env var the operator put their PAT into; broker
    registration reads the value from that env var at gateway
    startup and keeps the literal out of agent state thereafter.
    """


class GitLabAccountConfig(_ForgeAccountBase):
    """An agent's account on a GitLab forge.

    The credentials list typically holds a single ``"gitlab-pat"``
    entry referencing the env var the operator put their PAT into.
    """


def _git_https_basic_transform(raw: str) -> str:
    """Pre-encode a forge PAT for HTTP Basic against a git HTTPS endpoint.

    Git's HTTPS wire protocol expects ``Authorization: Basic
    <base64(user:password)>``.  Both GitHub and GitLab accept
    ``x:<pat>`` in the user:password slot (the legacy-by-convention
    "literal token in the password field" shape); we pick ``x`` as
    a stable, meaningless username so OneCLI's registered secret
    carries a byte-for-byte match for what ``git`` already sends.

    Run at broker-registration time, not at sandbox launch, so the
    base64 encoding is done once per agent load rather than per
    request; the output is what gets stored in OneCLI's encrypted
    secret store.  Exposed at module level (rather than inlined as
    a lambda) because ``functools.partial``-style lambdas capture
    closures awkwardly for the ``broker_credential_plans`` call
    path and this shape is naturally shared between the two forge
    implementations.
    """
    return base64.b64encode(f"x:{raw}".encode()).decode()


class ForgeHostService(BrokerableService, ABC):
    """API client plus credentials for HTTPS git against this forge host.

    Concrete subclasses implement the account-driven authentication
    methods below.  Account credentials are referenced from the
    agent's :class:`AccountConfig` (a :class:`GitHubAccountConfig` or
    :class:`GitLabAccountConfig`) and the literal secret value is
    read from ``os.environ`` at the point of use -- the agent state
    holds only the env var name.

    Subclasses also implement
    :meth:`BrokerableService.broker_credential_plans` so the gateway
    can register this forge's credentials with the upstream
    credential broker; that's how in-sandbox tools end up
    authenticating without ever seeing the literal token.
    """

    @abstractmethod
    def authenticated_client(
        self,
        account: AccountConfig,
    ) -> ForgeClient:
        """Build a forge client authenticated with *account*'s credentials.

        Implementations look up the appropriate credential on
        *account* (typically the first ``"pat"`` entry), read its
        value from ``os.environ``, and use the value to construct a
        client.  Raise :class:`TypeError` when *account* is not a
        typed account of the shape this forge expects.
        """

    @abstractmethod
    def git_https_password_for(
        self,
        account: AccountConfig,
    ) -> str:
        """Return the literal HTTPS password/token for git operations.

        Reads the underlying env var and returns the raw string
        (suitable for injection into git's HTTPS-auth env vars).
        Returns ``""`` when the account has no usable credential.
        """

    @abstractmethod
    def clone_url_for(self, native_id: str) -> str:
        """Derive the HTTPS clone URL for a project on this forge.

        Each forge backend knows how to construct a clone URL from
        its base URL and the forge-native project identifier.
        """


class GitLabForgeService(ForgeHostService):
    """Connection to a GitLab instance."""

    Config: ClassVar[type[BaseModel]] = GitLabForgeServiceConfig
    AccountConfig: ClassVar[type[AccountConfig] | None] = GitLabAccountConfig

    _CREDENTIAL_KIND: ClassVar[str] = "gitlab-pat"
    """Credential kind this forge expects on agent accounts.

    Lives as a ClassVar so the per-credential discrimination logic
    (looking up the credential on the account, planning broker
    registration, etc.) all reads from one place rather than
    repeating the literal in each method.
    """

    def __init__(self, config: GitLabForgeServiceConfig, *, service_name: str) -> None:
        self._config = config
        self._service_name = service_name

    @property
    def name(self) -> str:
        return self._service_name

    @property
    def url(self) -> str:
        """GitLab instance API base URL."""
        return self._config.url

    def _resolve_account(self, account: AccountConfig) -> GitLabAccountConfig:
        if not isinstance(account, GitLabAccountConfig):
            raise TypeError(
                f"GitLabForgeService requires GitLabAccountConfig, "
                f"got {type(account).__name__}"
            )
        return account

    def _read_credential_value(self, account: AccountConfig) -> ServiceCredential:
        gl_account = self._resolve_account(account)
        cred = require_credential(gl_account, kind=self._CREDENTIAL_KIND)
        return cred.read_value()

    def authenticated_client(
        self,
        account: AccountConfig,
    ) -> ForgeClient:
        from thorn.tools.gitlab import GitLabClient, GitLabConfig

        token = self._read_credential_value(account)
        gl_config = GitLabConfig(url=self._config.url, token=str(token))
        return GitLabForgeClient(GitLabClient(gl_config))

    def git_https_password_for(
        self,
        account: AccountConfig,
    ) -> str:
        return str(self._read_credential_value(account))

    def broker_credential_plans(
        self,
        account: AccountConfig,
    ) -> list[BrokerCredentialPlan]:
        """Plan broker registrations per ``gitlab-pat`` credential.

        Each credential contributes *two* plans:

        * An API plan: ``Authorization: Bearer <pat>`` against
          ``/api/*`` on the GitLab instance's host.  This covers
          ``python-gitlab`` and direct REST consumers.
        * A git HTTPS plan: ``Authorization: Basic <base64(x:pat)>``
          against ``/*`` on the same host, with
          ``git_extra_header_host`` set so the gateway renders a
          matching ``http.https://<host>/.extraHeader`` line in the
          per-agent gitconfig.  This is what wires
          ``git clone`` / ``git fetch`` through the broker.

        GitLab serves the API and git over the same host, so both
        plans share the host pattern; path-narrowing is how they
        stay distinct.
        """
        from urllib.parse import urlparse

        gl_account = self._resolve_account(account)
        host = urlparse(self._config.url).hostname or ""
        if not host:
            return []
        plans: list[BrokerCredentialPlan] = []
        for cred in gl_account.credentials:
            if cred.kind != self._CREDENTIAL_KIND:
                continue
            plans.append(BrokerCredentialPlan(
                env_var_name=cred.env_var_name,
                host_pattern=host,
                path_pattern="/api/*",
                injection=HeaderInjection(
                    header_name="Authorization",
                    value_format="Bearer {value}",
                ),
                secret_name_suffix="gitlab-pat",
            ))
            plans.append(BrokerCredentialPlan(
                env_var_name=cred.env_var_name,
                host_pattern=host,
                path_pattern="/*",
                injection=HeaderInjection(
                    header_name="Authorization",
                    value_format="Basic {value}",
                ),
                secret_name_suffix="gitlab-git-https",
                value_transform=_git_https_basic_transform,
                git_extra_header_host=host,
            ))
        return plans

    def clone_url_for(self, native_id: str) -> str:
        """Derive an HTTPS clone URL for a GitLab project.

        With the path-based ``native_id`` shape used by gateway
        config (e.g. ``"group/subgroup/project"``), the clone URL is
        just the GitLab instance URL with that path appended and a
        ``.git`` suffix.

        Returns an empty string when the native_id is purely numeric
        (legacy callers): we don't have enough information to
        reconstruct the path without a forge API call.
        """
        if not native_id or native_id.isdigit():
            return ""
        from urllib.parse import urlparse

        parsed = urlparse(self._config.url)
        host = parsed.hostname or ""
        scheme = parsed.scheme or "https"
        if not host:
            return ""
        return f"{scheme}://{host}/{native_id}.git"


class GitHubForgeService(ForgeHostService):
    """Connection to GitHub or GitHub Enterprise (PAT auth)."""

    Config: ClassVar[type[BaseModel]] = GitHubConnectionConfig
    AccountConfig: ClassVar[type[AccountConfig] | None] = GitHubAccountConfig

    _CREDENTIAL_KIND: ClassVar[str] = "pat"
    """Credential kind this forge expects on agent accounts.

    GitHub App auth (``"app"``) is intentionally not supported by
    the broker-routed path: OneCLI's substitution model handles
    static tokens only, not the JWT-signing + installation-token
    exchange flow App auth requires.  An account with only ``"app"``
    credentials produces an empty :meth:`broker_credential_plans`
    result; the operator should either move to PAT auth or run the
    gateway without sandboxing for that agent.
    """

    def __init__(self, config: GitHubConnectionConfig, *, service_name: str) -> None:
        self._config = config
        self._service_name = service_name

    @property
    def name(self) -> str:
        return self._service_name

    @property
    def base_url(self) -> str:
        """GitHub REST API base URL."""
        return self._config.base_url

    def _resolve_account(self, account: AccountConfig) -> GitHubAccountConfig:
        if not isinstance(account, GitHubAccountConfig):
            raise TypeError(
                f"GitHubForgeService requires GitHubAccountConfig, "
                f"got {type(account).__name__}"
            )
        return account

    def _read_credential_value(self, account: AccountConfig) -> ServiceCredential:
        gh_account = self._resolve_account(account)
        cred = require_credential(gh_account, kind=self._CREDENTIAL_KIND)
        return cred.read_value()

    def authenticated_client(
        self,
        account: AccountConfig,
    ) -> ForgeClient:
        from thorn.tools.github import GitHubClient

        value = self._read_credential_value(account)
        conn = GitHubConnectionConfig(
            base_url=self._config.base_url,
            auth=GitHubPatAuth(token=str(value)),
        )
        gh_client = GitHubClient(conn)
        return GitHubForgeClient(gh_client)

    def git_https_password_for(
        self,
        account: AccountConfig,
    ) -> str:
        return str(self._read_credential_value(account))

    def broker_credential_plans(
        self,
        account: AccountConfig,
    ) -> list[BrokerCredentialPlan]:
        """Plan broker registrations per ``pat`` credential.

        Each credential contributes *two* plans:

        * An API plan: ``Authorization: Bearer <pat>`` against the
          GitHub API host (``api.github.com`` for GitHub.com,
          otherwise the bare host from ``base_url``).  Covers
          ``pygithub`` and ``gh api`` / direct REST consumers.
        * A git HTTPS plan: ``Authorization: Basic <base64(x:pat)>``
          against ``/*`` on the *web* host (``github.com``, not the
          API host), with ``git_extra_header_host`` set so the
          gateway renders a matching ``http.https://<host>/.extraHeader``
          line in the per-agent gitconfig.  This wires ``git clone``
          / ``git fetch`` through the broker.

        GitHub splits API and git onto different hosts
        (``api.github.com`` vs ``github.com``), so the two plans
        carry different ``host_pattern`` values.  GitHub Enterprise
        collapses them back to a single host.
        """
        from urllib.parse import urlparse

        gh_account = self._resolve_account(account)
        parsed_host = urlparse(self._config.base_url).hostname or ""
        api_host = (
            "api.github.com" if parsed_host == "github.com" else parsed_host
        )
        if not api_host:
            return []
        # Git HTTPS goes against the bare web host, which differs
        # from the API host on GitHub.com (``api.github.com`` vs
        # ``github.com``).  GHE collapses the two onto a single host
        # served under ``/api/v3/``.  We mirror the same ``api.``
        # prefix strip that :meth:`clone_url_for` performs so both
        # callers agree on the forge's "web" identity.
        git_host = api_host[len("api."):] if api_host.startswith("api.") else api_host
        plans: list[BrokerCredentialPlan] = []
        for cred in gh_account.credentials:
            if cred.kind != self._CREDENTIAL_KIND:
                continue
            plans.append(BrokerCredentialPlan(
                env_var_name=cred.env_var_name,
                host_pattern=api_host,
                path_pattern="/*",
                injection=HeaderInjection(
                    header_name="Authorization",
                    value_format="Bearer {value}",
                ),
                secret_name_suffix="github-pat",
            ))
            plans.append(BrokerCredentialPlan(
                env_var_name=cred.env_var_name,
                host_pattern=git_host,
                path_pattern="/*",
                injection=HeaderInjection(
                    header_name="Authorization",
                    value_format="Basic {value}",
                ),
                secret_name_suffix="github-git-https",
                value_transform=_git_https_basic_transform,
                git_extra_header_host=git_host,
            ))
        return plans

    def clone_url_for(self, native_id: str) -> str:
        """Derive an HTTPS clone URL for a GitHub repository.

        For GitHub, ``native_id`` is ``owner/repo``, and the clone
        URL is ``https://<host>/<owner>/<repo>.git``.  The host is
        derived from the API base URL.
        """
        from urllib.parse import urlparse

        parsed = urlparse(self._config.base_url)
        host = parsed.hostname or "github.com"
        if host.startswith("api."):
            host = host[4:]
        return f"https://{host}/{native_id}.git"


# ---------------------------------------------------------------------------
# ProjectService
# ---------------------------------------------------------------------------


class ForkConfig(BaseModel):
    """A single fork of a project hosted on a forge.

    Built by :func:`thorn.gateway._config.instantiate_services` from a
    :class:`thorn.gateway._config.ForkSpec` -- ``native_id`` and
    ``clone_url`` are derived from the user-facing ``url`` so that
    consumers of ``ForkConfig`` have all the forge-specific identifiers
    they need without re-parsing.

    The ``name`` becomes the git remote name when the project is
    cloned locally (defaults are picked at config-resolution time:
    ``"origin"`` for single-fork projects, the forge name otherwise).
    """

    forge: str = Field(description="Name of the forge service hosting this fork")
    native_id: str = Field(
        description=(
            "Forge-native project identifier "
            "(``namespace/path`` for GitLab, ``owner/repo`` for GitHub)"
        ),
    )
    name: str = Field(
        default="",
        description="Local remote name for this fork (e.g. 'origin')",
    )
    clone_url: str = Field(
        default="",
        description="HTTPS clone URL for this fork",
    )
    default_branch: str = Field(
        default="",
        description=(
            "Per-fork override for the default branch.  When empty, "
            "the project-level default is used; when that is also "
            "empty, the value is fetched from the forge API on first "
            "access (see :meth:`ProjectService.resolve_default_branch`)."
        ),
    )


class ProjectServiceConfig(BaseModel):
    """Configuration for a project service.

    A project has one or more forks.  ``default_branch`` is an
    optional project-level override; when omitted, the value is
    resolved lazily by querying the forge API for the primary fork
    (see :meth:`ProjectService.resolve_default_branch`).
    """

    forks: list[ForkConfig] = Field(
        default_factory=list,
        description=(
            "List of forks of this project.  Defaults to an empty "
            "list so that a ``ProjectServiceConfig`` can be constructed "
            "incrementally; methods that need at least one fork (e.g. "
            ":meth:`ProjectService.get_fork`, "
            ":attr:`ProjectService.forge_name`) raise a clear "
            "``KeyError`` / ``ValueError`` when the list is still empty."
        ),
    )
    default_branch: str = Field(
        default="",
        description=(
            "Project-level default branch override.  When empty, "
            "the value is looked up from the forge for the primary "
            "fork on first access and cached for the process lifetime."
        ),
    )


class ProjectService(Service):
    """A named project within a forge (possibly with multiple forks).

    Carries the project's fork list and convenience accessors for
    the primary fork.  Resolves its ``ForgeClient`` by looking up the
    forge service on the ``Runtime``.

    The default branch for any given fork is resolved by
    :meth:`resolve_default_branch`, which consults a cascade of
    overrides before falling back to a forge API lookup.
    """

    Config: ClassVar[type[BaseModel]] = ProjectServiceConfig

    def __init__(
        self,
        config: ProjectServiceConfig,
        *,
        service_name: str,
    ) -> None:
        self._config = config
        self._service_name = service_name
        # Per-fork default-branch cache (filled by
        # ``resolve_default_branch`` on first lookup).  Keys are fork
        # names.  The cache is process-scoped because the default
        # branch on a forge changes infrequently and a stale value
        # is acceptable until the gateway restarts.
        self._default_branch_cache: dict[str, str] = {}

    @property
    def name(self) -> str:
        return self._service_name

    @property
    def forks(self) -> list[ForkConfig]:
        """The effective fork list for this project."""
        return list(self._config.forks)

    @property
    def forge_name(self) -> str:
        """Name of the forge hosting the primary fork."""
        forks = self.forks
        if not forks:
            raise ValueError(
                f"Project {self.name!r} has no forks configured"
            )
        return forks[0].forge

    @property
    def native_id(self) -> str:
        """Native identifier of the primary fork."""
        forks = self.forks
        if not forks:
            raise ValueError(
                f"Project {self.name!r} has no forks configured"
            )
        return forks[0].native_id

    @property
    def clone_url(self) -> str:
        """Clone URL of the primary fork."""
        forks = self.forks
        if not forks:
            return ""
        return forks[0].clone_url

    @property
    def default_branch(self) -> str:
        """Project-level default branch override (may be empty).

        This returns only the *configured* override; it does **not**
        trigger a forge API lookup.  Consumers that need a definitive
        value should call :meth:`resolve_default_branch`, which
        cascades through overrides and falls back to a live lookup.
        """
        return self._config.default_branch

    def resolve_default_branch(
        self,
        forge_client: ForgeClient,
        fork_name: str = "",
    ) -> str:
        """Resolve the default branch for *fork_name*, looking it up if needed.

        Resolution cascade:

        1. Per-fork override on :attr:`ForkConfig.default_branch`.
        2. Project-level override on :attr:`default_branch`.
        3. Process-cached previous lookup.
        4. Live ``get_project_info`` call via *forge_client*.

        The caller is responsible for supplying an authenticated
        :class:`ForgeClient` (typically via the forge service's
        ``authenticated_client(account)`` against the agent's
        :class:`AccountConfig`); this method only knows how to
        cascade through the override chain and cache the result.

        The result of step 4 is cached per fork for subsequent calls.
        Raises :class:`KeyError` for an unknown fork name.
        """
        fork = self.get_fork(fork_name)

        if fork.default_branch:
            return fork.default_branch
        if self._config.default_branch:
            return self._config.default_branch

        cache_key = fork.name
        cached = self._default_branch_cache.get(cache_key)
        if cached:
            return cached

        info = forge_client.get_project_info(fork.native_id)
        resolved = info.get("default_branch") or "main"
        self._default_branch_cache[cache_key] = resolved
        return resolved

    def get_fork(self, fork_name: str = "") -> ForkConfig:
        """Look up a fork by name, defaulting to the first fork.

        Raises :class:`KeyError` when the named fork doesn't exist.
        """
        forks = self.forks
        if not forks:
            raise KeyError(
                f"Project {self.name!r} has no forks configured"
            )
        if not fork_name:
            return forks[0]
        for fork in forks:
            if fork.name == fork_name:
                return fork
        raise KeyError(
            f"No fork named {fork_name!r} in project {self.name!r}. "
            f"Available: {[f.name for f in forks]}"
        )

# ---------------------------------------------------------------------------
# FORGE_TOOLS
# ---------------------------------------------------------------------------


def _resolve_with_forge_name(project: str) -> tuple[ForgeClient, str, str]:
    """Like :func:`_resolve` but also returns the forge service name.

    The forge service name is the same string that
    ``ActorIdentity.service`` and ``PeerAccount.service`` carry in
    the gateway's data model, which is what the registry indexes on.
    Surface it here so the body-wrapping helpers can ask the registry
    "is this user a peer on `<forge_name>`?" without re-walking the
    runtime's service graph.
    """
    from thorn.core._account import resolve_account

    ctx = get_context()
    if ctx.runtime is None:
        raise RuntimeError(
            "No Runtime available in the current ExecutionContext. "
            "Forge tools require a Runtime with registered services."
        )

    project_svc: ProjectService = ctx.runtime.get_service(project)
    if not isinstance(project_svc, ProjectService):
        raise TypeError(
            f"Service {project!r} is a "
            f"{type(project_svc).__name__}, not a ProjectService"
        )

    forge_svc: ForgeHostService = ctx.runtime.get_service(
        project_svc.forge_name,
    )
    if not isinstance(forge_svc, ForgeHostService):
        raise TypeError(
            f"Service {project_svc.forge_name!r} is a "
            f"{type(forge_svc).__name__}, not a ForgeHostService"
        )

    agent = ctx.agent
    if agent is None:
        raise RuntimeError(
            f"Forge tool for project {project!r} requires an active "
            "agent context with an account on forge "
            f"{forge_svc.name!r}, but no agent is in scope."
        )

    account = resolve_account(agent, forge_svc.name)
    return (
        forge_svc.authenticated_client(account),
        project_svc.native_id,
        forge_svc.name,
    )


def _actor_from_author(
    author: Any,
    *,
    fallback_login: str | None,
    service: str,
) -> Any:
    """Build an :class:`ActorIdentity` from a forge-adapter author dict.

    The shape that arrives here depends on which forge produced it:
    GitLab notes carry the raw user dict (``id``, ``username``,
    ``name``); GitHub comments carry the projection
    ``{"id", "login", "type"}``.  We accept either and pull out the
    immutable id (``id``) and textual handle (``login`` /
    ``username``) defensively.

    Returns ``None`` when *author* is missing or contains no usable
    handle -- the wrapper helper renders such bodies with
    ``actor=unknown`` rather than fabricating an identity.

    The :class:`ActorIdentity` import is deferred inside the helper
    so that ``thorn.tools.forge`` itself does not import any
    ``thorn.gateway`` module at module load time -- the gateway
    package re-exports ``thorn.tools.forge`` via
    ``thorn.gateway._agents``, and a top-level cross import would
    create a cycle.
    """
    from thorn.gateway._actor import ActorIdentity

    if isinstance(author, dict):
        immutable_id = author.get("id")
        login = author.get("login") or author.get("username")
        kind_hint = author.get("type")
        if immutable_id is None and login is None:
            return None
        return ActorIdentity(
            service=service,
            account_id=str(immutable_id) if immutable_id is not None else (
                login or ""
            ),
            display_name=author.get("name", "") or "",
            is_bot=(kind_hint == "Bot") if kind_hint is not None else None,
            secondary_account_ids=(login,) if login else (),
        )
    if fallback_login:
        return ActorIdentity(
            service=service,
            account_id=fallback_login,
            display_name="",
            is_bot=None,
            secondary_account_ids=(),
        )
    return None


def _peer_status_for(actor: Any) -> Any:
    """Resolve *actor* against the runtime's peer registry.

    Lives here rather than on the registry because the
    "no peer registry visible to this code" case is a common /
    expected one for in-process CLI runs that don't have a gateway.
    Returns ``PeerStatus.UNKNOWN`` when the registry is empty or
    when *actor* is ``None``.
    """
    from thorn.gateway._envelope import PeerStatus

    if actor is None:
        return PeerStatus.UNKNOWN
    ctx = get_context()
    runtime = ctx.runtime
    if runtime is None:
        return PeerStatus.UNKNOWN
    registry = getattr(runtime, "peer_registry", None)
    if registry is None:
        return PeerStatus.UNKNOWN
    spec = registry.lookup_actor(actor)
    if spec is not None:
        return PeerStatus.PEER
    # Empty registry = "no peers configured"; reporting NON_PEER in
    # that case would be misleading (no peers means we don't know
    # whether the actor *would* be one).  An empty registry is
    # detectable via no entries on the per-service bucket the actor
    # falls into.
    if not registry.all_peers():
        return PeerStatus.UNKNOWN
    return PeerStatus.NON_PEER


def _split_author_fields(info: dict[str, Any]) -> tuple[Any, str | None]:
    """Return structured author data plus a login fallback from an API record."""
    author_user = info.get("author_user")
    if author_user is not None:
        return author_user, None

    author = info.get("author")
    if isinstance(author, dict):
        return author, None
    if isinstance(author, str) and author.strip():
        return None, author.strip()
    return None, None


def _display_author(info: dict[str, Any]) -> str:
    """Return the best non-authoritative display handle for list metadata."""
    author = info.get("author")
    if isinstance(author, str) and author.strip():
        return author.strip()
    if isinstance(author, dict):
        handle = author.get("login") or author.get("username")
        if handle:
            return str(handle)
    author_user = info.get("author_user")
    if isinstance(author_user, dict):
        handle = author_user.get("login") or author_user.get("username")
        if handle:
            return str(handle)
    return "unknown"


def _combine_tool_result_content(
    content: str,
    *parts: ToolResultContent,
) -> ToolResultContent:
    peer_statuses: set[ExternalContentPeerStatus] = set()
    for part in parts:
        peer_statuses.update(part.external_content_peer_statuses)
    return ToolResultContent(
        content,
        external_content_peer_statuses=frozenset(peer_statuses),
    )


def _wrap_authored_text(
    *,
    text: str,
    author: Any,
    fallback_login: str | None,
    service: str,
    kind: AuthoredForgeTextKind,
    timestamp: Any,
) -> ToolResultContent:
    """Render user-authored forge text inside an external-content envelope.

    Single source of truth for forge-tool text wrapping: builds an
    :class:`ActorIdentity`, resolves peer status against the active
    registry, and delegates to :func:`wrap_external` so the envelope
    shape matches what the gateway-side notification formatter
    produces.  Empty / missing text is still wrapped (the envelope
    body becomes a "(no body)" line); the data-vs-instruction rule
    holds for a literal "this field has no body" case the same way
    it would for any other quoted content.
    """
    from thorn.gateway._envelope import wrap_external

    actor = _actor_from_author(
        author, fallback_login=fallback_login, service=service,
    )
    peer_status = _peer_status_for(actor)
    ts = ""
    if timestamp:
        ts = (
            timestamp.isoformat()
            if hasattr(timestamp, "isoformat")
            else str(timestamp)
        )
    # ``wrap_external`` already renders an empty body as a "(no body)"
    # blockquote line, so we pass the raw value through unchanged
    # rather than substituting a sentinel here.
    wrapped = wrap_external(
        body=text or "",
        actor=actor,
        source=service,
        kind=kind,
        peer_status=peer_status,
        timestamp=ts,
    )
    return ToolResultContent(
        wrapped,
        external_content_peer_statuses=frozenset(
            {ExternalContentPeerStatus(peer_status.value)}
        ),
    )


def _wrap_authored_body(
    *,
    body: str,
    author: Any,
    fallback_login: str | None,
    service: str,
    kind: AuthoredForgeTextKind,
    timestamp: Any,
) -> ToolResultContent:
    """Render an issue / change-request body or comment as external content."""
    return _wrap_authored_text(
        text=body,
        author=author,
        fallback_login=fallback_login,
        service=service,
        kind=kind,
        timestamp=timestamp,
    )


def _wrap_authored_title(
    *,
    info: dict[str, Any],
    service: str,
    kind: AuthoredForgeTextKind,
) -> ToolResultContent:
    """Render a forge title without making it trusted harness prose."""
    author, fallback_login = _split_author_fields(info)
    return _wrap_authored_text(
        text=str(info.get("title") or ""),
        author=author,
        fallback_login=fallback_login,
        service=service,
        kind=kind,
        timestamp=info.get("created_at"),
    )


def _resolve(project: str) -> tuple[ForgeClient, str]:
    """Resolve an authenticated ForgeClient + native ID for *project*.

    Convenience wrapper around :func:`_resolve_with_forge_name` for
    call sites that don't need the forge service name (most tools
    don't -- only the user-authored-body wrappers do).
    """
    client, native_id, _forge_name = _resolve_with_forge_name(project)
    return client, native_id


@tool(venue=ToolVenue.IN_PROCESS)
async def forge_read_issue(project: str, issue_id: int) -> str:
    """Read an issue from a project.

    *project* is the name of the project service.  *issue_id* is the
    issue number within the project.

    The issue title and description are wrapped in
    ``[external-content]`` envelopes so the agent's
    data-vs-instruction rule applies the same way it does for issue
    text that arrives via a notification.  Metadata fields (state,
    labels, assignees, URL) are not wrapped because they are
    platform-controlled structured data, not free-form text.
    """
    client, native_id, forge_name = _resolve_with_forge_name(project)
    info = await asyncio.to_thread(client.get_issue, native_id, issue_id)
    wrapped_title = _wrap_authored_title(
        info=info,
        service=forge_name,
        kind="issue_title",
    )
    wrapped_body = _wrap_authored_body(
        body=info.get("description") or "",
        author=info.get("author"),
        fallback_login=None,
        service=forge_name,
        kind="issue_body",
        timestamp=info.get("created_at"),
    )
    lines = [
        f"Issue #{info['id']}",
        "Title:",
        wrapped_title,
        f"State: {info['state']}",
        f"Labels: {', '.join(info.get('labels', [])) or '(none)'}",
        f"Assignees: {', '.join(info.get('assignees', [])) or '(none)'}",
        f"URL: {info['url']}",
        "",
        "Description:",
        wrapped_body,
    ]
    return _combine_tool_result_content(
        "\n".join(lines),
        wrapped_title,
        wrapped_body,
    )


@tool(venue=ToolVenue.IN_PROCESS)
async def forge_create_issue(
    project: str,
    title: str,
    description: str = "",
    labels: list[str] | None = None,
    assignees: list[str] | None = None,
) -> str:
    """Create a new issue in a project.

    *project* is the name of the project service.  *labels* and
    *assignees* are optional lists of label names and usernames.
    """
    client, native_id = _resolve(project)
    info = await asyncio.to_thread(
        client.create_issue,
        native_id,
        title=title,
        description=description,
        labels=labels or [],
        assignees=assignees or [],
    )
    lines = [
        f"Created issue #{info['id']}: {info['title']}",
        f"State: {info['state']}",
        f"URL: {info['url']}",
    ]
    if info.get("labels"):
        lines.append(f"Labels: {', '.join(info['labels'])}")
    if info.get("assignees"):
        lines.append(f"Assignees: {', '.join(info['assignees'])}")
    return "\n".join(lines)


@tool(venue=ToolVenue.IN_PROCESS)
async def forge_list_issues(
    project: str,
    state: IssueState = "open",
    labels: list[str] | None = None,
) -> str:
    """List issues in a project.

    Filters by *state* (default ``"open"``).  Optionally filter by
    *labels* (list of label names -- issues must have **all** listed
    labels).
    """
    client, native_id, forge_name = _resolve_with_forge_name(project)
    issues = await asyncio.to_thread(
        client.list_issues, native_id, state, labels,
    )
    if not issues:
        return f"No {state} issues in project {project!r}."
    entries: list[str] = []
    wrapped_results: list[ToolResultContent] = []
    for issue in issues:
        title = _wrap_authored_title(
            info=issue,
            service=forge_name,
            kind="issue_title",
        )
        wrapped_results.append(title)
        entries.append(
            "\n".join([
                f"  #{issue['id']} ({issue['state']}, by {_display_author(issue)})",
                title,
            ]),
        )
    header = f"{len(issues)} {state} issue(s) in project {project!r}:"
    return _combine_tool_result_content(
        "\n\n".join([header, *entries]),
        *wrapped_results,
    )


@tool(venue=ToolVenue.IN_PROCESS)
async def forge_update_issue(
    project: str,
    issue_id: int,
    title: str | None = None,
    description: str | None = None,
    state: IssueState | None = None,
    add_labels: list[str] | None = None,
    remove_labels: list[str] | None = None,
    add_assignees: list[str] | None = None,
    remove_assignees: list[str] | None = None,
) -> str:
    """Update an existing issue.

    All fields are optional -- only the ones provided are changed.
    Labels and assignees support add/remove semantics: the tool reads
    the current values and merges the requested changes before sending
    the update.
    """
    client, native_id, forge_name = _resolve_with_forge_name(project)

    labels: list[str] | None = None
    assignees: list[str] | None = None
    if add_labels or remove_labels or add_assignees or remove_assignees:
        current = await asyncio.to_thread(
            client.get_issue, native_id, issue_id,
        )
        if add_labels or remove_labels:
            current_labels = set(current.get("labels", []))
            current_labels |= set(add_labels or [])
            current_labels -= set(remove_labels or [])
            labels = sorted(current_labels)
        if add_assignees or remove_assignees:
            current_assignees = set(current.get("assignees", []))
            current_assignees |= set(add_assignees or [])
            current_assignees -= set(remove_assignees or [])
            assignees = sorted(current_assignees)

    info = await asyncio.to_thread(
        client.update_issue,
        native_id, issue_id,
        title=title, description=description,
        state=state, labels=labels, assignees=assignees,
    )
    wrapped_title = _wrap_authored_title(
        info=info,
        service=forge_name,
        kind="issue_title",
    )
    lines = [
        f"Updated issue #{info['id']}",
        "Title:",
        wrapped_title,
        f"State: {info['state']}",
        f"URL: {info['url']}",
    ]
    if info.get("labels"):
        lines.append(f"Labels: {', '.join(info['labels'])}")
    if info.get("assignees"):
        lines.append(f"Assignees: {', '.join(info['assignees'])}")
    return _combine_tool_result_content(
        "\n".join(lines),
        wrapped_title,
    )


@tool(venue=ToolVenue.IN_PROCESS)
async def forge_post_comment(
    project: str,
    target_type: CommentTargetKind,
    target_id: int,
    body: str,
) -> str:
    """Post a comment on an issue or change request.

    *target_type* must be ``"Issue"`` or ``"ChangeRequest"``.
    *target_id* is the issue or change-request number.
    """
    client, native_id = _resolve(project)
    await asyncio.to_thread(
        client.post_comment, native_id, target_type, target_id, body,
    )
    return (
        f"Posted comment on {target_type} "
        f"#{target_id} in project {project!r}."
    )


@tool(venue=ToolVenue.IN_PROCESS)
async def forge_create_change_request(
    project: str,
    source_branch: str,
    title: str,
    description: str = "",
    target_branch: str = "main",
) -> str:
    """Create a new change request (merge request / pull request).

    Opens a change request from *source_branch* into *target_branch*
    (default ``main``) in the specified project.
    """
    client, native_id = _resolve(project)
    info = await asyncio.to_thread(
        client.create_change_request,
        native_id,
        source_branch=source_branch,
        title=title,
        target_branch=target_branch,
        description=description,
    )
    return (
        f"Created change request #{info['id']}: {info['title']}\n"
        f"  {info['source_branch']} -> {info['target_branch']}\n"
        f"  URL: {info['url']}"
    )


@tool(venue=ToolVenue.IN_PROCESS)
async def forge_get_change_request(project: str, cr_id: int) -> str:
    """Read details of a change request (merge request / pull request).

    Returns the title, state, branches, and description.  The title
    and description are wrapped in ``[external-content]`` envelopes;
    structural metadata (branches, state, URL) is left as plain prose.
    """
    client, native_id, forge_name = _resolve_with_forge_name(project)
    info = await asyncio.to_thread(
        client.get_change_request, native_id, cr_id,
    )
    wrapped_title = _wrap_authored_title(
        info=info,
        service=forge_name,
        kind="pr_title",
    )
    wrapped_body = _wrap_authored_body(
        body=info.get("description") or "",
        author=info.get("author"),
        fallback_login=None,
        service=forge_name,
        kind="pr_body",
        timestamp=info.get("created_at"),
    )
    lines = [
        f"Change request #{info['id']}",
        "Title:",
        wrapped_title,
        f"State: {info['state']}",
        f"Branches: {info['source_branch']} -> {info['target_branch']}",
        f"URL: {info['url']}",
        "",
        "Description:",
        wrapped_body,
    ]
    return _combine_tool_result_content(
        "\n".join(lines),
        wrapped_title,
        wrapped_body,
    )


@tool(venue=ToolVenue.IN_PROCESS)
async def forge_list_change_requests(
    project: str,
    state: ChangeRequestState = "open",
) -> str:
    """List change requests (merge requests / pull requests) in a project.

    Filters by *state* (default ``"open"``).  Valid states are
    ``"open"``, ``"closed"``, ``"merged"``, and ``"all"``.
    """
    client, native_id, forge_name = _resolve_with_forge_name(project)
    crs = await asyncio.to_thread(
        client.list_change_requests, native_id, state,
    )
    if not crs:
        return f"No {state} change requests in project {project!r}."
    entries: list[str] = []
    wrapped_results: list[ToolResultContent] = []
    for cr in crs:
        title = _wrap_authored_title(
            info=cr,
            service=forge_name,
            kind="pr_title",
        )
        wrapped_results.append(title)
        entries.append(
            "\n".join([
                f"  #{cr['id']} ({cr['state']}, by {_display_author(cr)})",
                title,
            ]),
        )
    header = f"{len(crs)} {state} change request(s) in project {project!r}:"
    return _combine_tool_result_content(
        "\n\n".join([header, *entries]),
        *wrapped_results,
    )


@tool(venue=ToolVenue.IN_PROCESS)
async def forge_list_comments(
    project: str,
    target_type: CommentTargetKind,
    target_id: int,
    include_system: bool = False,
) -> str:
    """List comments on an issue or change request.

    *target_type* must be ``"Issue"`` or ``"ChangeRequest"``.
    Set *include_system* to ``True`` to also show auto-generated
    comments (label changes, bot comments, etc.).

    Each comment body is wrapped in an ``[external-content]``
    envelope with the author's peer status (``yes`` / ``no`` /
    ``unknown``).  This is the consistency point for the
    data-vs-instruction rule: an agent that re-reads a thread via
    this tool sees comments labelled the same way it does when they
    arrive as notifications.
    """
    client, native_id, forge_name = _resolve_with_forge_name(project)
    comments = await asyncio.to_thread(
        client.list_comments, native_id, target_type, target_id,
    )
    if not include_system:
        comments = [c for c in comments if not c.get("is_system", False)]

    if not comments:
        kind = "issue" if target_type == "Issue" else "change request"
        return (
            f"No comments on {kind} #{target_id} in project {project!r}."
        )

    lines: list[str] = []
    wrapped_results: list[ToolResultContent] = []
    for c in comments:
        wrapped = _wrap_authored_body(
            body=c.get("body") or "",
            author=c.get("author_user"),
            fallback_login=c.get("author"),
            service=forge_name,
            kind="comment",
            timestamp=c.get("created_at"),
        )
        wrapped_results.append(wrapped)
        lines.append(wrapped)
        lines.append("")
    return _combine_tool_result_content(
        "\n".join(lines).rstrip() + "\n",
        *wrapped_results,
    )


@tool(venue=ToolVenue.IN_PROCESS)
async def forge_get_project_info(project: str) -> str:
    """Get information about a project.

    Returns the project's name, HTTPS clone URL, default branch, and
    description.
    """
    client, native_id = _resolve(project)
    info = await asyncio.to_thread(client.get_project_info, native_id)
    lines = [
        f"Project: {info['name']}",
        f"Path: {info['path']}",
        f"Clone URL (HTTPS): {info['clone_url']}",
        "Clone URL (SSH): unavailable in Thorn's default sandbox/broker "
        "profile; use HTTPS unless explicit operator policy says SSH is "
        "available.",
        f"Default branch: {info['default_branch']}",
        f"URL: {info['url']}",
    ]
    if info.get("description"):
        lines.append(f"Description: {info['description']}")
    return "\n".join(lines)


@tool(venue=ToolVenue.IN_PROCESS)
async def forge_read_file(
    project: str,
    file_path: str,
    ref: str = "HEAD",
) -> str:
    """Read a file from a project's repository via the forge API.

    Useful for inspecting files without cloning the entire repository.
    *ref* can be a branch name, tag, or commit SHA.
    """
    client, native_id = _resolve(project)
    info = await asyncio.to_thread(
        client.read_file, native_id, file_path, ref,
    )
    return f"--- {info['file_path']} (ref: {info['ref']}) ---\n{info['content']}"


@tool(venue=ToolVenue.IN_PROCESS)
async def forge_mark_notification_done(
    project: str,
    notification_id: str,
) -> str:
    """Mark a notification as done/read.

    *notification_id* is the forge-native notification identifier
    (numeric string for both GitLab TODOs and GitHub notification
    threads).
    """
    client, _native_id = _resolve(project)
    await asyncio.to_thread(
        client.mark_notification_done, notification_id,
    )
    return (
        f"Marked notification {notification_id} as done "
        f"(project {project!r})."
    )


FORGE_TOOLS: list[object] = [
    forge_read_issue,
    forge_create_issue,
    forge_list_issues,
    forge_update_issue,
    forge_post_comment,
    forge_create_change_request,
    forge_get_change_request,
    forge_list_change_requests,
    forge_list_comments,
    forge_get_project_info,
    forge_read_file,
    forge_mark_notification_done,
]
"""All forge-neutral tools as a list, suitable for ``tools=[FORGE_TOOLS, ...]``."""


__all__ = [
    "ChangeRequestState",
    "CommentTargetKind",
    "FORGE_TOOLS",
    "ForgeClient",
    "ForgeUserIdentity",
    "ForkConfig",
    "ForgeHostService",
    "GitHubAccountConfig",
    "GitHubForgeService",
    "GitLabAccountConfig",
    "GitLabForgeService",
    "GitHubForgeClient",
    "GitLabForgeClient",
    "GitLabForgeServiceConfig",
    "IssueState",
    "ProjectService",
    "ProjectServiceConfig",
    "forge_create_change_request",
    "forge_create_issue",
    "forge_get_change_request",
    "forge_get_project_info",
    "forge_list_change_requests",
    "forge_list_comments",
    "forge_list_issues",
    "forge_mark_notification_done",
    "forge_post_comment",
    "forge_read_file",
    "forge_read_issue",
    "forge_update_issue",
]
