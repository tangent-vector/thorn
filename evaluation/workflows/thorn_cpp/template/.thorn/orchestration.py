"""Coordinator-driven hierarchical delegation with validation enforcement.

Provides the orchestration machinery for the development workflow:

- Per-role validation rules (declared on Agent subclasses via
  ``validation_rules``) with explicit enable/disable overrides
  propagated through the delegation chain via ContextVar
- Delegation tools for coordinators (delegate_to_role, delegate_to_child)
- A top-level ``coordinate`` tool discoverable by the concierge

This module deliberately avoids importing role classes from ``roles.py``.
It resolves roles at runtime through ``_delegatable_roles`` (populated by
``roles.py`` via ``register_role``) and the ``Agent`` registry.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
from typing import Any, Callable

from thorn import Agent, get_context, tool
from thorn.errors import SkillError

from .build_tools import build, run_tests
from .module_tools import list_submodules, qualify

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Validation system
# ---------------------------------------------------------------------------

_validation_enabled: contextvars.ContextVar[frozenset[str]] = (
    contextvars.ContextVar("validation_enabled", default=frozenset())
)

_validation_disabled: contextvars.ContextVar[frozenset[str]] = (
    contextvars.ContextVar("validation_disabled", default=frozenset())
)

ValidationCheck = Callable[..., Any]

VALIDATION_CHECKS: dict[str, ValidationCheck] = {
    "build": build,
    "test": run_tests,
}

MAX_VALIDATION_RETRIES: int = 3


def effective_validation_rules(agent: Agent) -> frozenset[str]:
    """Compute the effective validation rules for *agent*.

    Combines the agent's class-level ``validation_rules`` (accumulated
    through the MRO) with explicit overrides from the delegation context:

        effective = (role_defaults | ctx_enabled) - ctx_disabled
    """
    role_rules = frozenset(type(agent)._collect_validation_rules())
    enabled = _validation_enabled.get(frozenset())
    disabled = _validation_disabled.get(frozenset())
    return (role_rules | enabled) - disabled


async def _run_validation(rules: frozenset[str]) -> list[tuple[str, str]]:
    """Run the specified validation checks and return failures.

    Each failure is a ``(rule_name, error_detail)`` pair.  An empty list
    means all checks passed.
    """
    failures: list[tuple[str, str]] = []
    for name in sorted(rules):
        check_fn = VALIDATION_CHECKS.get(name)
        if check_fn is None:
            continue
        try:
            if asyncio.iscoroutinefunction(check_fn):
                result = await check_fn()
            else:
                result = check_fn()
            if isinstance(result, str) and "FAILED" in result:
                failures.append((name, result))
        except Exception as exc:
            failures.append((name, str(exc)))
    return failures


# ---------------------------------------------------------------------------
# Delegatable role registry (populated by roles.py at import time)
# ---------------------------------------------------------------------------

_delegatable_roles: dict[str, type[Agent]] = {}


def register_role(name: str, cls: type[Agent]) -> None:
    """Register a role class as available for coordinator delegation.

    Called by ``roles.py`` after each concrete workflow role is defined.
    """
    _delegatable_roles[name] = cls


def available_role_names() -> list[str]:
    """Return sorted list of registered delegatable role names."""
    return sorted(_delegatable_roles)


# ---------------------------------------------------------------------------
# Validation retry loop
# ---------------------------------------------------------------------------


async def _run_with_validation(
    agent: Agent,
    task: str,
    max_retries: int = MAX_VALIDATION_RETRIES,
) -> str:
    """Run an agent on a task, then validate and retry on failure.

    Validation rules are determined by the agent's class-level
    ``validation_rules`` combined with any overrides propagated through
    the delegation context.  If there are no effective rules, the agent
    runs without validation.

    The agent keeps its conversation history across retries so it can
    see what it did previously and what went wrong.
    """
    messages: list = []
    rules = effective_validation_rules(agent)

    summary = await agent.prompt(task, messages=messages)

    if not rules:
        return summary

    for retry in range(max_retries + 1):
        failures = await _run_validation(rules)
        if not failures:
            return summary

        if retry == max_retries:
            error_report = "\n".join(
                f"- {name}: {detail}" for name, detail in failures
            )
            raise SkillError(
                f"Validation still failing after {max_retries} retries:\n"
                f"{error_report}"
            )

        error_report = "\n".join(
            f"- {name}: {detail}" for name, detail in failures
        )
        logger.info(
            "Validation failed for %s (attempt %d/%d): %s",
            type(agent).__name__, retry + 1, max_retries,
            ", ".join(name for name, _ in failures),
        )
        summary = await agent.prompt(
            f"Validation failed after your changes:\n{error_report}\n\n"
            "Please fix the issues and try again.",
            messages=messages,
        )

    return summary  # pragma: no cover


# ---------------------------------------------------------------------------
# Delegation helpers
# ---------------------------------------------------------------------------


def _push_overrides(
    skip: list[str] | None,
    enable: list[str] | None,
) -> tuple[contextvars.Token[frozenset[str]], contextvars.Token[frozenset[str]]]:
    """Accumulate validation overrides and return reset tokens."""
    parent_enabled = _validation_enabled.get(frozenset())
    parent_disabled = _validation_disabled.get(frozenset())

    child_enabled = parent_enabled | frozenset(enable or [])
    child_disabled = parent_disabled | frozenset(skip or [])

    en_token = _validation_enabled.set(child_enabled)
    dis_token = _validation_disabled.set(child_disabled)
    return en_token, dis_token


def _pop_overrides(
    en_token: contextvars.Token[frozenset[str]],
    dis_token: contextvars.Token[frozenset[str]],
) -> None:
    """Restore previous validation overrides."""
    _validation_disabled.reset(dis_token)
    _validation_enabled.reset(en_token)


# ---------------------------------------------------------------------------
# Delegation tools (listed in Coordinator.tools, NOT @tool-decorated)
# ---------------------------------------------------------------------------


async def delegate_to_role(
    role: str,
    task: str,
    skip_validation: list[str] | None = None,
    enable_validation: list[str] | None = None,
) -> str:
    """Delegate a task to a development role at the current module.

    Creates an agent for the specified role, scoped to the coordinator's
    own module, and runs it with validation enforcement.  The sub-agent
    receives the task description and returns a summary of what it did.

    If the sub-agent cannot complete the task, the error is reported back
    to you so you can decide how to proceed.

    Args:
        role: Role name (e.g. "architect", "api_designer",
              "test_engineer", "implementer").
        task: Free-form description of what the role should do.
        skip_validation: Validation rules to disable for this delegation.
        enable_validation: Additional validation rules to enable.
    """
    ctx = get_context()
    module = ctx.agent.module

    role_cls = _delegatable_roles.get(role)
    if role_cls is None:
        available = ", ".join(available_role_names())
        raise ValueError(
            f"Unknown role: {role!r}. Available roles: {available}"
        )

    en_token, dis_token = _push_overrides(skip_validation, enable_validation)
    try:
        agent = role_cls(module=module)
        return await _run_with_validation(agent, task)
    except SkillError as exc:
        raise RuntimeError(
            f"{role}@{module} raised an error: {exc.detail}"
        ) from None
    finally:
        _pop_overrides(en_token, dis_token)


async def delegate_to_child(
    child: str,
    task: str,
    skip_validation: list[str] | None = None,
    enable_validation: list[str] | None = None,
) -> str:
    """Delegate a task to the coordinator of a child module.

    Creates a coordinator agent for the specified child module and runs
    it with validation enforcement.  The child must be a direct submodule
    of the current module (use ``list_submodules`` to discover children).

    If the child coordinator cannot complete the task, the error is
    reported back to you so you can decide how to proceed.

    Args:
        child: Unqualified name of the child module.
        task: Free-form description of what should be done.
        skip_validation: Validation rules to disable for this delegation.
        enable_validation: Additional validation rules to enable.
    """
    ctx = get_context()
    module = ctx.agent.module

    children = list_submodules(module)
    if child not in children:
        available = ", ".join(children) if children else "(none)"
        raise ValueError(
            f"{child!r} is not a submodule of {module!r}. "
            f"Available children: {available}"
        )

    qualified_child = qualify(module, child)

    coordinator_cls = Agent._registry.get("Coordinator")
    if coordinator_cls is None:
        raise RuntimeError("No Coordinator class registered")

    en_token, dis_token = _push_overrides(skip_validation, enable_validation)
    try:
        agent = coordinator_cls(module=qualified_child)
        return await _run_with_validation(agent, task)
    except SkillError as exc:
        raise RuntimeError(
            f"coordinator@{qualified_child} raised an error: {exc.detail}"
        ) from None
    finally:
        _pop_overrides(en_token, dis_token)


# ---------------------------------------------------------------------------
# Top-level entry point (discoverable by concierge via @tool)
# ---------------------------------------------------------------------------


@tool
async def coordinate(
    task: str,
    module: str = "main",
    skip_validation: list[str] | None = None,
    enable_validation: list[str] | None = None,
) -> str:
    """Coordinate a development task across the project's module hierarchy.

    This is the PRIMARY tool for any development work on the project.
    It creates a coordinator agent that autonomously decomposes the task,
    delegating to specialized roles (architect, API designer, test
    engineer, implementer) and child module coordinators as needed.

    USE THIS for: implementing features, fixing bugs, refactoring,
    designing APIs, adding tests, or any task that modifies source code.

    Do NOT attempt to modify source files directly -- always delegate
    development work through this tool.

    Args:
        task: Description of the development task.
        module: Root module to coordinate from (default: "main").
        skip_validation: Validation rules to skip.
        enable_validation: Additional validation rules to enable.
    """
    coordinator_cls = Agent._registry.get("Coordinator")
    if coordinator_cls is None:
        raise RuntimeError(
            "No Coordinator class registered.  "
            "Ensure roles.py defines a Coordinator(Developer) class."
        )

    en_token, dis_token = _push_overrides(skip_validation, enable_validation)
    try:
        agent = coordinator_cls(module=module)
        return await _run_with_validation(agent, task)
    except SkillError as exc:
        raise RuntimeError(
            f"coordinator@{module} could not complete the task: {exc.detail}"
        ) from None
    finally:
        _pop_overrides(en_token, dis_token)
