"""Hierarchical delegation with validation enforcement.

Provides the orchestration machinery for the development workflow:

- Per-module validation rules (declared on Agent subclasses via
  ``validation_rules``) with explicit enable/disable overrides
  propagated through the delegation chain via ContextVar
- A ``delegate_to_child`` tool for developers to delegate sub-module work
- A top-level ``coordinate`` tool discoverable by the concierge

This module deliberately avoids importing role classes from ``roles.py``.
It resolves the ``ModuleDeveloper`` class at runtime through the ``Agent``
registry.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
from typing import TYPE_CHECKING, Any, Callable

from thorn import Agent, get_context, tool

if TYPE_CHECKING:
    from thorn.core._context_injection import SeedContent
from thorn.core._validation_tracker import ValidationTracker
from thorn.errors import SkillError

from .build_tools import PROJECT_DIR, build, run_tests
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
# Validation retry loop
# ---------------------------------------------------------------------------


async def _run_with_validation(
    agent: Agent,
    task: str,
    max_retries: int = MAX_VALIDATION_RETRIES,
    recommended_context: list[SeedContent] | None = None,
) -> str:
    """Run an agent on a task, then validate and retry on failure.

    Validation rules are determined by the agent's class-level
    ``validation_rules`` combined with any overrides propagated through
    the delegation context.  If there are no effective rules, the agent
    runs without validation.

    The agent keeps its conversation history across retries so it can
    see what it did previously and what went wrong.

    When validation passes on the first attempt the original summary is
    returned as-is.  If one or more retry rounds were needed, the agent
    is asked for a fresh summary so the caller receives a coherent
    description of the completed work rather than a fix-oriented response.
    """
    rules = effective_validation_rules(agent)

    summary = await agent.prompt(
        task, recommended_context=recommended_context,
    )

    if not rules:
        return summary

    for retry in range(max_retries + 1):
        failures = await _run_validation(rules)
        if not failures:
            if retry == 0:
                return summary
            return await agent.prompt(
                "Validation now passes. Please provide a brief summary of "
                "the work you completed (covering the original task, not "
                "just the fixes).",
            )

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


def _get_developer_cls() -> type[Agent]:
    """Resolve the ModuleDeveloper class from the Agent registry.

    Raises RuntimeError if the class hasn't been registered (i.e.
    roles.py hasn't been loaded yet).
    """
    cls = Agent._registry.get("ModuleDeveloper")
    if cls is None:
        raise RuntimeError(
            "No ModuleDeveloper class registered.  "
            "Ensure roles.py defines a ModuleDeveloper(Developer) class."
        )
    return cls


# ---------------------------------------------------------------------------
# Delegation tool (listed in ModuleDeveloper.tools, NOT @tool-decorated)
# ---------------------------------------------------------------------------


async def delegate_to_child(
    child: str,
    task: str,
    recommended_files: list[str] | None = None,
    skip_validation: list[str] | None = None,
    enable_validation: list[str] | None = None,
) -> str:
    """Delegate a task to the developer of a child module.

    Creates a developer agent for the specified child module and runs it
    with validation enforcement.  The child must be a direct submodule
    of the current module (use ``list_submodules`` to discover children).

    If the child developer cannot complete the task, the error is
    reported back to you so you can decide how to proceed.

    Args:
        child: Unqualified name of the child module.
        task: Free-form description of what should be done.
        recommended_files: File paths the child should read for context,
            in priority order.  These supplement the child's own
            structurally-declared seeds.
        skip_validation: Validation rules to disable for this delegation.
        enable_validation: Additional validation rules to enable.
    """
    from thorn.core._context_injection import FileSeed

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
    developer_cls = _get_developer_cls()

    recommended_context = (
        [FileSeed(path=p) for p in recommended_files]
        if recommended_files
        else None
    )

    en_token, dis_token = _push_overrides(skip_validation, enable_validation)
    try:
        agent = developer_cls(module=qualified_child)
        return await _run_with_validation(
            agent, task, recommended_context=recommended_context,
        )
    except SkillError as exc:
        raise RuntimeError(
            f"developer@{qualified_child} raised an error: {exc.detail}"
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

    This is the PRIMARY entry point for ALL development work on the project.
    It creates a developer agent that autonomously handles the task --
    designing APIs, writing implementations, creating tests, and delegating
    sub-module work to child developers as needed.

    USE THIS for: implementing features, fixing bugs, refactoring,
    designing APIs, adding tests, or any task that modifies source code.
    Do NOT attempt to modify source files directly -- always delegate
    development work through this tool.

    IMPORTANT: The task description should convey the user's goal, not a
    detailed plan. The developer will read the codebase and determine the
    best approach. Elaborating beyond the user's original request wastes
    tokens and may constrain the developer from making optimal decisions.

    Args:
        task: Brief description of what should be achieved — state the
              goal, not the implementation approach.
        module: Root module to start from (default: "main").
        skip_validation: Validation rules to skip.
        enable_validation: Additional validation rules to enable.
    """
    ctx = get_context()
    if ctx.validation_tracker is None:
        tracker = ValidationTracker(root=PROJECT_DIR)
        tracker.add_target("build", file_patterns=[
            "src/**/*.h", "src/**/*.cpp", "CMakeLists.txt",
        ])
        tracker.add_target("test", file_patterns=[
            "src/**/*.h", "src/**/*.cpp", "tests/**/*.cpp",
        ], depends_on=["build"])
        ctx.validation_tracker = tracker

    developer_cls = _get_developer_cls()

    en_token, dis_token = _push_overrides(skip_validation, enable_validation)
    try:
        agent = developer_cls(module=module)
        return await _run_with_validation(agent, task)
    except SkillError as exc:
        raise RuntimeError(
            f"developer@{module} could not complete the task: {exc.detail}"
        ) from None
    finally:
        _pop_overrides(en_token, dis_token)
