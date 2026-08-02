"""thorn.core -- Agent primitives and core infrastructure.

This subpackage contains the foundational types, agent loop, provider
abstraction, history management, tool system, and all other primitives
that make up the core of Thorn.

The top-level ``thorn`` package keeps only the small script-oriented API.
Import broader core infrastructure explicitly from ``thorn.core``.
"""

from __future__ import annotations

from thorn.core._agent import Agent
from thorn.core._context import (
    ConsoleEventSink,
    EventSink,
    ExecutionContext,
    NullEventSink,
    Scope,
    StatusProvider,
    UsageTracker,
    Verbosity,
    get_context,
    reset_context,
    resolve_path,
    scoped_status_provider,
    set_context,
)
from thorn.core._context_injection import (
    DirectorySeed,
    FileSeed,
    SearchSeed,
    SeedContent,
)
from thorn.core._discovery import discover_tools
from thorn.core._event_bus import (
    EventBus,
    ScopeFilter,
    Subscription,
    accept_all,
    in_session,
)
from thorn.core._executor import (
    ExecutorRouter,
    InProcessToolExecutor,
    OnChunkCallback,
    ToolExecutor,
    ToolInvocation,
    ToolInvocationResult,
    ToolOutputChunk,
    ToolRegistry,
    ToolRegistryEntry,
    ToolVenue,
)
from thorn.core._file_access import (
    FileAccessLevel,
    FileAccessPolicy,
    FileAccessRule,
    RelativeTo,
)
from thorn.core._func import prompt, skill, tool, wrap_function
from thorn.core._history import (
    AdvisoryNode,
    ArchiveMarkerNode,
    CollapseState,
    CompactionResult,
    HistoryNode,
    HistoryTree,
    HousekeepingNode,
)
from thorn.core._housekeeping import (
    HousekeepingResult,
    perform_housekeeping,
    select_cut_point,
)
from thorn.core._journal import (
    JOURNAL_TOOLS,
    append_journal_entry,
    list_journal_dates,
    read_journal,
    read_journal_day,
    read_recent_journal,
    write_journal,
)
from thorn.core._loop import _WrappedTool, run_agent_loop
from thorn.core._mcp_config import MCPServerConfig
from thorn.core._module import ModulePath
from thorn.core._prompt_trace import (
    PromptTraceArtifact,
    PromptTraceCapture,
    PromptTraceContextSource,
    PromptTraceManifest,
    PromptTraceRecorder,
)
from thorn.core._provider import (
    LLMProvider,
    MockProvider,
    UsageChunk,
    load_provider_from_env,
)
from thorn.core._provider_telemetry import (
    ProviderAttemptNextAction,
    ProviderAttemptOutcome,
    ProviderAttemptTelemetry,
    ProviderContextMetrics,
)
from thorn.core._retry import RetryPolicy, bound_retries
from thorn.core._session import Session
from thorn.core._tools import (
    ALL_BUILTIN_TOOLS,
    FILE_READING,
    FILE_WRITING,
    FileEdit,
    create_file,
    delete_file,
    edit_file,
    find_files,
    list_directory,
    move_file,
    read_file,
    search_files,
)
from thorn.core._trace import CompositeEventSink, JsonLinesSink
from thorn.core._validation import ValidationRule
from thorn.core._validation_tracker import ValidationStatus, ValidationTracker
from thorn.core.errors import (
    AgentFailureError,
    LoopLimitError,
    ProviderError,
    ProviderFailureKind,
    ProviderUnavailableError,
    RateLimitError,
    SkillError,
    ThornError,
    TransientProviderError,
)

try:
    # ``serve_tools`` lives behind the optional ``mcp`` extra
    # because it imports ``mcp.server.fastmcp`` lazily; an
    # operator that does not install ``thorn-agent[mcp]`` simply does
    # not get the symbol.  ``MCPServerConfig`` itself has no
    # optional dependency and is always importable.
    from thorn.core._mcp import serve_tools
except ImportError:
    pass

__all__ = [
    # Agent / Session
    "Agent",
    "Session",
    # Context
    "ConsoleEventSink",
    "EventSink",
    "ExecutionContext",
    "NullEventSink",
    "Scope",
    "StatusProvider",
    "UsageTracker",
    "Verbosity",
    "get_context",
    "reset_context",
    "resolve_path",
    "scoped_status_provider",
    "set_context",
    # Context injection
    "DirectorySeed",
    "FileSeed",
    "SearchSeed",
    "SeedContent",
    # Discovery
    "discover_tools",
    # History
    "AdvisoryNode",
    "ArchiveMarkerNode",
    "CollapseState",
    "CompactionResult",
    "HistoryNode",
    "HistoryTree",
    "HousekeepingNode",
    # Module
    "ModulePath",
    # MCP
    "MCPServerConfig",
    "serve_tools",
    # Validation
    "ValidationRule",
    "ValidationStatus",
    "ValidationTracker",
    # File access
    "FileAccessLevel",
    "FileAccessPolicy",
    "FileAccessRule",
    "RelativeTo",
    # Function decorators
    "prompt",
    "skill",
    "tool",
    "wrap_function",
    # Retry
    "RetryPolicy",
    "bound_retries",
    # Loop
    "_WrappedTool",
    "run_agent_loop",
    # Tool execution seam
    "ExecutorRouter",
    "InProcessToolExecutor",
    "OnChunkCallback",
    "ToolExecutor",
    "ToolInvocation",
    "ToolInvocationResult",
    "ToolOutputChunk",
    "ToolRegistry",
    "ToolRegistryEntry",
    "ToolVenue",
    # Provider
    "LLMProvider",
    "MockProvider",
    "ProviderAttemptNextAction",
    "ProviderAttemptOutcome",
    "ProviderAttemptTelemetry",
    "ProviderContextMetrics",
    "PromptTraceArtifact",
    "PromptTraceCapture",
    "PromptTraceContextSource",
    "PromptTraceManifest",
    "PromptTraceRecorder",
    "UsageChunk",
    "load_provider_from_env",
    # Housekeeping
    "HousekeepingResult",
    "perform_housekeeping",
    "select_cut_point",
    # Journal
    "JOURNAL_TOOLS",
    "append_journal_entry",
    "list_journal_dates",
    "read_journal",
    "read_journal_day",
    "read_recent_journal",
    "write_journal",
    # Built-in tools
    "ALL_BUILTIN_TOOLS",
    "FILE_READING",
    "FILE_WRITING",
    "FileEdit",
    "create_file",
    "delete_file",
    "edit_file",
    "find_files",
    "list_directory",
    "move_file",
    "read_file",
    "search_files",
    # Errors
    "AgentFailureError",
    "LoopLimitError",
    "ProviderFailureKind",
    "ProviderError",
    "ProviderUnavailableError",
    "RateLimitError",
    "SkillError",
    "ThornError",
    "TransientProviderError",
    # Tracing
    "CompositeEventSink",
    "JsonLinesSink",
    # Event bus
    "EventBus",
    "ScopeFilter",
    "Subscription",
    "accept_all",
    "in_session",
]
