"""Message types for agent conversation histories."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ToolCall:
    """A tool invocation requested by the assistant.

    Attributes:
        call_id:   Provider-assigned identifier used to correlate results.
        name:      The tool name the assistant wants to invoke.
        arguments: JSON-encoded string of the arguments.
    """

    call_id: str
    name: str
    arguments: str


@dataclass
class Message:
    """Base class for all messages in a conversation history."""

    role: str


@dataclass
class UserMessage(Message):
    """A message from the user or from orchestration code."""

    content: str
    role: str = field(default="user", init=False)


@dataclass
class AssistantMessage(Message):
    """A message produced by the LLM."""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    role: str = field(default="assistant", init=False)


@dataclass
class ToolResultMessage(Message):
    """The result of executing a tool call, fed back to the assistant."""

    call_id: str
    content: str
    is_error: bool = False
    role: str = field(default="tool", init=False)
