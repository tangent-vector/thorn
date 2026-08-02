"""Message types for agent conversation histories."""

from __future__ import annotations

from dataclasses import dataclass, field

from thorn.core._external_content import ExternalContentPeerStatus


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
    """The result of executing a tool call, fed back to the assistant.

    ``external_content_peer_statuses`` is trusted runtime metadata for
    Thorn-rendered external-content envelopes in ``content``.  Raw tool
    output that merely contains marker-shaped text should not populate it.
    """

    call_id: str
    content: str
    is_error: bool = False
    external_content_peer_statuses: frozenset[ExternalContentPeerStatus] = (
        field(default_factory=frozenset)
    )
    role: str = field(default="tool", init=False)
