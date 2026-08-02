"""Validated GitHub notification identifiers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Self

_GITHUB_NOTIFICATION_THREAD_ID_RE = re.compile(r"[0-9]+")


def _validate_github_notification_thread_id(raw_value: str) -> str:
    if not isinstance(raw_value, str):
        raise TypeError(
            "GitHub notification thread ID must be a string of ASCII digits"
        )
    if not _GITHUB_NOTIFICATION_THREAD_ID_RE.fullmatch(raw_value):
        raise ValueError(
            "GitHub notification thread ID must contain only ASCII digits"
        )
    return raw_value


@dataclass(frozen=True)
class GitHubNotificationThreadID:
    """GitHub notification thread ID safe to embed in REST paths."""

    value: str

    def __post_init__(self) -> None:
        _validate_github_notification_thread_id(self.value)

    @classmethod
    def parse(cls, raw_value: str) -> Self:
        return cls(_validate_github_notification_thread_id(raw_value))

    def __str__(self) -> str:
        return self.value


__all__ = ["GitHubNotificationThreadID"]
