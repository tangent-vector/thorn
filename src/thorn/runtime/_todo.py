"""Session-scoped TODO lists for agent work tracking."""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import yaml

from thorn.runtime._notification import NotificationID


class TodoID(str):
    """Framework-assigned identifier for a session TODO item."""


class TodoStatus(str, Enum):
    """Lifecycle status of a session TODO item."""

    OPEN = "open"
    COMPLETED = "completed"
    ABANDONED = "abandoned"

    @property
    def is_resolved(self) -> bool:
        """Whether this status no longer blocks linked inbox completion."""
        return self in {TodoStatus.COMPLETED, TodoStatus.ABANDONED}


@dataclass(frozen=True)
class TodoItem:
    """One durable session TODO item."""

    id: TodoID
    title: str
    status: TodoStatus
    linked_inbox_item_ids: tuple[NotificationID, ...]
    notes: str | None
    resolution_rationale: str | None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def create(
        cls,
        *,
        id: TodoID,
        title: str,
        linked_inbox_item_id: NotificationID | None,
        notes: str | None,
        now: datetime,
    ) -> TodoItem:
        """Construct a new open TODO item."""
        return cls(
            id=id,
            title=_validated_title(title),
            status=TodoStatus.OPEN,
            linked_inbox_item_ids=(
                (linked_inbox_item_id,)
                if linked_inbox_item_id is not None else ()
            ),
            notes=_optional_text(notes),
            resolution_rationale=None,
            created_at=now,
            updated_at=now,
        )

    def with_updates(
        self,
        *,
        title: str | None = None,
        notes: str | None = None,
        linked_inbox_item_ids: tuple[NotificationID, ...] | None = None,
        now: datetime,
    ) -> TodoItem:
        """Return a copy with editable fields changed."""
        updates: dict[str, Any] = {"updated_at": now}
        if title is not None:
            updates["title"] = _validated_title(title)
        if notes is not None:
            updates["notes"] = _optional_text(notes)
        if linked_inbox_item_ids is not None:
            updates["linked_inbox_item_ids"] = linked_inbox_item_ids
        return replace(self, **updates)

    def resolve(
        self,
        *,
        status: TodoStatus,
        rationale: str,
        now: datetime,
    ) -> TodoItem:
        """Return a completed or abandoned copy with rationale recorded."""
        if not status.is_resolved:
            raise ValueError(f"Cannot resolve a TODO as {status.value!r}")
        cleaned_rationale = _required_text(rationale, "rationale")
        return replace(
            self,
            status=status,
            resolution_rationale=cleaned_rationale,
            updated_at=now,
        )

    def to_data(self) -> dict[str, Any]:
        """Return a YAML-serializable representation."""
        return {
            "id": str(self.id),
            "title": self.title,
            "status": self.status.value,
            "linked_inbox_item_ids": [
                str(item_id) for item_id in self.linked_inbox_item_ids
            ],
            "notes": self.notes,
            "resolution_rationale": self.resolution_rationale,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_data(cls, data: Mapping[str, Any]) -> TodoItem:
        """Reconstruct a TODO from front-matter data."""
        return cls(
            id=TodoID(str(data["id"])),
            title=_validated_title(str(data["title"])),
            status=TodoStatus(str(data["status"])),
            linked_inbox_item_ids=tuple(
                NotificationID(str(item_id))
                for item_id in (data.get("linked_inbox_item_ids") or ())
            ),
            notes=_optional_text(data.get("notes")),
            resolution_rationale=_optional_text(
                data.get("resolution_rationale"),
            ),
            created_at=_datetime_from_data(data["created_at"]),
            updated_at=_datetime_from_data(data["updated_at"]),
        )


@dataclass(frozen=True)
class LinkedTodoSummary:
    """Compact count/title summary for TODOs linked to an inbox item."""

    open_count: int = 0
    resolved_count: int = 0
    open_titles: tuple[str, ...] = ()

    @property
    def total_count(self) -> int:
        """Total number of linked TODOs represented by this summary."""
        return self.open_count + self.resolved_count

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "open_count": self.open_count,
            "resolved_count": self.resolved_count,
            "total_count": self.total_count,
            "open_titles": list(self.open_titles),
        }


class SessionTodoList:
    """File-backed TODO list for one session."""

    def __init__(self, todo_file: Path) -> None:
        self._todo_file = Path(todo_file)

    @property
    def todo_file(self) -> Path:
        """Path to the readable Markdown TODO file."""
        return self._todo_file

    def list_items(self) -> list[TodoItem]:
        """Return every TODO in file order."""
        return list(_load_items(self._todo_file))

    def create(
        self,
        *,
        title: str,
        linked_inbox_item_id: NotificationID | None = None,
        notes: str | None = None,
    ) -> TodoItem:
        """Create and persist a new open TODO item."""
        items = self.list_items()
        now = _now()
        item = TodoItem.create(
            id=_generate_todo_id({item.id for item in items}),
            title=title,
            linked_inbox_item_id=linked_inbox_item_id,
            notes=notes,
            now=now,
        )
        items.append(item)
        _write_items(self._todo_file, items)
        return item

    def update(
        self,
        todo_id: str,
        *,
        title: str | None = None,
        notes: str | None = None,
        linked_inbox_item_ids: tuple[NotificationID, ...] | None = None,
    ) -> TodoItem:
        """Update editable fields on an existing TODO."""
        items = self.list_items()
        index = _index_for(items, todo_id)
        updated = items[index].with_updates(
            title=title,
            notes=notes,
            linked_inbox_item_ids=linked_inbox_item_ids,
            now=_now(),
        )
        items[index] = updated
        _write_items(self._todo_file, items)
        return updated

    def complete(self, todo_id: str, *, rationale: str) -> TodoItem:
        """Mark a TODO completed with an explicit rationale."""
        return self._resolve(
            todo_id,
            status=TodoStatus.COMPLETED,
            rationale=rationale,
        )

    def abandon(self, todo_id: str, *, rationale: str) -> TodoItem:
        """Mark a TODO abandoned with an explicit rationale."""
        return self._resolve(
            todo_id,
            status=TodoStatus.ABANDONED,
            rationale=rationale,
        )

    def unresolved_linked_to(
        self,
        inbox_item_id: NotificationID | str,
    ) -> list[TodoItem]:
        """Return open TODOs linked to *inbox_item_id*."""
        notification_id = NotificationID(str(inbox_item_id))
        return [
            item
            for item in self.list_items()
            if item.status is TodoStatus.OPEN
            and notification_id in item.linked_inbox_item_ids
        ]

    def linked_summary_by_inbox_item(
        self,
    ) -> dict[NotificationID, LinkedTodoSummary]:
        """Return TODO summary counts keyed by linked inbox item ID."""
        builders: dict[NotificationID, _LinkedTodoSummaryBuilder] = {}
        for item in self.list_items():
            for inbox_item_id in item.linked_inbox_item_ids:
                builder = builders.setdefault(
                    inbox_item_id,
                    _LinkedTodoSummaryBuilder(),
                )
                builder.add(item)
        return {
            inbox_item_id: builder.build()
            for inbox_item_id, builder in builders.items()
        }

    def _resolve(
        self,
        todo_id: str,
        *,
        status: TodoStatus,
        rationale: str,
    ) -> TodoItem:
        items = self.list_items()
        index = _index_for(items, todo_id)
        updated = items[index].resolve(
            status=status,
            rationale=rationale,
            now=_now(),
        )
        items[index] = updated
        _write_items(self._todo_file, items)
        return updated


@dataclass
class _LinkedTodoSummaryBuilder:
    open_count: int = 0
    resolved_count: int = 0
    open_titles: list[str] | None = None

    def add(self, item: TodoItem) -> None:
        if item.status is TodoStatus.OPEN:
            self.open_count += 1
            if self.open_titles is None:
                self.open_titles = []
            self.open_titles.append(item.title)
            return
        if item.status.is_resolved:
            self.resolved_count += 1

    def build(self) -> LinkedTodoSummary:
        return LinkedTodoSummary(
            open_count=self.open_count,
            resolved_count=self.resolved_count,
            open_titles=tuple(self.open_titles or ()),
        )


_FRONT_MATTER_DELIMITER = "---"
_TODO_FILE_VERSION = 1


def _load_items(path: Path) -> tuple[TodoItem, ...]:
    if not path.exists():
        return ()
    text = path.read_text(encoding="utf-8")
    data = _load_front_matter(text)
    raw_items = data.get("items") or ()
    if not isinstance(raw_items, list):
        raise ValueError("TODO front matter field 'items' must be a list")
    try:
        return tuple(TodoItem.from_data(item) for item in raw_items)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"TODO front matter contains an invalid item: {exc}"
        ) from exc


def _load_front_matter(text: str) -> dict[str, Any]:
    if not text.startswith(f"{_FRONT_MATTER_DELIMITER}\n"):
        raise ValueError("TODO.md must start with YAML front matter")
    marker = f"\n{_FRONT_MATTER_DELIMITER}\n"
    end = text.find(marker, len(_FRONT_MATTER_DELIMITER) + 1)
    if end < 0:
        raise ValueError("TODO.md front matter is missing its closing delimiter")
    raw = text[len(_FRONT_MATTER_DELIMITER) + 1:end]
    loaded = yaml.safe_load(raw) or {}
    if not isinstance(loaded, dict):
        raise ValueError("TODO.md front matter must be a mapping")
    return dict(loaded)


def _write_items(path: Path, items: list[TodoItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": _TODO_FILE_VERSION,
        "items": [item.to_data() for item in items],
    }
    front_matter = yaml.safe_dump(
        payload,
        allow_unicode=False,
        sort_keys=False,
    )
    text = (
        f"{_FRONT_MATTER_DELIMITER}\n"
        f"{front_matter}"
        f"{_FRONT_MATTER_DELIMITER}\n\n"
        f"{_render_markdown_body(items)}"
    )
    temp_path = path.with_name(f".tmp-{path.name}")
    temp_path.write_text(text, encoding="utf-8")
    os.replace(temp_path, path)


def _render_markdown_body(items: list[TodoItem]) -> str:
    lines = [
        "# Session TODOs",
        "",
        "Edit the YAML front matter above if the structured tools are too narrow.",
        "The summary below is regenerated by Thorn.",
        "",
    ]
    _append_section(
        lines,
        "Open",
        [item for item in items if item.status is TodoStatus.OPEN],
    )
    _append_section(
        lines,
        "Completed",
        [item for item in items if item.status is TodoStatus.COMPLETED],
    )
    _append_section(
        lines,
        "Abandoned",
        [item for item in items if item.status is TodoStatus.ABANDONED],
    )
    return "\n".join(lines).rstrip() + "\n"


def _append_section(
    lines: list[str],
    heading: str,
    items: list[TodoItem],
) -> None:
    lines.extend([f"## {heading}", ""])
    if not items:
        lines.extend(["- None", ""])
        return
    for item in items:
        marker = " " if item.status is TodoStatus.OPEN else "x"
        lines.append(f"- [{marker}] `{item.id}` {_one_line(item.title)}")
        if item.linked_inbox_item_ids:
            linked = ", ".join(
                f"`{item_id}`" for item_id in item.linked_inbox_item_ids
            )
            lines.append(f"  - Linked inbox: {linked}")
        if item.notes:
            lines.append(f"  - Notes: {_one_line(item.notes)}")
        if item.resolution_rationale:
            lines.append(
                f"  - Rationale: {_one_line(item.resolution_rationale)}"
            )
    lines.append("")


def _index_for(items: list[TodoItem], todo_id: str) -> int:
    for index, item in enumerate(items):
        if item.id == todo_id:
            return index
    raise KeyError(todo_id)


def _generate_todo_id(existing_ids: set[TodoID]) -> TodoID:
    while True:
        candidate = TodoID(f"todo-{secrets.token_urlsafe(6)}")
        if candidate not in existing_ids:
            return candidate


def _validated_title(title: str) -> str:
    return _required_text(title, "title")


def _required_text(value: str, field_name: str) -> str:
    cleaned = str(value).strip()
    if not cleaned:
        raise ValueError(f"{field_name} must not be empty")
    return cleaned


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _one_line(value: str) -> str:
    return " ".join(value.splitlines()).strip()


def _datetime_from_data(value: object) -> datetime:
    parsed = datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _now() -> datetime:
    return datetime.now(timezone.utc)


__all__ = [
    "LinkedTodoSummary",
    "SessionTodoList",
    "TodoID",
    "TodoItem",
    "TodoStatus",
]
