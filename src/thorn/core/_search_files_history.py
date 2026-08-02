"""Session-local duplicate-result tracking for ``search_files`` calls."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from thorn.core._prompt_visibility import file_content_key_for_path


@dataclass(frozen=True)
class SearchFilesCallKey:
    """Normalized arguments that determine a ``search_files`` result."""

    pattern: str
    resolved_path: Path
    glob: str | None
    use_regex: bool
    ignore_case: bool
    context_lines: int

    @property
    def fingerprint(self) -> str:
        """Short stable hash for telemetry and diagnostics."""
        return hashlib.sha256(self.to_json_bytes()).hexdigest()

    def to_json_bytes(self) -> bytes:
        payload = {
            "pattern": self.pattern,
            "resolved_path": self.resolved_path.as_posix(),
            "glob": self.glob,
            "use_regex": self.use_regex,
            "ignore_case": self.ignore_case,
            "context_lines": self.context_lines,
        }
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")


@dataclass(frozen=True)
class SearchFilesObservation:
    """One observed ``search_files`` result payload."""

    call_id: str
    render_id: str | None
    call_key: SearchFilesCallKey
    result_hash: str


@dataclass(frozen=True)
class SearchFilesDuplicateObservation:
    """Facts from comparing a new search result against prior results."""

    call_id: str
    render_id: str | None
    call_key: SearchFilesCallKey
    result_hash: str
    prior_call_id: str | None
    prior_render_id: str | None
    hint_emitted: bool


@dataclass
class SearchFilesResultHistory:
    """Bounded session memory of normalized ``search_files`` results."""

    max_observations_per_key: int = 8
    _observations_by_key: dict[
        SearchFilesCallKey,
        dict[str, SearchFilesObservation],
    ] = field(default_factory=dict)

    def observe(
        self,
        *,
        call_id: str,
        render_id: str | None,
        arguments: Mapping[str, Any],
        result_content: str,
        workspace_root: Path | None,
    ) -> SearchFilesDuplicateObservation | None:
        """Record one successful result and report exact prior duplicates.

        The comparison is intentionally post-execution only: callers have
        already returned the real tool output, and this tracker only decides
        whether that output exactly matches a previous normalized search.
        """
        call_key = search_files_call_key_from_arguments(
            arguments,
            workspace_root=workspace_root,
        )
        if call_key is None:
            return None

        result_hash = search_files_result_hash(result_content)
        observations_for_key = self._observations_by_key.setdefault(call_key, {})
        prior = observations_for_key.get(result_hash)

        observation = SearchFilesObservation(
            call_id=call_id,
            render_id=render_id,
            call_key=call_key,
            result_hash=result_hash,
        )
        observations_for_key[result_hash] = observation
        while len(observations_for_key) > self.max_observations_per_key:
            oldest_result_hash = next(iter(observations_for_key))
            observations_for_key.pop(oldest_result_hash, None)

        return SearchFilesDuplicateObservation(
            call_id=call_id,
            render_id=render_id,
            call_key=call_key,
            result_hash=result_hash,
            prior_call_id=prior.call_id if prior is not None else None,
            prior_render_id=prior.render_id if prior is not None else None,
            hint_emitted=prior is not None,
        )


def search_files_call_key_from_arguments(
    arguments: Mapping[str, Any],
    *,
    workspace_root: Path | None,
) -> SearchFilesCallKey | None:
    """Normalize arguments that affect ``search_files`` output."""
    pattern = arguments.get("pattern")
    if not isinstance(pattern, str):
        return None

    raw_path = arguments.get("path", ".")
    if not isinstance(raw_path, str):
        return None

    glob = arguments.get("glob")
    if glob is not None and not isinstance(glob, str):
        return None

    use_regex = _bool_argument(arguments.get("use_regex", False))
    if use_regex is None:
        return None
    ignore_case = _bool_argument(arguments.get("ignore_case", False))
    if ignore_case is None:
        return None
    context_lines = _int_argument(arguments.get("context_lines", 0))
    if context_lines is None:
        return None

    file_key = file_content_key_for_path(raw_path, workspace_root=workspace_root)
    if file_key is None:
        return None

    return SearchFilesCallKey(
        pattern=pattern,
        resolved_path=file_key.resolved_path,
        glob=glob,
        use_regex=use_regex,
        ignore_case=ignore_case,
        context_lines=context_lines,
    )


def search_files_result_hash(result_content: str) -> str:
    """Hash a normalized rendered ``search_files`` payload."""
    normalized = _normalize_search_files_result(result_content)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _normalize_search_files_result(result_content: str) -> str:
    normalized = result_content.replace("\r\n", "\n").replace("\r", "\n")
    return normalized.rstrip("\n")


def _bool_argument(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _int_argument(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None
