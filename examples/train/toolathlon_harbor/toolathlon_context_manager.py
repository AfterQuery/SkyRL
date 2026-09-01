"""Bounded active context with durable, searchable Harbor trajectory history."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

LOCAL_TOOL_SCHEMAS = [
    {
        "name": "context_check_status",
        "description": "Report active context usage and prior compaction events.",
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "context_manage",
        "description": "Schedule deterministic removal of older conversation exchanges.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "enum": ["keep_recent_turns", "keep_recent_percent", "delete_first_turns", "delete_first_percent"],
                },
                "value": {"type": "number", "exclusiveMinimum": 0},
            },
            "required": ["method", "value"],
            "additionalProperties": False,
        },
    },
    {
        "name": "history_search",
        "description": "Search the complete durable conversation history, including compacted turns.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "history_view",
        "description": "View bounded records from the complete durable conversation history.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "start_seq": {"type": "integer", "minimum": 0},
                "count": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["start_seq"],
            "additionalProperties": False,
        },
    },
    {
        "name": "tool_output_search",
        "description": "Regex-search an offloaded tool output and start a paginated search session.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "artifact_id": {"type": "string"},
                "pattern": {"type": "string"},
                "page_size": {"type": "integer", "minimum": 1, "maximum": 50},
                "context_size": {"type": "integer", "minimum": 1},
            },
            "required": ["artifact_id", "pattern"],
            "additionalProperties": False,
        },
    },
    {
        "name": "tool_output_search_navigate",
        "description": "Navigate a search session created by tool_output_search.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "search_session_id": {"type": "string"},
                "action": {
                    "type": "string",
                    "enum": ["next_page", "prev_page", "jump_to_page", "first_page", "last_page"],
                },
                "target_page": {"type": "integer", "minimum": 1},
            },
            "required": ["search_session_id", "action"],
            "additionalProperties": False,
        },
    },
    {
        "name": "tool_output_view",
        "description": "View the first character page of an offloaded tool output.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "artifact_id": {"type": "string"},
                "page_size": {"type": "integer", "minimum": 1, "maximum": 100000},
            },
            "required": ["artifact_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "tool_output_view_navigate",
        "description": "Navigate a view session created by tool_output_view.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "view_session_id": {"type": "string"},
                "action": {
                    "type": "string",
                    "enum": ["next_page", "prev_page", "jump_to_page", "first_page", "last_page"],
                },
                "target_page": {"type": "integer", "minimum": 1},
            },
            "required": ["view_session_id", "action"],
            "additionalProperties": False,
        },
    },
]


@dataclass
class ContextPolicy:
    max_context_tokens: int
    max_output_tokens: int
    safety_tokens: int = 2048
    warning_ratio: float = 0.75
    compact_ratio: float = 0.85
    target_ratio: float = 0.70
    keep_recent_turns: int = 4
    keep_reasoning_turns: int = 1
    inline_tool_output_chars: int = 12_000
    preview_chars: int = 2_000
    max_resets: int = 2

    @property
    def prompt_budget(self) -> int:
        return self.max_context_tokens - self.max_output_tokens - self.safety_tokens

    def validate(self) -> None:
        if self.prompt_budget <= 0:
            raise ValueError("max_context_tokens must exceed max_tokens + context_safety_tokens")
        if not 0 < self.target_ratio < self.warning_ratio < self.compact_ratio < 1:
            raise ValueError("context ratios must satisfy target < warning < compact < 1")


@dataclass
class ManagedContext:
    instruction: str
    artifact_dir: Path
    policy: ContextPolicy
    full_messages: list[dict[str, Any]] = field(default_factory=list)
    active_messages: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    latest_prompt_tokens: int = 0
    additions_since_prompt: list[dict[str, Any]] = field(default_factory=list)
    pending_compaction: dict[str, Any] | None = None
    reset_count: int = 0
    search_sessions: dict[str, dict[str, Any]] = field(default_factory=dict)
    view_sessions: dict[str, dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.policy.validate()
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        (self.artifact_dir / "tool_outputs").mkdir(exist_ok=True)
        initial = {"role": "user", "content": self.instruction}
        self.full_messages.append(initial.copy())
        self.active_messages.append(initial.copy())
        self._append_history(initial, active=True)

    @property
    def tool_schemas(self) -> list[dict[str, Any]]:
        return LOCAL_TOOL_SCHEMAS

    @property
    def history_path(self) -> Path:
        return self.artifact_dir / "context_history.jsonl"

    def _append_history(self, message: dict[str, Any], *, active: bool) -> None:
        record = {"seq": self._history_count(), "active_at_write": active, "message": message}
        with self.history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _history_count(self) -> int:
        if not self.history_path.exists():
            return 0
        with self.history_path.open("r", encoding="utf-8") as stream:
            return sum(1 for _ in stream)

    def append(self, message: dict[str, Any]) -> None:
        full = json.loads(json.dumps(message, ensure_ascii=False))
        active = json.loads(json.dumps(message, ensure_ascii=False))
        self.full_messages.append(full)
        self.active_messages.append(active)
        self.additions_since_prompt.append(active)
        self._append_history(full, active=True)

    def append_tool_result(self, message: dict[str, Any]) -> dict[str, Any]:
        content = str(message.get("content") or "")
        if len(content) <= self.policy.inline_tool_output_chars:
            self.append(message)
            return message
        digest = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()[:16]
        artifact_id = f"tool-{digest}"
        artifact_path = self.artifact_dir / "tool_outputs" / f"{artifact_id}.json"
        artifact_path.write_text(json.dumps({"artifact_id": artifact_id, "content": content}, ensure_ascii=False), encoding="utf-8")
        n = self.policy.preview_chars
        preview = (
            f"[Tool output offloaded as {artifact_id}; {len(content)} characters. "
            "Use tool_output_search or tool_output_view for complete content.]\n"
            f"{content[:n]}\n... [offloaded] ...\n{content[-n:]}"
        )
        active_message = dict(message)
        active_message["content"] = preview
        durable_message = dict(active_message)
        durable_message["artifact_id"] = artifact_id
        self.full_messages.append(durable_message)
        self.active_messages.append(active_message)
        self.additions_since_prompt.append(active_message)
        self._append_history(durable_message, active=True)
        self._event("tool_output_offloaded", artifact_id=artifact_id, original_chars=len(content))
        return durable_message

    @staticmethod
    def _estimate(messages: list[dict[str, Any]]) -> int:
        # Conservative tokenizer-independent estimate; calibrated by real prompt usage after each request.
        return sum(max(1, len(json.dumps(message, ensure_ascii=False)) // 3) for message in messages)

    def estimated_prompt_tokens(self) -> int:
        if not self.latest_prompt_tokens:
            return self._estimate(self.active_messages)
        return self.latest_prompt_tokens + self._estimate(self.additions_since_prompt)

    def observe_prompt_tokens(self, tokens: int) -> None:
        self.latest_prompt_tokens = max(0, tokens)
        self.additions_since_prompt.clear()

    def _event(self, kind: str, **details: Any) -> None:
        self.events.append({"type": kind, **details})

    def prepare_for_request(self) -> None:
        estimate = self.estimated_prompt_tokens()
        ratio = estimate / self.policy.prompt_budget
        if ratio >= self.policy.warning_ratio:
            self._event("context_warning", estimated_tokens=estimate, ratio=ratio)
        if ratio >= self.policy.compact_ratio:
            self.compact_automatic(estimate)

    def _exchange_ranges(self) -> list[tuple[int, int]]:
        ranges: list[tuple[int, int]] = []
        start: int | None = None
        for index, message in enumerate(self.active_messages):
            if message.get("role") == "assistant":
                if start is not None:
                    ranges.append((start, index))
                start = index
        if start is not None:
            ranges.append((start, len(self.active_messages)))
        return ranges

    def _prune_old_reasoning(self) -> int:
        assistant_indices = [i for i, m in enumerate(self.active_messages) if m.get("role") == "assistant"]
        eligible = assistant_indices[:-self.policy.keep_reasoning_turns] if self.policy.keep_reasoning_turns else assistant_indices
        removed = 0
        for index in eligible:
            reasoning = self.active_messages[index].pop("reasoning_content", None)
            if reasoning is not None:
                removed += max(1, len(str(reasoning)) // 3)
        if removed:
            self._event("reasoning_pruned", estimated_tokens_removed=removed, assistant_turns=len(eligible))
        return removed

    def compact_automatic(self, estimate_before: int) -> None:
        current_estimate = max(0, estimate_before - self._prune_old_reasoning())
        target = int(self.policy.prompt_budget * self.policy.target_ratio)
        ranges = self._exchange_ranges()
        removable = max(0, len(ranges) - self.policy.keep_recent_turns)
        removed = 0
        while removable > 0 and current_estimate > target:
            ranges = self._exchange_ranges()
            start, end = ranges[0]
            current_estimate = max(
                0, current_estimate - self._estimate(self.active_messages[start:end])
            )
            del self.active_messages[start:end]
            removed += 1
            removable -= 1
        self.latest_prompt_tokens = 0
        self.additions_since_prompt.clear()
        self._event(
            "automatic_compaction",
            estimated_tokens_before=estimate_before,
            estimated_tokens_after=current_estimate,
            exchanges_removed=removed,
        )

    def schedule_manual(self, args: dict[str, Any]) -> dict[str, Any]:
        method = str(args.get("method") or "")
        value = args.get("value")
        valid = {"keep_recent_turns", "keep_recent_percent", "delete_first_turns", "delete_first_percent"}
        if method not in valid or not isinstance(value, (int, float)) or value <= 0:
            return {"status": "error", "message": "invalid context compaction method or value"}
        if "percent" in method and value >= 100:
            return {"status": "error", "message": "percentage must be less than 100"}
        self.pending_compaction = {"method": method, "value": value}
        return {"status": "scheduled", "method": method, "value": value}

    def apply_pending(self) -> None:
        if not self.pending_compaction:
            return
        request = self.pending_compaction
        self.pending_compaction = None
        ranges = self._exchange_ranges()
        eligible = max(0, len(ranges) - self.policy.keep_recent_turns)
        method, value = request["method"], request["value"]
        if method == "keep_recent_turns":
            delete = max(0, len(ranges) - max(self.policy.keep_recent_turns, int(value)))
        elif method == "keep_recent_percent":
            keep = max(self.policy.keep_recent_turns, int(len(ranges) * float(value) / 100))
            delete = max(0, len(ranges) - keep)
        elif method == "delete_first_turns":
            delete = min(eligible, int(value))
        else:
            delete = min(eligible, int(eligible * float(value) / 100))
        for _ in range(delete):
            start, end = self._exchange_ranges()[0]
            del self.active_messages[start:end]
        self.latest_prompt_tokens = 0
        self.additions_since_prompt.clear()
        self._event("manual_compaction", method=method, value=value, exchanges_removed=delete)

    def status(self) -> dict[str, Any]:
        estimate = self.estimated_prompt_tokens()
        return {
            "estimated_prompt_tokens": estimate,
            "prompt_budget": self.policy.prompt_budget,
            "usage_ratio": estimate / self.policy.prompt_budget,
            "active_messages": len(self.active_messages),
            "full_messages": len(self.full_messages),
            "reset_count": self.reset_count,
            "recent_events": self.events[-10:],
        }

    def _read_history(self) -> list[dict[str, Any]]:
        if not self.history_path.exists():
            return []
        return [json.loads(line) for line in self.history_path.read_text(encoding="utf-8").splitlines() if line]

    def _artifact_content(self, artifact_id: str) -> str:
        if not re.fullmatch(r"tool-[0-9a-f]{16}", artifact_id):
            raise ValueError("invalid artifact_id")
        path = self.artifact_dir / "tool_outputs" / f"{artifact_id}.json"
        return str(json.loads(path.read_text(encoding="utf-8"))["content"])

    @staticmethod
    def _target_page(session: dict[str, Any], args: dict[str, Any], total_pages: int) -> int:
        current = int(session.get("current_page", 1))
        action = str(args.get("action", "next_page"))
        if action == "next_page":
            return min(current + 1, total_pages)
        if action == "prev_page":
            return max(current - 1, 1)
        if action == "first_page":
            return 1
        if action == "last_page":
            return total_pages
        if action == "jump_to_page":
            target = int(args.get("target_page", 0))
            if not 1 <= target <= total_pages:
                raise ValueError(f"target_page must be between 1 and {total_pages}")
            return target
        raise ValueError(f"invalid navigation action: {action}")

    @staticmethod
    def _format_search_page(session_id: str, session: dict[str, Any], page: int) -> dict[str, Any]:
        matches = session["matches"]
        page_size = session["page_size"]
        total_pages = max(1, (len(matches) + page_size - 1) // page_size)
        start = (page - 1) * page_size
        results = []
        for match in matches[start : start + page_size]:
            results.append(
                {
                    "match_text": match["match_text"],
                    "start_pos": match["start_pos"],
                    "end_pos": match["end_pos"],
                    "line_num": match["line_num"],
                    "context": (
                        match["before_context"]
                        + f">>>{match['match_text']}<<<"
                        + match["after_context"]
                    ),
                }
            )
        return {
            "artifact_id": session["artifact_id"],
            "pattern": session["pattern"],
            "search_session_id": session_id,
            "total_matches": len(matches),
            "current_page": page,
            "total_pages": total_pages,
            "page_size": page_size,
            "file_size_chars": session["content_length"],
            "results": results,
        }

    def _search_output(self, args: dict[str, Any]) -> dict[str, Any]:
        artifact_id = str(args["artifact_id"])
        pattern = str(args["pattern"]).strip()
        page_size = int(args.get("page_size", 10))
        context_size = int(args.get("context_size", 1000))
        if not pattern:
            raise ValueError("pattern is required")
        if not 1 <= page_size <= 50:
            raise ValueError("page_size must be between 1 and 50")
        if context_size < 1:
            raise ValueError("context_size must be positive")
        try:
            regex = re.compile(pattern, re.IGNORECASE | re.MULTILINE | re.DOTALL)
        except re.error as exc:
            raise ValueError(f"invalid regex pattern: {exc}") from exc
        content = self._artifact_content(artifact_id)
        matches = []
        for match in regex.finditer(content):
            start, end = match.span()
            context_start = max(0, start - context_size // 2)
            context_end = min(len(content), end + context_size // 2)
            matches.append(
                {
                    "match_text": match.group(0),
                    "start_pos": start,
                    "end_pos": end,
                    "line_num": content[:start].count("\n") + 1,
                    "before_context": content[context_start:start],
                    "after_context": content[end:context_end],
                }
            )
        session_id = uuid.uuid4().hex[:8]
        session = {
            "artifact_id": artifact_id,
            "pattern": pattern,
            "matches": matches,
            "page_size": page_size,
            "context_size": context_size,
            "content_length": len(content),
            "current_page": 1,
        }
        self.search_sessions[session_id] = session
        return self._format_search_page(session_id, session, 1)

    def _navigate_search(self, args: dict[str, Any]) -> dict[str, Any]:
        session_id = str(args["search_session_id"])
        if session_id not in self.search_sessions:
            raise ValueError("invalid or expired search_session_id")
        session = self.search_sessions[session_id]
        total_pages = max(1, (len(session["matches"]) + session["page_size"] - 1) // session["page_size"])
        page = self._target_page(session, args, total_pages)
        session["current_page"] = page
        return self._format_search_page(session_id, session, page)

    @staticmethod
    def _format_view_page(session_id: str, session: dict[str, Any], content: str, page: int) -> dict[str, Any]:
        page_size = session["page_size"]
        total_pages = max(1, (len(content) + page_size - 1) // page_size)
        start = (page - 1) * page_size
        end = min(start + page_size, len(content))
        return {
            "artifact_id": session["artifact_id"],
            "view_session_id": session_id,
            "current_page": page,
            "total_pages": total_pages,
            "page_size": page_size,
            "start_pos": start,
            "end_pos": end,
            "file_size_chars": len(content),
            "content": content[start:end],
        }

    def _view_output(self, args: dict[str, Any]) -> dict[str, Any]:
        artifact_id = str(args["artifact_id"])
        page_size = int(args.get("page_size", 10_000))
        if not 1 <= page_size <= 100_000:
            raise ValueError("page_size must be between 1 and 100000")
        content = self._artifact_content(artifact_id)
        session_id = uuid.uuid4().hex[:8]
        session = {"artifact_id": artifact_id, "page_size": page_size, "current_page": 1}
        self.view_sessions[session_id] = session
        return self._format_view_page(session_id, session, content, 1)

    def _navigate_view(self, args: dict[str, Any]) -> dict[str, Any]:
        session_id = str(args["view_session_id"])
        if session_id not in self.view_sessions:
            raise ValueError("invalid or expired view_session_id")
        session = self.view_sessions[session_id]
        content = self._artifact_content(session["artifact_id"])
        total_pages = max(1, (len(content) + session["page_size"] - 1) // session["page_size"])
        page = self._target_page(session, args, total_pages)
        session["current_page"] = page
        return self._format_view_page(session_id, session, content, page)

    def call_local_tool(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == "context_check_status":
            return self.status()
        if name == "context_manage":
            return self.schedule_manual(args)
        if name == "history_view":
            start, count = int(args["start_seq"]), min(20, int(args.get("count", 5)))
            return {"records": self._read_history()[start : start + count]}
        if name == "history_search":
            query = str(args["query"]).casefold()
            limit = min(20, int(args.get("max_results", 10)))
            found = []
            for record in self._read_history():
                text = json.dumps(record.get("message"), ensure_ascii=False)
                pos = text.casefold().find(query)
                if pos >= 0:
                    found.append({"seq": record["seq"], "snippet": text[max(0, pos - 500) : pos + len(query) + 500]})
                if len(found) >= limit:
                    break
            return {"results": found}
        if name == "tool_output_search":
            return self._search_output(args)
        if name == "tool_output_search_navigate":
            return self._navigate_search(args)
        if name == "tool_output_view":
            return self._view_output(args)
        if name == "tool_output_view_navigate":
            return self._navigate_view(args)
        raise ValueError(f"unknown local tool: {name}")

    def emergency_reset(self) -> bool:
        if self.reset_count >= self.policy.max_resets:
            return False
        self.reset_count += 1
        recent = self.full_messages[-12:]
        overview = json.dumps(recent, ensure_ascii=False)
        overview = overview[:12_000]
        recovery = {
            "role": "user",
            "content": (
                "[Context reset] The previous active context exceeded the model limit. "
                "Continue the original task using this recent structural overview. Use history_search, "
                "history_view, tool_output_search, or tool_output_view for omitted details.\n\n"
                f"Recent history:\n{overview}"
            ),
        }
        self.active_messages = [{"role": "user", "content": self.instruction}, recovery]
        self.full_messages.append(recovery.copy())
        self._append_history(recovery, active=True)
        self.latest_prompt_tokens = 0
        self.additions_since_prompt.clear()
        self._event("emergency_reset", reset_count=self.reset_count)
        return True


def create_managed_context(
    *,
    instruction: str,
    artifact_dir: Path,
    max_context_tokens: int,
    max_output_tokens: int,
    **policy_kwargs: Any,
) -> ManagedContext:
    """Create the Toolathlon-inspired managed context for the generic runner."""
    return ManagedContext(
        instruction=instruction,
        artifact_dir=artifact_dir,
        policy=ContextPolicy(
            max_context_tokens=max_context_tokens,
            max_output_tokens=max_output_tokens,
            **policy_kwargs,
        ),
    )
