"""Events yielded by the agent loop. The only contract between agent and frontend."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TextDelta:
    text: str


@dataclass(frozen=True, slots=True)
class ToolCall:
    call_id: str
    name: str
    arguments: dict[str, object]


@dataclass(frozen=True, slots=True)
class ToolResult:
    call_id: str
    name: str
    output: str


type Event = TextDelta | ToolCall | ToolResult
