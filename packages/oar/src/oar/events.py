"""Events yielded by the agent loop. The only contract between agent and frontend."""

from dataclasses import dataclass


@dataclass
class TextDelta:
    text: str


@dataclass
class ToolCall:
    call_id: str
    name: str
    arguments: dict


@dataclass
class ToolResult:
    call_id: str
    name: str
    output: str


Event = TextDelta | ToolCall | ToolResult
