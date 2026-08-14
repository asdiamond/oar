"""The agent loop: send input, stream, execute tool calls, feed results back, repeat."""

import json
import os
import platform
from collections.abc import Iterator
from pathlib import Path

from openai import OpenAI

from .events import Event, TextDelta, ToolCall, ToolResult
from .session import Session
from .tools import DEFINITIONS, DISPATCH

DEFAULT_MODEL = os.environ.get("OAR_MODEL", "gpt-5.1")

INSTRUCTIONS = (Path(__file__).parent / "system_prompt.md").read_text()


class Agent:
    def __init__(self, model: str = DEFAULT_MODEL, session: Session | None = None):
        self.client = OpenAI()
        self.model = model
        self.instructions = INSTRUCTIONS.format(cwd=Path.cwd(), platform=platform.platform())
        self.items: list = []
        self.session = session or Session()

    def run(self, user_text: str) -> Iterator[Event]:
        self._append({"role": "user", "content": user_text})
        while True:
            response = yield from self._stream_turn()
            calls = self._record(response)
            if not calls:
                return
            yield from self._execute(calls)

    def _append(self, item) -> None:
        """The single write path for the transcript: in-memory history + session file."""
        self.items.append(item)
        data = item if isinstance(item, dict) else item.model_dump(mode="json")
        self.session.append(data.get("type") or data.get("role"), item=data)

    def _stream_turn(self):
        """One API call. Yields TextDeltas as they arrive; returns the completed Response."""
        with self.client.responses.stream(
            model=self.model,
            instructions=self.instructions,
            tools=DEFINITIONS,
            input=self.items,
            store=False,
        ) as stream:
            for event in stream:
                if event.type == "response.output_text.delta":
                    yield TextDelta(event.delta)
            return stream.get_final_response()

    def _record(self, response) -> list:
        """Record every output item (reasoning included); return the function calls."""
        for item in response.output:
            self._append(item)
        return [i for i in response.output if i.type == "function_call"]

    def _execute(self, calls) -> Iterator[Event]:
        for call in calls:
            args = json.loads(call.arguments)
            yield ToolCall(call.call_id, call.name, args)
            try:
                output = DISPATCH[call.name](**args)
            except Exception as e:
                output = f"Error: {e}"
            yield ToolResult(call.call_id, call.name, output)
            self._append(
                {"type": "function_call_output", "call_id": call.call_id, "output": output}
            )
