"""The agent loop: send input, stream, execute tool calls, feed results back, repeat."""

import json
import os
import platform
from collections.abc import Generator, Sequence
from pathlib import Path
from typing import cast

from openai import OpenAI
from pydantic import BaseModel
from openai.types.responses import (
    Response,
    ResponseFunctionToolCall,
    ResponseOutputItem,
)
from openai.types.responses.response_input_param import ResponseInputItemParam

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
        self.items: list[ResponseInputItemParam] = []
        self.session = session or Session()

    def run(self, user_text: str) -> Generator[Event]:
        self._append({"role": "user", "content": user_text})
        while True:
            response = yield from self._stream_turn()
            calls = self._record(response)
            if not calls:
                return
            yield from self._execute(calls)

    def _append(self, item: ResponseInputItemParam | ResponseOutputItem) -> None:
        """The single write path for the transcript: in-memory history + session file.

        SDK output objects are valid input items at runtime; the stubs only admit
        TypedDicts, hence the cast.
        """
        self.items.append(cast(ResponseInputItemParam, item))
        data = item.model_dump(mode="json") if isinstance(item, BaseModel) else item
        kind = data.get("type") or data.get("role") or "item"
        self.session.append(str(kind), item=data)

    def _stream_turn(self) -> Generator[TextDelta, None, Response]:
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

    def _record(self, response: Response) -> list[ResponseFunctionToolCall]:
        """Record every output item (reasoning included); return the function calls."""
        for item in response.output:
            self._append(item)
        return [i for i in response.output if i.type == "function_call"]

    def _execute(self, calls: Sequence[ResponseFunctionToolCall]) -> Generator[Event]:
        for call in calls:
            args: dict[str, object] = json.loads(call.arguments)
            yield ToolCall(call.call_id, call.name, args)
            try:
                output = DISPATCH[call.name](**args)
            except Exception as e:
                output = f"Error: {e}"
            yield ToolResult(call.call_id, call.name, output)
            self._append(
                {"type": "function_call_output", "call_id": call.call_id, "output": output}
            )
