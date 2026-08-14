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


def run(
    user_text: str,
    *,
    model: str = DEFAULT_MODEL,
    client: OpenAI | None = None,
    session: Session | None = None,
    items: list[ResponseInputItemParam] | None = None,
) -> Generator[Event]:
    """Run the agent on one user message, yielding events until the model stops calling tools.

    `items` is the conversation history; pass the same list across calls to continue
    a conversation. It is mutated in place and mirrored to `session`.
    """
    client = client or OpenAI()
    session = session or Session()
    items = items if items is not None else []
    instructions = INSTRUCTIONS.format(cwd=Path.cwd(), platform=platform.platform())

    def append(item: ResponseInputItemParam | ResponseOutputItem) -> None:
        # single write path: in-memory history + session file. SDK output objects are
        # valid input items at runtime; the stubs only admit TypedDicts, hence the cast.
        items.append(cast(ResponseInputItemParam, item))
        data = item.model_dump(mode="json") if isinstance(item, BaseModel) else item
        kind = data.get("type") or data.get("role") or "item"
        session.append(str(kind), item=data)

    def stream_turn() -> Generator[TextDelta, None, Response]:
        # one API call: yield TextDeltas as they arrive, return the completed Response
        with client.responses.stream(
            model=model,
            instructions=instructions,
            tools=DEFINITIONS,
            input=items,
            store=False,
        ) as stream:
            for event in stream:
                if event.type == "response.output_text.delta":
                    yield TextDelta(event.delta)
            return stream.get_final_response()

    def record(response: Response) -> list[ResponseFunctionToolCall]:
        # record every output item (reasoning included); return the function calls
        for item in response.output:
            append(item)
        return [i for i in response.output if i.type == "function_call"]

    def execute(calls: Sequence[ResponseFunctionToolCall]) -> Generator[Event]:
        for call in calls:
            args: dict[str, object] = json.loads(call.arguments)
            yield ToolCall(call.call_id, call.name, args)
            try:
                output = DISPATCH[call.name](**args)
            except Exception as e:
                output = f"Error: {e}"
            yield ToolResult(call.call_id, call.name, output)
            append({"type": "function_call_output", "call_id": call.call_id, "output": output})

    append({"role": "user", "content": user_text})
    while True:
        response = yield from stream_turn()
        calls = record(response)
        if not calls:
            return
        yield from execute(calls)
