"""The agent loop: send input, stream, execute tool calls, feed results back, repeat."""

import json
import os
import platform
from collections.abc import Generator
from pathlib import Path
from typing import cast

from openai import OpenAI
from openai.types.responses import ResponseFunctionToolCall
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

    items.append({"role": "user", "content": user_text})
    session.append("user", item={"role": "user", "content": user_text})

    while True:
        # one API call: yield text deltas as they arrive, then take the completed response
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
            response = stream.get_final_response()

        # record every output item (reasoning included) into history + session, and
        # collect the function calls. SDK output objects are valid input items at
        # runtime; the stubs only admit TypedDicts, hence the cast.
        calls: list[ResponseFunctionToolCall] = []
        for item in response.output:
            items.append(cast(ResponseInputItemParam, item))
            data = item.model_dump(mode="json")
            session.append(str(data.get("type") or "item"), item=data)
            if item.type == "function_call":
                calls.append(item)

        if not calls:
            return

        # execute each call and feed its output back for the next turn
        for call in calls:
            args: dict[str, object] = json.loads(call.arguments)
            yield ToolCall(call.call_id, call.name, args)
            try:
                output = DISPATCH[call.name](**args)
            except Exception as e:
                output = f"Error: {e}"
            yield ToolResult(call.call_id, call.name, output)
            result: ResponseInputItemParam = {
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": output,
            }
            items.append(result)
            session.append("function_call_output", item=result)
