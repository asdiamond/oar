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
        self.items.append({"role": "user", "content": user_text})
        self.session.append("user", text=user_text)

        while True:
            stream = self.client.responses.create(
                model=self.model,
                instructions=self.instructions,
                tools=DEFINITIONS,
                input=self.items,
                store=False,
                stream=True,
            )

            response = None
            for event in stream:
                if event.type == "response.output_text.delta":
                    yield TextDelta(event.delta)
                elif event.type == "response.completed":
                    response = event.response
                elif event.type == "response.failed":
                    raise RuntimeError(f"Response failed: {event.response.error}")

            if response is None:
                raise RuntimeError("Stream ended without a completed response")

            calls = []
            for item in response.output:
                self.items.append(item)
                self.session.append("assistant_item", item=item.model_dump(mode="json"))
                if item.type == "function_call":
                    calls.append(item)

            if not calls:
                return

            for call in calls:
                args = json.loads(call.arguments)
                yield ToolCall(call.call_id, call.name, args)
                try:
                    output = DISPATCH[call.name](**args)
                except Exception as e:
                    output = f"Error: {e}"
                yield ToolResult(call.call_id, call.name, output)
                self.items.append(
                    {"type": "function_call_output", "call_id": call.call_id, "output": output}
                )
                self.session.append(
                    "function_call_output", call_id=call.call_id, name=call.name, output=output
                )
