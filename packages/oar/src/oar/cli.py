"""oar CLI: `oar -p "prompt"` runs the agent loop and streams to stdout."""

import argparse
import sys

from . import __version__
from .agent import Agent
from .events import TextDelta, ToolCall, ToolResult

DIM = "\033[2m"
RESET = "\033[0m"


def _summarize(args: dict) -> str:
    parts = [str(v) for v in args.values()]
    summary = ", ".join(parts)
    first_line = summary.splitlines()[0] if summary else ""
    return first_line[:80] + ("…" if len(summary) > 80 or "\n" in summary else "")


def main() -> int:
    parser = argparse.ArgumentParser(prog="oar", description="A bare-metal coding harness for the OpenAI Responses API.")
    parser.add_argument("-p", "--prompt", required=True, help="Prompt to run")
    parser.add_argument("--model", default=None, help="Model to use (default: $OAR_MODEL or gpt-5.1)")
    parser.add_argument("--version", action="version", version=f"oar {__version__}")
    args = parser.parse_args()

    agent = Agent(model=args.model) if args.model else Agent()

    dim = sys.stdout.isatty()
    for event in agent.run(args.prompt):
        match event:
            case TextDelta(text):
                sys.stdout.write(text)
                sys.stdout.flush()
            case ToolCall(_, name, call_args):
                line = f"⏺ {name}({_summarize(call_args)})"
                print(f"\n{DIM}{line}{RESET}" if dim else f"\n{line}")
            case ToolResult(_, _, output):
                first = output.splitlines()[0] if output else "(no output)"
                line = f"  ⎿ {first[:100]}"
                print(f"{DIM}{line}{RESET}" if dim else line)

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
