"""The four tools: read, write, edit, bash."""

import subprocess
from collections.abc import Callable
from pathlib import Path

from openai.types.responses import FunctionToolParam

MAX_OUTPUT = 30_000
BASH_TIMEOUT = 120


def _truncate(text: str) -> str:
    if len(text) <= MAX_OUTPUT:
        return text
    return text[:MAX_OUTPUT] + f"\n... [truncated, {len(text)} chars total]"


def read(path: str) -> str:
    return _truncate(Path(path).read_text())


def write(path: str, content: str) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"Wrote {len(content)} chars to {path}"


def edit(path: str, old_string: str, new_string: str) -> str:
    p = Path(path)
    content = p.read_text()
    count = content.count(old_string)
    if count == 0:
        raise ValueError(f"old_string not found in {path}")
    if count > 1:
        raise ValueError(f"old_string appears {count} times in {path}; provide more context to make it unique")
    p.write_text(content.replace(old_string, new_string, 1))
    return f"Edited {path}"


def bash(command: str) -> str:
    try:
        proc = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=BASH_TIMEOUT
        )
    except subprocess.TimeoutExpired:
        return f"Command timed out after {BASH_TIMEOUT}s"
    output = proc.stdout + proc.stderr
    if proc.returncode != 0:
        output += f"\n[exit code {proc.returncode}]"
    return _truncate(output) or "(no output)"


DISPATCH: dict[str, Callable[..., str]] = {"read": read, "write": write, "edit": edit, "bash": bash}


def _tool(name: str, description: str, properties: dict[str, object]) -> FunctionToolParam:
    return {
        "type": "function",
        "name": name,
        "description": description,
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        },
    }


DEFINITIONS: list[FunctionToolParam] = [
    _tool(
        "read",
        "Read a file and return its contents.",
        {"path": {"type": "string", "description": "Path to the file"}},
    ),
    _tool(
        "write",
        "Write content to a file, creating it (and parent directories) if needed, overwriting if it exists.",
        {
            "path": {"type": "string", "description": "Path to the file"},
            "content": {"type": "string", "description": "Full file content"},
        },
    ),
    _tool(
        "edit",
        "Replace an exact string in a file. Fails if the string is missing or appears more than once — include enough surrounding context to make it unique.",
        {
            "path": {"type": "string", "description": "Path to the file"},
            "old_string": {"type": "string", "description": "Exact text to replace"},
            "new_string": {"type": "string", "description": "Replacement text"},
        },
    ),
    _tool(
        "bash",
        f"Run a shell command and return its output. Times out after {BASH_TIMEOUT}s. "
        "Use this for searching, listing, git, tests, and everything else.",
        {"command": {"type": "string", "description": "The command to run"}},
    ),
]
