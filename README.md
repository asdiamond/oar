# oar

A bare-metal coding harness for the OpenAI Responses API. No bells, no whistles — one tool, you do the rowing.

> **Status:** names claimed, code coming soon.

This is a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/) with two packages:

| Package | PyPI | What it is |
|---|---|---|
| [`oar`](packages/oar) | `pip install oar` | The harness: Responses API loop, tools, CLI |
| [`oar-tui`](packages/oar-tui) | `pip install oar-tui` | Minimal TUI library: native scrollback, differential rendering, zero deps |

`oar-tui` knows nothing about agents; the agent loop knows nothing about terminals; the CLI is the only glue.

## Links

- Homepage: [oar.run](https://oar.run)

## License

MIT
