# Privacy

pandas-doctor runs entirely on your machine. It collects no data, has no telemetry, and makes no network requests of its own.

- `check` reads the files you point it at.
- `run` and `%%doctor` run your code locally, compare results in memory, and print a report. With `--fix`, `--diff` or `--json` it writes only where you tell it to.
- The MCP server and the Claude Code skill run the same local commands. Whatever your AI assistant does with the output is covered by that assistant's own policy.

Installing it (pip, uvx) downloads the package from PyPI, which has its own [privacy notice](https://policies.python.org/pypi.org/Privacy-Notice/).

Questions: [open an issue](https://github.com/furqan-khan07/pandas-doctor/issues).
