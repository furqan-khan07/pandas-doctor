"""GitHub Action helper, MCP tools, pre-commit hook and plugin manifests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pandas_doctor import github, mcp_server

ROOT = Path(__file__).resolve().parents[1]

FINDING = {
    "code": "PDD001",
    "message": "`df.iterrows()` builds a Series for every row",
    "path": "src/job.py",
    "cell": None,
    "line": 12,
    "column": 1,
    "suggestion": 'df["t"] = df["a"] * 2',
}


def test_annotations_escape_newlines() -> None:
    (line,) = github.annotation_lines([FINDING])
    assert line.startswith("::warning file=src/job.py,line=12,col=1,title=PDD001::")
    assert "\n" not in line and "%0A" in line


def test_markdown_summary() -> None:
    md = github.markdown(
        [FINDING, {**FINDING, "code": "PDD003", "line": 30}], ["x.py: could not parse"]
    )
    assert md.startswith(github.MARKER)
    assert "Found **2** slow pandas patterns (1 x PDD001, 1 x PDD003)." in md
    assert "`src/job.py:12`" in md and "couldn't be checked" in md


def test_markdown_clean() -> None:
    assert "No slow pandas patterns found" in github.markdown([], [])


def test_github_main(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    report = tmp_path / "pdd.json"
    report.write_text(json.dumps({"findings": [FINDING], "errors": []}))
    assert github.main([str(report), "--annotate"]) == 0
    out = capsys.readouterr()
    assert "### pandas-doctor" in out.out and "::warning" in out.err


def test_mcp_check_code() -> None:
    result = mcp_server.check_code("for i, r in df.iterrows():\n    df.at[i, 'b'] = r['a'] + 1\n")
    assert result["error"] is None
    assert result["findings"][0]["code"] == "PDD001"
    assert result["findings"][0]["suggestion"] == "df['b'] = df['a'] + 1"


def test_mcp_verify_rewrite() -> None:
    result = mcp_server.verify_rewrite(
        original="for i, r in df.iterrows():\n    df.at[i, 'b'] = r['a'] * 2",
        candidate="df['b'] = df['a'] * 2",
        setup="import pandas as pd\ndf = pd.DataFrame({'a': [1.0, 2.0, 3.0]})\ndf['b'] = 0.0",
    )
    assert result["verdict"] == "identical"
    assert result["scope"] == "verified on the given inputs only"
    assert result["original_time"] and result["candidate_time"]


def test_mcp_verify_rewrite_scalar_types() -> None:
    result = mcp_server.verify_rewrite(
        original="out = []\nfor _, r in df.iterrows():\n    out.append(r['a'] * 2)",
        candidate="out = (df['a'] * 2).tolist()",
        setup="import pandas as pd\ndf = pd.DataFrame({'a': [1.0, 2.0, 3.0]})",
    )
    assert result["verdict"] == "scalar_types"
    assert result["details"]["scalar_types"] == [["numpy.float64", "float"]]
    assert "equal values, different scalar types" in result["description"]
    assert result["outputs"]["out"]["kind"] == "scalar_types"


def test_mcp_list_rules() -> None:
    codes = {r["code"] for r in mcp_server.list_rules()}
    assert {"PDD001", "PDD003", "PDD006", "PDD008"} <= codes


def test_mcp_server_registers_tools() -> None:
    pytest.importorskip("mcp")
    server = mcp_server.build_server()
    assert server.name == "pandas-doctor"


def test_manifests_are_valid_json() -> None:
    for path in [
        ".claude-plugin/marketplace.json",
        "claude-plugin/.claude-plugin/plugin.json",
        "claude-plugin/.mcp.json",
    ]:
        json.loads((ROOT / path).read_text())
    skill = (ROOT / "claude-plugin/skills/pandas-doctor/SKILL.md").read_text()
    assert skill.startswith("---\nname: pandas-doctor\n")


def test_precommit_hook_points_at_check() -> None:
    text = (ROOT / ".pre-commit-hooks.yaml").read_text()
    assert "id: pandas-doctor" in text and "entry: pandas-doctor check" in text
    assert "types_or: [python, jupyter]" in text


def test_action_installs_itself() -> None:
    text = (ROOT / "action.yml").read_text()
    assert 'pip install --quiet "${{ github.action_path }}"' in text
    assert "--format json --exit-zero" in text


def test_mcp_stdio_round_trip() -> None:
    """Speak raw JSON-RPC to `pandas-doctor mcp` the way an assistant's client would."""
    pytest.importorskip("mcp")
    import subprocess
    import sys

    proc = subprocess.Popen(
        [sys.executable, "-m", "pandas_doctor", "mcp"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    assert proc.stdin is not None and proc.stdout is not None

    def send(msg: dict[str, object]) -> None:
        proc.stdin.write(json.dumps(msg) + "\n")  # type: ignore[union-attr]
        proc.stdin.flush()  # type: ignore[union-attr]

    try:
        send(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "0"},
                },
            }
        )
        init = json.loads(proc.stdout.readline())
        assert init["result"]["serverInfo"]["name"] == "pandas-doctor"
        send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        send({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        tools = json.loads(proc.stdout.readline())
        names = {t["name"] for t in tools["result"]["tools"]}
        assert names == {"check_code", "verify_rewrite", "list_rules"}
    finally:
        proc.kill()
