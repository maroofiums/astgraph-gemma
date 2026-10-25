import json, textwrap
from pathlib import Path
import pytest
from astgraph import RepoIndex
from astgraph.parser import extract_file
from astgraph.tools import ToolKit, TOOL_SPECS
from astgraph.cli import main as cli_main

SRC_A = textwrap.dedent('''
    class Parser(Base):
        def parse_date(self, s: str) -> str:
            if not s:
                return None
            for ch in s:
                self.convert(ch)
            return self.convert(s)

        def convert(self, s):
            try:
                return s
            except ValueError:
                return None

    def helper(x):
        return Parser().parse_date(x)
''')


@pytest.fixture()
def repo(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg/a.py").write_text(SRC_A)
    (tmp_path / "pkg/b.py").write_text("class Base:\n    pass\n\ndef unrelated():\n    return 1\n")
    for i in range(30):
        (tmp_path / f"pkg/m{i}.py").write_text(f"def f{i}(a):\n    return g{i}(a)\n\ndef g{i}(a):\n    return a\n")
    return tmp_path


def test_parser_extracts_signature_calls_control(tmp_path):
    p = tmp_path / "a.py"; p.write_text(SRC_A)
    syms = {s.qualname: s for s in extract_file(p, tmp_path)}
    assert syms["Parser"].kind == "class" and syms["Parser"].bases == ["Base"]
    m = syms["Parser.parse_date"]
    assert m.kind == "method" and m.signature == "def parse_date(self, s: str) -> str"
    assert m.calls == {"convert"} and m.control == {"if": 1, "for": 1}
    assert syms["helper"].calls == {"Parser", "parse_date"}
    assert syms["Parser.convert"].control == {"try": 1}


@pytest.mark.parametrize("budget", [64, 128, 512])
@pytest.mark.parametrize("edges,control", [(False, False), (True, True)])
def test_budget_is_never_exceeded(repo, budget, edges, control):
    idx = RepoIndex.from_path(repo)
    r = idx.context("Parser.parse_date returns None for empty string", budget=budget, edges=edges, control=control)
    assert r.tokens <= budget and idx.count(r.text) <= budget


def test_context_ranks_the_right_file_first(repo):
    idx = RepoIndex.from_path(repo)
    r = idx.context('File "pkg/a.py", line 4, in parse_date\nParser.parse_date returns None for empty `s`', budget=256)
    assert r.files[0] == "pkg/a.py"
    assert "def parse_date(self, s: str) -> str" in r.text
    assert " >" not in r.text and " #" not in r.text         # default: signatures only, no callee/control suffixes


def test_edges_and_control_are_opt_in(repo):
    idx = RepoIndex.from_path(repo)
    r = idx.context("Parser.parse_date empty", budget=512, edges=True, control=True)
    assert ">convert" in r.text and "#for1 if1" in r.text


def test_parse_cache_skips_unchanged_files(repo):
    cache = {}
    RepoIndex.from_path(repo, cache=cache)
    n = len(cache)
    assert n >= 32
    (repo / "pkg/a.py").write_text(SRC_A + "\ndef extra():\n    pass\n")
    idx = RepoIndex.from_path(repo, cache=cache)
    assert len(cache) == n + 1 and any(s.qualname == "extra" for s in idx.symbols)


def test_tools(repo):
    kit = ToolKit(RepoIndex.from_path(repo))
    assert {t["function"]["name"] for t in TOOL_SPECS} == {"repo_map", "find_symbol", "read_symbol", "callers", "callees", "outline"}
    assert "pkg/a.py" in kit.dispatch("repo_map", {"query": "Parser.parse_date empty string", "budget": 200})
    assert "pkg/a.py::Parser.parse_date" in kit.dispatch("find_symbol", '{"name": "parse_date"}')
    body = kit.dispatch("read_symbol", {"name": "Parser.parse_date"})
    assert "def parse_date" in body and "self.convert(s)" in body
    assert "Parser.parse_date" in kit.dispatch("callers", {"name": "Parser.convert"})
    assert "Parser.convert" in kit.dispatch("callees", {"name": "Parser.parse_date"})
    assert "def convert" in kit.dispatch("outline", {"path": "pkg/a.py"})
    assert "Unknown tool" in kit.dispatch("rm_rf", {})
    assert "Tool error" in kit.dispatch("read_symbol", {"nope": 1})
    assert "No symbol" in kit.dispatch("read_symbol", {"name": "does_not_exist"})


def test_ambiguous_name_asks_for_qualification(repo):
    (repo / "pkg/c.py").write_text("class Other:\n    def convert(self):\n        pass\n")
    kit = ToolKit(RepoIndex.from_path(repo))
    out = kit.dispatch("read_symbol", {"name": "convert"})
    assert "match" in out and "pkg/c.py::Other.convert" in out
    assert "def convert(self, s)" in kit.dispatch("read_symbol", {"name": "pkg/a.py::Parser.convert"})


def test_cli(repo, capsys):
    assert cli_main(["map", str(repo), "--issue", "Parser.parse_date empty", "--budget", "200", "--json"]) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["files"][0] == "pkg/a.py" and d["tokens"] <= 200
    assert cli_main(["stats", str(repo)]) == 0
    assert json.loads(capsys.readouterr().out)["symbols"] > 30


def test_agent_loop_with_scripted_model(repo):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations"))
    from tool_calling import run_agent
    script = iter([
        {"content": "", "tool_calls": [{"name": "repo_map", "arguments": {"query": "Parser.parse_date empty", "budget": 200}}]},
        {"content": '{"tool": "read_symbol", "arguments": {"name": "Parser.parse_date"}}', "tool_calls": []},
        {"content": '{"files": ["pkg/a.py"], "symbol": "Parser.parse_date"}', "tool_calls": []},
    ])
    final, msgs = run_agent(RepoIndex.from_path(repo), "Parser.parse_date returns None for empty input", lambda m, t: next(script))
    assert json.loads(final)["files"] == ["pkg/a.py"]
    tool_msgs = [m for m in msgs if m["role"] == "tool"]
    assert [m["name"] for m in tool_msgs] == ["repo_map", "read_symbol"]
    assert "def parse_date" in tool_msgs[1]["content"]


def test_mcp_server_registers_tools(repo):
    pytest.importorskip("mcp")
    import asyncio, sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations"))
    from mcp_server import build
    names = {t.name for t in asyncio.run(build(str(repo)).list_tools())}
    assert names == {"repo_map", "find_symbol", "read_symbol", "callers", "callees", "outline"}
