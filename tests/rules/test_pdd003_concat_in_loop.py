from __future__ import annotations

import textwrap

import pytest

from tests.helpers import findings, fixed

HEADER = "import pandas as pd\n"


def only(source: str):  # type: ignore[no-untyped-def]
    found = findings(HEADER + textwrap.dedent(source), "PDD003")
    assert len(found) == 1, found
    return found[0]


class TestDetection:
    @pytest.mark.parametrize(
        "source",
        [
            """
            for p in parts:
                out = pd.concat([out, p])
            """,
            """
            while more():
                out = pd.concat([out, nxt()])
            """,
            """
            for p in parts:
                out = pd.concat([p, out])
            """,
            """
            for p in parts:
                out = pd.concat([out, p]).reset_index(drop=True)
            """,
            """
            from pandas import concat
            for p in parts:
                out = concat((out, p))
            """,
        ],
    )
    def test_flagged(self, source: str) -> None:
        only(source)

    def test_removed_append(self) -> None:
        f = only(
            """
            for p in parts:
                out = out.append(p)
            """
        )
        assert "removed in pandas 2.0" in f.message
        assert f.fix is None

    @pytest.mark.parametrize(
        "source",
        [
            "out = pd.concat([out, p])",  # not in a loop
            """
            for p in parts:
                other = pd.concat([out, p])
            """,  # doesn't grow `other`
            """
            for p in parts:
                out.append(p)
            """,  # list append is fine
            """
            for t in range(10):
                res = res.append(endog.iloc[t : t + 1], refit=False)
            """,  # statsmodels results.append, found in the corpus
            """
            def f():
                out = pd.concat([out, p])
            for p in parts:
                f()
            """,
        ],
    )
    def test_not_flagged(self, source: str) -> None:
        assert findings(HEADER + textwrap.dedent(source), "PDD003") == []


class TestFixOffered:
    @pytest.mark.parametrize(
        "source",
        [
            """
            for p in parts:
                out = pd.concat([out, p])
            """,
            """
            for p in parts:
                out = pd.concat([out, p], ignore_index=True)
            """,
            """
            for f in files:
                out = pd.concat([out, pd.read_csv(f)], axis=0, sort=False)
            """,
            """
            for f in files:
                chunk = pd.read_csv(f)
                chunk["src"] = f
                out = pd.concat([out, chunk])
            """,
            """
            for f in files:
                if f.endswith(".csv"):
                    out = pd.concat([out, load(f)])
                else:
                    continue
            """,
        ],
    )
    def test_safe_shapes(self, source: str) -> None:
        assert only(source).fix is not None


class TestFixRefused:
    @pytest.mark.parametrize(
        ("source", "reason"),
        [
            ("while x:\n    out = pd.concat([out, nxt()])", "not a `for` loop"),
            ("for p in ps:\n    out = pd.concat([out, p])\nelse:\n    pass", "`else:`"),
            ("for p in ps:\n    out = pd.concat([out, p])\n    print(len(out))", "read elsewhere"),
            ("for p in ps:\n    out = pd.concat([p, out])", "prepended"),
            ("for p in ps:\n    out = pd.concat([out, p, p])", "exactly [frame, part]"),
            ("for p in ps:\n    out = pd.concat([out, p], keys=['a', 'b'])", "`keys=`"),
            ("for p in ps:\n    out = pd.concat([out, p], ignore_index=flag)", "plain literal"),
            ("for p in ps:\n    out = pd.concat([out, p], axis=1)", "plain literal"),
            ("for p in ps:\n    out = pd.concat([out, p]).copy()", "bigger expression"),
            (
                "try:\n    for p in ps:\n        out = pd.concat([out, p])\nexcept E:\n    pass",
                "try/with",
            ),
            ("with ctx:\n    for p in ps:\n        out = pd.concat([out, p])", "try/with"),
            (
                "def g():\n    for p in ps:\n        out = pd.concat([out, p])\n        yield 1",
                "yield",
            ),
            (
                "def g():\n    global out\n    for p in ps:\n        out = pd.concat([out, p])",
                "global",
            ),
            # the loop variable is modified before being appended
            ("for p in ps:\n    p['x'] = 1\n    out = pd.concat([out, p])", "loop variable"),
            # the part is changed after being appended
            (
                "for f in fs:\n    c = load(f)\n    out = pd.concat([out, c])\n    c['x'] = 1",
                "used after",
            ),
            # the part isn't fresh: same object each time
            ("for f in fs:\n    c = template\n    out = pd.concat([out, c])", "freshly created"),
            # used before its own assignment in the iteration: previous iteration's object
            (
                "for f in fs:\n    c['n'] = f\n    c = load(f)\n    out = pd.concat([out, c])",
                "freshly created",
            ),
            ("for i in range(3):\n    out = pd.concat([out, frames[i]])", "can't prove"),
            ("for p in ps:\n    out = pd.concat([out, p])\n    return out", "read elsewhere"),
        ],
    )
    def test_reason(self, source: str, reason: str) -> None:
        f = only(source)
        assert f.fix is None
        assert f.fix_skipped is not None and reason in f.fix_skipped, f.fix_skipped


class TestFixOutput:
    def test_golden(self) -> None:
        before = HEADER + textwrap.dedent(
            """
            out = pd.DataFrame()
            # stack the monthly files
            for p in parts:
                out = pd.concat([out, p], ignore_index=True)  # grow
            print(out.shape)
            """
        )
        after = HEADER + textwrap.dedent(
            """
            out = pd.DataFrame()
            # stack the monthly files
            out_parts = []
            for p in parts:
                out_parts.append(p)  # grow
            if out_parts:
                out = pd.concat([out, *out_parts], ignore_index=True)
            print(out.shape)
            """
        )
        assert fixed(before) == after

    def test_inside_function_keeps_indentation(self) -> None:
        before = textwrap.dedent(
            """\
            import pandas as pd

            def load_all(files):
              out = pd.DataFrame()
              for f in files:
                if f:
                  out = pd.concat([out, pd.read_csv(f)])
              return out
            """
        )
        after = textwrap.dedent(
            """\
            import pandas as pd

            def load_all(files):
              out = pd.DataFrame()
              out_parts = []
              for f in files:
                if f:
                  out_parts.append(pd.read_csv(f))
              if out_parts:
                out = pd.concat([out, *out_parts])
              return out
            """
        )
        assert fixed(before) == after

    def test_name_clash_gets_a_suffix(self) -> None:
        out = fixed(HEADER + "out_parts = 1\nfor p in ps:\n    out = pd.concat([out, p])\n")
        assert "out_parts_2 = []" in out

    def test_fix_is_idempotent(self) -> None:
        source = HEADER + "for p in ps:\n    out = pd.concat([out, p])\n"
        once = fixed(source)
        assert fixed(once) == once

    def test_nested_loops_fix_inner_loop_only(self) -> None:
        source = HEADER + textwrap.dedent(
            """
            for group in groups:
                for p in group:
                    out = pd.concat([out, p])
            """
        )
        result = fixed(source)
        compile(result, "x", "exec")
        assert result.count(".append(") == 1
