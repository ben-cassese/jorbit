#!/usr/bin/env python3
"""Run codespell on only the Markdown cells of Jupyter notebooks.

Pre-commit hook so notebook prose gets spell-checked without codespell
choking on base64 / rich-output payloads (codespell has no native notebook
support and treats a .ipynb as one big JSON blob). Extracted markdown is fed
to codespell via stdin, so the repo's [tool.codespell] config in pyproject.toml
(e.g. ignore-words-list) is still honoured.
"""

import json
import subprocess
import sys


def markdown_text(path: str) -> str:
    """Extract the markdown text from a Jupyter notebook."""
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)
    cells = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        src = cell.get("source", "")
        cells.append("".join(src) if isinstance(src, list) else src)
    return "\n".join(cells)


def main(paths: list[str]) -> int:
    """Run codespell on the markdown cells of the given notebook paths."""
    rc = 0
    for path in paths:
        text = markdown_text(path)
        if not text.strip():
            continue
        # codespell reads stdin via "-"; line numbers are relative to the
        # extracted markdown, not the notebook.
        result = subprocess.run(
            ["codespell", "-"], input=text, text=True, capture_output=True
        )
        if result.returncode != 0:
            rc = result.returncode
            print(f"{path} (markdown cells):")
            sys.stdout.write(result.stdout)
    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
