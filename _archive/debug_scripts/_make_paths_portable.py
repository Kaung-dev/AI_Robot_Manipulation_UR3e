"""
One-shot refactor: replace every '/home/user/Desktop/ur_pick' absolute path
in scripts/*.py with a PROJECT_ROOT expression computed from __file__, so
the scripts work no matter where the repo is cloned.

Run from the project root:  python3 scripts/_make_paths_portable.py
"""

import re
from pathlib import Path

SENTINEL = "/home/user/Desktop/ur_pick"
ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"

PATH_LITERAL = re.compile(
    r'(["\'])'
    + re.escape(SENTINEL)
    + r'(?:/([^"\']*))?'
    + r'\1'
)


def replace_literals(text: str) -> tuple[str, int]:
    def repl(m):
        rest = m.group(2)
        if rest:
            parts = " / ".join(f'"{p}"' for p in rest.split("/"))
            return f"str(PROJECT_ROOT / {parts})"
        return "str(PROJECT_ROOT)"

    return PATH_LITERAL.subn(repl, text)


def insert_header(text: str) -> str:
    """Insert `from pathlib import Path` (if missing) and PROJECT_ROOT after
    the last top-level import, never inside the docstring."""
    if "PROJECT_ROOT = Path(__file__).resolve().parent.parent" in text:
        return text

    lines = text.splitlines(keepends=True)
    # Find indices of import lines at module level (col 0).
    last_import_idx = -1
    has_path_import = False
    in_docstring = False
    docstring_quote = None
    for i, line in enumerate(lines):
        s = line.rstrip("\n")
        # Track docstring state — never insert inside one.
        stripped = s.strip()
        if not in_docstring:
            if stripped.startswith('"""') or stripped.startswith("'''"):
                q = stripped[:3]
                # one-liner docstring like """x"""
                if stripped.count(q) >= 2 and len(stripped) > 3:
                    continue
                in_docstring = True
                docstring_quote = q
                continue
        else:
            if docstring_quote in s:
                in_docstring = False
                docstring_quote = None
            continue

        # Top-level import?
        if s.startswith("import ") or s.startswith("from "):
            last_import_idx = i
            if "from pathlib import Path" in s or "import pathlib" in s:
                has_path_import = True

    insert_lines = []
    if not has_path_import:
        insert_lines.append("from pathlib import Path\n")
    insert_lines.append("PROJECT_ROOT = Path(__file__).resolve().parent.parent\n")
    insert_lines.append("\n")

    if last_import_idx == -1:
        # No imports at all (rare). Insert AFTER the leading docstring (if any).
        # Find first non-docstring, non-comment, non-blank line.
        in_docstring = False
        docstring_quote = None
        insert_at = 0
        for i, line in enumerate(lines):
            s = line.strip()
            if in_docstring:
                if docstring_quote in s:
                    in_docstring = False
                    docstring_quote = None
                    insert_at = i + 1
                continue
            if s.startswith('"""') or s.startswith("'''"):
                q = s[:3]
                if s.count(q) >= 2 and len(s) > 3:
                    insert_at = i + 1
                    continue
                in_docstring = True
                docstring_quote = q
                continue
            if s and not s.startswith("#"):
                insert_at = i
                break
        lines[insert_at:insert_at] = insert_lines
    else:
        lines[last_import_idx + 1 : last_import_idx + 1] = insert_lines

    return "".join(lines)


def main():
    changed = []
    for py in sorted(SCRIPTS.glob("*.py")):
        if py.name == "_make_paths_portable.py":
            continue
        text = py.read_text()
        if SENTINEL not in text:
            continue
        new, n = replace_literals(text)
        if n == 0:
            continue
        new = insert_header(new)
        py.write_text(new)
        changed.append((py.name, n))

    print(f"Refactored {len(changed)} scripts:")
    for name, n in changed:
        print(f"  {name}  ({n} path literal(s) replaced)")


if __name__ == "__main__":
    main()
