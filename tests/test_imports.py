"""Import hygiene for src/falafelona. No data is read here.

Our module `falafelona/io.py` shares its name with Python's built-in `io`. A bare
`from io import Match` therefore finds the built-in module and fails. Sibling modules must be
imported with the package prefix: `from falafelona.io import Match`, `from falafelona import ball`.
"""

import ast
import io as stdlib_io
import subprocess
import sys

import pytest

from falafelona import vendor

PACKAGE = vendor.REPO_ROOT / "src" / "falafelona"
MODULES = sorted(p.stem for p in PACKAGE.glob("*.py") if p.stem != "__main__")


def _bare_sibling_imports(path):
    """Lines that import a falafelona module without the `falafelona.` prefix."""
    bad = []
    for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
        if isinstance(node, ast.Import):
            for alias in node.names:
                # `import io` alone is the built-in module and is fine.
                if alias.name in MODULES and alias.name != "io":
                    bad.append((node.lineno, f"import {alias.name}"))
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module in MODULES:
            names = [a.name for a in node.names]
            # `from io import StringIO` is the built-in module; `from io import Match` is not.
            if node.module == "io" and all(hasattr(stdlib_io, n) for n in names):
                continue
            bad.append((node.lineno, f"from {node.module} import {', '.join(names)}"))
    return bad


@pytest.mark.parametrize("module", MODULES)
def test_sibling_imports_use_package_prefix(module):
    bad = _bare_sibling_imports(PACKAGE / f"{module}.py")
    hint = "; ".join(f"line {n}: `{s}`" for n, s in bad)
    assert not bad, f"{module}.py: use `from falafelona.<module> import ...` instead of {hint}"


@pytest.mark.parametrize("module", MODULES)
def test_module_imports_without_data(module):
    # Fresh interpreter with no MTAH_DATA_DIR: importing must work and must not load the match.
    env = {"PATH": "", "PYTHONPATH": str(PACKAGE.parent)}
    result = subprocess.run(
        [sys.executable, "-c", f"import falafelona.{module}"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"importing falafelona.{module} failed:\n{result.stderr[-800:]}"
