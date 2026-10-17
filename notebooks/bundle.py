"""Write notebooks/full_benchmark.bundled.ipynb: the benchmark notebook with the
openvitals wheel embedded, so a Colab runtime (e.g. via the VS Code extension)
needs no GitHub token or clone. Rebuild after code changes.

    uv run python notebooks/bundle.py
"""

import base64
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / "full_benchmark.ipynb"
OUT = ROOT / "notebooks" / "full_benchmark.bundled.ipynb"


def main() -> None:
    with tempfile.TemporaryDirectory() as d:
        subprocess.run(["uv", "build", "--wheel", "--out-dir", d, str(ROOT)], check=True,
                       capture_output=True)
        (whl,) = Path(d).glob("openvitals-*.whl")
        b64 = base64.b64encode(whl.read_bytes()).decode()
        name = whl.name
    nb = json.loads(SRC.read_text())
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    nb["cells"].insert(1, {
        "cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
        "source": [f"# Embedded by notebooks/bundle.py from commit {commit}. Don't edit.\n",
                   f"WHEEL_NAME = {name!r}\n", f"WHEEL_B64 = {b64!r}\n"],
    })
    OUT.write_text(json.dumps(nb, indent=1))
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1024:.0f} KB, {name}, commit {commit})",
          file=sys.stderr)


if __name__ == "__main__":
    main()
