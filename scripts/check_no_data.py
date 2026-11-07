"""Guard used by .githooks/pre-commit and by CI.

    python scripts/check_no_data.py <file> [<file> ...]    -> exit 1 if any file is refused

Refuses:
- anything under data/, cache/ or outputs/ (except outputs/.gitkeep)
- data file types: *.mp4, *_atd.txt, *_atd.xml, *_smart_tagging.xml, *_homography.csv,
  *_possession_gt*.csv, *.pkl, *.npz, *.npy, *.parquet, *.pdf, ...
- CSV files outside experiments/ (a submission is frame-level labels)
- any file larger than 5 MB
- Jupyter notebooks that still contain cell outputs (outputs can show real rows)

Also warns if data/ or vendor/challenge/data/ has anything besides its README. Git would not
commit it, but AI tools can still see it, so it should be moved outside the repo.

Prints file NAMES only, never contents.
"""

import json
import sys
from pathlib import Path

MAX_BYTES = 5 * 1024 * 1024
BLOCKED_DIRS = ("data", "cache", "outputs")
ALLOWED = {"outputs/.gitkeep", "data/README.md"}
BLOCKED_SUFFIXES = (
    "_atd.txt", "_atd.xml", "_smart_tagging.xml", "_homography.csv",
    ".mp4", ".mov", ".pkl", ".npz", ".npy", ".parquet", ".feather", ".h5", ".pdf", ".gzip",
    ".pl3container",
)
WATCHED_DATA_DIRS = ("data", "vendor/challenge/data")


def reason_refused(path):
    """Why `path` must not be committed, or None if it is fine. `path` is repo-relative."""
    name = path.replace("\\", "/")
    lower = name.lower()
    if name in ALLOWED:
        return None
    if name.split("/", 1)[0] in BLOCKED_DIRS:
        return "inside a data/output folder"
    if lower.endswith(BLOCKED_SUFFIXES):
        return "match-data file type"
    if "_possession_gt" in lower:
        return "ground-truth file"
    if lower.endswith(".csv") and not lower.startswith("experiments/"):
        return "CSV outside experiments/"

    file = Path(name)
    if file.is_file():
        if file.stat().st_size > MAX_BYTES:
            return "larger than 5 MB"
        if lower.endswith(".ipynb") and _notebook_has_outputs(file):
            return "notebook with outputs (clear them first)"
    return None


def _notebook_has_outputs(path):
    try:
        cells = json.loads(path.read_text()).get("cells", [])
    except (ValueError, OSError):
        return True   # unreadable notebook: refuse rather than guess
    return any(cell.get("outputs") for cell in cells)


def data_dirs_not_empty(root="."):
    """Watched data folders that hold anything besides README.md. Names of folders only."""
    found = []
    for folder in WATCHED_DATA_DIRS:
        path = Path(root) / folder
        try:
            if any(entry.name != "README.md" for entry in path.iterdir()):
                found.append(folder)
        except OSError:
            continue
    return found


def main(argv):
    refused = [(path, why) for path in argv if (why := reason_refused(path))]
    for path, why in refused:
        print(f"BLOCKED  {path}: {why}", file=sys.stderr)
    for folder in data_dirs_not_empty():
        print(f"WARNING  {folder}/ has files in it. Move the data outside the repo "
              "(CONTRIBUTING.md).", file=sys.stderr)
    if refused:
        print("Commit refused: match data must never enter this repo.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
