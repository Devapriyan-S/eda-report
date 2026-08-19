#!/usr/bin/env python3
"""Copy edakit into web/py/ so the browser build stays in sync."""
import json, shutil
from pathlib import Path
ROOT = Path(__file__).parent
SRC, DEST = ROOT / "edakit", ROOT / "web" / "py" / "edakit"
if DEST.exists():
    shutil.rmtree(DEST)
DEST.mkdir(parents=True)
names = []
for f in sorted(SRC.glob("*.py")):
    shutil.copy2(f, DEST / f.name)
    names.append(f.name)
(DEST.parent / "manifest.json").write_text(
    json.dumps({"package": "edakit", "files": names}, indent=2))
print(f"Copied {len(names)} modules -> {DEST}")
