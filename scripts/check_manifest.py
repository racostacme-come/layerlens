"""Check that the distributed paper's recorded inputs match this checkout."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
report = json.loads((ROOT / "paper/validation.json").read_text())
for name, digest in report["inputs_sha256"].items():
    actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    if actual != digest:
        raise SystemExit(f"Stale paper input: {name}; rebuild the paper and review it.")
print(f"Verified {len(report['inputs_sha256'])} manuscript input hashes.")
