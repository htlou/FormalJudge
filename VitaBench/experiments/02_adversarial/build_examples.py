"""Generate L1–L4 variants as JSONL."""
from pathlib import Path
import json
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from experimentkit.attacks import make_attacks

if __name__ == "__main__":
    for row in make_attacks(json.loads((HERE / "seed.json").read_text())):
        print(json.dumps(row, ensure_ascii=False))
