import csv
import json
import pathlib
import datetime as dt
import re
import tempfile
import traceback


def mk_exp_dir(seed: str) -> pathlib.Path:
    """Allocate a fresh experiment directory for a nonempty, sanitized seed."""
    slug = "_".join(seed.lower().split()[:3])
    slug = re.sub(r"[^a-z0-9]+", "_", slug).strip("_")[:80]
    if not slug:
        raise ValueError("Seed must contain letters or digits")
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S")
    root = pathlib.Path("experiments")
    root.mkdir(parents=True, exist_ok=True)
    return pathlib.Path(tempfile.mkdtemp(prefix=f"{slug}_{ts}_", dir=root))


class Recorder:
    def __init__(self, exp_dir: pathlib.Path):
        self.csv = open(exp_dir / "chain.csv", "w", newline="")
        self.jl = open(exp_dir / "chain.jsonl", "w")
        self.w = csv.DictWriter(self.csv, fieldnames=["cycle", "phase", "model", "text", "score"])
        self.w.writeheader()

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            print(f"Exception occurred in Recorder context: {exc_type.__name__}: {exc_val}")
            if exc_tb is not None:
                print("Traceback:")
                traceback.print_tb(exc_tb)

        self.csv.close()
        self.jl.close()

    def log(self, **row):
        self.w.writerow(row)
        self.csv.flush()
        self.jl.write(json.dumps(row) + "\n")
        self.jl.flush()
