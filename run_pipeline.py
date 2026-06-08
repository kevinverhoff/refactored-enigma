"""
Run the full DLGF memo pipeline in sequence.

Steps:
  1  get_docs.py           Download documents and write metadata.json
  2  ingest.py             Extract text -> documents.parquet
  3  document_clustering.py  Topic modeling -> cluster columns in parquet
  4  build_vectorstore.py  Embed and index -> chroma_db/

Usage:
    python run_pipeline.py              # run all steps
    python run_pipeline.py --from 2     # start from step 2 (e.g. docs already downloaded)
    python run_pipeline.py --only 4     # run a single step
    python run_pipeline.py --only 2 3   # run specific steps
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

STEPS = [
    (1, "get_docs.py",              "Download documents + write metadata.json"),
    (2, "ingest.py",                "Extract text -> documents.parquet"),
    (3, "document_clustering.py",   "Topic modeling -> cluster columns in parquet"),
    (4, "build_vectorstore.py",     "Embed + index -> chroma_db/"),
]


def fmt_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s"


def run_step(num: int, script: str, description: str) -> bool:
    print(f"\n{'=' * 60}")
    print(f"  Step {num}: {script}")
    print(f"  {description}")
    print(f"{'=' * 60}")

    start = time.time()
    result = subprocess.run([sys.executable, script])
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"\n  Step {num} complete ({fmt_elapsed(elapsed)})")
        return True
    else:
        print(f"\n  Step {num} FAILED (exit code {result.returncode}) after {fmt_elapsed(elapsed)}")
        print(f"  To retry from this step: python run_pipeline.py --from {num}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run the DLGF memo pipeline end-to-end",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(f"  {n}  {s}  --  {d}" for n, s, d in STEPS),
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--from", dest="from_step", type=int, metavar="N",
        help="Start from step N (skip earlier steps)"
    )
    group.add_argument(
        "--only", nargs="+", type=int, metavar="N",
        help="Run only these step numbers"
    )
    args = parser.parse_args()

    if args.only:
        invalid = [n for n in args.only if n not in {s[0] for s in STEPS}]
        if invalid:
            parser.error(f"Invalid step numbers: {invalid}. Choose from 1-{len(STEPS)}.")
        steps_to_run = [s for s in STEPS if s[0] in args.only]
    elif args.from_step:
        if not (1 <= args.from_step <= len(STEPS)):
            parser.error(f"--from must be between 1 and {len(STEPS)}.")
        steps_to_run = [s for s in STEPS if s[0] >= args.from_step]
    else:
        steps_to_run = STEPS

    print(f"Pipeline: {len(steps_to_run)} step(s) to run")
    for n, script, desc in steps_to_run:
        print(f"  {n}. {script}")

    total_start = time.time()

    for num, script, description in steps_to_run:
        if not Path(script).exists():
            print(f"\nERROR: {script} not found in current directory.")
            sys.exit(1)

        success = run_step(num, script, description)
        if not success:
            sys.exit(1)

    total = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"  All steps complete in {fmt_elapsed(total)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()