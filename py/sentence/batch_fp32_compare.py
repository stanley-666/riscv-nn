#!/usr/bin/env python3
"""
Batch run FP32 C vs Python logits comparison for many embeddings.

It repeatedly calls run_fp32_pipeline.py for each row, collects final logit diff,
and writes:
  - compare_out/all_diff.csv (row, c_logit, py_logit, diff)
  - compare_out/diff_hist.png (histogram of diff)

WARNING: Each row rebuilds the binary and runs Spike; full dataset will be very slow.
Use --end or --limit to sample fewer rows.

Usage examples:
  # run first 10 rows
  python tools/batch_fp32_compare.py --end 9

  # run specific range [100,199]
  python tools/batch_fp32_compare.py --start 100 --end 199
"""

import argparse
import csv
import pathlib
import re
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="py/sentence/embeddings/test_rest_int8.csv")
    ap.add_argument("--weight", default="py/sentence/weights/best_weights.pth")
    ap.add_argument("--env-sh", default="~/chipyard_1.13.0/chipyard/env.sh")
    ap.add_argument("--python-bin", default=sys.executable)
    ap.add_argument("--start", type=int, default=0, help="start row (inclusive)")
    ap.add_argument("--end", type=int, default=None, help="end row (inclusive)")
    ap.add_argument("--limit", type=int, default=None, help="run at most this many rows")
    ap.add_argument("--out-dir", default="compare_out")
    ap.add_argument("--keep-logs", action="store_true", help="keep per-row pipeline output")
    return ap.parse_args()


def count_rows(csv_path: pathlib.Path) -> int:
    with open(csv_path) as f:
        return sum(1 for _ in csv.DictReader(f))


def run_pipeline(row, csv_path, weight_path, env_sh, py_bin, log_dir):
    cmd = [
        "bash",
        "-lc",
        f"cd '{ROOT}' && PYTHON={py_bin} python tools/run_fp32_pipeline.py "
        f"--row {row} --csv '{csv_path}' --weight '{weight_path}' "
        f"--env-sh '{env_sh}' --python-bin {py_bin}",
    ]
    out = subprocess.check_output(cmd, text=True)
    if log_dir:
        (log_dir / f"row_{row}.compare.txt").write_text(out)
    c = re.search(r"C logit:\s*([0-9.eE+-]+)", out)
    p = re.search(r"Py logit:\s*([0-9.eE+-]+)", out)
    if not (c and p):
        return None
    return float(c.group(1)), float(p.group(1)), float(c.group(1)) - float(p.group(1))


def main():
    args = parse_args()
    csv_path = (ROOT / args.csv).resolve()
    weight_path = (ROOT / args.weight).resolve()
    env_sh = str(pathlib.Path(args.env_sh).expanduser())
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(exist_ok=True)
    log_dir = out_dir if args.keep_logs else None

    total_rows = count_rows(csv_path)
    start = args.start
    end = args.end if args.end is not None else total_rows - 1
    if args.limit is not None:
        end = min(end, start + args.limit - 1)
    end = min(end, total_rows - 1)

    print(f"Running rows [{start}, {end}] (total rows {total_rows})")
    rows = range(start, end + 1)

    diffs = []
    csv_out = out_dir / "all_diff.csv"
    with open(csv_out, "w", newline="") as fout:
        w = csv.writer(fout)
        w.writerow(["row", "c_logit", "py_logit", "diff"])
        for r in rows:
            print(f"== row {r} ==")
            try:
                res = run_pipeline(r, csv_path, weight_path, env_sh, args.python_bin, log_dir)
            except subprocess.CalledProcessError as e:
                print(f"  row {r} failed: {e}")
                continue
            if res is None:
                print(f"  row {r} missing logits")
                continue
            c_logit, p_logit, diff = res
            w.writerow([r, c_logit, p_logit, diff])
            diffs.append(diff)

    if not diffs:
        print("No diffs collected.")
        return

    diffs = np.array(diffs)
    print(f"samples={len(diffs)}, mean={diffs.mean():.6f}, mean_abs={np.abs(diffs).mean():.6f}, max_abs={np.abs(diffs).max():.6f}")

    plt.figure(figsize=(8, 4))
    plt.hist(diffs, bins=50, color="#4682b4", edgecolor="black")
    plt.title("C - Py logits diff")
    plt.xlabel("logit diff")
    plt.ylabel("count")
    plt.tight_layout()
    png_out = out_dir / "diff_hist.png"
    plt.savefig(png_out, dpi=150)
    print(f"saved CSV: {csv_out}")
    print(f"saved hist: {png_out}")


if __name__ == "__main__":
    main()
