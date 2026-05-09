#!/usr/bin/env python3
import argparse

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Export q0~q383 CSV rows as a Gemmini baremetal dataset header")
    parser.add_argument("--csv", default="embeddings/test_rest_int8.csv")
    parser.add_argument("--out", default="../../../bareMetalC/sentence_dataset_gemmini.h")
    parser.add_argument("--limit", type=int, default=256, help="Number of rows to export; use 0 for all rows")
    parser.add_argument("--offset", type=int, default=0, help="Starting row")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    features = [f"q{i}" for i in range(384)]
    if args.offset < 0:
        raise ValueError("--offset must be non-negative")
    end = None if args.limit == 0 else args.offset + args.limit
    subset = df.iloc[args.offset:end].copy()
    x = subset[features].to_numpy(dtype=np.int8)
    y = subset["label"].to_numpy(dtype=np.int32) if "label" in subset.columns else np.zeros(len(subset), dtype=np.int32)

    with open(args.out, "w") as f:
        f.write("#ifndef SENTENCE_DATASET_GEMMINI_H\n")
        f.write("#define SENTENCE_DATASET_GEMMINI_H\n\n")
        f.write("#include <stdint.h>\n")
        f.write("#include <stdbool.h>\n\n")
        f.write(f"#define SENTENCE_DATASET_SIZE {len(subset)}\n")
        f.write("#define SENTENCE_DATASET_IN_COL_DIM 384\n\n")
        f.write("static const elem_t sentence_inputs[SENTENCE_DATASET_SIZE][1][384][1] = {\n")
        for row in x:
            vals = ",".join(f"{{{int(v)}}}" for v in row)
            f.write(f"    {{{{{vals}}}}},\n")
        f.write("};\n\n")
        f.write("static const bool sentence_labels[SENTENCE_DATASET_SIZE] = {\n    ")
        f.write(",".join("true" if int(v) != 0 else "false" for v in y))
        f.write("\n};\n\n")
        f.write("#endif // SENTENCE_DATASET_GEMMINI_H\n")

    print(f"[export] wrote {args.out}, rows={len(subset)}, offset={args.offset}")


if __name__ == "__main__":
    main()
