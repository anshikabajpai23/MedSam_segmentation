import argparse
import numpy as np
import pandas as pd
from pathlib import Path

def check_file(path):
    try:
        data = np.load(path, allow_pickle=True)
        _ = data["image"]
        _ = data["mask"]
        _ = data["box"]
        return True, None
    except Exception as e:
        return False, str(e)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    bad = []
    for i, row in df.iterrows():
        path = row["npz_path"]
        ok, err = check_file(path)
        if not ok:
            bad.append((i, path, err))
            print(f"BAD: {path}")
            print(f"  error: {err}")

    print("\n====================")
    print(f"total rows: {len(df)}")
    print(f"bad rows  : {len(bad)}")

    if bad:
        out = Path("bad_npz_files.csv")
        pd.DataFrame(bad, columns=["row_idx", "npz_path", "error"]).to_csv(out, index=False)
        print(f"saved bad file list to {out.resolve()}")
    else:
        print("all npz files are readable")

if __name__ == "__main__":
    main()