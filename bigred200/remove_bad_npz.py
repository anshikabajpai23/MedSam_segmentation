import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--bad_csv", type=str, required=True)
    parser.add_argument("--out_csv", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    bad_df = pd.read_csv(args.bad_csv)

    bad_paths = set(bad_df["npz_path"].tolist())
    clean_df = df[~df["npz_path"].isin(bad_paths)].copy()
    clean_df.to_csv(args.out_csv, index=False)

    print("original rows:", len(df))
    print("removed rows :", len(df) - len(clean_df))
    print("clean rows   :", len(clean_df))
    print("saved to     :", args.out_csv)

if __name__ == "__main__":
    main()