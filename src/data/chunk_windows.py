import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--win_s", type=float, default=30)
    parser.add_argument("--hop_s", type=float, default=10)
    args = parser.parse_args()
    print("stub: chunk_windows", args)

if __name__ == "__main__":
    main()
