import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print("stub: build_manifest", args)

if __name__ == "__main__":
    main()
