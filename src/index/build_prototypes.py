import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--windows", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print("stub: build_prototypes", args)

if __name__ == "__main__":
    main()
