import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--audio", required=True)
    args = parser.parse_args()
    print("stub: streaming", args)

if __name__ == "__main__":
    main()
