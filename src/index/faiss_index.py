import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protos", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print("stub: faiss_index", args)

if __name__ == "__main__":
    main()
