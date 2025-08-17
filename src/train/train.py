import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=False)
    args = parser.parse_args()
    print("stub: train", args)

if __name__ == "__main__":
    main()
