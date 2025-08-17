import argparse

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--path", required=True)
    args = p.parse_args()
    print("stub: serve cli", args)

if __name__ == "__main__":
    main()
