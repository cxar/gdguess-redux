class DatasetStub:
    def __init__(self):
        pass

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        raise IndexError
