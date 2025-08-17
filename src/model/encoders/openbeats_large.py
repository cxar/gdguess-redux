class OpenBEATsLargeStub:
    def __init__(self, freeze: bool = True):
        self.freeze = freeze

    def __call__(self, x):
        return x
