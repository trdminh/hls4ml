class HLSModel:

    def __init__(self, name="model"):

        self.name = name

        self.layers = []

    def add(self, layer):

        self.layers.append(layer)

    def export(self):

        return [layer.export() for layer in self.layers]

    def summary(self):

        print("=" * 200)
        print(self.name)
        print("=" * 200)

        for layer in self.layers:
            layer.summary()