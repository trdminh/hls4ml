from abc import ABC, abstractmethod


class HLSLayer(ABC):

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def export(self):
        pass

    def summary(self):
        print(self.name)