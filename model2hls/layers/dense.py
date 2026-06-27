from .base import HLSLayer


class Dense(HLSLayer):
    def __init__(
        self,
        name,
        weights,
        bias,
        input_dim,
        output_dim,
        quantizer,
        activation="relu",
    ):
        super().__init__(name)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.quantizer = quantizer

        self.weights = self.quantizer.quantize_weight(weights)
        self.bias = self.quantizer.quantize_bias(bias)

    def export(self):
        return {
            "type": "Dense",
            "name": self.name,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "activation": self.activation,
            "relu": self.activation == "relu",
            "weights": self.weights,
            "bias": self.bias,
            "c_type": self.quantizer.get_c_type(),
            "min_value": self.quantizer.get_min_value(),
            "max_value": self.quantizer.get_max_value(),
            "requant_scale": self.quantizer.requant_scale,
        }

    def summary(self):
        print(
            f"{self.name}: {self.input_dim} -> {self.output_dim} "
            f"activation={self.activation}"
        )