import tensorflow as tf

from model2hls.quantizer import Quantizer
from model2hls.model.model import HLSModel
from model2hls.layers.dense import Dense
from model2hls.exporter import HLSExporter


keras_model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(276,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(6, activation="softmax"),
    ]
)

quantizer = Quantizer(
    dtype="int8",
    input_scale=64,
    weight_scale=64,
    requant_scale=6,
)

hls_model = HLSModel("MLP")

for i, layer in enumerate(keras_model.layers):
    if not isinstance(layer, tf.keras.layers.Dense):
        continue

    weights, bias = layer.get_weights()

    activation = layer.activation.__name__

    if activation == "softmax":
        activation = "linear"

    hls_model.add(
        Dense(
            name=f"dense_{i}",
            weights=weights,
            bias=bias,
            input_dim=weights.shape[0],
            output_dim=weights.shape[1],
            quantizer=quantizer,
            activation=activation,
        )
    )

hls_model.summary()

exporter = HLSExporter(
    hls_model,
    output_dir="./build",
)

exporter.export()