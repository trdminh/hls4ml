import os
import numpy as np
import tensorflow as tf

def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def quantize_to_int(x, scale, dtype=np.int32):
    return np.round(x * scale).astype(dtype)


def requantize_round(x_int, divisor):
    return np.round(x_int / divisor).astype(np.int32)


def relu_int(x):
    return np.maximum(x, 0).astype(np.int32)


def dump_float_weights_to_txt_and_bin(
    model,
    save_dir="dump_results/dump_results_weights",
    input_scale=128,
    weight_scale=128,
):
    ensure_dir(save_dir)

    dense_idx = 0
    for layer in model.layers:
        if not isinstance(layer, tf.keras.layers.Dense):
            print(f"Skipping layer {layer.name}")
            continue

        weights, biases = layer.get_weights()

        weights_int = quantize_to_int(weights, weight_scale, dtype=np.int32)
        biases_int = quantize_to_int(biases, input_scale * weight_scale, dtype=np.int32)

        base_name = f"quant_dense_{dense_idx}"

        np.savetxt(
            os.path.join(save_dir, f"{base_name}_kernel.txt"),
            weights_int.flatten(),
            fmt="%d",
        )
        np.savetxt(
            os.path.join(save_dir, f"{base_name}_bias.txt"),
            biases_int.flatten(),
            fmt="%d",
        )

        weights_int.astype(np.int16).tofile(
            os.path.join(save_dir, f"{base_name}_kernel.bin")
        )
        biases_int.astype(np.int32).tofile(
            os.path.join(save_dir, f"{base_name}_bias.bin")
        )

        print(
            f"Saved {base_name}: "
            f"w range [{weights_int.min()}, {weights_int.max()}], "
            f"b range [{biases_int.min()}, {biases_int.max()}]"
        )
        dense_idx += 1


class Utils:
    def __init__(self, path="cpp/"):
        self.path = path

        self.num_classes = 6
        self.input_dim = 276
        self.dims = [276, 128, 64, 32, 6]

        self.input_scale = 128
        self.weight_scale = 128
        self.activation_scale = 128

        self.log_scales = [7, 7, 7, 7]

        self.load_data()
    def load_data(self):
        X = np.load("X.npy")
        Y = np.load("Y.npy")

        X_test = np.load("x_test.npy")
        Y_test = np.load("y_test.npy")

        Y = Y[:, 0] - 1
        Y_test = Y_test[:, 0] - 1
        self.x_mean = np.mean(X)
        self.x_std = np.std(X) + 1e-8

        X = (X - self.x_mean) / self.x_std
        X_test = (X_test - self.x_mean) / self.x_std

        Y = tf.keras.utils.to_categorical(Y, self.num_classes)
        Y_test = tf.keras.utils.to_categorical(Y_test, self.num_classes)

        self.x_train = X.astype(np.float32)
        self.y_train = Y.astype(np.float32)
        self.x_test = X_test.astype(np.float32)
        self.y_test = Y_test.astype(np.float32)


    def build_model(self):
        inputs = tf.keras.Input(shape=(self.input_dim,))
        x = tf.keras.layers.Dense(128, activation="relu", name="dense_0")(inputs)
        x = tf.keras.layers.Dense(64, activation="relu", name="dense_1")(x)
        x = tf.keras.layers.Dense(32, activation="relu", name="dense_2")(x)
        outputs = tf.keras.layers.Dense(self.num_classes, activation="softmax", name="dense_3")(x)

        model = tf.keras.Model(inputs, outputs)
        model.compile(
            loss="categorical_crossentropy",
            optimizer="adam",
            metrics=["accuracy"],
        )
        return model

    def train_or_load_float_model(self, train=False, model_path="model.h5"):
        if train or not os.path.exists(model_path):
            model = self.build_model()
            model.fit(
                self.x_train,
                self.y_train,
                epochs=20,
                batch_size=32,
                validation_split=0.1,
                verbose=1,
            )
            score = model.evaluate(self.x_test, self.y_test, verbose=0)
            print("Float accuracy:", score[1])
            model.save(model_path)
        else:
            model = tf.keras.models.load_model(model_path)
            score = model.evaluate(self.x_test, self.y_test, verbose=0)
            print("Loaded float accuracy:", score[1])

        self.float_model = model
        return model

    def check_float_manual_inference(self):
        model = self.float_model
        weights = model.get_weights()

        err = 0
        for i in range(len(self.x_test)):
            data = self.x_test[i]

            for layer_idx in range(4):
                w = weights[2 * layer_idx]
                b = weights[2 * layer_idx + 1]
                data = data @ w + b
                if layer_idx < 3:
                    data = np.maximum(data, 0)

            pred = np.argmax(data)
            label = np.argmax(self.y_test[i])
            if pred != label:
                err += 1

        acc = 1 - err / len(self.x_test)
        print("Manual float logits accuracy:", acc)
        return acc

    def dump_quant_params_from_float(self):
        dump_float_weights_to_txt_and_bin(
            self.float_model,
            save_dir="dump_results/dump_results_weights",
            input_scale=self.input_scale,
            weight_scale=self.weight_scale,
        )

    def load_quant_params(self, weight_dir="dump_results/dump_results_weights"):
        weights_int = []
        biases_int = []

        for i in range(4):
            w_path = os.path.join(weight_dir, f"quant_dense_{i}_kernel.txt")
            b_path = os.path.join(weight_dir, f"quant_dense_{i}_bias.txt")

            if not os.path.exists(w_path):
                raise FileNotFoundError(f"Missing weight file: {w_path}")
            if not os.path.exists(b_path):
                raise FileNotFoundError(f"Missing bias file: {b_path}")

            w_data = np.loadtxt(w_path, dtype=np.int32)
            b_data = np.loadtxt(b_path, dtype=np.int32)

            w_data = w_data.reshape(self.dims[i], self.dims[i + 1])
            b_data = b_data.reshape(self.dims[i + 1])

            weights_int.append(w_data)
            biases_int.append(b_data)

            print(
                f"Loaded layer {i}: "
                f"W {w_data.shape}, B {b_data.shape}, "
                f"W range [{w_data.min()}, {w_data.max()}], "
                f"B range [{b_data.min()}, {b_data.max()}]"
            )

        self.weights_int = weights_int
        self.bias_int = biases_int
        return weights_int, biases_int

    def infer_one_int(self, x_float):
        data = quantize_to_int(x_float, self.input_scale, dtype=np.int32)
        layer_outputs = []
        layer_accs = []

        for layer_idx in range(4):
            acc = data @ self.weights_int[layer_idx] + self.bias_int[layer_idx]
            layer_accs.append(acc.copy())

            data = requantize_round(acc, self.weight_scale)

            # ReLU for hidden layers only.
            if layer_idx < 3:
                data = relu_int(data)

            layer_outputs.append(data.copy())

        return data, layer_outputs, layer_accs

    def evaluate_int(self):
        err = 0
        all_outputs = []
        all_accs = []

        for i in range(len(self.x_test)):
            logits_int, layer_outputs, layer_accs = self.infer_one_int(self.x_test[i])

            all_outputs.extend(layer_outputs)
            all_accs.extend(layer_accs)

            pred = np.argmax(logits_int)
            label = np.argmax(self.y_test[i])
            if pred != label:
                err += 1

        acc = 1 - err / len(self.x_test)
        print("Integer fixed-point accuracy:", acc)
        print("Errors:", err, "/", len(self.x_test))

        self.out = all_outputs
        self.accs = all_accs

        out_min = min(np.min(o) for o in all_outputs)
        out_max = max(np.max(o) for o in all_outputs)
        acc_min = min(np.min(a) for a in all_accs)
        acc_max = max(np.max(a) for a in all_accs)

        print("Layer output range:", out_min, out_max)
        print("Accumulator range:", acc_min, acc_max)

        if out_min < -32768 or out_max > 32767:
            print("WARNING: layer output does not fit int16")
        if acc_min < -2147483648 or acc_max > 2147483647:
            print("WARNING: accumulator does not fit int32")

        return acc

    def write_model_h(self):
        ensure_dir(self.path)

        x_sample = quantize_to_int(
            self.x_test[0],
            self.input_scale,
            dtype=np.int16,
        ).flatten()

        w_layers = [w.astype(np.int16).flatten() for w in self.weights_int]

        b_all = np.concatenate([b.flatten() for b in self.bias_int]).astype(np.int32)

        with open(os.path.join(self.path, "model.h"), "w") as file:
            file.write("#ifndef MODEL_H\n")
            file.write("#define MODEL_H\n\n")
            file.write("#include <stdint.h>\n\n")

            file.write(f"#define INPUT_SCALE {self.input_scale}\n")
            file.write(f"#define WEIGHT_SCALE {self.weight_scale}\n")
            file.write(f"#define ACTIVATION_SCALE {self.activation_scale}\n\n")

            for i, w in enumerate(w_layers, 1):
                file.write(f"static const int16_t weights{i}[{len(w)}] = {{")
                for j, val in enumerate(w):
                    if j % 15 == 0:
                        file.write("\n    ")
                    file.write(f"{int(val)}, ")
                file.write("\n};\n\n")

            file.write(f"static const int32_t bias[{len(b_all)}] = {{")
            for j, val in enumerate(b_all):
                if j % 10 == 0:
                    file.write("\n    ")
                file.write(f"{int(val)}, ")
            file.write("\n};\n\n")

            file.write(f"static const int16_t im[{len(x_sample)}] = {{")
            for j, val in enumerate(x_sample):
                if j % 15 == 0:
                    file.write("\n    ")
                file.write(f"{int(val)}, ")
            file.write("\n};\n\n")

            file.write(
                f"static const int16_t dims[{len(self.dims)}] = "
                f"{{ {', '.join(map(str, self.dims))} }};\n"
            )
            file.write(
                f"static const int8_t scales[{len(self.log_scales)}] = "
                f"{{ {', '.join(map(str, self.log_scales))} }};\n"
            )

            file.write("\n#endif\n")

        print(f"Saved header: {os.path.join(self.path, 'model.h')}")

    def pynq_dpu(self):
        """
        Keep this only for Vitis AI quantization.
        Note: The manual fixed-point inference above does not use Vitis internal scales.
        If you use VitisQuantizer, prefer evaluating quantized_model directly.
        """
        from tensorflow_model_optimization.quantization.keras import vitis_quantize

        model = self.train_or_load_float_model(train=True)

        print(model.predict(self.x_test[0][np.newaxis, ...]))

        quantizer = vitis_quantize.VitisQuantizer(model)
        quantized_model = quantizer.quantize_model(
            calib_dataset=self.x_test[1:1024],
            weight_bit=16,
            activation_bit=16,
        )

        quantized_model.compile(
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        score = quantized_model.evaluate(
            self.x_test,
            self.y_test,
            verbose=0,
            batch_size=32,
        )
        print("Vitis quantized score:", score)

        quantized_model.save("model_quant.h5")

        quantizer.dump_model(
            quantized_model,
            dataset=self.x_test[1:1024],
            output_dir="./dump_results",
            dump_float=True,
        )

        quantizer.dump_model(
            quantized_model,
            dataset=self.x_test[1:1024],
            output_dir="./dump_results",
            dump_float=False,
        )


if __name__ == "__main__":
    utils_obj = Utils("cpp/")

    model = utils_obj.train_or_load_float_model(train=True)

    utils_obj.check_float_manual_inference()

    utils_obj.dump_quant_params_from_float()

    utils_obj.load_quant_params()

    utils_obj.evaluate_int()

    utils_obj.write_model_h()

    # Optional Vitis AI flow:
    # utils_obj.pynq_dpu()
