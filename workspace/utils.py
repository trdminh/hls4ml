import os
import csv
import numpy as np
import tensorflow as tf

INT8_MIN = -128
INT8_MAX = 127


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def quantize_to_int(x, scale, dtype=np.int32, clip=None):
    q = np.round(x * scale)
    if clip is not None:
        q = np.clip(q, clip[0], clip[1])
    return q.astype(dtype)


def clip_int8(x):
    return np.clip(x, INT8_MIN, INT8_MAX).astype(np.int32)


def requantize_round(x_int, divisor):
    return np.round(x_int / divisor).astype(np.int32)


def relu_int(x):
    return np.maximum(x, 0).astype(np.int32)


def binary_roc_auc(y_true, y_score):
    y_true = np.asarray(y_true).astype(bool)
    y_score = np.asarray(y_score)

    positives = np.sum(y_true)
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        return np.nan

    order = np.argsort(y_score)
    sorted_scores = y_score[order]
    ranks = np.empty(len(y_score), dtype=np.float64)

    start = 0
    while start < len(y_score):
        end = start + 1
        while end < len(y_score) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end

    positive_rank_sum = np.sum(ranks[y_true])
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def weighted_classification_metrics(y_true, y_pred, y_score, num_classes):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_score = np.asarray(y_score)

    precisions = []
    recalls = []
    f1_scores = []
    auc_scores = []
    supports = []

    for class_idx in range(num_classes):
        true_positive = np.sum((y_true == class_idx) & (y_pred == class_idx))
        false_positive = np.sum((y_true != class_idx) & (y_pred == class_idx))
        false_negative = np.sum((y_true == class_idx) & (y_pred != class_idx))
        support = np.sum(y_true == class_idx)

        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive > 0
            else 0.0
        )
        recall = true_positive / support if support > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )
        auc = binary_roc_auc(y_true == class_idx, y_score[:, class_idx])

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        auc_scores.append(auc)
        supports.append(support)

    supports = np.asarray(supports, dtype=np.float64)
    valid_auc = ~np.isnan(auc_scores)

    return {
        "weighted_average_precision": float(np.average(precisions, weights=supports)),
        "weighted_average_recall": float(np.average(recalls, weights=supports)),
        "weighted_average_f1_score": float(np.average(f1_scores, weights=supports)),
        "area_under_roc_curve": float(
            np.average(np.asarray(auc_scores)[valid_auc], weights=supports[valid_auc])
        ),
    }


def dump_float_weights_to_txt_and_bin(
    model,
    save_dir="dump_results/dump_results_weights",
    input_scale=64,
    weight_scale=64,
):
    ensure_dir(save_dir)

    dense_idx = 0
    for layer in model.layers:
        if not isinstance(layer, tf.keras.layers.Dense):
            print(f"Skipping layer {layer.name}")
            continue

        weights, biases = layer.get_weights()

        weights_int = quantize_to_int(
            weights,
            weight_scale,
            dtype=np.int32,
            clip=(INT8_MIN, INT8_MAX),
        )
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

        weights_int.astype(np.int8).tofile(
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
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.path = path if os.path.isabs(path) else os.path.join(self.base_dir, path)

        self.num_classes = 6
        self.input_dim = 276
        self.dims = [276, 128, 64, 32, 6]

        self.input_scale = 64
        self.weight_scale = 64
        self.activation_scale = 64

        self.log_scales = [6, 6, 6, 6]

        self.load_data()
    def load_data(self):
        X = np.load(os.path.join(self.base_dir, "X.npy"))
        Y = np.load(os.path.join(self.base_dir, "Y.npy"))

        X_test = np.load(os.path.join(self.base_dir, "x_test.npy"))
        Y_test = np.load(os.path.join(self.base_dir, "y_test.npy"))

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

    def train_model(self, train=False, model_path="model.h5"):
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
            save_dir=os.path.join(self.base_dir, "dump_results", "dump_results_weights"),
            input_scale=self.input_scale,
            weight_scale=self.weight_scale,
        )

    def load_quant_params(self, weight_dir=None):
        if weight_dir is None:
            weight_dir = os.path.join(self.base_dir, "dump_results", "dump_results_weights")

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
        data = quantize_to_int(
            x_float,
            self.input_scale,
            dtype=np.int32,
            clip=(INT8_MIN, INT8_MAX),
        )
        layer_outputs = []
        layer_accs = []

        for layer_idx in range(4):
            acc = data @ self.weights_int[layer_idx] + self.bias_int[layer_idx]
            layer_accs.append(acc.copy())

            data = requantize_round(acc, self.weight_scale)

            # ReLU for hidden layers only.
            if layer_idx < 3:
                data = relu_int(data)

            data = clip_int8(data)
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

        if out_min < INT8_MIN or out_max > INT8_MAX:
            print("WARNING: layer output does not fit int8")
        if acc_min < -2147483648 or acc_max > 2147483647:
            print("WARNING: accumulator does not fit int32")

        return acc

    def predict_int_dataset(self, x_data):
        y_score = []
        y_pred = []

        for x in x_data:
            logits_int, _, _ = self.infer_one_int(x)
            y_score.append(logits_int)
            y_pred.append(np.argmax(logits_int))

        return np.asarray(y_pred), np.asarray(y_score)

    def build_metric_row(self, model_name, train_pred, test_pred, test_score):
        y_train_true = np.argmax(self.y_train, axis=1)
        y_test_true = np.argmax(self.y_test, axis=1)

        row = {
            "model": model_name,
            "accuracy": float(np.mean(train_pred == y_train_true)),
            "test_accuracy": float(np.mean(test_pred == y_test_true)),
        }
        row.update(
            weighted_classification_metrics(
                y_test_true,
                test_pred,
                test_score,
                self.num_classes,
            )
        )
        return row

    def write_metrics_report(self, save_path=None):
        if save_path is None:
            save_path = os.path.join(self.base_dir, "dump_results", "model_metrics.csv")

        ensure_dir(os.path.dirname(save_path))

        float_train_score = self.float_model.predict(self.x_train, verbose=0)
        float_test_score = self.float_model.predict(self.x_test, verbose=0)
        float_train_pred = np.argmax(float_train_score, axis=1)
        float_test_pred = np.argmax(float_test_score, axis=1)

        int_train_pred, _ = self.predict_int_dataset(self.x_train)
        int_test_pred, int_test_score = self.predict_int_dataset(self.x_test)

        rows = [
            self.build_metric_row(
                "float",
                float_train_pred,
                float_test_pred,
                float_test_score,
            ),
            self.build_metric_row(
                "int8",
                int_train_pred,
                int_test_pred,
                int_test_score,
            ),
        ]

        fieldnames = [
            "model",
            "accuracy",
            "test_accuracy",
            "area_under_roc_curve",
            "weighted_average_precision",
            "weighted_average_recall",
            "weighted_average_f1_score",
        ]
        with open(save_path, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"Saved metrics report: {save_path}")
        return rows

    def write_model_h(self):
        ensure_dir(self.path)

        x_sample = quantize_to_int(
            self.x_test[0],
            self.input_scale,
            dtype=np.int8,
            clip=(INT8_MIN, INT8_MAX),
        ).flatten()

        w_layers = [w.astype(np.int8).flatten() for w in self.weights_int]

        b_all = np.concatenate([b.flatten() for b in self.bias_int]).astype(np.int32)

        with open(os.path.join(self.path, "model.h"), "w") as file:
            file.write("#ifndef MODEL_H\n")
            file.write("#define MODEL_H\n\n")
            file.write("#include <stdint.h>\n\n")

            file.write(f"#define INPUT_SCALE {self.input_scale}\n")
            file.write(f"#define WEIGHT_SCALE {self.weight_scale}\n")
            file.write(f"#define ACTIVATION_SCALE {self.activation_scale}\n\n")

            for i, w in enumerate(w_layers, 1):
                file.write(f"static const int8_t weights{i}[{len(w)}] = {{")
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

            file.write(f"static const int8_t im[{len(x_sample)}] = {{")
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

    def write_test_data_h(self, max_samples=5102):
        ensure_dir(self.path)

        num_samples = min(max_samples, len(self.x_test))
        x_test_int = quantize_to_int(
            self.x_test[:num_samples],
            self.input_scale,
            dtype=np.int8,
            clip=(INT8_MIN, INT8_MAX),
        )
        y_test_int = np.argmax(self.y_test[:num_samples], axis=1).astype(np.int8)

        out_min = int(x_test_int.min())
        out_max = int(x_test_int.max())
        if out_min < INT8_MIN or out_max > INT8_MAX:
            print("WARNING: test input does not fit int8")

        test_path = os.path.join(self.path, "test_data.h")
        with open(test_path, "w") as file:
            file.write("#ifndef TEST_DATA_H\n")
            file.write("#define TEST_DATA_H\n\n")
            file.write("#include <stdint.h>\n\n")

            file.write(f"#define NUM_TEST_SAMPLES {x_test_int.shape[0]}\n")
            file.write(f"#define TEST_INPUT_DIM {x_test_int.shape[1]}\n\n")

            file.write(
                "static const int8_t x_test_data"
                f"[NUM_TEST_SAMPLES][TEST_INPUT_DIM] = {{"
            )
            for i, row in enumerate(x_test_int):
                file.write("\n    {")
                for j, val in enumerate(row):
                    if j % 15 == 0:
                        file.write("\n        ")
                    file.write(f"{int(val)}, ")
                file.write("\n    },")
            file.write("\n};\n\n")

            file.write("static const int8_t y_test_label[NUM_TEST_SAMPLES] = {")
            for i, val in enumerate(y_test_int):
                if i % 20 == 0:
                    file.write("\n    ")
                file.write(f"{int(val)}, ")
            file.write("\n};\n\n")

            file.write("#endif\n")

        print(f"Saved test data header: {test_path}")
        print(f"Test samples exported: {num_samples}")

    # def pynq_dpu(self):
    #     """
    #     Keep this only for Vitis AI quantization.
    #     Note: The manual fixed-point inference above does not use Vitis internal scales.
    #     If you use VitisQuantizer, prefer evaluating quantized_model directly.
    #     """
    #     from tensorflow_model_optimization.quantization.keras import vitis_quantize

    #     model = self.train_or_load_float_model(train=True)

    #     print(model.predict(self.x_test[0][np.newaxis, ...]))

    #     quantizer = vitis_quantize.VitisQuantizer(model)
    #     quantized_model = quantizer.quantize_model(
    #         calib_dataset=self.x_test[1:1024],
    #         weight_bit=8,
    #         activation_bit=8,
    #     )

    #     quantized_model.compile(
    #         loss="categorical_crossentropy",
    #         metrics=["accuracy"],
    #     )

    #     score = quantized_model.evaluate(
    #         self.x_test,
    #         self.y_test,
    #         verbose=0,
    #         batch_size=32,
    #     )
    #     print("Vitis quantized score:", score)

    #     quantized_model.save("model_quant.h5")

    #     quantizer.dump_model(
    #         quantized_model,
    #         dataset=self.x_test[1:1024],
    #         output_dir="./dump_results",
    #         dump_float=True,
    #     )

    #     quantizer.dump_model(
    #         quantized_model,
    #         dataset=self.x_test[1:1024],
    #         output_dir="./dump_results",
    #         dump_float=False,
    #     )


if __name__ == "__main__":
    utils_obj = Utils("cpp/")

    model = utils_obj.train_model(train=False)

    utils_obj.check_float_manual_inference()

    utils_obj.dump_quant_params_from_float()

    utils_obj.load_quant_params()

    utils_obj.evaluate_int()
    utils_obj.write_metrics_report()

    utils_obj.write_model_h()
    utils_obj.write_test_data_h(max_samples=5102)

    # Optional Vitis AI flow:
    # utils_obj.pynq_dpu()
