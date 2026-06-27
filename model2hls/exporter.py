import os


class HLSExporter:
    def __init__(self, model, output_dir="build"):
        self.model = model
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def export(self):
        layers = self.model.export()

        dense_layers = [
            layer for layer in layers
            if layer["type"] == "Dense"
        ]

        self.write_model_h(dense_layers)
        self.write_dense_h(dense_layers)
        self.write_dense_cpp(dense_layers)

        print(f"Exported HLS files to: {self.output_dir}")

    def _write_array(self, f, c_type, name, data, per_line=16):
        flat = data.flatten()

        f.write(f"static const {c_type} {name}[{len(flat)}] = {{\n")

        for i, value in enumerate(flat):
            if i % per_line == 0:
                f.write("    ")
            f.write(f"{int(value)}")
            if i != len(flat) - 1:
                f.write(", ")
            if (i + 1) % per_line == 0:
                f.write("\n")

        if len(flat) % per_line != 0:
            f.write("\n")

        f.write("};\n\n")

    def write_model_h(self, layers):
        path = os.path.join(self.output_dir, "model.h")

        with open(path, "w", encoding="utf-8") as f:
            f.write("#pragma once\n\n")
            f.write("#include <stdint.h>\n\n")

            network_info = [layers[0]["input_dim"]]
            for layer in layers:
                network_info.append(layer["output_dim"])

            f.write(f"static const int n_layers = {len(layers)};\n\n")

            f.write(
                "static const int network_info[] = {"
                + ", ".join(map(str, network_info))
                + "};\n\n"
            )

            offsets = []
            current = 0
            for layer in layers:
                offsets.append(current)
                current += layer["output_dim"]

            f.write(
                "static const int bias_offset[] = {"
                + ", ".join(map(str, offsets))
                + "};\n\n"
            )

            scales = [layer["requant_scale"] for layer in layers]
            f.write(
                "static const int8_t scales[] = {"
                + ", ".join(map(str, scales))
                + "};\n\n"
            )

            for idx, layer in enumerate(layers, start=1):
                self._write_array(
                    f,
                    layer["c_type"],
                    f"weights{idx}",
                    layer["weights"],
                )

            all_bias = []
            for layer in layers:
                all_bias.extend(layer["bias"].flatten())

            f.write(f"static const int32_t bias[{len(all_bias)}] = {{\n")
            for i, value in enumerate(all_bias):
                if i % 12 == 0:
                    f.write("    ")
                f.write(f"{int(value)}")
                if i != len(all_bias) - 1:
                    f.write(", ")
                if (i + 1) % 12 == 0:
                    f.write("\n")
            if len(all_bias) % 12 != 0:
                f.write("\n")
            f.write("};\n")

    def write_dense_h(self, layers):
        path = os.path.join(self.output_dir, "dense.h")

        c_type = layers[0]["c_type"]

        with open(path, "w", encoding="utf-8") as f:
            f.write("#pragma once\n\n")
            f.write("#include <stdint.h>\n")
            f.write('#include "model.h"\n\n')

            f.write(
                "void FullyConnectedLayer(\n"
                f"    const {c_type} A[],\n"
                f"    const {c_type} B[],\n"
                f"    {c_type} C[],\n"
                "    const int32_t bias[],\n"
                "    const int8_t scale,\n"
                "    int K,\n"
                "    int N,\n"
                "    bool relu\n"
                ");\n"
            )

    def write_dense_cpp(self, layers):
        path = os.path.join(self.output_dir, "dense.cpp")

        c_type = layers[0]["c_type"]
        min_value = layers[0]["min_value"]
        max_value = layers[0]["max_value"]

        with open(path, "w", encoding="utf-8") as f:
            f.write('#include "dense.h"\n\n')
            f.write("#include <stdint.h>\n")
            f.write("#include <stdbool.h>\n\n")

            f.write(
                f"static {c_type} SaturateValue(int32_t value) {{\n"
                f"    if (value > {max_value}) {{\n"
                f"        return {max_value};\n"
                "    }\n"
                f"    if (value < {min_value}) {{\n"
                f"        return {min_value};\n"
                "    }\n"
                f"    return ({c_type})value;\n"
                "}\n\n"
            )

            f.write(
                f"static {c_type} RequantizeRoundEven(int32_t value, int8_t scale) {{\n"
                "    const int32_t divisor = (int32_t)1 << scale;\n"
                "    const int32_t half = divisor >> 1;\n"
                "    const bool negative = value < 0;\n"
                "    uint32_t magnitude = negative ? (uint32_t)(-value) : (uint32_t)value;\n\n"
                "    uint32_t quotient = magnitude >> scale;\n"
                "    uint32_t remainder = magnitude & (uint32_t)(divisor - 1);\n\n"
                "    if (remainder > (uint32_t)half ||\n"
                "        (remainder == (uint32_t)half && (quotient & 1U) != 0U)) {\n"
                "        ++quotient;\n"
                "    }\n\n"
                "    int32_t rounded = negative ? -(int32_t)quotient : (int32_t)quotient;\n"
                "    return SaturateValue(rounded);\n"
                "}\n\n"
            )

            f.write(
                "void FullyConnectedLayer(\n"
                f"    const {c_type} A[],\n"
                f"    const {c_type} B[],\n"
                f"    {c_type} C[],\n"
                "    const int32_t bias[],\n"
                "    const int8_t scale,\n"
                "    int K,\n"
                "    int N,\n"
                "    bool relu\n"
                ") {\n"
                "    for (int j = 0; j < N; j++) {\n"
                "        #pragma HLS PIPELINE II=1\n\n"
                "        int32_t sum = bias[j];\n\n"
                "        for (int k = 0; k < K; k++) {\n"
                "            #pragma HLS UNROLL factor=16\n"
                "            sum += (int32_t)A[k] * (int32_t)B[k * N + j];\n"
                "        }\n\n"
                "        auto res = RequantizeRoundEven(sum, scale);\n\n"
                "        if (relu) {\n"
                f"            C[j] = (res < 0) ? ({c_type})0 : res;\n"
                "        } else {\n"
                "            C[j] = res;\n"
                "        }\n"
                "    }\n"
                "}\n"
            )