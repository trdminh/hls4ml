import numpy as np


class Quantizer:
    dtype_info = {
        "int8": {
            "np_dtype": np.int8,
            "c_type": "int8_t",
            "min": -128,
            "max": 127,
        },
        "int16": {
            "np_dtype": np.int16,
            "c_type": "int16_t",
            "min": -32768,
            "max": 32767,
        },
    }

    def __init__(self, dtype="int8", input_scale=64, weight_scale=64, requant_scale=6):
        if dtype not in self.dtype_info:
            raise ValueError(f"Unsupported dtype: {dtype}")

        self.dtype_name = dtype
        self.np_dtype = self.dtype_info[dtype]["np_dtype"]
        self.c_type = self.dtype_info[dtype]["c_type"]
        self.min_value = self.dtype_info[dtype]["min"]
        self.max_value = self.dtype_info[dtype]["max"]

        self.input_scale = input_scale
        self.weight_scale = weight_scale
        self.requant_scale = requant_scale

    def quantize(self, x, scale=None):
        if scale is None:
            scale = self.weight_scale

        q = np.round(x * scale)
        q = np.clip(q, self.min_value, self.max_value)
        return q.astype(self.np_dtype)

    def quantize_input(self, x):
        return self.quantize(x, self.input_scale)

    def quantize_weight(self, w):
        return self.quantize(w, self.weight_scale)

    def quantize_bias(self, bias, input_scale=None, weight_scale=None):
        if input_scale is None:
            input_scale = self.input_scale
        if weight_scale is None:
            weight_scale = self.weight_scale

        q = np.round(bias * input_scale * weight_scale)
        return q.astype(np.int32)

    def get_c_type(self):
        return self.c_type

    def get_min_value(self):
        return self.min_value

    def get_max_value(self):
        return self.max_value