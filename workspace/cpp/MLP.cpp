#include "MLP.h"
#include <stdio.h>

static int8_t SaturateInt8(int32_t value) {
    if (value > 127) {
        return 127;
    }
    if (value < -128) {
        return -128;
    }
    return (int8_t)value;
}

static int8_t RequantizeRoundEven(int32_t value, int8_t scale) {
    const int32_t divisor = (int32_t)1 << scale;
    const int32_t half = divisor >> 1;
    const bool negative = value < 0;
    uint32_t magnitude = negative ? (uint32_t)(-value) : (uint32_t)value;

    uint32_t quotient = magnitude >> scale;
    uint32_t remainder = magnitude & (uint32_t)(divisor - 1);

    if (remainder > (uint32_t)half ||
        (remainder == (uint32_t)half && (quotient & 1U) != 0U)) {
        ++quotient;
    }

    int32_t rounded = negative ? -(int32_t)quotient : (int32_t)quotient;
    return SaturateInt8(rounded);
}

void FullyConnectedLayer(
    const int8_t A[],
    const int8_t B[],
    int8_t C[],
    const int32_t bias[],
    const int8_t scale,
    int K,
    int N,
    bool relu
) {
    for (int j = 0; j < N; j++) {
        #pragma HLS PIPELINE II=1

        int32_t sum = bias[j];

        for (int k = 0; k < K; k++) {
            #pragma HLS UNROLL factor=16
            sum += (int32_t)A[k] * (int32_t)B[k * N + j];
        }

        int8_t res = RequantizeRoundEven(sum, scale);

        if (relu) {
            C[j] = (res < 0) ? (int8_t)0 : res;
        } else {
            C[j] = res;
        }
    }
}

void MultilayerPerceptron(const int8_t im[], int8_t out[]) {
    #pragma HLS INTERFACE m_axi port=im bundle=gmem0 offset=slave depth=276
    #pragma HLS INTERFACE m_axi port=out bundle=gmem1 offset=slave depth=6
    #pragma HLS INTERFACE s_axilite port=im bundle=control
    #pragma HLS INTERFACE s_axilite port=out bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    int8_t data1[128];
    int8_t data2[64];
    int8_t data3[32];
    int8_t data4[6];

    FullyConnectedLayer(im, weights1, data1,
                        &bias[bias_offset[0]], scales[0], 276, 128, true);
    FullyConnectedLayer(data1, weights2, data2,
                        &bias[bias_offset[1]], scales[1], 128, 64, true);
    FullyConnectedLayer(data2, weights3, data3,
                        &bias[bias_offset[2]], scales[2], 64, 32, true);
    FullyConnectedLayer(data3, weights4, data4,
                        &bias[bias_offset[3]], scales[3], 32, 6, false);

    int8_t max_val = -128;
    int8_t argmax = 0;
    for (int j = 0; j < 6; j++) {
        #pragma HLS PIPELINE II=1
        out[j] = data4[j];
        if (data4[j] > max_val) {
            max_val = data4[j];
            argmax = (int8_t)j;
        }
    }

    out[0] = argmax;
}
