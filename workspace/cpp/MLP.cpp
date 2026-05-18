#include "MLP.h"
#include <stdio.h>

static int16_t RequantizeRoundEven(int32_t value, int8_t scale) {
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
    return (int16_t)rounded;
}

void FullyConnectedLayer(
    const int16_t A[],
    const int16_t B[],
    int16_t C[],
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

        int16_t res = RequantizeRoundEven(sum, scale);

        if (relu) {
            C[j] = (res < 0) ? (int16_t)0 : res;
        } else {
            C[j] = res;
        }
    }
}

void MultilayerPerceptron(const int16_t im[], int16_t out[]) {
    #pragma HLS INTERFACE m_axi port=im bundle=gmem0 offset=slave depth=276
    #pragma HLS INTERFACE m_axi port=out bundle=gmem1 offset=slave depth=6
    #pragma HLS INTERFACE s_axilite port=im bundle=control
    #pragma HLS INTERFACE s_axilite port=out bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    int16_t data1[128];
    int16_t data2[64];
    int16_t data3[32];
    int16_t data4[6];

    FullyConnectedLayer(im, weights1, data1,
                        &bias[bias_offset[0]], scales[0], 276, 128, true);
    FullyConnectedLayer(data1, weights2, data2,
                        &bias[bias_offset[1]], scales[1], 128, 64, true);
    FullyConnectedLayer(data2, weights3, data3,
                        &bias[bias_offset[2]], scales[2], 64, 32, true);
    FullyConnectedLayer(data3, weights4, data4,
                        &bias[bias_offset[3]], scales[3], 32, 6, false);

    int16_t max_val = -32768;
    int16_t argmax = 0;
    for (int j = 0; j < 6; j++) {
        #pragma HLS PIPELINE II=1
        out[j] = data4[j];
        if (data4[j] > max_val) {
            max_val = data4[j];
            argmax = (int16_t)j;
        }
    }

    out[0] = argmax;
}
