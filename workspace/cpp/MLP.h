#pragma once

#include <stdint.h>
#include "model.h"

constexpr int network_info[] = {276, 128, 64, 32, 6};

constexpr int bias_offset[] = {
    0,
    network_info[1],
    network_info[1] + network_info[2],
    network_info[1] + network_info[2] + network_info[3]
};

constexpr int n_layers = 4;

void MultilayerPerceptron(const int8_t im[276], int8_t out[6]);

void FullyConnectedLayer(
    const int8_t A[],
    const int8_t B[],
    int8_t C[],
    const int32_t bias[],
    const int8_t scale,
    int K,
    int N,
    bool relu
);
