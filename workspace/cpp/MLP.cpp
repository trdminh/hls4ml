#include "MLP.h"
#include <stdio.h>

void FullyConnectedLayer(
    const int16_t A[],      // Input vector (kích thước K)
    const Vec_t B[],        // Weights Matrix (đã Pack)
    int16_t C[],            // Output vector (kích thước N)
    const int16_t bias[],   // Bias vector (kích thước N)
    const int8_t scale,     // Shift amount
    int K,                  // Số input neurons
    int N,                  // Số output neurons
    bool relu               // Kích hoạt ReLU
) {

    for (int j = 0; j < N; j++) {
        #pragma HLS PIPELINE II=1
        
        int32_t sum = 0; 


        for (int k = 0; k < K; k++) {
            #pragma HLS UNROLL factor=16 

            sum += (int32_t)A[k] * (int32_t)B[j * (K / 16) + (k / 16)][k % 16];
        }

        int32_t res = (sum + (int32_t)bias[j]) >> scale;

        if (relu) {
            C[j] = (res < 0) ? (int16_t)0 : (int16_t)res;
        } else {
            C[j] = (int16_t)res;
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

    FullyConnectedLayer(im, reinterpret_cast<Vec_t const *>(weights1), data1, 
                        &bias[bias_offset[0]], scales[0], 276, 128, true);
    #ifndef __SYNTHESIS__
    for(int i = 0; i < 128; i++){
        if(data1[i] != res_layers[i])
            printf("Layer 1 Err at %d: res=%d, exp=%d\n", i, data1[i], res_layers[i]);
    }
    #endif

    FullyConnectedLayer(data1, reinterpret_cast<Vec_t const *>(weights2), data2, 
                        &bias[bias_offset[1]], scales[1], 128, 64, true);
    #ifndef __SYNTHESIS__
    for(int i = 0; i < 64; i++){
        if(data2[i] != res_layers[i + 128])
            printf("Layer 2 Err at %d: res=%d, exp=%d\n", i, data2[i], res_layers[i + 128]);
    }
    #endif

    FullyConnectedLayer(data2, reinterpret_cast<Vec_t const *>(weights3), data3, 
                        &bias[bias_offset[2]], scales[2], 64, 32, true);
    #ifndef __SYNTHESIS__
    for(int i = 0; i < 32; i++){
        if(data3[i] != res_layers[i + 128 + 64])
            printf("Layer 3 Err at %d: res=%d, exp=%d\n", i, data3[i], res_layers[i + 128 + 64]);
    }
    #endif

    FullyConnectedLayer(data3, reinterpret_cast<Vec_t const *>(weights4), data4, 
                        &bias[bias_offset[3]], scales[3], 32, 6, false);
    #ifndef __SYNTHESIS__
    for(int i = 0; i < 6; i++){
        if(data4[i] != res_layers[i + 128 + 64 + 32])
            printf("Layer 4 Err at %d: res=%d, exp=%d\n", i, data4[i], res_layers[i + 128 + 64 + 32]);
    }
    #endif

    int16_t max_val = -32768;
    int8_t argmax = 0;
    for (int j = 0; j < 6; j++) {
        #pragma HLS PIPELINE II=1
        if (data4[j] > max_val) {
            max_val = data4[j];
            argmax = j;
        }
    }
    out[0] = argmax;
}