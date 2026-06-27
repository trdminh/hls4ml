#include <stdint.h>
#include <stdio.h>
#include "MLP.h"
#include "test_data.h"

int main() {
    int correct = 0;
    int printed_mismatches = 0;
    int8_t out[6];

    for (int i = 0; i < NUM_TEST_SAMPLES; ++i) {
        MultilayerPerceptron(x_test_data[i], out);

        const int pred = (int)out[0];
        const int label = (int)y_test_label[i];

        if (pred == label) {
            ++correct;
        } else if (printed_mismatches < 20) {
            printf("Mismatch %d: pred=%d label=%d\n", i, pred, label);
            ++printed_mismatches;
        }
    }

    const int errors = NUM_TEST_SAMPLES - correct;
    const double accuracy = (double)correct / (double)NUM_TEST_SAMPLES;

    printf("Correct: %d / %d\n", correct, NUM_TEST_SAMPLES);
    printf("Errors: %d / %d\n", errors, NUM_TEST_SAMPLES);
    printf("Accuracy: %.6f\n", accuracy);

    return 0;
}
