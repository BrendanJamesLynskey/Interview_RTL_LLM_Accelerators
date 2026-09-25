// =============================================================================
// softmax_golden.c — C golden model for challenge_02_softmax_testbench.sv
// =============================================================================
//
// Numerically stable softmax over n signed 16-bit inputs, returning Q0.8
// probabilities: output[i] = round(prob[i] * 256), clamped to [0, 255].
//
// Called from SystemVerilog through DPI-C. The SV import passes fixed-size
// arrays, which arrive here as plain pointers (shortint -> short, byte -> char).
// =============================================================================

#include <math.h>
#include <stdint.h>

void softmax_golden_c(const short *input, char *output, int n) {
    float exp_vals[n];
    float sum = 0.0f;
    int max_val = input[0];

    // Find max for numerical stability
    for (int i = 1; i < n; i++)
        if (input[i] > max_val) max_val = input[i];

    // Compute exp(x - max) and accumulate sum (int arithmetic: no overflow)
    for (int i = 0; i < n; i++) {
        exp_vals[i] = expf((float)(input[i] - max_val));
        sum += exp_vals[i];
    }

    // Normalise and convert to Q0.8 (round half up)
    for (int i = 0; i < n; i++) {
        float prob = exp_vals[i] / sum;
        int rounded = (int)(prob * 256.0f + 0.5f);
        if (rounded > 255) rounded = 255;
        if (rounded < 0)   rounded = 0;
        output[i] = (char)(uint8_t)rounded;
    }
}
