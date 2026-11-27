#ifndef AIDGE_EXPORT_C_FULLYCONNECTED_H
#define AIDGE_EXPORT_C_FULLYCONNECTED_H

#include "kernels/c/activation.h"
#include "utils/c/typedefs.h"

static inline void export_c_fullyconnected_forward(
    const float *input,
    float *output,
    const float *weights,
    const float *biases,
    int batch,
    int in_features,
    int out_features,
    int activation)
{
    for (int b = 0; b < batch; ++b) {
        const float *x = input + (size_t)b * (size_t)in_features;
        for (int o = 0; o < out_features; ++o) {
            const float *w = weights + (size_t)o * (size_t)in_features;
            float sum = biases != NULL ? biases[o] : 0.0f;
            for (int i = 0; i < in_features; ++i) {
                sum += x[i] * w[i];
            }
            output[(size_t)b * (size_t)out_features + (size_t)o] =
                export_c_activation_value(sum, activation);
        }
    }
}

static inline void export_c_fullyconnected_forward_int8(
    const int8_t *input,
    int8_t *output,
    const int8_t *weights,
    const int32_t *biases,
    int batch,
    int in_features,
    int out_features,
    int activation,
    int coef,
    int shift)
{
    for (int b = 0; b < batch; ++b) {
        const int8_t *x = input + (size_t)b * (size_t)in_features;
        for (int o = 0; o < out_features; ++o) {
            const int8_t *w = weights + (size_t)o * (size_t)in_features;
            int32_t sum = biases != NULL ? biases[o] : 0;
            for (int i = 0; i < in_features; ++i) {
                sum += (int32_t)x[i] * (int32_t)w[i];
            }
            output[(size_t)b * (size_t)out_features + (size_t)o] =
                export_c_activation_int8(sum, activation, coef, shift);
        }
    }
}

#endif /* AIDGE_EXPORT_C_FULLYCONNECTED_H */
