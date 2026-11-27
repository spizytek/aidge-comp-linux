#ifndef AIDGE_EXPORT_C_ACTIVATION_H
#define AIDGE_EXPORT_C_ACTIVATION_H

#include "utils/c/typedefs.h"

static inline float export_c_activation_value(float x, int activation)
{
    if (activation == EXPORT_C_ACT_RELU) {
        return x > 0.0f ? x : 0.0f;
    }
    return x;
}

static inline void export_c_activation_forward(
    const float *input,
    float *output,
    int size,
    int activation)
{
    for (int i = 0; i < size; ++i) {
        output[i] = export_c_activation_value(input[i], activation);
    }
}

static inline void export_c_activation_forward_int8(
    const int8_t *input,
    int8_t *output,
    int size,
    int activation,
    int coef,
    int shift)
{
    for (int i = 0; i < size; ++i) {
        output[i] = export_c_activation_int8(
            (int32_t)input[i], activation, coef, shift);
    }
}

#endif /* AIDGE_EXPORT_C_ACTIVATION_H */
