#ifndef AIDGE_EXPORT_C_ELEMWISE_H
#define AIDGE_EXPORT_C_ELEMWISE_H

#include "kernels/c/activation.h"
#include "utils/c/typedefs.h"

static inline float export_c_elemwise_apply(float a, float b, int op)
{
    switch (op) {
    case EXPORT_C_ELEM_SUB:
        return a - b;
    case EXPORT_C_ELEM_MUL:
        return a * b;
    case EXPORT_C_ELEM_DIV:
        return a / b;
    case EXPORT_C_ELEM_ADD:
    default:
        return a + b;
    }
}

static inline void export_c_elemwise_forward(
    float *output,
    const float *input1,
    const float *input2,
    int nb_mat,
    int op,
    int input1_cont_size,
    int input2_cont_size,
    int output_cont_size,
    const size_t *offset_in1,
    const size_t *offset_in2,
    int activation)
{
    for (int stack = 0; stack < nb_mat; ++stack) {
        const size_t in1_offset = offset_in1[stack] * (size_t)input1_cont_size;
        const size_t in2_offset = offset_in2[stack] * (size_t)input2_cont_size;
        const size_t out_offset = (size_t)stack * (size_t)output_cont_size;

        for (int i = 0; i < output_cont_size; ++i) {
            const size_t in1_id = (input1_cont_size != 1) ? (size_t)i : 0;
            const size_t in2_id = (input2_cont_size != 1) ? (size_t)i : 0;
            const float value = export_c_elemwise_apply(
                input1[in1_offset + in1_id],
                input2[in2_offset + in2_id],
                op);
            output[out_offset + (size_t)i] =
                export_c_activation_value(value, activation);
        }
    }
}

static inline int32_t export_c_elemwise_apply_int8(int32_t a, int32_t b, int op)
{
    switch (op) {
    case EXPORT_C_ELEM_SUB:
        return a - b;
    case EXPORT_C_ELEM_MUL:
        return a * b;
    case EXPORT_C_ELEM_DIV:
        return a / b;
    case EXPORT_C_ELEM_ADD:
    default:
        return a + b;
    }
}

static inline void export_c_elemwise_forward_int8(
    int8_t *output,
    const int8_t *input1,
    const int8_t *input2,
    int nb_mat,
    int op,
    int input1_cont_size,
    int input2_cont_size,
    int output_cont_size,
    const size_t *offset_in1,
    const size_t *offset_in2,
    int activation,
    int coef,
    int shift)
{
    for (int stack = 0; stack < nb_mat; ++stack) {
        const size_t in1_offset = offset_in1[stack] * (size_t)input1_cont_size;
        const size_t in2_offset = offset_in2[stack] * (size_t)input2_cont_size;
        const size_t out_offset = (size_t)stack * (size_t)output_cont_size;

        for (int i = 0; i < output_cont_size; ++i) {
            const size_t in1_id = (input1_cont_size != 1) ? (size_t)i : 0;
            const size_t in2_id = (input2_cont_size != 1) ? (size_t)i : 0;
            const int32_t value = export_c_elemwise_apply_int8(
                (int32_t)input1[in1_offset + in1_id],
                (int32_t)input2[in2_offset + in2_id],
                op);
            output[out_offset + (size_t)i] =
                export_c_activation_int8(value, activation, coef, shift);
        }
    }
}

#endif /* AIDGE_EXPORT_C_ELEMWISE_H */
