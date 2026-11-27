#ifndef AIDGE_EXPORT_C_TYPEDEFS_H
#define AIDGE_EXPORT_C_TYPEDEFS_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#define EXPORT_C_ACT_LINEAR 0
#define EXPORT_C_ACT_RELU 1

#define EXPORT_C_POOL_MAX 0
#define EXPORT_C_POOL_AVERAGE 1

#define EXPORT_C_ELEM_ADD 0
#define EXPORT_C_ELEM_SUB 1
#define EXPORT_C_ELEM_MUL 2
#define EXPORT_C_ELEM_DIV 3

static inline int32_t export_c_apply_scale_int8(int32_t value, int coef, int shift)
{
    if (coef == 1) {
        if (shift > 0) {
            return ((value >> (shift - 1)) + 1) >> 1;
        }
        if (shift < 0) {
            return value << (-shift);
        }
        return value;
    }

    int64_t scaled = (int64_t)value * (int64_t)coef;
    if (shift > 0) {
        scaled += ((int64_t)1 << (shift - 1));
        scaled >>= shift;
    }
    return (int32_t)scaled;
}

static inline int8_t export_c_saturate_int8(int32_t value)
{
    if (value > 127) {
        return (int8_t)127;
    }
    if (value < -128) {
        return (int8_t)-128;
    }
    return (int8_t)value;
}

static inline int8_t export_c_activation_int8(
    int32_t value,
    int activation,
    int coef,
    int shift)
{
    if (activation == EXPORT_C_ACT_RELU && value <= 0) {
        value = 0;
    }
    return export_c_saturate_int8(export_c_apply_scale_int8(value, coef, shift));
}

#endif /* AIDGE_EXPORT_C_TYPEDEFS_H */
