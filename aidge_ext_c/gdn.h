/* Hardware-friendly GDN kernel for aidge_export_c (NCHW, in-place safe).
 *
 * y[o,h,w] = x[o,h,w] * rsqrt(beta[o] + sum_j gamma[o,j] * x[j,h,w]^2)
 *
 * The two-pass form computes all channel norms before writing any output, so
 * the output buffer may alias the input buffer (in_place ProdConso model).
 */
#ifndef EXPORT_C_GDN_H
#define EXPORT_C_GDN_H

#include <stdint.h>

#define EXPORT_C_GDN_MAX_C 512

static inline float export_c_gdn_rsqrt(float z)
{
    if (z < 1e-12f) z = 1e-12f;
    union { float f; uint32_t i; } u;
    u.f = z;
    u.i = 0x5f3759dfu - (u.i >> 1);
    float y = u.f;
    y = y * (1.5f - 0.5f * z * y * y);
    y = y * (1.5f - 0.5f * z * y * y);
    return y;
}

static inline void export_c_gdn_forward(
    const float *in, float *out,
    const float *gamma, const float *beta,
    int channels, int height, int width)
{
    const int hw = height * width;
    float norms[EXPORT_C_GDN_MAX_C];
    for (int p = 0; p < hw; ++p) {
        for (int o = 0; o < channels; ++o) {
            const float *go = gamma + (long)o * channels;
            float norm = beta[o];
            for (int j = 0; j < channels; ++j) {
                const float v = in[(long)j * hw + p];
                norm += go[j] * v * v;
            }
            norms[o] = norm;
        }
        for (int o = 0; o < channels; ++o) {
            out[(long)o * hw + p] = in[(long)o * hw + p] *
                                    export_c_gdn_rsqrt(norms[o]);
        }
    }
}

static inline void export_c_gdn_forward_int8(
    const int8_t *in, float *out,
    const float *gamma, const float *beta,
    int channels, int height, int width, float in_scale)
{
    const int hw = height * width;
    float norms[EXPORT_C_GDN_MAX_C];
    for (int p = 0; p < hw; ++p) {
        for (int o = 0; o < channels; ++o) {
            const float *go = gamma + (long)o * channels;
            float norm = beta[o];
            for (int j = 0; j < channels; ++j) {
                const float v = (float)in[(long)j * hw + p] * in_scale;
                norm += go[j] * v * v;
            }
            norms[o] = norm;
        }
        for (int o = 0; o < channels; ++o) {
            out[(long)o * hw + p] = (float)in[(long)o * hw + p] * in_scale *
                                    export_c_gdn_rsqrt(norms[o]);
        }
    }
}

#endif /* EXPORT_C_GDN_H */
