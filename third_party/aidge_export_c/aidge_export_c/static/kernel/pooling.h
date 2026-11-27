#ifndef AIDGE_EXPORT_C_POOLING_H
#define AIDGE_EXPORT_C_POOLING_H

#include <float.h>
#include <math.h>

#include "kernels/c/activation.h"
#include "utils/c/typedefs.h"

static inline void export_c_pool2d_forward(
    const float *input,
    float *output,
    int batch,
    int in_h,
    int in_w,
    int in_c,
    int out_h,
    int out_w,
    int out_c,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_y,
    int pad_x,
    int pooling_type,
    int activation)
{
    for (int b = 0; b < batch; ++b) {
        const float *batch_in = input + (size_t)b * (size_t)in_h * (size_t)in_w * (size_t)in_c;
        float *batch_out = output + (size_t)b * (size_t)out_h * (size_t)out_w * (size_t)out_c;

        for (int oy = 0; oy < out_h; ++oy) {
            for (int ox = 0; ox < out_w; ++ox) {
                for (int oc = 0; oc < out_c; ++oc) {
                    if (pooling_type == EXPORT_C_POOL_AVERAGE) {
                        float sum = 0.0f;
                        for (int ky = 0; ky < kernel_h; ++ky) {
                            const int iy = oy * stride_h - pad_y + ky;
                            if (iy < 0 || iy >= in_h) {
                                continue;
                            }
                            for (int kx = 0; kx < kernel_w; ++kx) {
                                const int ix = ox * stride_w - pad_x + kx;
                                if (ix < 0 || ix >= in_w) {
                                    continue;
                                }
                                sum += batch_in[((size_t)iy * (size_t)in_w + (size_t)ix) *
                                                    (size_t)in_c +
                                                (size_t)oc];
                            }
                        }
                        const float value =
                            sum / (float)(kernel_h * kernel_w);
                        batch_out[((size_t)oy * (size_t)out_w + (size_t)ox) * (size_t)out_c +
                                  (size_t)oc] = export_c_activation_value(value, activation);
                    } else {
                        float value = -FLT_MAX;
                        for (int ky = 0; ky < kernel_h; ++ky) {
                            const int iy = oy * stride_h - pad_y + ky;
                            if (iy < 0 || iy >= in_h) {
                                continue;
                            }
                            for (int kx = 0; kx < kernel_w; ++kx) {
                                const int ix = ox * stride_w - pad_x + kx;
                                if (ix < 0 || ix >= in_w) {
                                    continue;
                                }
                                const float candidate =
                                    batch_in[((size_t)iy * (size_t)in_w + (size_t)ix) *
                                                 (size_t)in_c +
                                             (size_t)oc];
                                if (candidate > value) {
                                    value = candidate;
                                }
                            }
                        }
                        batch_out[((size_t)oy * (size_t)out_w + (size_t)ox) * (size_t)out_c +
                                  (size_t)oc] = export_c_activation_value(value, activation);
                    }
                }
            }
        }
    }
}

static inline void export_c_pool2d_forward_int8(
    const int8_t *input,
    int8_t *output,
    int batch,
    int in_h,
    int in_w,
    int in_c,
    int out_h,
    int out_w,
    int out_c,
    int kernel_h,
    int kernel_w,
    int stride_h,
    int stride_w,
    int pad_y,
    int pad_x,
    int pooling_type,
    int activation)
{
    for (int b = 0; b < batch; ++b) {
        const int8_t *batch_in =
            input + (size_t)b * (size_t)in_h * (size_t)in_w * (size_t)in_c;
        int8_t *batch_out =
            output + (size_t)b * (size_t)out_h * (size_t)out_w * (size_t)out_c;

        for (int oy = 0; oy < out_h; ++oy) {
            for (int ox = 0; ox < out_w; ++ox) {
                for (int oc = 0; oc < out_c; ++oc) {
                    if (pooling_type == EXPORT_C_POOL_AVERAGE) {
                        float sum = 0.0f;
                        for (int ky = 0; ky < kernel_h; ++ky) {
                            const int iy = oy * stride_h - pad_y + ky;
                            if (iy < 0 || iy >= in_h) {
                                continue;
                            }
                            for (int kx = 0; kx < kernel_w; ++kx) {
                                const int ix = ox * stride_w - pad_x + kx;
                                if (ix < 0 || ix >= in_w) {
                                    continue;
                                }
                                sum += (float)batch_in
                                    [((size_t)iy * (size_t)in_w + (size_t)ix) *
                                         (size_t)in_c +
                                     (size_t)oc];
                            }
                        }
                        int32_t value =
                            (int32_t)roundf(sum / (float)(kernel_h * kernel_w));
                        if (activation == EXPORT_C_ACT_RELU && value < 0) {
                            value = 0;
                        }
                        batch_out[((size_t)oy * (size_t)out_w + (size_t)ox) *
                                      (size_t)out_c +
                                  (size_t)oc] = export_c_saturate_int8(value);
                    } else {
                        int32_t value = -128;
                        for (int ky = 0; ky < kernel_h; ++ky) {
                            const int iy = oy * stride_h - pad_y + ky;
                            if (iy < 0 || iy >= in_h) {
                                continue;
                            }
                            for (int kx = 0; kx < kernel_w; ++kx) {
                                const int ix = ox * stride_w - pad_x + kx;
                                if (ix < 0 || ix >= in_w) {
                                    continue;
                                }
                                const int32_t candidate = (int32_t)batch_in
                                    [((size_t)iy * (size_t)in_w + (size_t)ix) *
                                         (size_t)in_c +
                                     (size_t)oc];
                                if (candidate > value) {
                                    value = candidate;
                                }
                            }
                        }
                        if (activation == EXPORT_C_ACT_RELU && value < 0) {
                            value = 0;
                        }
                        batch_out[((size_t)oy * (size_t)out_w + (size_t)ox) *
                                      (size_t)out_c +
                                  (size_t)oc] = export_c_saturate_int8(value);
                    }
                }
            }
        }
    }
}

#endif /* AIDGE_EXPORT_C_POOLING_H */
