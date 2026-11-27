#ifndef AIDGE_EXPORT_C_CONVOLUTION_TRANSPOSE_H
#define AIDGE_EXPORT_C_CONVOLUTION_TRANSPOSE_H

#include <stddef.h>

#include "kernels/c/activation.h"
#include "utils/c/typedefs.h"

static inline void export_c_conv2d_transpose_forward(
    const float *input,
    float *output,
    const float *weights,
    const float *biases,
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
    int dilation_h,
    int dilation_w,
    int pad_top,
    int pad_left,
    int pad_bottom,
    int pad_right,
    int activation)
{
    const int oy_min = pad_top > 0 ? pad_top : 0;
    const int oy_max = out_h - (pad_bottom > 0 ? pad_bottom : 0);
    const int ox_min = pad_left > 0 ? pad_left : 0;
    const int ox_max = out_w - (pad_right > 0 ? pad_right : 0);

    for (int b = 0; b < batch; ++b) {
        float *batch_out =
            output + (size_t)b * (size_t)out_h * (size_t)out_w * (size_t)out_c;
        const float *batch_in =
            input + (size_t)b * (size_t)in_h * (size_t)in_w * (size_t)in_c;

        for (int oy = 0; oy < out_h; ++oy) {
            for (int ox = 0; ox < out_w; ++ox) {
                const int outside =
                    oy < oy_min || oy >= oy_max || ox < ox_min || ox >= ox_max;
                for (int oc = 0; oc < out_c; ++oc) {
                    batch_out[((size_t)oy * (size_t)out_w + (size_t)ox) *
                                  (size_t)out_c +
                              (size_t)oc] =
                        outside ? 0.0f : (biases != NULL ? biases[oc] : 0.0f);
                }
            }
        }

        for (int iy = 0; iy < in_h; ++iy) {
            for (int ix = 0; ix < in_w; ++ix) {
                for (int ky = 0; ky < kernel_h; ++ky) {
                    const int oy = iy * stride_h + ky * dilation_h + pad_top;
                    if (oy < oy_min || oy >= oy_max) {
                        continue;
                    }
                    for (int kx = 0; kx < kernel_w; ++kx) {
                        const int ox = ix * stride_w + kx * dilation_w + pad_left;
                        if (ox < ox_min || ox >= ox_max) {
                            continue;
                        }

                        const float *in_pixel =
                            batch_in + ((size_t)iy * (size_t)in_w + (size_t)ix) *
                                           (size_t)in_c;
                        for (int oc = 0; oc < out_c; ++oc) {
                            float sum = 0.0f;
                            for (int ic = 0; ic < in_c; ++ic) {
                                const float w = weights
                                    [(((size_t)ic * (size_t)kernel_h + (size_t)ky) *
                                          (size_t)kernel_w +
                                      (size_t)kx) *
                                         (size_t)out_c +
                                     (size_t)oc];
                                sum += in_pixel[ic] * w;
                            }
                            batch_out[((size_t)oy * (size_t)out_w + (size_t)ox) *
                                          (size_t)out_c +
                                      (size_t)oc] += sum;
                        }
                    }
                }
            }
        }

        if (activation == EXPORT_C_ACT_RELU) {
            for (int oy = 0; oy < out_h; ++oy) {
                for (int ox = 0; ox < out_w; ++ox) {
                    for (int oc = 0; oc < out_c; ++oc) {
                        const size_t index =
                            ((size_t)oy * (size_t)out_w + (size_t)ox) *
                                (size_t)out_c +
                            (size_t)oc;
                        if (batch_out[index] < 0.0f) {
                            batch_out[index] = 0.0f;
                        }
                    }
                }
            }
        }
    }
}

#endif /* AIDGE_EXPORT_C_CONVOLUTION_TRANSPOSE_H */
