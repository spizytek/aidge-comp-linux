#ifndef AIDGE_EXPORT_C_CONVOLUTION_DEPTHWISE_H
#define AIDGE_EXPORT_C_CONVOLUTION_DEPTHWISE_H

#include "kernels/c/activation.h"
#include "utils/c/typedefs.h"

/* NHWC depthwise convolution with O,H,W weights. */
static inline void export_c_conv2d_depthwise_forward(
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
    int pad_y,
    int pad_x,
    int activation)
{
    for (int b = 0; b < batch; ++b) {
        const float *batch_in = input + (size_t)b * (size_t)in_h * (size_t)in_w * (size_t)in_c;
        float *batch_out = output + (size_t)b * (size_t)out_h * (size_t)out_w * (size_t)out_c;

        for (int oy = 0; oy < out_h; ++oy) {
            for (int ox = 0; ox < out_w; ++ox) {
                for (int oc = 0; oc < out_c; ++oc) {
                    const int channel = (oc * in_c) / out_c;
                    float sum = biases != NULL ? biases[oc] : 0.0f;

                    for (int ky = 0; ky < kernel_h; ++ky) {
                        const int iy = oy * stride_h - pad_y + ky * dilation_h;
                        if (iy < 0 || iy >= in_h) {
                            continue;
                        }
                        for (int kx = 0; kx < kernel_w; ++kx) {
                            const int ix = ox * stride_w - pad_x + kx * dilation_w;
                            if (ix < 0 || ix >= in_w) {
                                continue;
                            }

                            const float in_value =
                                batch_in[((size_t)iy * (size_t)in_w + (size_t)ix) *
                                             (size_t)in_c +
                                         (size_t)channel];
                            const float w_value =
                                weights[((size_t)oc * (size_t)kernel_h + (size_t)ky) *
                                            (size_t)kernel_w +
                                        (size_t)kx];
                            sum += in_value * w_value;
                        }
                    }

                    batch_out[((size_t)oy * (size_t)out_w + (size_t)ox) * (size_t)out_c +
                              (size_t)oc] = export_c_activation_value(sum, activation);
                }
            }
        }
    }
}

static inline void export_c_conv2d_depthwise_forward_int8(
    const int8_t *input,
    int8_t *output,
    const int8_t *weights,
    const int32_t *biases,
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
    int pad_y,
    int pad_x,
    int activation,
    int coef,
    int shift)
{
    for (int b = 0; b < batch; ++b) {
        const int8_t *batch_in =
            input + (size_t)b * (size_t)in_h * (size_t)in_w * (size_t)in_c;
        int8_t *batch_out =
            output + (size_t)b * (size_t)out_h * (size_t)out_w * (size_t)out_c;

        for (int oy = 0; oy < out_h; ++oy) {
            for (int ox = 0; ox < out_w; ++ox) {
                for (int oc = 0; oc < out_c; ++oc) {
                    const int channel = (oc * in_c) / out_c;
                    int32_t sum = biases != NULL ? biases[oc] : 0;

                    for (int ky = 0; ky < kernel_h; ++ky) {
                        const int iy = oy * stride_h - pad_y + ky * dilation_h;
                        if (iy < 0 || iy >= in_h) {
                            continue;
                        }
                        for (int kx = 0; kx < kernel_w; ++kx) {
                            const int ix = ox * stride_w - pad_x + kx * dilation_w;
                            if (ix < 0 || ix >= in_w) {
                                continue;
                            }
                            const int8_t in_value =
                                batch_in[((size_t)iy * (size_t)in_w + (size_t)ix) *
                                             (size_t)in_c +
                                         (size_t)channel];
                            const int8_t w_value =
                                weights[((size_t)oc * (size_t)kernel_h + (size_t)ky) *
                                            (size_t)kernel_w +
                                        (size_t)kx];
                            sum += (int32_t)in_value * (int32_t)w_value;
                        }
                    }

                    batch_out[((size_t)oy * (size_t)out_w + (size_t)ox) *
                                  (size_t)out_c +
                              (size_t)oc] =
                        export_c_activation_int8(sum, activation, coef, shift);
                }
            }
        }
    }
}

#endif /* AIDGE_EXPORT_C_CONVOLUTION_DEPTHWISE_H */
