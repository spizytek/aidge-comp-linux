/* GDN accelerator for the PYNQ bring-up demo.
 *
 *   y[o,h,w] = x[o,h,w] * rsqrt(beta[o] + sum_j gamma[o,j] * x[j,h,w]^2)
 *
 * 1/sqrt is a Newton iteration seeded with 0.5*(1+z); only add/mul/sub and
 * comparisons are used (no sqrtf, no division).  The test input/parameters
 * keep the norm in a range where this seed converges in 5 steps.
 *
 * Layout: NCHW, C=16, H=W=8, batch=1.  gamma/beta are compile-time constants
 * in params.h (small enough to live in the accelerator ROM for this demo).
 */
#include "io.h"
#include "params.h"

static inline float rsqrt_newton(float z)
{
    if (z < 1.0e-6f) {
        z = 1.0e-6f;
    }

    float r = 0.5f * (1.0f + z);
    for (int i = 0; i < 5; ++i) {
        float t = 3.0f - z * r * r;
        if (t < 0.0f) {
            t = 0.0f;
        }
        if (t > 3.0f) {
            t = 3.0f;
        }
        r = 0.5f * r * t;
    }
    return r;
}

void gdn(struct param_gdn *param, struct result_gdn *result)
{
    const int hw = GDN_H * GDN_W;

    for (int o = 0; o < GDN_C; ++o) {
        for (int p = 0; p < hw; ++p) {
            float norm = GDN_BETA[o];
            for (int j = 0; j < GDN_C; ++j) {
                const float v = param->in[j * hw + p];
                norm += GDN_GAMMA[o][j] * v * v;
            }
            result->out[o * hw + p] =
                param->in[o * hw + p] * rsqrt_newton(norm);
        }
    }
}
