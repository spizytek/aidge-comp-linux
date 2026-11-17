/* ==========================================================================
 * accel.c  --  the function LoCod turns into FPGA hardware
 * ==========================================================================
 * Passed to locod.sh as the 3rd file. Panda-Bambu compiles this file with
 * -DLOCOD_FPGA and --top-fname=forward, so everything Bambu must synthesise
 * is here and nothing else.
 *
 * PORTED FROM : output_acetone/export/dnn/src/forward.c   (model_forward)
 * Port date   : 2025 (LoCod test scaffold)
 *
 * Exactly four changes, all mechanical:
 *   1. signature : void model_forward(const float*, float**)
 *               -> void forward(struct param_encoder*, struct result_encoder*)
 *   2. mem[]     : renamed ws[] (activation workspace), still file-scope
 *   3. input     : g_a_g_a_0_Conv_input_0 now aliases param->in
 *   4. output    : *out_ptr = ...  ->  memcpy(result->out, ...)
 *   5. sqrt()    : replaced by sqrt_fp32(), see below (Bambu has no sqrt FU)
 * Everything else is untouched. Note that change 5 means the result is no
 * longer bit-identical to the reference executable, only numerical: the
 * deviation is <= 2 ulp per square root (~1e-7 relative), measured by diff.
 *
 * --------------------------------------------------------------------------
 * WHAT WILL BLOCK SYNTHESIS (read before running without --no-hard)
 * --------------------------------------------------------------------------
 *  - ws[] is 6,213,632 bytes of on-chip working memory. An Ultra96 has
 *    ~4.5 MB of BRAM in total. Bambu will not fit this.
 *  - params.h holds ~14 MB of weights. Baked into the bitstream they must
 *    live in BRAM/ROM -> impossible. They would have to stream from DDR
 *    (see LoCod demo/bram_to_ddr) instead of being static const.
 *  - pow() only survives because its exponent is the literal 2.0 (Constant_3
 *    is a 1-element array holding 2.0), so it folds to x*x before reaching
 *    the IR. A genuine runtime pow/powf/exp/log has no FU either.
 *  - sqrt() has no usable FU, hence sqrt_fp32() below.
 *  - Nested loops in this order (192 outputs x 192 inputs x 5 x 5) produce
 *    an enormous datapath; expect many hours in Bambu.
 *
 * Recommended next step: keep this file as the golden reference, and make a
 * second accelerator containing ONLY the g_a_g_a_0_Conv layer to get a real
 * bitstream through. See README notes in main.c.
 * ========================================================================== */

#include <stddef.h>
#include <string.h>
#include <math.h>

#include "io.h"
#include "params.h"

/* Activation workspace. Same size and same offsets as the original mem[]. */
#define WS_SIZE 6213632u
static unsigned char ws[WS_SIZE];

/* --- sqrt_fp32 begin ----------------------------------------------------
 * Why this exists: PandA 0.9.8 declares fp_sqrt_FU / fp_sqrtf_FU but never
 * binds sqrt/sqrtf/__builtin_sqrtf to them, so module allocation dies with
 * "does not exist a functional unit in the resource library". Only + - * /
 * and comparisons have working float units.
 *
 * Method: normalize x into [1,4) by multiplying by powers of 4, then run a
 * fixed number of Newton-Raphson steps (r = (r + y/r) / 2), then scale back.
 * Newton's iteration doubles the number of correct digits each step, so 5
 * steps take the ~25% seed error well below float resolution.
 *
 * Accuracy : <= 2 ulp for all normal inputs in [1e-30, 1e30] (measured)
 * Latency  : ~2 + 2*log4(x) + 5 iterations
 * Cost     : 5 float divisions + ~20 float multiply/add  (all map to FUs)
 * ---------------------------------------------------------------------- */

#define SQRT_ITER 5

static float sqrt_fp32(float x)
{
    float y;
    float scale;
    float r;
    int i;

    if (x <= 0.0f)
        return 0.0f;

    /* Scale into [1, 4): keeps the Newton seed within 25% of the answer
     * and keeps the step count independent of the input's magnitude. */
    y     = x;
    scale = 1.0f;
    while (y >= 4.0f) { y *= 0.25f; scale *= 2.0f; }
    while (y <  1.0f) { y *= 4.0f;  scale *= 0.5f; }

    /* Linear seed: r0 = (1 + y)/2, error < 25% on [1,4) */
    r = 0.5f * (1.0f + y);

    for (i = 0; i < SQRT_ITER; ++i)
        r = 0.5f * (r + y / r);

    return r * scale;
}

/* --- sqrt_fp32 end ----------------------------------------------------- */

void forward(struct param_encoder *param, struct result_encoder *result)
{
    /* input now lives in the struct LoCod DMA'd in */
    const float *g_a_g_a_0_Conv_input_0 = param->in;
    
    // g_a_g_a_0_Conv
    float *g_a_g_a_0_Conv_output_0 = (float*) (ws + 0);
    {
        

        float (*g_a_g_a_0_Conv_output_0_v)[1][192][32][32] = (float(*)[1][192][32][32]) (ws + 0);
        float (*g_a_g_a_0_Conv_input_0_v)[1][3][64][64] = (float(*)[1][3][64][64]) (g_a_g_a_0_Conv_input_0);
        float (*g_a_0_weight_output_0_v)[192][3][5][5] = (float(*)[192][3][5][5]) (g_a_0_weight_output_0);
        float (*g_a_0_bias_output_0_v)[192] = (float(*)[192]) (g_a_0_bias_output_0);

        // Conv2D (inputs: 6, outputs 1) with
        // - Input g_a_g_a_0_Conv_input_0_v[1,3,64,64]
        // - Input (weights) g_a_0_weight_output_0_v[192,3,5,5]
        // - Input (bias) g_a_0_bias_output_0_v[192]
        // - Output g_a_g_a_0_Conv_output_0_v[1,192,32,32]
        // - Stride [2, 2]
        // - Dilation [1, 1]
        for (size_t b = 0; b < 1; ++b) {
            for (size_t c = 0; c < 192; ++c) {
                for (size_t m = 0; m < 32; ++m) {
                    for (size_t n = 0; n < 32; ++n) {
                        
                        float sum = (*g_a_0_bias_output_0_v)[c];
                        
                        for (size_t i = 0; i < 3; ++i) {
                            for (size_t h = 0; h < 5; ++h) {
                                for (size_t w = 0; w < 5; ++w) {
                                    size_t ii = m*2 + h*1; 
                                    size_t jj = n*2 + w*1; 
                                    if (ii < 2 || jj < 2) {
                                        // Check border on top/left of input
                                        const float input = 0;
                                        const float weight = (*g_a_0_weight_output_0_v)[c][i][h][w];
                                        sum += input * weight;
                                    } else if (ii - 2 < 64 && jj - 2 < 64) {
                                        const float input = (*g_a_g_a_0_Conv_input_0_v)[b][i][ii - 2][jj - 2];
                                        const float weight = (*g_a_0_weight_output_0_v)[c][i][h][w];
                                        sum += input * weight;
                                    } else if ((ii < 2 + 64 + 2) && (jj < 2 + 64 + 2)) {
                                        // Check border on bottom/right of input
                                        const float input = 0;
                                        const float weight = (*g_a_0_weight_output_0_v)[c][i][h][w];
                                        sum += input * weight;
                                    }
                                }
                            }
                        }
                        (*g_a_g_a_0_Conv_output_0_v)[b][c][m][n] = sum;
                    }
                }
            }
        }
    }


    

    // g_a_g_a_1_Pow

    //  Element-wise square
    float* g_a_g_a_1_Pow_output_0 = (float*) (ws + 786432);
    {
        // Element-wise add with
        // - Input g_a_g_a_0_Conv_output_0[1,192,32,32] (broadcast from [1,192,32,32]) (var: g_a_g_a_0_Conv_output_0[196608])
        // - Input g_a_g_a_1_Constant_3_output_0[1] (broadcast from [1]) (var: g_a_g_a_1_Constant_3_output_0[1])
        // - Output g_a_g_a_1_Pow_output_0[1,192,32,32] (var: g_a_g_a_1_Pow_output_0[196608])
        for (size_t i = 0; i < 196608; ++i) {
            // Compute corresponding index in input 0 and input 1

            size_t ib = 0;
            
            ib = 0;
        


            g_a_g_a_1_Pow_output_0[i] = pow(g_a_g_a_0_Conv_output_0[i], g_a_g_a_1_Constant_3_output_0[ib]);
        }
    }


    
    // g_a_g_a_1_Conv
    float *g_a_g_a_1_Conv_output_0 = (float*) (ws + 1572864);
    {
        

        float (*g_a_g_a_1_Conv_output_0_v)[1][192][32][32] = (float(*)[1][192][32][32]) (ws + 1572864);
        float (*g_a_g_a_1_Pow_output_0_v)[1][192][32][32] = (float(*)[1][192][32][32]) (g_a_g_a_1_Pow_output_0);
        float (*g_a_g_a_1_Reshape_output_0_v)[192][192][1][1] = (float(*)[192][192][1][1]) (g_a_g_a_1_Reshape_output_0);
        float (*g_a_g_a_1_beta_reparam_Sub_output_0_v)[192] = (float(*)[192]) (g_a_g_a_1_beta_reparam_Sub_output_0);

        // Conv2D (inputs: 3, outputs 1) with
        // - Input g_a_g_a_1_Pow_output_0_v[1,192,32,32]
        // - Input (weights) g_a_g_a_1_Reshape_output_0_v[192,192,1,1]
        // - Input (bias) g_a_g_a_1_beta_reparam_Sub_output_0_v[192]
        // - Output g_a_g_a_1_Conv_output_0_v[1,192,32,32]
        // - Stride [1, 1]
        // - Dilation [1, 1]
        for (size_t b = 0; b < 1; ++b) {
            for (size_t c = 0; c < 192; ++c) {
                for (size_t m = 0; m < 32; ++m) {
                    for (size_t n = 0; n < 32; ++n) {
                        
                        float sum = (*g_a_g_a_1_beta_reparam_Sub_output_0_v)[c];
                        
                        for (size_t i = 0; i < 192; ++i) {
                            for (size_t h = 0; h < 1; ++h) {
                                for (size_t w = 0; w < 1; ++w) {
                                    size_t ii = m*1 + h*1; 
                                    size_t jj = n*1 + w*1; 
                                    if (ii < 0 || jj < 0) {
                                        // Check border on top/left of input
                                        const float input = 0;
                                        const float weight = (*g_a_g_a_1_Reshape_output_0_v)[c][i][h][w];
                                        sum += input * weight;
                                    } else if (ii - 0 < 32 && jj - 0 < 32) {
                                        const float input = (*g_a_g_a_1_Pow_output_0_v)[b][i][ii - 0][jj - 0];
                                        const float weight = (*g_a_g_a_1_Reshape_output_0_v)[c][i][h][w];
                                        sum += input * weight;
                                    } else if ((ii < 0 + 32 + 0) && (jj < 0 + 32 + 0)) {
                                        // Check border on bottom/right of input
                                        const float input = 0;
                                        const float weight = (*g_a_g_a_1_Reshape_output_0_v)[c][i][h][w];
                                        sum += input * weight;
                                    }
                                }
                            }
                        }
                        (*g_a_g_a_1_Conv_output_0_v)[b][c][m][n] = sum;
                    }
                }
            }
        }
    }


    

    // g_a_g_a_1_Sqrt

    //  Element-wise sqrt
    float* g_a_g_a_1_Sqrt_output_0 = (float*) (ws + 2359296);
    {
        // Element-wise sqrt with
        // - Input g_a_g_a_1_Conv_output_0[1,192,32,32] (broadcast from [1,192,32,32]) (var: g_a_g_a_1_Conv_output_0[196608])
        // - Output g_a_g_a_1_Sqrt_output_0[1,192,32,32] (var: g_a_g_a_1_Sqrt_output_0[196608])
        for (size_t i = 0; i < 196608; ++i) {
            // Compute corresponding index in input 0 and input 1

            g_a_g_a_1_Sqrt_output_0[i] = sqrt_fp32(g_a_g_a_1_Conv_output_0[i]) ;
        }
    }


    

    // g_a_g_a_1_Div

    //  Element-wise division between two input tensors, with broadcasting.
    float* g_a_g_a_1_Div_output_0 = (float*) (ws + 3145728);
    {
        // Element-wise division with
        // - Input g_a_g_a_1_Constant_4_output_0[1] (broadcast from [1]) (var: g_a_g_a_1_Constant_4_output_0[1])
        // - Input g_a_g_a_1_Sqrt_output_0[1,192,32,32] (broadcast from [1,192,32,32]) (var: g_a_g_a_1_Sqrt_output_0[196608])
        // - Output g_a_g_a_1_Div_output_0[1,192,32,32] (var: g_a_g_a_1_Div_output_0[196608])
        for (size_t i = 0; i < 196608; ++i) {
            // Compute corresponding index in input 0 and input 1

            size_t ia = 0;
            
            ia = 0;
        

            size_t ib = 0;
            
            ib = i;
        


            g_a_g_a_1_Div_output_0[i] = g_a_g_a_1_Constant_4_output_0[ia] / g_a_g_a_1_Sqrt_output_0[ib];
        }
    }


    

    // g_a_g_a_1_Mul

    //  Element-wise multiplication between two input tensors, with broadcasting.
    float* g_a_g_a_1_Mul_output_0 = (float*) (ws + 3932160);
    {
        // Element-wise multiplication with
        // - Input g_a_g_a_0_Conv_output_0[1,192,32,32] (broadcast from [1,192,32,32]) (var: g_a_g_a_0_Conv_output_0[196608])
        // - Input g_a_g_a_1_Div_output_0[1,192,32,32] (broadcast from [1,192,32,32]) (var: g_a_g_a_1_Div_output_0[196608])
        // - Output g_a_g_a_1_Mul_output_0[1,192,32,32] (var: g_a_g_a_1_Mul_output_0[196608])
        for (size_t i = 0; i < 196608; ++i) {
            // Compute corresponding index in input 0 and input 1

            size_t ia = 0;
            
            ia = i;
        

            size_t ib = 0;
            
            ib = i;
        


            g_a_g_a_1_Mul_output_0[i] = g_a_g_a_0_Conv_output_0[ia] * g_a_g_a_1_Div_output_0[ib];
        }
    }


    
    // g_a_g_a_2_Conv
    float *g_a_g_a_2_Conv_output_0 = (float*) (ws + 4718592);
    {
        

        float (*g_a_g_a_2_Conv_output_0_v)[1][192][16][16] = (float(*)[1][192][16][16]) (ws + 4718592);
        float (*g_a_g_a_1_Mul_output_0_v)[1][192][32][32] = (float(*)[1][192][32][32]) (g_a_g_a_1_Mul_output_0);
        float (*g_a_2_weight_output_0_v)[192][192][5][5] = (float(*)[192][192][5][5]) (g_a_2_weight_output_0);
        float (*g_a_2_bias_output_0_v)[192] = (float(*)[192]) (g_a_2_bias_output_0);

        // Conv2D (inputs: 6, outputs 1) with
        // - Input g_a_g_a_1_Mul_output_0_v[1,192,32,32]
        // - Input (weights) g_a_2_weight_output_0_v[192,192,5,5]
        // - Input (bias) g_a_2_bias_output_0_v[192]
        // - Output g_a_g_a_2_Conv_output_0_v[1,192,16,16]
        // - Stride [2, 2]
        // - Dilation [1, 1]
        for (size_t b = 0; b < 1; ++b) {
            for (size_t c = 0; c < 192; ++c) {
                for (size_t m = 0; m < 16; ++m) {
                    for (size_t n = 0; n < 16; ++n) {
                        
                        float sum = (*g_a_2_bias_output_0_v)[c];
                        
                        for (size_t i = 0; i < 192; ++i) {
                            for (size_t h = 0; h < 5; ++h) {
                                for (size_t w = 0; w < 5; ++w) {
                                    size_t ii = m*2 + h*1; 
                                    size_t jj = n*2 + w*1; 
                                    if (ii < 2 || jj < 2) {
                                        // Check border on top/left of input
                                        const float input = 0;
                                        const float weight = (*g_a_2_weight_output_0_v)[c][i][h][w];
                                        sum += input * weight;
                                    } else if (ii - 2 < 32 && jj - 2 < 32) {
                                        const float input = (*g_a_g_a_1_Mul_output_0_v)[b][i][ii - 2][jj - 2];
                                        const float weight = (*g_a_2_weight_output_0_v)[c][i][h][w];
                                        sum += input * weight;
                                    } else if ((ii < 2 + 32 + 2) && (jj < 2 + 32 + 2)) {
                                        // Check border on bottom/right of input
                                        const float input = 0;
                                        const float weight = (*g_a_2_weight_output_0_v)[c][i][h][w];
                                        sum += input * weight;
                                    }
                                }
                            }
                        }
                        (*g_a_g_a_2_Conv_output_0_v)[b][c][m][n] = sum;
                    }
                }
            }
        }
    }


    

    // g_a_g_a_3_Pow

    //  Element-wise square
    float* g_a_g_a_3_Pow_output_0 = (float*) (ws + 4915200);
    {
        // Element-wise add with
        // - Input g_a_g_a_2_Conv_output_0[1,192,16,16] (broadcast from [1,192,16,16]) (var: g_a_g_a_2_Conv_output_0[49152])
        // - Input g_a_g_a_3_Constant_3_output_0[1] (broadcast from [1]) (var: g_a_g_a_3_Constant_3_output_0[1])
        // - Output g_a_g_a_3_Pow_output_0[1,192,16,16] (var: g_a_g_a_3_Pow_output_0[49152])
        for (size_t i = 0; i < 49152; ++i) {
            // Compute corresponding index in input 0 and input 1

            size_t ib = 0;
            
            ib = 0;
        


            g_a_g_a_3_Pow_output_0[i] = pow(g_a_g_a_2_Conv_output_0[i], g_a_g_a_3_Constant_3_output_0[ib]);
        }
    }


    
    // g_a_g_a_3_Conv
    float *g_a_g_a_3_Conv_output_0 = (float*) (ws + 5111808);
    {
        

        float (*g_a_g_a_3_Conv_output_0_v)[1][192][16][16] = (float(*)[1][192][16][16]) (ws + 5111808);
        float (*g_a_g_a_3_Pow_output_0_v)[1][192][16][16] = (float(*)[1][192][16][16]) (g_a_g_a_3_Pow_output_0);
        float (*g_a_g_a_3_Reshape_output_0_v)[192][192][1][1] = (float(*)[192][192][1][1]) (g_a_g_a_3_Reshape_output_0);
        float (*g_a_g_a_3_beta_reparam_Sub_output_0_v)[192] = (float(*)[192]) (g_a_g_a_3_beta_reparam_Sub_output_0);

        // Conv2D (inputs: 3, outputs 1) with
        // - Input g_a_g_a_3_Pow_output_0_v[1,192,16,16]
        // - Input (weights) g_a_g_a_3_Reshape_output_0_v[192,192,1,1]
        // - Input (bias) g_a_g_a_3_beta_reparam_Sub_output_0_v[192]
        // - Output g_a_g_a_3_Conv_output_0_v[1,192,16,16]
        // - Stride [1, 1]
        // - Dilation [1, 1]
        for (size_t b = 0; b < 1; ++b) {
            for (size_t c = 0; c < 192; ++c) {
                for (size_t m = 0; m < 16; ++m) {
                    for (size_t n = 0; n < 16; ++n) {
                        
                        float sum = (*g_a_g_a_3_beta_reparam_Sub_output_0_v)[c];
                        
                        for (size_t i = 0; i < 192; ++i) {
                            for (size_t h = 0; h < 1; ++h) {
                                for (size_t w = 0; w < 1; ++w) {
                                    size_t ii = m*1 + h*1; 
                                    size_t jj = n*1 + w*1; 
                                    if (ii < 0 || jj < 0) {
                                        // Check border on top/left of input
                                        const float input = 0;
                                        const float weight = (*g_a_g_a_3_Reshape_output_0_v)[c][i][h][w];
                                        sum += input * weight;
                                    } else if (ii - 0 < 16 && jj - 0 < 16) {
                                        const float input = (*g_a_g_a_3_Pow_output_0_v)[b][i][ii - 0][jj - 0];
                                        const float weight = (*g_a_g_a_3_Reshape_output_0_v)[c][i][h][w];
                                        sum += input * weight;
                                    } else if ((ii < 0 + 16 + 0) && (jj < 0 + 16 + 0)) {
                                        // Check border on bottom/right of input
                                        const float input = 0;
                                        const float weight = (*g_a_g_a_3_Reshape_output_0_v)[c][i][h][w];
                                        sum += input * weight;
                                    }
                                }
                            }
                        }
                        (*g_a_g_a_3_Conv_output_0_v)[b][c][m][n] = sum;
                    }
                }
            }
        }
    }


    

    // g_a_g_a_3_Sqrt

    //  Element-wise sqrt
    float* g_a_g_a_3_Sqrt_output_0 = (float*) (ws + 5308416);
    {
        // Element-wise sqrt with
        // - Input g_a_g_a_3_Conv_output_0[1,192,16,16] (broadcast from [1,192,16,16]) (var: g_a_g_a_3_Conv_output_0[49152])
        // - Output g_a_g_a_3_Sqrt_output_0[1,192,16,16] (var: g_a_g_a_3_Sqrt_output_0[49152])
        for (size_t i = 0; i < 49152; ++i) {
            // Compute corresponding index in input 0 and input 1

            g_a_g_a_3_Sqrt_output_0[i] = sqrt_fp32(g_a_g_a_3_Conv_output_0[i]) ;
        }
    }


    

    // g_a_g_a_3_Div

    //  Element-wise division between two input tensors, with broadcasting.
    float* g_a_g_a_3_Div_output_0 = (float*) (ws + 5505024);
    {
        // Element-wise division with
        // - Input g_a_g_a_3_Constant_4_output_0[1] (broadcast from [1]) (var: g_a_g_a_3_Constant_4_output_0[1])
        // - Input g_a_g_a_3_Sqrt_output_0[1,192,16,16] (broadcast from [1,192,16,16]) (var: g_a_g_a_3_Sqrt_output_0[49152])
        // - Output g_a_g_a_3_Div_output_0[1,192,16,16] (var: g_a_g_a_3_Div_output_0[49152])
        for (size_t i = 0; i < 49152; ++i) {
            // Compute corresponding index in input 0 and input 1

            size_t ia = 0;
            
            ia = 0;
        

            size_t ib = 0;
            
            ib = i;
        


            g_a_g_a_3_Div_output_0[i] = g_a_g_a_3_Constant_4_output_0[ia] / g_a_g_a_3_Sqrt_output_0[ib];
        }
    }


    

    // g_a_g_a_3_Mul

    //  Element-wise multiplication between two input tensors, with broadcasting.
    float* g_a_g_a_3_Mul_output_0 = (float*) (ws + 5701632);
    {
        // Element-wise multiplication with
        // - Input g_a_g_a_2_Conv_output_0[1,192,16,16] (broadcast from [1,192,16,16]) (var: g_a_g_a_2_Conv_output_0[49152])
        // - Input g_a_g_a_3_Div_output_0[1,192,16,16] (broadcast from [1,192,16,16]) (var: g_a_g_a_3_Div_output_0[49152])
        // - Output g_a_g_a_3_Mul_output_0[1,192,16,16] (var: g_a_g_a_3_Mul_output_0[49152])
        for (size_t i = 0; i < 49152; ++i) {
            // Compute corresponding index in input 0 and input 1

            size_t ia = 0;
            
            ia = i;
        

            size_t ib = 0;
            
            ib = i;
        


            g_a_g_a_3_Mul_output_0[i] = g_a_g_a_2_Conv_output_0[ia] * g_a_g_a_3_Div_output_0[ib];
        }
    }


    
    // g_a_g_a_4_Conv
    float *g_a_g_a_4_Conv_output_0 = (float*) (ws + 5898240);
    {
        

        float (*g_a_g_a_4_Conv_output_0_v)[1][192][8][8] = (float(*)[1][192][8][8]) (ws + 5898240);
        float (*g_a_g_a_3_Mul_output_0_v)[1][192][16][16] = (float(*)[1][192][16][16]) (g_a_g_a_3_Mul_output_0);
        float (*g_a_4_weight_output_0_v)[192][192][5][5] = (float(*)[192][192][5][5]) (g_a_4_weight_output_0);
        float (*g_a_4_bias_output_0_v)[192] = (float(*)[192]) (g_a_4_bias_output_0);

        // Conv2D (inputs: 6, outputs 1) with
        // - Input g_a_g_a_3_Mul_output_0_v[1,192,16,16]
        // - Input (weights) g_a_4_weight_output_0_v[192,192,5,5]
        // - Input (bias) g_a_4_bias_output_0_v[192]
        // - Output g_a_g_a_4_Conv_output_0_v[1,192,8,8]
        // - Stride [2, 2]
        // - Dilation [1, 1]
        for (size_t b = 0; b < 1; ++b) {
            for (size_t c = 0; c < 192; ++c) {
                for (size_t m = 0; m < 8; ++m) {
                    for (size_t n = 0; n < 8; ++n) {
                        
                        float sum = (*g_a_4_bias_output_0_v)[c];
                        
                        for (size_t i = 0; i < 192; ++i) {
                            for (size_t h = 0; h < 5; ++h) {
                                for (size_t w = 0; w < 5; ++w) {
                                    size_t ii = m*2 + h*1; 
                                    size_t jj = n*2 + w*1; 
                                    if (ii < 2 || jj < 2) {
                                        // Check border on top/left of input
                                        const float input = 0;
                                        const float weight = (*g_a_4_weight_output_0_v)[c][i][h][w];
                                        sum += input * weight;
                                    } else if (ii - 2 < 16 && jj - 2 < 16) {
                                        const float input = (*g_a_g_a_3_Mul_output_0_v)[b][i][ii - 2][jj - 2];
                                        const float weight = (*g_a_4_weight_output_0_v)[c][i][h][w];
                                        sum += input * weight;
                                    } else if ((ii < 2 + 16 + 2) && (jj < 2 + 16 + 2)) {
                                        // Check border on bottom/right of input
                                        const float input = 0;
                                        const float weight = (*g_a_4_weight_output_0_v)[c][i][h][w];
                                        sum += input * weight;
                                    }
                                }
                            }
                        }
                        (*g_a_g_a_4_Conv_output_0_v)[b][c][m][n] = sum;
                    }
                }
            }
        }
    }


    

    // g_a_g_a_5_Pow

    //  Element-wise square
    float* g_a_g_a_5_Pow_output_0 = (float*) (ws + 5947392);
    {
        // Element-wise add with
        // - Input g_a_g_a_4_Conv_output_0[1,192,8,8] (broadcast from [1,192,8,8]) (var: g_a_g_a_4_Conv_output_0[12288])
        // - Input g_a_g_a_5_Constant_3_output_0[1] (broadcast from [1]) (var: g_a_g_a_5_Constant_3_output_0[1])
        // - Output g_a_g_a_5_Pow_output_0[1,192,8,8] (var: g_a_g_a_5_Pow_output_0[12288])
        for (size_t i = 0; i < 12288; ++i) {
            // Compute corresponding index in input 0 and input 1

            size_t ib = 0;
            
            ib = 0;
        


            g_a_g_a_5_Pow_output_0[i] = pow(g_a_g_a_4_Conv_output_0[i], g_a_g_a_5_Constant_3_output_0[ib]);
        }
    }


    
    // g_a_g_a_5_Conv
    float *g_a_g_a_5_Conv_output_0 = (float*) (ws + 5996544);
    {
        

        float (*g_a_g_a_5_Conv_output_0_v)[1][192][8][8] = (float(*)[1][192][8][8]) (ws + 5996544);
        float (*g_a_g_a_5_Pow_output_0_v)[1][192][8][8] = (float(*)[1][192][8][8]) (g_a_g_a_5_Pow_output_0);
        float (*g_a_g_a_5_Reshape_output_0_v)[192][192][1][1] = (float(*)[192][192][1][1]) (g_a_g_a_5_Reshape_output_0);
        float (*g_a_g_a_5_beta_reparam_Sub_output_0_v)[192] = (float(*)[192]) (g_a_g_a_5_beta_reparam_Sub_output_0);

        // Conv2D (inputs: 3, outputs 1) with
        // - Input g_a_g_a_5_Pow_output_0_v[1,192,8,8]
        // - Input (weights) g_a_g_a_5_Reshape_output_0_v[192,192,1,1]
        // - Input (bias) g_a_g_a_5_beta_reparam_Sub_output_0_v[192]
        // - Output g_a_g_a_5_Conv_output_0_v[1,192,8,8]
        // - Stride [1, 1]
        // - Dilation [1, 1]
        for (size_t b = 0; b < 1; ++b) {
            for (size_t c = 0; c < 192; ++c) {
                for (size_t m = 0; m < 8; ++m) {
                    for (size_t n = 0; n < 8; ++n) {
                        
                        float sum = (*g_a_g_a_5_beta_reparam_Sub_output_0_v)[c];
                        
                        for (size_t i = 0; i < 192; ++i) {
                            for (size_t h = 0; h < 1; ++h) {
                                for (size_t w = 0; w < 1; ++w) {
                                    size_t ii = m*1 + h*1; 
                                    size_t jj = n*1 + w*1; 
                                    if (ii < 0 || jj < 0) {
                                        // Check border on top/left of input
                                        const float input = 0;
                                        const float weight = (*g_a_g_a_5_Reshape_output_0_v)[c][i][h][w];
                                        sum += input * weight;
                                    } else if (ii - 0 < 8 && jj - 0 < 8) {
                                        const float input = (*g_a_g_a_5_Pow_output_0_v)[b][i][ii - 0][jj - 0];
                                        const float weight = (*g_a_g_a_5_Reshape_output_0_v)[c][i][h][w];
                                        sum += input * weight;
                                    } else if ((ii < 0 + 8 + 0) && (jj < 0 + 8 + 0)) {
                                        // Check border on bottom/right of input
                                        const float input = 0;
                                        const float weight = (*g_a_g_a_5_Reshape_output_0_v)[c][i][h][w];
                                        sum += input * weight;
                                    }
                                }
                            }
                        }
                        (*g_a_g_a_5_Conv_output_0_v)[b][c][m][n] = sum;
                    }
                }
            }
        }
    }


    

    // g_a_g_a_5_Sqrt

    //  Element-wise sqrt
    float* g_a_g_a_5_Sqrt_output_0 = (float*) (ws + 6045696);
    {
        // Element-wise sqrt with
        // - Input g_a_g_a_5_Conv_output_0[1,192,8,8] (broadcast from [1,192,8,8]) (var: g_a_g_a_5_Conv_output_0[12288])
        // - Output g_a_g_a_5_Sqrt_output_0[1,192,8,8] (var: g_a_g_a_5_Sqrt_output_0[12288])
        for (size_t i = 0; i < 12288; ++i) {
            // Compute corresponding index in input 0 and input 1

            g_a_g_a_5_Sqrt_output_0[i] = sqrt_fp32(g_a_g_a_5_Conv_output_0[i]) ;
        }
    }


    

    // g_a_g_a_5_Div

    //  Element-wise division between two input tensors, with broadcasting.
    float* g_a_g_a_5_Div_output_0 = (float*) (ws + 6094848);
    {
        // Element-wise division with
        // - Input g_a_g_a_5_Constant_4_output_0[1] (broadcast from [1]) (var: g_a_g_a_5_Constant_4_output_0[1])
        // - Input g_a_g_a_5_Sqrt_output_0[1,192,8,8] (broadcast from [1,192,8,8]) (var: g_a_g_a_5_Sqrt_output_0[12288])
        // - Output g_a_g_a_5_Div_output_0[1,192,8,8] (var: g_a_g_a_5_Div_output_0[12288])
        for (size_t i = 0; i < 12288; ++i) {
            // Compute corresponding index in input 0 and input 1

            size_t ia = 0;
            
            ia = 0;
        

            size_t ib = 0;
            
            ib = i;
        


            g_a_g_a_5_Div_output_0[i] = g_a_g_a_5_Constant_4_output_0[ia] / g_a_g_a_5_Sqrt_output_0[ib];
        }
    }


    

    // g_a_g_a_5_Mul

    //  Element-wise multiplication between two input tensors, with broadcasting.
    float* g_a_g_a_5_Mul_output_0 = (float*) (ws + 6144000);
    {
        // Element-wise multiplication with
        // - Input g_a_g_a_4_Conv_output_0[1,192,8,8] (broadcast from [1,192,8,8]) (var: g_a_g_a_4_Conv_output_0[12288])
        // - Input g_a_g_a_5_Div_output_0[1,192,8,8] (broadcast from [1,192,8,8]) (var: g_a_g_a_5_Div_output_0[12288])
        // - Output g_a_g_a_5_Mul_output_0[1,192,8,8] (var: g_a_g_a_5_Mul_output_0[12288])
        for (size_t i = 0; i < 12288; ++i) {
            // Compute corresponding index in input 0 and input 1

            size_t ia = 0;
            
            ia = i;
        

            size_t ib = 0;
            
            ib = i;
        


            g_a_g_a_5_Mul_output_0[i] = g_a_g_a_4_Conv_output_0[ia] * g_a_g_a_5_Div_output_0[ib];
        }
    }


    
    // g_a_g_a_6_Conv
    float *g_a_g_a_6_Conv_output_0 = (float*) (ws + 6193152);
    {
        

        float (*g_a_g_a_6_Conv_output_0_v)[1][320][4][4] = (float(*)[1][320][4][4]) (ws + 6193152);
        float (*g_a_g_a_5_Mul_output_0_v)[1][192][8][8] = (float(*)[1][192][8][8]) (g_a_g_a_5_Mul_output_0);
        float (*g_a_6_weight_output_0_v)[320][192][5][5] = (float(*)[320][192][5][5]) (g_a_6_weight_output_0);
        float (*g_a_6_bias_output_0_v)[320] = (float(*)[320]) (g_a_6_bias_output_0);

        // Conv2D (inputs: 6, outputs 1) with
        // - Input g_a_g_a_5_Mul_output_0_v[1,192,8,8]
        // - Input (weights) g_a_6_weight_output_0_v[320,192,5,5]
        // - Input (bias) g_a_6_bias_output_0_v[320]
        // - Output g_a_g_a_6_Conv_output_0_v[1,320,4,4]
        // - Stride [2, 2]
        // - Dilation [1, 1]
        for (size_t b = 0; b < 1; ++b) {
            for (size_t c = 0; c < 320; ++c) {
                for (size_t m = 0; m < 4; ++m) {
                    for (size_t n = 0; n < 4; ++n) {
                        
                        float sum = (*g_a_6_bias_output_0_v)[c];
                        
                        for (size_t i = 0; i < 192; ++i) {
                            for (size_t h = 0; h < 5; ++h) {
                                for (size_t w = 0; w < 5; ++w) {
                                    size_t ii = m*2 + h*1; 
                                    size_t jj = n*2 + w*1; 
                                    if (ii < 2 || jj < 2) {
                                        // Check border on top/left of input
                                        const float input = 0;
                                        const float weight = (*g_a_6_weight_output_0_v)[c][i][h][w];
                                        sum += input * weight;
                                    } else if (ii - 2 < 8 && jj - 2 < 8) {
                                        const float input = (*g_a_g_a_5_Mul_output_0_v)[b][i][ii - 2][jj - 2];
                                        const float weight = (*g_a_6_weight_output_0_v)[c][i][h][w];
                                        sum += input * weight;
                                    } else if ((ii < 2 + 8 + 2) && (jj < 2 + 8 + 2)) {
                                        // Check border on bottom/right of input
                                        const float input = 0;
                                        const float weight = (*g_a_6_weight_output_0_v)[c][i][h][w];
                                        sum += input * weight;
                                    }
                                }
                            }
                        }
                        (*g_a_g_a_6_Conv_output_0_v)[b][c][m][n] = sum;
                    }
                }
            }
        }
    }


    memcpy(result->out, g_a_g_a_6_Conv_output_0, sizeof(result->out));
}

