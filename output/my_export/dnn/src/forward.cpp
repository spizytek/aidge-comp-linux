
#include <stdint.h>
#include <stddef.h>


#include "forward.hpp"

// Layer & memory configurations

#include "layers/_2_FC_0_weights.h"
#include "parameters/_2_FC_0_biases.hpp"
#include "parameters/_2_FC_0_weights.hpp"
#include "kernels/cpp/fullyconnected.hpp"
#include "layers/_2_FC_0.h"
#include "layers/_0_FCAct_0_biases.h"
#include "layers/_0_FCAct_0.h"
#include "layers/_1_FCAct_1_weights.h"
#include "parameters/_1_FCAct_1_weights.hpp"
#include "parameters/_0_FCAct_0_biases.hpp"
#include "layers/_0_FCAct_0_weights.h"
#include "layers/_1_FCAct_1_biases.h"
#include "layers/_1_FCAct_1.h"
#include "layers/_2_FC_0_biases.h"
#include "parameters/_1_FCAct_1_biases.hpp"
#include "parameters/_0_FCAct_0_weights.hpp"

// Memory block
static unsigned char mem_0[400];


void model_forward (const float* _0_FCAct_0_input_0,float** _2_FC_0_output_0_ptr)
{
    
    

    float* _0_FCAct_0_output_0 = (float*) (mem_0 + _0_FCACT_0_OUTPUT_0_MEM_OFFSET);

    export_cpp::fullyconnected_forward<_0_FCACT_0_INPUT_0_IN_BATCH,
                              _0_FCACT_0_INPUT_0_NB_CHANNELS,
                              _0_FCACT_0_INPUT_0_IN_HEIGHT,
                              _0_FCACT_0_INPUT_0_IN_WIDTH,
                              _0_FCACT_0_OUTPUT_0_NB_OUTPUTS,
                              _0_FCACT_0_OUTPUT_0_OUT_HEIGHT,
                              _0_FCACT_0_OUTPUT_0_OUT_WIDTH,
                              _0_FCACT_0_ACTIVATION,
                              _0_FCACT_0_INPUT_0_MEM_CONT_OFFSET,
                              _0_FCACT_0_INPUT_0_MEM_CONT_SIZE,
                              _0_FCACT_0_INPUT_0_MEM_WRAP_OFFSET,
                              _0_FCACT_0_INPUT_0_MEM_WRAP_SIZE,
                              _0_FCACT_0_INPUT_0_MEM_STRIDE,
                              _0_FCACT_0_OUTPUT_0_MEM_CONT_OFFSET,
                              _0_FCACT_0_OUTPUT_0_MEM_CONT_SIZE,
                              _0_FCACT_0_OUTPUT_0_MEM_WRAP_OFFSET,
                              _0_FCACT_0_OUTPUT_0_MEM_WRAP_SIZE,
                              _0_FCACT_0_OUTPUT_0_MEM_STRIDE>
                              (_0_FCAct_0_input_0, _0_FCAct_0_output_0, _0_FCAct_0_weights_output_0, _0_FCAct_0_biases_output_0, _0_FCACT_0_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _0_FCAct_0\n");
    FILE* _0_FCACT_0_OUTPUT_0_STREAM = fopen("data/export_outputs/_0_FCAct_0_output_0.txt", "w");
    export_cpp::saveOutputs<_0_FCACT_0_OUTPUT_0_NB_OUTPUTS,
                            _0_FCACT_0_OUTPUT_0_OUT_HEIGHT,
                            _0_FCACT_0_OUTPUT_0_OUT_WIDTH,
                            _0_FCACT_0_OUTPUT_0_MEM_CONT_OFFSET,
                            _0_FCACT_0_OUTPUT_0_MEM_CONT_SIZE,
                            _0_FCACT_0_OUTPUT_0_MEM_WRAP_OFFSET,
                            _0_FCACT_0_OUTPUT_0_MEM_WRAP_SIZE,
                            _0_FCACT_0_OUTPUT_0_DEV_FMT>
                            (_0_FCAct_0_output_0,
                            _0_FCACT_0_OUTPUT_0_STREAM);
    fclose(_0_FCACT_0_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _1_FCAct_1_output_0 = (float*) (mem_0 + _1_FCACT_1_OUTPUT_0_MEM_OFFSET);

    export_cpp::fullyconnected_default_forward<_0_FCACT_0_OUTPUT_0_IN_BATCH,
                              _0_FCACT_0_OUTPUT_0_NB_CHANNELS,
                              _0_FCACT_0_OUTPUT_0_IN_HEIGHT,
                              _0_FCACT_0_OUTPUT_0_IN_WIDTH,
                              _1_FCACT_1_OUTPUT_0_NB_OUTPUTS,
                              _1_FCACT_1_OUTPUT_0_OUT_HEIGHT,
                              _1_FCACT_1_OUTPUT_0_OUT_WIDTH,
                              _1_FCACT_1_ACTIVATION,
                              _0_FCACT_0_OUTPUT_0_MEM_CONT_OFFSET,
                              _0_FCACT_0_OUTPUT_0_MEM_CONT_SIZE,
                              _0_FCACT_0_OUTPUT_0_MEM_WRAP_OFFSET,
                              _0_FCACT_0_OUTPUT_0_MEM_WRAP_SIZE,
                              _0_FCACT_0_OUTPUT_0_MEM_STRIDE,
                              _1_FCACT_1_OUTPUT_0_MEM_CONT_OFFSET,
                              _1_FCACT_1_OUTPUT_0_MEM_CONT_SIZE,
                              _1_FCACT_1_OUTPUT_0_MEM_WRAP_OFFSET,
                              _1_FCACT_1_OUTPUT_0_MEM_WRAP_SIZE,
                              _1_FCACT_1_OUTPUT_0_MEM_STRIDE>
                              (_0_FCAct_0_output_0, _1_FCAct_1_output_0, _1_FCAct_1_weights_output_0, _1_FCAct_1_biases_output_0, _1_FCACT_1_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _1_FCAct_1\n");
    FILE* _1_FCACT_1_OUTPUT_0_STREAM = fopen("data/export_outputs/_1_FCAct_1_output_0.txt", "w");
    export_cpp::saveOutputs<_1_FCACT_1_OUTPUT_0_NB_OUTPUTS,
                            _1_FCACT_1_OUTPUT_0_OUT_HEIGHT,
                            _1_FCACT_1_OUTPUT_0_OUT_WIDTH,
                            _1_FCACT_1_OUTPUT_0_MEM_CONT_OFFSET,
                            _1_FCACT_1_OUTPUT_0_MEM_CONT_SIZE,
                            _1_FCACT_1_OUTPUT_0_MEM_WRAP_OFFSET,
                            _1_FCACT_1_OUTPUT_0_MEM_WRAP_SIZE,
                            _1_FCACT_1_OUTPUT_0_DEV_FMT>
                            (_1_FCAct_1_output_0,
                            _1_FCACT_1_OUTPUT_0_STREAM);
    fclose(_1_FCACT_1_OUTPUT_0_STREAM);
    #endif




    
    
    

    float* _2_FC_0_output_0 = (float*) (mem_0 + _2_FC_0_OUTPUT_0_MEM_OFFSET);

    export_cpp::fullyconnected_default_forward<_1_FCACT_1_OUTPUT_0_IN_BATCH,
                              _1_FCACT_1_OUTPUT_0_NB_CHANNELS,
                              _1_FCACT_1_OUTPUT_0_IN_HEIGHT,
                              _1_FCACT_1_OUTPUT_0_IN_WIDTH,
                              _2_FC_0_OUTPUT_0_NB_OUTPUTS,
                              _2_FC_0_OUTPUT_0_OUT_HEIGHT,
                              _2_FC_0_OUTPUT_0_OUT_WIDTH,
                              _2_FC_0_ACTIVATION,
                              _1_FCACT_1_OUTPUT_0_MEM_CONT_OFFSET,
                              _1_FCACT_1_OUTPUT_0_MEM_CONT_SIZE,
                              _1_FCACT_1_OUTPUT_0_MEM_WRAP_OFFSET,
                              _1_FCACT_1_OUTPUT_0_MEM_WRAP_SIZE,
                              _1_FCACT_1_OUTPUT_0_MEM_STRIDE,
                              _2_FC_0_OUTPUT_0_MEM_CONT_OFFSET,
                              _2_FC_0_OUTPUT_0_MEM_CONT_SIZE,
                              _2_FC_0_OUTPUT_0_MEM_WRAP_OFFSET,
                              _2_FC_0_OUTPUT_0_MEM_WRAP_SIZE,
                              _2_FC_0_OUTPUT_0_MEM_STRIDE>
                              (_1_FCAct_1_output_0, _2_FC_0_output_0, _2_FC_0_weights_output_0, _2_FC_0_biases_output_0, _2_FC_0_RESCALING);

    #ifdef SAVE_OUTPUTS
    printf("[NOTICE] - Saving outputs of node _2_FC_0\n");
    FILE* _2_FC_0_OUTPUT_0_STREAM = fopen("data/export_outputs/_2_FC_0_output_0.txt", "w");
    export_cpp::saveOutputs<_2_FC_0_OUTPUT_0_NB_OUTPUTS,
                            _2_FC_0_OUTPUT_0_OUT_HEIGHT,
                            _2_FC_0_OUTPUT_0_OUT_WIDTH,
                            _2_FC_0_OUTPUT_0_MEM_CONT_OFFSET,
                            _2_FC_0_OUTPUT_0_MEM_CONT_SIZE,
                            _2_FC_0_OUTPUT_0_MEM_WRAP_OFFSET,
                            _2_FC_0_OUTPUT_0_MEM_WRAP_SIZE,
                            _2_FC_0_OUTPUT_0_DEV_FMT>
                            (_2_FC_0_output_0,
                            _2_FC_0_OUTPUT_0_STREAM);
    fclose(_2_FC_0_OUTPUT_0_STREAM);
    #endif




    
    *_2_FC_0_output_0_ptr = _2_FC_0_output_0;
}
