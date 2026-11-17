#ifndef _2_FC_0_LAYER_H
#define _2_FC_0_LAYER_H

#include "utils/cpp/rescaling_utils.hpp"
#include "utils/cpp/typedefs.hpp"



// INPUT CONF
#define _1_FCACT_1_OUTPUT_0_IN_BATCH 1
#define _1_FCACT_1_OUTPUT_0_NB_CHANNELS 50
#define _1_FCACT_1_OUTPUT_0_IN_HEIGHT 1
#define _1_FCACT_1_OUTPUT_0_IN_WIDTH 1

// OUTPUT CONF
#define _2_FC_0_OUTPUT_0_NB_OUTPUTS 10
#define _2_FC_0_OUTPUT_0_OUT_HEIGHT 1
#define _2_FC_0_OUTPUT_0_OUT_WIDTH 1
#define _2_FC_0_OUTPUT_0_OUT_BATCH 1

// MEMINFO CONF





#define _2_FC_0_OUTPUT_0_MEM_SIZE 40
#define _2_FC_0_OUTPUT_0_MEM_OFFSET 0
#define _2_FC_0_OUTPUT_0_MEM_STRIDE 40
#define _2_FC_0_OUTPUT_0_MEM_LENGTH 1
#define _2_FC_0_OUTPUT_0_MEM_CONT_OFFSET 0
#define _2_FC_0_OUTPUT_0_MEM_CONT_SIZE 0
#define _2_FC_0_OUTPUT_0_MEM_WRAP_OFFSET 0
#define _2_FC_0_OUTPUT_0_MEM_WRAP_SIZE 0


#define _2_FC_0_ACTIVATION export_cpp::ActivationFunction_T::Linear

static const export_cpp::NoScaling _2_FC_0_RESCALING = {};
#define _2_FC_0_WEIGHTS_SIZE 500
#define _2_FC_0_BIASES_SIZE 10

#define _2_FC_0_OUTPUT_0_DEV_FMT export_cpp::DataFormat_T::DEFAULT

#endif /* _2_FC_0_LAYER_H */
