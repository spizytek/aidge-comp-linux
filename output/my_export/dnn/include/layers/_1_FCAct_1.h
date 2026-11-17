#ifndef _1_FCACT_1_LAYER_H
#define _1_FCACT_1_LAYER_H

#include "utils/cpp/rescaling_utils.hpp"
#include "utils/cpp/typedefs.hpp"



// INPUT CONF
#define _0_FCACT_0_OUTPUT_0_IN_BATCH 1
#define _0_FCACT_0_OUTPUT_0_NB_CHANNELS 50
#define _0_FCACT_0_OUTPUT_0_IN_HEIGHT 1
#define _0_FCACT_0_OUTPUT_0_IN_WIDTH 1

// OUTPUT CONF
#define _1_FCACT_1_OUTPUT_0_NB_OUTPUTS 50
#define _1_FCACT_1_OUTPUT_0_OUT_HEIGHT 1
#define _1_FCACT_1_OUTPUT_0_OUT_WIDTH 1
#define _1_FCACT_1_OUTPUT_0_OUT_BATCH 1

// MEMINFO CONF





#define _1_FCACT_1_OUTPUT_0_MEM_SIZE 200
#define _1_FCACT_1_OUTPUT_0_MEM_OFFSET 200
#define _1_FCACT_1_OUTPUT_0_MEM_STRIDE 200
#define _1_FCACT_1_OUTPUT_0_MEM_LENGTH 1
#define _1_FCACT_1_OUTPUT_0_MEM_CONT_OFFSET 0
#define _1_FCACT_1_OUTPUT_0_MEM_CONT_SIZE 0
#define _1_FCACT_1_OUTPUT_0_MEM_WRAP_OFFSET 0
#define _1_FCACT_1_OUTPUT_0_MEM_WRAP_SIZE 0


#define _1_FCACT_1_ACTIVATION export_cpp::ActivationFunction_T::Rectifier

static const export_cpp::NoScaling _1_FCACT_1_RESCALING = {};
#define _1_FCACT_1_WEIGHTS_SIZE 2500
#define _1_FCACT_1_BIASES_SIZE 50

#define _1_FCACT_1_OUTPUT_0_DEV_FMT export_cpp::DataFormat_T::DEFAULT

#endif /* _1_FCACT_1_LAYER_H */
