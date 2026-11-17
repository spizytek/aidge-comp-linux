#include <cstdio>
#include "forward.hpp"
#include "data/_0_FCAct_0_input_0.h"

int main()
{
    // Initialize the output arrays
    float* _2_FC_0_output_0 = nullptr;
    

    // Call the forward function
    model_forward(_0_FCAct_0_input_0, &_2_FC_0_output_0);

    // Print the results

    printf("_2_FC_0_output_0:\n");
    for (int o = 0; o < 10; ++o) {
        printf("%f ", _2_FC_0_output_0[o]);
    }
    printf("\n");
    return 0;
}
