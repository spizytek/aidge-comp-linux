/* ==========================================================================
 * main.c  --  CPU driver for the LoCod encoder-model test
 * ==========================================================================
 * This file MUST be the FIRST argument after -f in locod.sh.
 * Reason: locod.sh sets  MAIN_FILE="${FILES[0]}"  and only scans that one
 * file for the init_locod and FPGA calls. Anything in the other three files
 * is invisible to the accelerator generator.
 *
 * Two hard rules, both enforced by grep in locod.sh, not by a C parser:
 *
 *   1. The init_locod argument must be a bare integer literal.
 *   2. Each FPGA call must sit on ONE line with a literal accelerator index.
 *
 *   - Wrapping either call across lines  -> accelerator silently missing
 *   - Using #define or a variable         -> garbage in locod_top.vhd
 *
 * WARNING: do not write the call syntax in comments or #if 0 blocks.
 * locod.sh greps the raw text of this file, so a commented-out example is
 * counted as a real accelerator. Keep the words "FPGA" and "init_locod"
 * separated from their "(" here.
 *
 * Everything that cannot be synthesized is inside #ifndef LOCOD_FPGA,
 * because Panda-Bambu compiles THIS file too (with -DLOCOD_FPGA).
 * ========================================================================== */

#include <stddef.h>
#include <stdio.h>

#ifndef LOCOD_FPGA
/* Inside a locod.sh build, locod.h is copied into the flat build directory
 * and is picked up here -> the real hardware API.
 * On a plain desktop build it does not exist, so we fall back to the
 * software shim and this becomes an ordinary CPU program. */
#  if defined(__has_include) && __has_include("locod.h")
#    include "locod.h"
#  else
#    include "locod_host.h"
#  endif
#endif

#include "io.h"

#ifndef LOCOD_FPGA

int main(void)
{
    /* static: 48 KB + 20 KB, keep them off the stack (the board's default
     * pthread stack is only 8 MB and the DMA buffer must be plain memory) */
    static struct param_encoder  p;
    static struct result_encoder r;

    /* Load the golden test vector (identical to reference bin/run_export) */
    for (size_t i = 0; i < ENC_IN_SIZE; ++i)
        p.in[i] = g_a_g_a_0_Conv_input_0[i];

    /* One hardware accelerator will be instantiated at index 0 */
    init_locod(1);

    /* Fire-and-forget: copies param+result into accelerator memory and
     * starts accel 0. Does NOT block, does NOT return anything. */
    FPGA(forward, &p, &r, 0);

    /* Results are only valid after this returns */
    wait_accelerator(&r, 0);

    /* Same format as the reference executable, so the two outputs can be
     * diffed directly:  ./bin/run_export vs ./locod-cpu */
    for (size_t i = 0; i < ENC_OUT_SIZE; ++i)
        printf("%.10f ", r.out[i]);
    printf("\n");

    deinit_locod();
    return 0;
}

#endif /* LOCOD_FPGA */
