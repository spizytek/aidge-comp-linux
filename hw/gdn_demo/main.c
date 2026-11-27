/* CPU driver for the PYNQ GDN bring-up demo.
 *
 * This file must be the FIRST -f argument to locod.sh: the LoCod script only
 * scans the first file for the startup call and the accelerator call.
 *
 * Keep those two calls on single lines with literal arguments.  Do not write
 * the call syntax in comments: locod.sh greps the raw text of this file.
 */
#include <stddef.h>

#ifndef LOCOD_FPGA
#  include <stdio.h>
#endif

#include "io.h"

#ifndef LOCOD_FPGA
#  if defined(__has_include) && __has_include("locod.h")
#    include "locod.h"
#  else
#    include "locod_host.h"
#  endif
#endif

#ifndef LOCOD_FPGA

int main(void)
{
    static struct param_gdn p;
    static struct result_gdn r_fpga;
    static struct result_gdn r_cpu;

    /* Deterministic input in [-0.5, 0.5]; keeps GDN norms in a sane range. */
    for (int i = 0; i < GDN_IN_SIZE; ++i) {
        p.in[i] = ((float)((i * 37) % 101) - 50.0f) * 0.01f;
    }

    init_locod(1);

    FPGA(gdn, &p, &r_fpga, 0);
    wait_accelerator(&r_fpga, 0);

    /* CPU reference: same source, same parameters. */
    gdn(&p, &r_cpu);

    const unsigned int t_ns = get_time_ns_FPGA(0);

    float max_err = 0.0f;
    for (int i = 0; i < GDN_OUT_SIZE; ++i) {
        float e = r_fpga.out[i] - r_cpu.out[i];
        if (e < 0.0f) {
            e = -e;
        }
        if (e > max_err) {
            max_err = e;
        }
    }

    printf("FPGA time = %u ns\n", t_ns);
    printf("max |FPGA - CPU| = %g\n", max_err);
    for (int i = 0; i < 8; ++i) {
        printf("out[%d] FPGA=%.9g CPU=%.9g\n",
               i, r_fpga.out[i], r_cpu.out[i]);
    }
    printf("RESULT: %s\n", (max_err < 1e-3f) ? "PASS" : "FAIL");

    deinit_locod();
    return (max_err < 1e-3f) ? 0 : 1;
}

#endif /* LOCOD_FPGA */
