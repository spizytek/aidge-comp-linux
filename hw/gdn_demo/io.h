/* Shared declarations for the PYNQ GDN bring-up demo.
 *
 * Fixed shape: C=16, H=8, W=8, NCHW, batch=1.
 * The structs are the exact bytes LoCod memcpy's to/from the accelerator,
 * so they contain fixed arrays only (no pointers).
 */
#ifndef GDN_DEMO_IO_H
#define GDN_DEMO_IO_H

#define GDN_C       16
#define GDN_H       8
#define GDN_W       8
#define GDN_IN_SIZE  (GDN_C * GDN_H * GDN_W)
#define GDN_OUT_SIZE (GDN_C * GDN_H * GDN_W)

struct param_gdn {
    float in[GDN_IN_SIZE];
};

struct result_gdn {
    float out[GDN_OUT_SIZE];
};

/* Accelerator entry point (LoCod contract: void, two struct pointers). */
void gdn(struct param_gdn *param, struct result_gdn *result);

#endif /* GDN_DEMO_IO_H */
