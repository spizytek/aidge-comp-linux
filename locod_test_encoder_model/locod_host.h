/* ==========================================================================
 * locod_host.h  --  desktop stand-in for LoCod's locod.h
 * ==========================================================================
 * Used ONLY when the real locod.h is not on the include path, i.e. when you
 * build with plain `make` on your laptop instead of through locod.sh.
 *
 * It reproduces the exact semantics main.c relies on:
 *
 *   FPGA(f, &p, &r, a)   on hardware : DMA p+r into the FPGA, start accel a,
 *                                      return immediately (asynchronous)
 *                        here       : just call f(&p, &r) synchronously
 *   wait_accelerator(&r) on hardware : block until the accelerator is done
 *                                      and copy the result back
 *                        here       : no-op, the result is already in r
 *
 * So the number crunching is identical; only the transport changes. That is
 * what makes `./bin/run_encoder` produce byte-identical output to the
 * reference `./bin/run_export`.
 *
 * This file is deliberately NOT listed in the locod.sh -f arguments: inside
 * a LoCod build the real locod.h is present, so this header is never reached.
 * ========================================================================== */

#ifndef LOCOD_HOST_H
#define LOCOD_HOST_H

/* Same definitions as locod-core/src/locod.h */
#define CPU(fct, param_ptr, result_ptr) do { fct(param_ptr, result_ptr); } while (0)

#define FPGA(fct, param_ptr, result_ptr, accel) do { \
	(void)(accel);                                  \
	fct(param_ptr, result_ptr);                     \
} while (0)

static inline int init_locod(int nb_acc) { (void)nb_acc; return 0; }

static inline int wait_accelerator(void *result_addr, int accel)
{
	(void)result_addr;
	(void)accel;
	return 0;
}

static inline int deinit_locod(void) { return 0; }

static inline int get_time_ns_FPGA(int accel) { (void)accel; return 0; }

#endif /* LOCOD_HOST_H */
