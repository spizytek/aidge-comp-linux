/* Desktop stand-in for LoCod's locod.h.
 *
 * Used when the real locod.h is not on the include path (plain `make`).
 * On hardware, LoCod copies its own locod.h into the build directory and this
 * file is never reached.
 */
#ifndef LOCOD_HOST_H
#define LOCOD_HOST_H

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

static inline unsigned int get_time_ns_FPGA(int accel)
{
    (void)accel;
    return 0u;
}

#endif /* LOCOD_HOST_H */
