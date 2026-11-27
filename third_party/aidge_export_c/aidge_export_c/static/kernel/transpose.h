#ifndef AIDGE_EXPORT_C_TRANSPOSE_H
#define AIDGE_EXPORT_C_TRANSPOSE_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>

static inline void export_c_transpose_nd(
    const void *input,
    void *output,
    int nb_dims,
    int nb_elts,
    const size_t *permute,
    const size_t *in_dims,
    const size_t *out_strides,
    size_t elem_size)
{
    if (nb_dims <= 0 || nb_dims > 8) {
        return;
    }

    const unsigned char *in = (const unsigned char *)input;
    unsigned char *out = (unsigned char *)output;
    size_t idx[8] = {0};

    for (int e = 0; e < nb_elts; ++e) {
        size_t out_index = 0;
        for (int i = 0; i < nb_dims; ++i) {
            out_index += idx[permute[i]] * out_strides[i];
        }
        memcpy(out + out_index * elem_size, in + (size_t)e * elem_size, elem_size);

        for (int i = nb_dims - 1; i >= 0; --i) {
            if (++idx[i] < in_dims[i]) {
                break;
            }
            idx[i] = 0;
        }
    }
}

#endif /* AIDGE_EXPORT_C_TRANSPOSE_H */
