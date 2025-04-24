#ifndef CLIFFORD_LINEAR_H
#define CLIFFORD_LINEAR_H

#include <stdlib.h>
#include <stdbool.h>

/// Holds the Clifford signature (g array) and derived dims.
typedef struct
{
    int dim;      ///< number of signature elements (1,2,3)
    int n_blades; ///< 2^dim
    float *g;     ///< array of length dim
} CliffordSignature;

/// Main linear‐layer object.
typedef struct
{
    CliffordSignature sig;
    int in_channels;
    int out_channels;
    float *weight; ///< [n_blades][out_channels][in_channels]
    float *bias;   ///< [n_blades][out_channels]  (NULL if no bias)

    /// Points to one of the three kernel‐gen functions below.
    void (*get_kernel)(
        const float *w, // weight ptr
        const float *g, // signature ptr
        int in_ch,
        int out_ch,
        int *out_nblades,
        float **out_matrix // alloc’d inside
    );
} CliffordLinear;

/// Initialize a signature from an integer array g_in of length dim (1,2,3).
/// Caller is responsible for freeing sig->g.
void clifford_signature_init(CliffordSignature *sig, const int *g_in, int dim);

/// Create a new CliffordLinear; allocates weight & bias (zeroed).
/// get_kernel is set based on dim.  Returns NULL on unsupported dim.
CliffordLinear *clifford_linear_create(
    const int *g_in,
    int dim,
    int in_channels,
    int out_channels,
    bool use_bias);

/// Free all memory inside and the struct itself.
void clifford_linear_destroy(CliffordLinear *L);

/// Run the forward pass:
///   x:   [B][in_channels][n_blades]
///   out: [B][out_channels][n_blades]
/// Caller allocates x and out; this does its own temporaries.
void clifford_linear_forward(
    const CliffordLinear *L,
    const float *x,
    int B,
    float *out);

#endif // CLIFFORD_LINEAR_H
