#include "clifford_linear.h"
#include <string.h>
#include <math.h>
#include "instrument.h"

/// Kernel generators:
/// Each allocates a float array of shape
///   (out_ch * n_blades) × (in_ch * n_blades)
/// which must be freed by the caller.

static void get_1d_clifford_kernel(
    const float *w, // [2][out_ch][in_ch]
    const float *g, // [1]
    int in_ch,
    int out_ch,
    int *out_nblades,
    float **out_matrix)
{
    int nB = 2;
    int R = out_ch * nB;
    int C = in_ch * nB;
    float *K = malloc(sizeof(float) * R * C);

    float g0 = g[0];
    // row 0..out_ch-1 => k0 = [ w[0], g0*w[1] ]
    // row out_ch..2*out_ch-1 => k1 = [ w[1], w[0] ]
    for (int r = 0; r < nB; r++)
    {
        for (int oc = 0; oc < out_ch; oc++)
        {
            for (int blade_in = 0; blade_in < nB; blade_in++)
            {
                for (int ic = 0; ic < in_ch; ic++)
                {
                    // pick which w[?][oc][ic]
                    float val;
                    if (r == 0 && blade_in == 0)
                        val = w[(0 * out_ch + oc) * in_ch + ic];
                    else if (r == 0 && blade_in == 1)
                        val = g0 * w[(1 * out_ch + oc) * in_ch + ic];
                    else if (r == 1 && blade_in == 0)
                        val = w[(1 * out_ch + oc) * in_ch + ic];
                    else /*r==1&&blade_in==1*/
                        val = w[(0 * out_ch + oc) * in_ch + ic];

                    int row = r * out_ch + oc;
                    int col = blade_in * in_ch + ic;
                    K[row * C + col] = val;
                }
            }
        }
    }

    *out_nblades = nB;
    *out_matrix = K;
}

static void get_2d_clifford_kernel(
    const float *w, // [4][out_ch][in_ch]
    const float *g, // [2]
    int in_ch,
    int out_ch,
    int *out_nblades,
    float **out_matrix)
{
    int nB = 4;
    int R = out_ch * nB;
    int C = in_ch * nB;
    float *K = malloc(sizeof(float) * R * C);

    float g0 = g[0];
    float g1 = g[1];
    // Hard‐coded from Python:
    // k0 = [w0,  g0*w1,  g1*w2, -g0*g1*w3]
    // k1 = [w1,  w0,    -g1*w3, g1*w2]
    // k2 = [w2,  g0*w3, w0,    -g0*w1]
    // k3 = [w3,  w2,    -w1,    w0   ]
    for (int r = 0; r < 4; r++)
    {
        for (int oc = 0; oc < out_ch; oc++)
        {
            for (int blade_in = 0; blade_in < 4; blade_in++)
            {
                for (int ic = 0; ic < in_ch; ic++)
                {
                    float val = 0.f;
                    float w0 = w[(0 * out_ch + oc) * in_ch + ic];
                    float w1 = w[(1 * out_ch + oc) * in_ch + ic];
                    float w2 = w[(2 * out_ch + oc) * in_ch + ic];
                    float w3 = w[(3 * out_ch + oc) * in_ch + ic];
                    switch (r)
                    {
                    case 0:
                        if (blade_in == 0)
                            val = w0;
                        else if (blade_in == 1)
                            val = g0 * w1;
                        else if (blade_in == 2)
                            val = g1 * w2;
                        else
                            val = -g0 * g1 * w3;
                        break;
                    case 1:
                        if (blade_in == 0)
                            val = w1;
                        else if (blade_in == 1)
                            val = w0;
                        else if (blade_in == 2)
                            val = -g1 * w3;
                        else
                            val = g1 * w2;
                        break;
                    case 2:
                        if (blade_in == 0)
                            val = w2;
                        else if (blade_in == 1)
                            val = g0 * w3;
                        else if (blade_in == 2)
                            val = w0;
                        else
                            val = -g0 * w1;
                        break;
                    default: // r==3
                        if (blade_in == 0)
                            val = w3;
                        else if (blade_in == 1)
                            val = w2;
                        else if (blade_in == 2)
                            val = -w1;
                        else
                            val = w0;
                        break;
                    }
                    int row = r * out_ch + oc;
                    int col = blade_in * in_ch + ic;
                    K[row * C + col] = val;
                }
            }
        }
    }

    *out_nblades = nB;
    *out_matrix = K;
}

static void get_3d_clifford_kernel(
    const float *w, // [8][out_ch][in_ch]
    const float *g, // [3]
    int in_ch,
    int out_ch,
    int *out_nblades,
    float **out_matrix)
{
    int nB = 8;
    int R = out_ch * nB;
    int C = in_ch * nB;
    float *K = malloc(sizeof(float) * R * C);

    float g0 = g[0], g1 = g[1], g2 = g[2];
    // Hard‐coded from Python version:
    for (int r = 0; r < 8; r++)
    {
        for (int oc = 0; oc < out_ch; oc++)
        {
            for (int blade_in = 0; blade_in < 8; blade_in++)
            {
                for (int ic = 0; ic < in_ch; ic++)
                {
                    // extract all w[i] for this oc,ic
                    float W[8];
                    for (int b = 0; b < 8; b++)
                    {
                        W[b] = w[(b * out_ch + oc) * in_ch + ic];
                    }
                    float val = 0.f;
                    switch (r)
                    {
                    case 0:
                        switch (blade_in)
                        {
                        case 0:
                            val = W[0];
                            break;
                        case 1:
                            val = W[1] * g0;
                            break;
                        case 2:
                            val = W[2] * g1;
                            break;
                        case 3:
                            val = W[3] * g2;
                            break;
                        case 4:
                            val = -W[4] * g0 * g1;
                            break;
                        case 5:
                            val = -W[5] * g0 * g2;
                            break;
                        case 6:
                            val = -W[6] * g1 * g2;
                            break;
                        default:
                            val = -W[7] * g0 * g1 * g2;
                            break;
                        }
                        break;
                    case 1:
                        switch (blade_in)
                        {
                        case 0:
                            val = W[1];
                            break;
                        case 1:
                            val = W[0];
                            break;
                        case 2:
                            val = -W[4] * g1;
                            break;
                        case 3:
                            val = -W[5] * g2;
                            break;
                        case 4:
                            val = W[2] * g1;
                            break;
                        case 5:
                            val = W[3] * g2;
                            break;
                        case 6:
                            val = -W[7] * g1 * g2;
                            break;
                        default:
                            val = -W[6] * g2 * g1;
                            break;
                        }
                        break;
                    case 2:
                        switch (blade_in)
                        {
                        case 0:
                            val = W[2];
                            break;
                        case 1:
                            val = W[4] * g0;
                            break;
                        case 2:
                            val = W[0];
                            break;
                        case 3:
                            val = -W[6] * g2;
                            break;
                        case 4:
                            val = -W[1] * g0;
                            break;
                        case 5:
                            val = W[7] * g0 * g2;
                            break;
                        case 6:
                            val = W[3] * g2;
                            break;
                        default:
                            val = W[5] * g2 * g0;
                            break;
                        }
                        break;
                    case 3:
                        switch (blade_in)
                        {
                        case 0:
                            val = W[3];
                            break;
                        case 1:
                            val = W[5] * g0;
                            break;
                        case 2:
                            val = W[6] * g1;
                            break;
                        case 3:
                            val = W[0];
                            break;
                        case 4:
                            val = -W[7] * g0 * g1;
                            break;
                        case 5:
                            val = -W[1] * g0;
                            break;
                        case 6:
                            val = -W[2] * g1;
                            break;
                        default:
                            val = -W[4] * g0 * g1;
                            break;
                        }
                        break;

                    case 4:
                        switch (blade_in)
                        {
                        case 0:
                            val = W[4];
                            break;
                        case 1:
                            val = W[2];
                            break;
                        case 2:
                            val = -W[1];
                            break;
                        case 3:
                            val = W[7] * g2;
                            break;
                        case 4:
                            val = W[0];
                            break;
                        case 5:
                            val = -W[6] * g2;
                            break;
                        case 6:
                            val = W[5] * g2;
                            break;
                        default:
                            val = W[3] * g2;
                            break;
                        }
                        break;

                    case 5:
                        switch (blade_in)
                        {
                        case 0:
                            val = W[5];
                            break;
                        case 1:
                            val = W[3];
                            break;
                        case 2:
                            val = -W[7] * g1;
                            break;
                        case 3:
                            val = -W[1];
                            break;
                        case 4:
                            val = W[6] * g1;
                            break;
                        case 5:
                            val = W[0];
                            break;
                        case 6:
                            val = -W[4] * g1;
                            break;
                        default:
                            val = -W[2] * g1;
                            break;
                        }
                        break;

                    case 6:
                        switch (blade_in)
                        {
                        case 0:
                            val = W[6];
                            break;
                        case 1:
                            val = W[7] * g0;
                            break;
                        case 2:
                            val = W[3];
                            break;
                        case 3:
                            val = -W[2];
                            break;
                        case 4:
                            val = -W[5] * g0;
                            break;
                        case 5:
                            val = W[4] * g0;
                            break;
                        case 6:
                            val = W[0];
                            break;
                        default:
                            val = W[1] * g0;
                            break;
                        }
                        break;

                    default: // r==7
                        switch (blade_in)
                        {
                        case 0:
                            val = W[7];
                            break;
                        case 1:
                            val = W[6];
                            break;
                        case 2:
                            val = -W[5];
                            break;
                        case 3:
                            val = W[4];
                            break;
                        case 4:
                            val = W[3];
                            break;
                        case 5:
                            val = -W[2];
                            break;
                        case 6:
                            val = W[1];
                            break;
                        default:
                            val = W[0];
                            break;
                        }
                        break;
                    }
                    int row = r * out_ch + oc;
                    int col = blade_in * in_ch + ic;
                    K[row * C + col] = val;
                }
            }
        }
    }

    *out_nblades = nB;
    *out_matrix = K;
}

/// Initialize signature struct.
void clifford_signature_init(CliffordSignature *sig, const int *g_in, int dim)
{
    sig->dim = dim;
    sig->n_blades = 1 << dim;
    sig->g = malloc(sizeof(float) * dim);
    for (int i = 0; i < dim; i++)
    {
        sig->g[i] = (float)g_in[i];
    }
}

/// Create and zero‐initialize weights/bias.
CliffordLinear *clifford_linear_create(
    const int *g_in,
    int dim,
    int in_ch,
    int out_ch,
    bool use_bias)
{
    CliffordLinear *L = malloc(sizeof(*L));
    L->in_channels = in_ch;
    L->out_channels = out_ch;
    clifford_signature_init(&L->sig, g_in, dim);

    // choose kernel generator
    if (dim == 1)
        L->get_kernel = get_1d_clifford_kernel;
    else if (dim == 2)
        L->get_kernel = get_2d_clifford_kernel;
    else if (dim == 3)
        L->get_kernel = get_3d_clifford_kernel;
    else
    {
        free(L);
        return NULL;
    }

    int B = L->sig.n_blades * out_ch * in_ch;
    L->weight = calloc(B, sizeof(float));
    if (use_bias)
    {
        L->bias = calloc(L->sig.n_blades * out_ch, sizeof(float));
    }
    else
    {
        L->bias = NULL;
    }
    return L;
}

void clifford_linear_destroy(CliffordLinear *L)
{
    if (!L)
        return;
    free(L->sig.g);
    free(L->weight);
    free(L->bias);
    free(L);
}

/// Performs: out[b] = K * flatten(x[b]) + bias, reshaped back

void clifford_linear_forward(
    const CliffordLinear *L,
    const float *x,
    int B,
    float *out)
{
    // reset flop counter
    __flop_count = 0;

    int I = L->sig.n_blades;
    int Cin = L->in_channels;
    int Cout = L->out_channels;
    int Dflat = Cin * I;
    int Nflat = Cout * I;

    // flatten input
    float *Xf = malloc(sizeof(float) * B * Dflat);
    for (int b = 0; b < B; b++) {
        for (int i = 0; i < I; i++) {
            for (int c = 0; c < Cin; c++) {
                Xf[b*Dflat + i*Cin + c] = x[b*(Cin*I) + c*I + i];
            }
        }
    }

    // build kernel
    int nb;
    float *K;
    L->get_kernel(L->weight, L->sig.g, Cin, Cout, &nb, &K);

    // mat-vec + bias with instrumentation
    float *Yf = malloc(sizeof(float) * B * Nflat);
    for (int b = 0; b < B; b++) {
        for (int r = 0; r < Nflat; r++) {
            float acc = 0.f;
            for (int d = 0; d < Dflat; d++) {
                float prod = MUL(K[r*Dflat + d], Xf[b*Dflat + d]);
                acc = ADD(acc, prod);
            }
            if (L->bias) {
                acc = ADD(acc, L->bias[r]);
            }
            Yf[b*Nflat + r] = acc;
        }
    }

    free(Xf);
    free(K);

    // reshape back
    for (int b = 0; b < B; b++) {
        for (int i = 0; i < I; i++) {
            for (int c = 0; c < Cout; c++) {
                out[b*(Cout*I) + c*I + i] = Yf[b*Nflat + i*Cout + c];
            }
        }
    }
    free(Yf);

    // report total flops
    REPORT_FLOPS();
}
