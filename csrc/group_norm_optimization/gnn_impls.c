#include "gnn_bench.h"
#include "gnn_setup.h"
#include "../clifford_groupnorm.h"
#include "../flops.h"
#include <stdlib.h>
#include <string.h>

#include <assert.h>
#include <stdlib.h>
#include <stdint.h> 
#include <immintrin.h> // For AVX2 intrinsics
#include <stdio.h>     // For fprintf, and for commented-out debug prints
#include <stdio.h>     // For fprintf in case of errors (though not strictly used in final proposal)
#include <math.h>      // For fmaf

//baseline with memory alloc
/*
static void baseline_groupnorm(
    const float *x,
    int           B,
    int           C,
    int           D,
    int           I,
    int           num_groups,
    bool          running,
    float        *running_mean_orig,
    float        *running_cov_orig,
    bool          scaling,
    const float  *weight_orig,
    const float  *bias_orig,
    bool          training,
    float         momentum,
    float         eps,
    float        *x_norm
) {
    clifford_groupnorm(
        x, B, C, D, I, num_groups,
        running,
        running_mean_orig,
        running_cov_orig,
        scaling,
        weight_orig,
        bias_orig,
        training,
        momentum,
        eps,
        x_norm
    );
}
*/
static void  baseline_groupnorm(
    const float *x,
    int           B,
    int           C,
    int           D,
    int           _I,
    int           num_groups,
    bool          running,
    float        *running_mean_orig,
    float        *running_cov_orig,
    bool          scaling,
    const float  *weight_orig,
    const float  *bias_orig,
    bool          training,
    float         momentum,
    float         eps,
    float        *x_norm
){

    // Track bytes for input data
    size_t input_bytes = B * C * D * _I * sizeof(float);
    BYTES_READ(input_bytes);
    
    // Track bytes for running mean and covariance if used
    if (running) {
        size_t running_mean_bytes = _I * (C / num_groups) * sizeof(float);
        size_t running_cov_bytes = _I * _I * (C / num_groups) * sizeof(float);
        BYTES_READ(running_mean_bytes);
        BYTES_READ(running_cov_bytes);
        // These will be updated, so also count as written
        BYTES_WRITTEN(running_mean_bytes);
        BYTES_WRITTEN(running_cov_bytes);
    }
    
    // Track bytes for weight and bias if used
    if (scaling) {
        if (weight_orig) {
            size_t weight_bytes = _I * (C / num_groups) * sizeof(float);
            BYTES_READ(weight_bytes);
        }
        if (bias_orig) {
            size_t bias_bytes = _I * (C / num_groups) * sizeof(float);
            BYTES_READ(bias_bytes);
        }
    }
    
    // Track bytes for output data
    size_t output_bytes = B * C * D * _I * sizeof(float);
    BYTES_WRITTEN(output_bytes);

    //basic constants
    int dim0 = (B*C)/num_groups;
    int group_size = C/num_groups;
    
    //blow up running mean and cov along the batch dim
    if ( running ){
        for(int i = 0; i < _I; i++){ //COST: 0 flops
            for(int j = 0; j < group_size; j++){
                for(int l = 0; l < B; l++){
                    gn_running_mean_temp[i*group_size*B + j*B + l] = running_mean_orig[i*group_size + ((j*B + l)%group_size)];
                }
            }
        }
        for(int i = 0; i < _I; i++){
            for(int ii = 0; ii < _I; ii++){
                for(int j = 0; j < group_size; j++){
                    for(int l = 0; l < B; l++){
                        gn_running_cov_temp[i*_I*group_size*B + ii*group_size*B + j*B + l] = running_cov_orig[i*_I*group_size + ii*group_size + ((j*B + l)%group_size)];
                    }
                }
            }
        }
    }

    //mean
    if (training || (!running)) {
        int mean_size = dim0 * _I;
        for (int i = 0; i < mean_size; i++){
            gn_mean_temp[i] = 0.0f;
        }
        for(int i = 0; i < dim0; i++){
            for(int j = 0; j < num_groups; j++){
                for(int k = 0; k < D; k++){
                    for(int l = 0; l < _I; l++){ //COST: dim0*num_groups*D*I = B*C*D*I flops
                        FLOP(1);
                        gn_mean_temp[i*_I + l] += x[i*num_groups*D*_I + j*D*_I + k*_I + l];
                    }
                }
            }
        }
        for(int i = 0; i < dim0; i++){
            for(int l = 0; l < _I; l++){ //COST: dim0*I
                FLOP(1);
                gn_mean_temp[i*_I + l] /= (D*num_groups);
            }
        }
        if (running){ //update running mean
            for(int i = 0; i < _I; i++){
                for(int j = 0; j < dim0; j++){
                    FLOP(2);
                    FLOP(1);
                    gn_running_mean_temp[i*dim0 + j] += momentum *(gn_mean_temp[j*_I + i] - gn_running_mean_temp[i*dim0 + j]);
                }
            }
        }
    } else { //if not training use running mean
        for(int i = 0; i < _I; i++){
            for(int j = 0; j < dim0; j++){
                    gn_mean_temp[j*_I + i] = gn_running_mean_temp[i*dim0 + j];
            }
        }
    }

    //subtract mean form x
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){
                for(int l = 0; l < _I; l++){
                    FLOP(1);
                    x_norm[i*num_groups*D*_I + j*D*_I + k*_I + l] = x[i*num_groups*D*_I + j*D*_I + k*_I + l] - gn_mean_temp[i*_I + l];
                }
            }
        }
    }

    //permute: dim0, num_groups, D, _I -> dim0, _I, num_groups, D
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){
                for(int l = 0; l < _I; l++){
                    gn_X_temp[i*_I*num_groups*D + l*num_groups*D + j*D +k] = x_norm[i*num_groups*D*_I + j*D*_I + k*_I + l];
                }
            }
        }
    }

    //calc cov
    if (training || (!running)){
        int cov_size = dim0*_I*_I;
        for (int i = 0; i < cov_size; i++){
            gn_cov_temp[i] = 0.0f;
        }
        for(int i = 0; i < dim0; i++){
            for(int j = 0; j < _I; j++){ //row selector
                for(int l = 0; l < _I; l++){ //col selector
                    for(int k = 0; k < num_groups*D; k++){
                        FLOP(1);
                        FLOP(1);
                        gn_cov_temp[i*_I*_I + j*_I + l] += gn_X_temp[i*_I*num_groups*D + j*num_groups*D + k]* x_norm[i*num_groups*D*_I + k*_I + l];
        }}}}
        //scale all of cov
        for(int i = 0; i < cov_size; i++){
            FLOP(1);
            gn_cov_temp[i] /= (num_groups * D);
        }
        if (running) { //upadate running cov
            for(int i = 0; i < _I; i++){            
                for(int ii = 0; ii < _I; ii++){
                    for(int j = 0; j < dim0; j++){
                        FLOP(2);
                        FLOP(1);
                        gn_running_cov_temp[i*_I*dim0 + ii*dim0 + j] += momentum *(gn_cov_temp[j*_I*_I + i*_I + ii] - gn_running_cov_temp[i*_I*dim0 + ii*dim0 + j]);
                    }
                }
            }
        }
    } else {
        for(int i = 0; i < _I; i++){ //if not training use running cov
            for(int ii = 0; ii < _I; ii++){
                for(int j = 0; j < dim0; j++){
                    gn_cov_temp[j*_I*_I + i*_I + ii] = gn_running_cov_temp[i*_I*dim0 + ii*dim0 + j];
                }
            }
        }
    }

    //write runnning mean and cov back
    if( running ){
        //mean
        for(int i = 0; i < _I*group_size; i++){
            running_mean_orig[i] = 0;
        }
        for(int i = 0; i < _I; i++){
            for(int j = 0; j < group_size; j++){
                for(int l = 0; l < B; l++){
                    FLOP(1);
                    running_mean_orig[i*group_size + j] += gn_running_mean_temp[i*group_size*B + l*group_size + j];
                }
            }
        }
        for(int i = 0; i < _I*group_size; i++){
            FLOP(1);
            running_mean_orig[i] /= B;
        }
        //cov
        for(int i = 0; i < _I*_I*group_size; i++){
            running_cov_orig[i] = 0;
        }
        for(int i = 0; i < _I; i++){
            for(int ii = 0; ii < _I; ii++){
                for(int j = 0; j < group_size; j++){
                    for(int l = 0; l < B; l++){
                        FLOP(1);
                        running_cov_orig[i*_I*group_size + ii*group_size + j] += gn_running_cov_temp[i*_I*group_size*B + ii*group_size*B + l*group_size + j];
                    }
                }
            }
        }
        for(int i = 0; i < _I*_I*group_size; i++){
            FLOP(1);
            running_cov_orig[i] /= B;
        }

    }

    //get max for each groups cov matrix
    for(int i = 0; i < dim0; i++){
        gn_max_temp[i] = gn_cov_temp[i*_I*_I];
        for(int j = 1; j < _I*_I; j++){
            gn_max_temp[i] = gn_cov_temp[i*_I*_I + j] > gn_max_temp[i] ? gn_cov_temp[i*_I*_I + j] : gn_max_temp[i];
        }
    }

    //add pertubation to diagonal
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < _I; j++){
            FLOP(2);
            gn_cov_temp[i*_I*_I + j*_I + j] += gn_max_temp[i]*eps;
    }}

    //Src: https://github.com/mdwarfgeek/lib/blob/master/cholesky.c
    /* In-place Cholesky decomposition for L in A = L L^T where L is a
    lower triangular matrix.  Uses the Cholesky-Crout algorithm,
    based on Stoer & Bulirsch "Introduction to numerical analysis"
    Sect. 4.3, but with the loops for i = j and i != j separated and
    changed to access only the lower triangular part of the matrix.
    Output is computed column by column so the accesses in the
    innermost loop (indexed by k) are sequential in memory. */

    for(int i = 0; i < dim0; i++){
        int n = _I;
        float *a = &gn_cov_temp[i*_I*_I];
        float x, r;
        int i, j, k;
        /* Loop over columns */
        for(j = 0; j < n; j++) {
            /* i = j */
            x = a[j*n+j];  /* A_jj */

            for(k = 0; k < j; k++){
                FLOP(2);
                x -= a[j*n+k] * a[j*n+k];  /* L_jk L_jk */
            }

            if(x < 0)
                printf("clofford_groupnorm: Cholesky decomposition x was negative\n");

            FLOP(30);
            x = sqrt(x);

            a[j*n+j] = x;  /* L_jj */
            FLOP(1);
            r = 1.0 / x;

            /* i != j */
            for(i = j+1; i < n; i++) {
                x = a[i*n+j];  /* A_ij */

                for(k = 0; k < j; k++){
                    FLOP(2);
                    x -= a[i*n+k] * a[j*n+k];  /* L_ik L_ij */
                }
                FLOP(1);
                a[i*n+j] = x * r;  /* L_ij = x / L_jj */
            }
        }
        //baseline just transpose back inplace 
        for(i = 0; i < n; i++){
            for(j = i; j < n; j++){
                x = a[i*n +j];
                a[i*n +j] = a[j*n+i];
                a[j*n+i] = x;
            }
        }
    }

    //solve triangular
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){ //cost: BCDI(I-1)/2 mults and adds, BCDI div
                    float *mat = gn_cov_temp + (i*_I*_I);
                    float *vec = x_norm + (i*num_groups*D*_I + j*D*_I + k*_I);
                    for(int r = _I-1; r >= 0; r--){//from last row to first
                        float rowsum = 0.0f;
                        for(int c = _I-1; c > r; c--){//do the partioal vec row mult
                            FLOP(1);
                            FLOP(1);
                            rowsum += mat[r*_I + c]*vec[c];
                        }
                        FLOP(1);
                        rowsum = vec[r] - rowsum;
                        FLOP(1);
                        vec[r] = rowsum / mat[r*_I + r];
                    }
            }
        }
    }

    //blow up weight and bias
    if ( scaling ){
        for(int i = 0; i < _I; i++){
            for(int ii = 0; ii < _I; ii++){
                for(int j = 0; j < group_size; j++){
                    for(int l = 0; l < B; l++){
                        gn_weight_temp[i*_I*group_size*B + ii*group_size*B + j*B + l] = weight_orig[i*_I*group_size + ii*group_size + ((j*B + l)%group_size)];
                    }
                }
            }
        }
        
        for(int i = 0; i < _I; i++){
            for(int j = 0; j < group_size; j++){
                for(int l = 0; l < B; l++){
                    gn_bias_temp[i*group_size*B + j*B + l] = bias_orig[i*group_size + ((j*B + l)%group_size)];
                }
            }
        }
    }

    //scaling or just copy over
    if ( scaling ){
        //permute weight from (_I0, _I1, dim0) to (dim0, _I0, _I1)
        //think their code is wrong since they dont permute the bias and just reshape it...
        for(int i = 0; i < _I*_I; i++){
            for(int j = 0; j < dim0; j++){
                gn_weight_perm_temp[j*_I*_I + i] = gn_weight_temp[i*dim0 + j];
            }
        }

        //mat mult and add 
        float vecmult[8]; //put this on the stack since 3D is max size anyways
        for(int i = 0; i < dim0; i++){
            for(int j = 0; j < num_groups; j++){
                for(int l = 0; l < D; l++){
                    //xn = Wx + b
                    float *W = gn_weight_perm_temp + i*_I*_I;
                    float *xv = x_norm + i*num_groups*D*_I + j*D*_I + l*_I;
                    float *b = gn_bias_temp + i*_I;
                    //(I,I)x(I,1)
                    for(int ii = 0; ii < _I; ii++){
                        vecmult[ii] = 0.0f;
                        for(int jj = 0; jj < _I; jj++){
                            FLOP(1);
                            FLOP(1);
                            vecmult[ii] += W[ii*_I + jj] * xv[jj];
                        }
                        FLOP(1);
                        vecmult[ii] += b[ii];
                    }
                    for(int ii = 0; ii < _I; ii++){
                        xv[ii] = vecmult[ii];
                    }
                }
            }
        }
    }

    return;
}

static void optimized_groupnorm_v0(
    const float *x,
    int           B,
    int           C,
    int           D,
    int           I,
    int           num_groups,
    bool          running,
    float        *running_mean_orig,
    float        *running_cov_orig,
    bool          scaling,
    const float  *weight_orig,
    const float  *bias_orig,
    bool          training,
    float         momentum,
    float         eps,
    float        *x_norm
) {

    // Track bytes for input data
    size_t input_bytes = B * C * D * I * sizeof(float);
    BYTES_READ(input_bytes);
    
    // Track bytes for running mean and covariance if used
    if (running) {
        size_t running_mean_bytes = I * (C / num_groups) * sizeof(float);
        size_t running_cov_bytes = I * I * (C / num_groups) * sizeof(float);
        BYTES_READ(running_mean_bytes);
        BYTES_READ(running_cov_bytes);
        // These will be updated, so also count as written
        BYTES_WRITTEN(running_mean_bytes);
        BYTES_WRITTEN(running_cov_bytes);
    }
    
    // Track bytes for weight and bias if used
    if (scaling) {
        if (weight_orig) {
            size_t weight_bytes = I * (C / num_groups) * sizeof(float);
            BYTES_READ(weight_bytes);
        }
        if (bias_orig) {
            size_t bias_bytes = I * (C / num_groups) * sizeof(float);
            BYTES_READ(bias_bytes);
        }
    }
    
    // Track bytes for output data
    size_t output_bytes = B * C * D * I * sizeof(float);
    BYTES_WRITTEN(output_bytes);

    int dim0 = (B*C)/num_groups;
    int group_size = C/num_groups;
    
    //blow up running mean and cov along the batch dim
    if ( running ){
        for(int i = 0; i < I; i++){ //COST: 0 flops
            for(int j = 0; j < group_size; j++){
                for(int l = 0; l < B; l++){
                     gn_running_mean_temp[i*group_size*B + j*B + l] = running_mean_orig[i*group_size + ((j*B + l)%group_size)];
                }
            }
        }
        for(int i = 0; i < I; i++){
            for(int ii = 0; ii < I; ii++){
                for(int j = 0; j < group_size; j++){
                    for(int l = 0; l < B; l++){
                         gn_running_cov_temp[i*I*group_size*B + ii*group_size*B + j*B + l] = running_cov_orig[i*I*group_size + ii*group_size + ((j*B + l)%group_size)];
                    }
                }
            }
        }
    }

    //mean
    if (training || (!running)) {
        int mean_size = dim0 * I;
        for (int i = 0; i < mean_size; i++){
            gn_mean_temp[i] = 0.0f;
        }
        for(int i = 0; i < dim0; i++){
            for(int j = 0; j < num_groups; j++){
                for(int k = 0; k < D; k++){
                    for(int l = 0; l < I; l++){ //COST: dim0*num_groups*D*I = B*C*D*I flops
                        FLOP(1);
                        gn_mean_temp[i*I + l] += x[i*num_groups*D*I + j*D*I + k*I + l];
                    }
                }
            }
        }
        for(int i = 0; i < dim0; i++){
            for(int l = 0; l < I; l++){ //COST: dim0*I
                FLOP(1);
                gn_mean_temp[i*I + l] /= (D*num_groups);
            }
        }
        if (running){ //update running mean
            for(int i = 0; i < I; i++){
                for(int j = 0; j < dim0; j++){
                    FLOP(2);
                    FLOP(1);
                     gn_running_mean_temp[i*dim0 + j] += momentum *(gn_mean_temp[j*I + i] -  gn_running_mean_temp[i*dim0 + j]);
                }
            }
        }
    } else { //if not training use running mean
        for(int i = 0; i < I; i++){
            for(int j = 0; j < dim0; j++){
                    gn_mean_temp[j*I + i] =  gn_running_mean_temp[i*dim0 + j];
            }
        }
    }

    //subtract mean form x
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){
                for(int l = 0; l < I; l++){
                    FLOP(1);
                    x_norm[i*num_groups*D*I + j*D*I + k*I + l] = x[i*num_groups*D*I + j*D*I + k*I + l] - gn_mean_temp[i*I + l];
                }
            }
        }
    }

    //permute: dim0, num_groups, D, I -> dim0, I, num_groups, D
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){
                for(int l = 0; l < I; l++){
                    gn_X_temp[i*I*num_groups*D + l*num_groups*D + j*D +k] = x_norm[i*num_groups*D*I + j*D*I + k*I + l];
                }
            }
        }
    }

    //calc cov
    if (training || (!running)){
        int cov_size = dim0*I*I;
        for (int i = 0; i < cov_size; i++){
            gn_cov_temp[i] = 0.0f;
        }
        for(int i = 0; i < dim0; i++){
            for(int j = 0; j < I; j++){ //row selector
                for(int l = 0; l < I; l++){ //col selector
                    for(int k = 0; k < num_groups*D; k++){
                        FLOP(1);
                        FLOP(1);
                        gn_cov_temp[i*I*I + j*I + l] += gn_X_temp[i*I*num_groups*D + j*num_groups*D + k]* x_norm[i*num_groups*D*I + k*I + l];
        }}}}
        //scale all of cov
        for(int i = 0; i < cov_size; i++){
            FLOP(1);
            gn_cov_temp[i] /= (num_groups * D);
        }
        if (running) { //upadate running cov
            for(int i = 0; i < I; i++){            
                for(int ii = 0; ii < I; ii++){
                    for(int j = 0; j < dim0; j++){
                        FLOP(2);
                        FLOP(1);
                         gn_running_cov_temp[i*I*dim0 + ii*dim0 + j] += momentum *(gn_cov_temp[j*I*I + i*I + ii] -  gn_running_cov_temp[i*I*dim0 + ii*dim0 + j]);
                    }
                }
            }
        }
    } else {
        for(int i = 0; i < I; i++){ //if not training use running cov
            for(int ii = 0; ii < I; ii++){
                for(int j = 0; j < dim0; j++){
                    gn_cov_temp[j*I*I + i*I + ii] =  gn_running_cov_temp[i*I*dim0 + ii*dim0 + j];
                }
            }
        }
    }

    //write runnning mean and cov back
    if( running ){
        //mean
        for(int i = 0; i < I*group_size; i++){
            running_mean_orig[i] = 0;
        }
        for(int i = 0; i < I; i++){
            for(int j = 0; j < group_size; j++){
                for(int l = 0; l < B; l++){
                    FLOP(1);
                    running_mean_orig[i*group_size + j] +=  gn_running_mean_temp[i*group_size*B + l*group_size + j];
                }
            }
        }
        for(int i = 0; i < I*group_size; i++){
            FLOP(1);
            running_mean_orig[i] /= B;
        }
        //cov
        for(int i = 0; i < I*I*group_size; i++){
            running_cov_orig[i] = 0;
        }
        for(int i = 0; i < I; i++){
            for(int ii = 0; ii < I; ii++){
                for(int j = 0; j < group_size; j++){
                    for(int l = 0; l < B; l++){
                        FLOP(1);
                        running_cov_orig[i*I*group_size + ii*group_size + j] +=  gn_running_cov_temp[i*I*group_size*B + ii*group_size*B + l*group_size + j];
                    }
                }
            }
        }
        for(int i = 0; i < I*I*group_size; i++){
            FLOP(1);
            running_cov_orig[i] /= B;
        }

    }

    //get max for each groups cov matrix
    for(int i = 0; i < dim0; i++){
        gn_max_temp[i] = gn_cov_temp[i*I*I];
        for(int j = 1; j < I*I; j++){
            gn_max_temp[i] = gn_cov_temp[i*I*I + j] > gn_max_temp[i] ? gn_cov_temp[i*I*I + j] : gn_max_temp[i];
        }
    }

    //add pertubation to diagonal
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < I; j++){
            FLOP(2);
            gn_cov_temp[i*I*I + j*I + j] += gn_max_temp[i]*eps;
    }}

    //Src: https://github.com/mdwarfgeek/lib/blob/master/cholesky.c
    /* In-place Cholesky decomposition for L in A = L L^T where L is a
    lower triangular matrix.  Uses the Cholesky-Crout algorithm,
    based on Stoer & Bulirsch "Introduction to numerical analysis"
    Sect. 4.3, but with the loops for i = j and i != j separated and
    changed to access only the lower triangular part of the matrix.
    Output is computed column by column so the accesses in the
    innermost loop (indexed by k) are sequential in memory. */

    for(int i = 0; i < dim0; i++){
        int n = I;
        float *a = &gn_cov_temp[i*I*I];
        float x, r;
        int i, j, k;
        /* Loop over columns */
        for(j = 0; j < n; j++) {
            /* i = j */
            x = a[j*n+j];  /* A_jj */

            for(k = 0; k < j; k++){
                FLOP(2);
                x -= a[j*n+k] * a[j*n+k];  /* L_jk L_jk */
            }

            if(x < 0)
                printf("clofford_groupnorm: Cholesky decomposition x was negative\n");

            FLOP(30);
            x = sqrt(x);

            a[j*n+j] = x;  /* L_jj */
            FLOP(1);
            r = 1.0 / x;

            /* i != j */
            for(i = j+1; i < n; i++) {
                x = a[i*n+j];  /* A_ij */

                for(k = 0; k < j; k++){
                    FLOP(2);
                    x -= a[i*n+k] * a[j*n+k];  /* L_ik L_ij */
                }
                FLOP(1);
                a[i*n+j] = x * r;  /* L_ij = x / L_jj */
            }
        }
        //baseline just transpose back inplace 
        for(i = 0; i < n; i++){
            for(j = i; j < n; j++){
                x = a[i*n +j];
                a[i*n +j] = a[j*n+i];
                a[j*n+i] = x;
            }
        }
    }

    //invert the diagaonal first then
    for(int i = 0; i < dim0; i++){
        for(int d = 0; d < I; d++){
            gn_cov_temp[i*I*I + d*I + d] = 1.0f / gn_cov_temp[i*I*I + d*I + d];
            FLOP(1);
        }
    }

    //solve triangular with the already inverted diagonal
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){ //cost: BCDI(I-1)/2 mults and adds, BCDI div
                    float *mat = gn_cov_temp + (i*I*I);
                    float *vec = x_norm + (i*num_groups*D*I + j*D*I + k*I);
                    for(int r = I-1; r >= 0; r--){//from last row to first
                        float rowsum = 0.0f;
                        for(int c = I-1; c > r; c--){//do the partioal vec row mult
                            FLOP(1);
                            FLOP(1);
                            rowsum += mat[r*I + c]*vec[c];
                        }
                        FLOP(1);
                        rowsum = vec[r] - rowsum;
                        FLOP(1);
                        vec[r] = rowsum * mat[r*I + r];
                    }
            }
        }
    }

    //blow up weight and bias
    if ( scaling ){
        for(int i = 0; i < I; i++){
            for(int ii = 0; ii < I; ii++){
                for(int j = 0; j < group_size; j++){
                    for(int l = 0; l < B; l++){
                        gn_weight_temp[i*I*group_size*B + ii*group_size*B + j*B + l] = weight_orig[i*I*group_size + ii*group_size + ((j*B + l)%group_size)];
                    }
                }
            }
        }
        
        for(int i = 0; i < I; i++){
            for(int j = 0; j < group_size; j++){
                for(int l = 0; l < B; l++){
                    gn_bias_temp[i*group_size*B + j*B + l] = bias_orig[i*group_size + ((j*B + l)%group_size)];
                }
            }
        }
    }

    //scaling or just copy over
    if ( scaling ){
        //permute weight from (I0, I1, dim0) to (dim0, I0, I1)
        //think their code is wrong since they dont permute the bias and just reshape it...
        for(int i = 0; i < I*I; i++){
            for(int j = 0; j < dim0; j++){
                gn_weight_perm_temp[j*I*I + i] = gn_weight_temp[i*dim0 + j];
            }
        }

        //mat mult and add 
        float vecmult[8]; //put this on the stack since 3D is max size anyways
        for(int i = 0; i < dim0; i++){
            for(int j = 0; j < num_groups; j++){
                for(int l = 0; l < D; l++){
                    //xn = Wx + b
                    float *W = gn_weight_perm_temp + i*I*I;
                    float *xv = x_norm + i*num_groups*D*I + j*D*I + l*I;
                    float *b = gn_bias_temp + i*I;
                    //(I,I)x(I,1)
                    for(int ii = 0; ii < I; ii++){
                        vecmult[ii] = 0.0f;
                        for(int jj = 0; jj < I; jj++){
                            FLOP(1);
                            FLOP(1);
                            vecmult[ii] += W[ii*I + jj] * xv[jj];
                        }
                        FLOP(1);
                        vecmult[ii] += b[ii];
                    }
                    for(int ii = 0; ii < I; ii++){
                        xv[ii] = vecmult[ii];
                    }
                }
            }
        }
    }

    return;
}

static void optimized_groupnorm_v1(
    const float *x,
    int           B,
    int           C,
    int           D,
    int           I,
    int           num_groups,
    bool          running,
    float        *running_mean_orig,
    float        *running_cov_orig,
    bool          scaling,
    const float  *weight_orig,
    const float  *bias_orig,
    bool          training,
    float         momentum,
    float         eps,
    float        *x_norm
) {

    // Track bytes for input data
    size_t input_bytes = B * C * D * I * sizeof(float);
    BYTES_READ(input_bytes);
    
    // Track bytes for running mean and covariance if used
    if (running) {
        size_t running_mean_bytes = I * (C / num_groups) * sizeof(float);
        size_t running_cov_bytes = I * I * (C / num_groups) * sizeof(float);
        BYTES_READ(running_mean_bytes);
        BYTES_READ(running_cov_bytes);
        BYTES_WRITTEN(running_mean_bytes);
        BYTES_WRITTEN(running_cov_bytes);
    }
    
    // Track bytes for weight and bias if used
    if (scaling) {
        if (weight_orig) {
            size_t weight_bytes = I * (C / num_groups) * sizeof(float);
            BYTES_READ(weight_bytes);
        }
        if (bias_orig) {
            size_t bias_bytes = I * (C / num_groups) * sizeof(float);
            BYTES_READ(bias_bytes);
        }
    }
    
    // Track bytes for output data
    size_t output_bytes = B * C * D * I * sizeof(float);
    BYTES_WRITTEN(output_bytes);

    int dim0 = (B*C)/num_groups;
    int group_size = C/num_groups;
    
    assert(I == 8); //optimized for 3D

    //mean
    /*
    int mean_size = dim0 * I;
    for (int i = 0; i < mean_size; i++){
        gn_mean_temp[i] = 0.0f;
    }
    */
   __m256 mean_scale = _mm256_set1_ps(1.0 / ((float)(D*num_groups)));
    for(int i = 0; i < dim0; i++){
        //calc mean
        __m256 accum = _mm256_setzero_ps();
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){
                __m256 row = _mm256_loadu_ps(&x[i*num_groups*D*I + j*D*I + k*I]);
                FLOP(8);
                accum = _mm256_add_ps(accum, row);
                /*
                for(int l = 0; l < I; l++){ //COST: dim0*num_groups*D*I = B*C*D*I flops
                    gn_mean_temp[i*I + l] += x[i*num_groups*D*I + j*D*I + k*I + l];
                }
                */
            }
        }
        FLOP(8);
        accum = _mm256_mul_ps(accum, mean_scale);
        //center around mean
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){
                __m256 row = _mm256_loadu_ps(&x[i*num_groups*D*I + j*D*I + k*I]);
                FLOP(8);
                row = _mm256_sub_ps(row, accum);
                _mm256_storeu_ps(&x_norm[i*num_groups*D*I + j*D*I + k*I], row);
            }
        }
    }
    /*
    for(int i = 0; i < dim0; i++){
        for(int l = 0; l < I; l++){ //COST: dim0*I
            FLOP(1);
            gn_mean_temp[i*I + l] /= (D*num_groups);
        }
    }
    */

    //subtract mean form x
    /*
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){
                for(int l = 0; l < I; l++){
                    FLOP(1);
                    x_norm[i*num_groups*D*I + j*D*I + k*I + l] = x[i*num_groups*D*I + j*D*I + k*I + l] - gn_mean_temp[i*I + l];
                }
            }
        }
    }
    */

    //permute: dim0, num_groups, D, I -> dim0, I, num_groups, D
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){
                for(int l = 0; l < I; l++){
                    gn_X_temp[i*I*num_groups*D + l*num_groups*D + j*D +k] = x_norm[i*num_groups*D*I + j*D*I + k*I + l];
                }
            }
        }
    }

    //calc cov
    int cov_size = dim0*I*I;
    for (int i = 0; i < cov_size; i++){
        gn_cov_temp[i] = 0.0f;
    }
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < I; j++){ //row selector
            for(int l = 0; l < I; l++){ //col selector
                for(int k = 0; k < num_groups*D; k++){
                    FLOP(1);
                    FLOP(1);
                    gn_cov_temp[i*I*I + j*I + l] += gn_X_temp[i*I*num_groups*D + j*num_groups*D + k]* x_norm[i*num_groups*D*I + k*I + l];
    }}}}
    //scale all of cov
    for(int i = 0; i < cov_size; i++){
        FLOP(1);
        gn_cov_temp[i] /= (num_groups * D);
    }

    //get max for each groups cov matrix
    for(int i = 0; i < dim0; i++){
        gn_max_temp[i] = gn_cov_temp[i*I*I];
        for(int j = 1; j < I*I; j++){
            gn_max_temp[i] = gn_cov_temp[i*I*I + j] > gn_max_temp[i] ? gn_cov_temp[i*I*I + j] : gn_max_temp[i];
        }
    }

    //add pertubation to diagonal
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < I; j++){
            FLOP(2);
            gn_cov_temp[i*I*I + j*I + j] += gn_max_temp[i]*eps;
    }}

    //cholesky decomp
    //Src: https://github.com/mdwarfgeek/lib/blob/master/cholesky.c
    /* In-place Cholesky decomposition for L in A = L L^T where L is a
    lower triangular matrix.  Uses the Cholesky-Crout algorithm,
    based on Stoer & Bulirsch "Introduction to numerical analysis"
    Sect. 4.3, but with the loops for i = j and i != j separated and
    changed to access only the lower triangular part of the matrix.
    Output is computed column by column so the accesses in the
    innermost loop (indexed by k) are sequential in memory. */
    for(int i = 0; i < dim0; i++){
        int n = I;
        float *a = &gn_cov_temp[i*I*I];
        float x, r;
        int i, j, k;
        /* Loop over columns */
        for(j = 0; j < n; j++) {
            /* i = j */
            x = a[j*n+j];  /* A_jj */

            for(k = 0; k < j; k++){
                FLOP(2);
                x -= a[j*n+k] * a[j*n+k];  /* L_jk L_jk */
            }

            if(x < 0)
                printf("clofford_groupnorm: Cholesky decomposition x was negative\n");

            FLOP(30);
            x = sqrt(x);

            a[j*n+j] = x;  /* L_jj */
            FLOP(1);
            r = 1.0 / x;

            /* i != j */
            for(i = j+1; i < n; i++) {
                x = a[i*n+j];  /* A_ij */

                for(k = 0; k < j; k++){
                    FLOP(2);
                    x -= a[i*n+k] * a[j*n+k];  /* L_ik L_ij */
                }
                FLOP(1);
                a[i*n+j] = x * r;  /* L_ij = x / L_jj */
            }
        }
        //baseline just transpose back inplace 
        for(i = 0; i < n; i++){
            for(j = i; j < n; j++){
                x = a[i*n +j];
                a[i*n +j] = a[j*n+i];
                a[j*n+i] = x;
            }
        }
    }

    //invert the diagaonal first then
    for(int i = 0; i < dim0; i++){
        for(int d = 0; d < I; d++){
            gn_cov_temp[i*I*I + d*I + d] = 1.0f / gn_cov_temp[i*I*I + d*I + d];
            FLOP(1);
        }
    }

    //solve triangular with the already inverted diagonal
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){ //cost: BCDI(I-1)/2 mults and adds, BCDI div
                    float *mat = gn_cov_temp + (i*I*I);
                    float *vec = x_norm + (i*num_groups*D*I + j*D*I + k*I);
                    for(int r = I-1; r >= 0; r--){//from last row to first
                        float rowsum = 0.0f;
                        for(int c = I-1; c > r; c--){//do the partioal vec row mult
                            FLOP(1);
                            FLOP(1);
                            rowsum += mat[r*I + c]*vec[c];
                        }
                        FLOP(1);
                        rowsum = vec[r] - rowsum;
                        FLOP(1);
                        vec[r] = rowsum * mat[r*I + r];
                    }
            }
        }
    }

    //blow up weight and bias
    if ( scaling ){
        for(int i = 0; i < I; i++){
            for(int ii = 0; ii < I; ii++){
                for(int j = 0; j < group_size; j++){
                    for(int l = 0; l < B; l++){
                        gn_weight_temp[i*I*group_size*B + ii*group_size*B + j*B + l] = weight_orig[i*I*group_size + ii*group_size + ((j*B + l)%group_size)];
                    }
                }
            }
        }
        
        for(int i = 0; i < I; i++){
            for(int j = 0; j < group_size; j++){
                for(int l = 0; l < B; l++){
                    gn_bias_temp[i*group_size*B + j*B + l] = bias_orig[i*group_size + ((j*B + l)%group_size)];
                }
            }
        }
    
        //scaling or just copy over
        //permute weight from (I0, I1, dim0) to (dim0, I0, I1)
        //think their code is wrong since they dont permute the bias and just reshape it...
        for(int i = 0; i < I*I; i++){
            for(int j = 0; j < dim0; j++){
                gn_weight_perm_temp[j*I*I + i] = gn_weight_temp[i*dim0 + j];
            }
        }

        //mat mult and add 
        float vecmult[8]; //put this on the stack since 3D is max size anyways
        for(int i = 0; i < dim0; i++){
            for(int j = 0; j < num_groups; j++){
                for(int l = 0; l < D; l++){
                    //xn = Wx + b
                    float *W = gn_weight_perm_temp + i*I*I;
                    float *xv = x_norm + i*num_groups*D*I + j*D*I + l*I;
                    float *b = gn_bias_temp + i*I;
                    //(I,I)x(I,1)
                    for(int ii = 0; ii < I; ii++){
                        vecmult[ii] = 0.0f;
                        for(int jj = 0; jj < I; jj++){
                            FLOP(1);
                            FLOP(1);
                            vecmult[ii] += W[ii*I + jj] * xv[jj];
                        }
                        FLOP(1);
                        vecmult[ii] += b[ii];
                    }
                    for(int ii = 0; ii < I; ii++){
                        xv[ii] = vecmult[ii];
                    }
                }
            }
        }
    }

    return;
}


static void optimized_groupnorm_v2(
    const float *x,
    int           B,
    int           C,
    int           D,
    int           I,
    int           num_groups,
    bool          running,
    float        *running_mean_orig,
    float        *running_cov_orig,
    bool          scaling,
    const float  *weight_orig,
    const float  *bias_orig,
    bool          training,
    float         momentum,
    float         eps,
    float        *x_norm
) {

    // Track bytes for input data
    size_t input_bytes = B * C * D * I * sizeof(float);
    BYTES_READ(input_bytes);
    
    // Track bytes for running mean and covariance if used
    if (running) {
        size_t running_mean_bytes = I * (C / num_groups) * sizeof(float);
        size_t running_cov_bytes = I * I * (C / num_groups) * sizeof(float);
        BYTES_READ(running_mean_bytes);
        BYTES_READ(running_cov_bytes);
        BYTES_WRITTEN(running_mean_bytes);
        BYTES_WRITTEN(running_cov_bytes);
    }
    
    // Track bytes for weight and bias if used
    if (scaling) {
        if (weight_orig) {
            size_t weight_bytes = I * (C / num_groups) * sizeof(float);
            BYTES_READ(weight_bytes);
        }
        if (bias_orig) {
            size_t bias_bytes = I * (C / num_groups) * sizeof(float);
            BYTES_READ(bias_bytes);
        }
    }
    
    // Track bytes for output data
    size_t output_bytes = B * C * D * I * sizeof(float);
    BYTES_WRITTEN(output_bytes);

    int dim0 = (B*C)/num_groups;
    int group_size = C/num_groups;
    
    assert(I == 8); //optimized for 3D

    //mean
   __m256 mean_scale = _mm256_set1_ps(1.0 / ((float)(D*num_groups)));
    for(int i = 0; i < dim0; i++){
        //calc mean
        __m256 accum = _mm256_setzero_ps();
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){
                __m256 row = _mm256_loadu_ps(&x[i*num_groups*D*I + j*D*I + k*I]);
                FLOP(8);
                accum = _mm256_add_ps(accum, row);
            }
        }
        FLOP(8);
        accum = _mm256_mul_ps(accum, mean_scale);
        //center around mean
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){
                __m256 row = _mm256_loadu_ps(&x[i*num_groups*D*I + j*D*I + k*I]);
                FLOP(8);
                row = _mm256_sub_ps(row, accum);
                _mm256_storeu_ps(&x_norm[i*num_groups*D*I + j*D*I + k*I], row);
            }
        }
    }

    /*
    //permute: dim0, num_groups, D, I -> dim0, I, num_groups, D
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){
                for(int l = 0; l < I; l++){
                    gn_X_temp[i*I*num_groups*D + l*num_groups*D + j*D +k] = x_norm[i*num_groups*D*I + j*D*I + k*I + l];
                }
            }
        }
    }
    */

    //calc cov
    __m256 cov_scale = _mm256_set1_ps(1.0 / ((float)(D*num_groups)));
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < I; j++){ //row selector
            __m256 accum = _mm256_setzero_ps();
            for(int k = 0; k < num_groups*D; k++){
                __m256 j_part = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + k*I + j]);
                __m256 l_part = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + k*I]);
                FLOP(16);
                accum = _mm256_fmadd_ps(j_part, l_part, accum);
            }
            FLOP(8);
            accum = _mm256_mul_ps(accum, cov_scale);
            _mm256_storeu_ps(&gn_cov_temp[i*I*I + j*I], accum);
        }
    }
    /*
    int cov_size = dim0*I*I;
    for (int i = 0; i < cov_size; i++){
        gn_cov_temp[i] = 0.0f;
    }
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < I; j++){ //row selector
            for(int l = 0; l < I; l++){ //col selector
                for(int k = 0; k < num_groups*D; k++){
                    FLOP(1);
                    FLOP(1);
                    gn_cov_temp[i*I*I + j*I + l] += gn_X_temp[i*I*num_groups*D + j*num_groups*D + k]* x_norm[i*num_groups*D*I + k*I + l];
    }}}}
    //scale all of cov
    for(int i = 0; i < cov_size; i++){
        FLOP(1);
        gn_cov_temp[i] /= (num_groups * D);
    }
    */

    //get max for each groups cov matrix
    for(int i = 0; i < dim0; i++){
        gn_max_temp[i] = gn_cov_temp[i*I*I];
        for(int j = 1; j < I*I; j++){
            gn_max_temp[i] = gn_cov_temp[i*I*I + j] > gn_max_temp[i] ? gn_cov_temp[i*I*I + j] : gn_max_temp[i];
        }
    }

    //add pertubation to diagonal
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < I; j++){
            FLOP(2);
            gn_cov_temp[i*I*I + j*I + j] += gn_max_temp[i]*eps;
    }}

    //cholesky decomp
    //Src: https://github.com/mdwarfgeek/lib/blob/master/cholesky.c
    /* In-place Cholesky decomposition for L in A = L L^T where L is a
    lower triangular matrix.  Uses the Cholesky-Crout algorithm,
    based on Stoer & Bulirsch "Introduction to numerical analysis"
    Sect. 4.3, but with the loops for i = j and i != j separated and
    changed to access only the lower triangular part of the matrix.
    Output is computed column by column so the accesses in the
    innermost loop (indexed by k) are sequential in memory. */
    for(int i = 0; i < dim0; i++){
        int n = I;
        float *a = &gn_cov_temp[i*I*I];
        float x, r;
        int i, j, k;
        /* Loop over columns */
        for(j = 0; j < n; j++) {
            /* i = j */
            x = a[j*n+j];  /* A_jj */

            for(k = 0; k < j; k++){
                FLOP(2);
                x -= a[j*n+k] * a[j*n+k];  /* L_jk L_jk */
            }

            if(x < 0)
                printf("clofford_groupnorm: Cholesky decomposition x was negative\n");

            FLOP(30);
            x = sqrt(x);

            a[j*n+j] = x;  /* L_jj */
            FLOP(1);
            r = 1.0 / x;

            /* i != j */
            for(i = j+1; i < n; i++) {
                x = a[i*n+j];  /* A_ij */

                for(k = 0; k < j; k++){
                    FLOP(2);
                    x -= a[i*n+k] * a[j*n+k];  /* L_ik L_ij */
                }
                FLOP(1);
                a[i*n+j] = x * r;  /* L_ij = x / L_jj */
            }
        }
        //baseline just transpose back inplace 
        for(i = 0; i < n; i++){
            for(j = i; j < n; j++){
                x = a[i*n +j];
                a[i*n +j] = a[j*n+i];
                a[j*n+i] = x;
            }
        }
    }

    //invert the diagaonal first then
    for(int i = 0; i < dim0; i++){
        for(int d = 0; d < I; d++){
            gn_cov_temp[i*I*I + d*I + d] = 1.0f / gn_cov_temp[i*I*I + d*I + d];
            FLOP(1);
        }
    }

    //solve triangular with the already inverted diagonal
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){ //cost: BCDI(I-1)/2 mults and adds, BCDI div
                    float *mat = gn_cov_temp + (i*I*I);
                    float *vec = x_norm + (i*num_groups*D*I + j*D*I + k*I);
                    for(int r = I-1; r >= 0; r--){//from last row to first
                        float rowsum = 0.0f;
                        for(int c = I-1; c > r; c--){//do the partioal vec row mult
                            FLOP(1);
                            FLOP(1);
                            rowsum += mat[r*I + c]*vec[c];
                        }
                        FLOP(1);
                        rowsum = vec[r] - rowsum;
                        FLOP(1);
                        vec[r] = rowsum * mat[r*I + r];
                    }
            }
        }
    }

    //blow up weight and bias
    if ( scaling ){
        for(int i = 0; i < I; i++){
            for(int ii = 0; ii < I; ii++){
                for(int j = 0; j < group_size; j++){
                    for(int l = 0; l < B; l++){
                        gn_weight_temp[i*I*group_size*B + ii*group_size*B + j*B + l] = weight_orig[i*I*group_size + ii*group_size + ((j*B + l)%group_size)];
                    }
                }
            }
        }
        
        for(int i = 0; i < I; i++){
            for(int j = 0; j < group_size; j++){
                for(int l = 0; l < B; l++){
                    gn_bias_temp[i*group_size*B + j*B + l] = bias_orig[i*group_size + ((j*B + l)%group_size)];
                }
            }
        }
    
        //scaling or just copy over
        //permute weight from (I0, I1, dim0) to (dim0, I0, I1)
        //think their code is wrong since they dont permute the bias and just reshape it...
        for(int i = 0; i < I*I; i++){
            for(int j = 0; j < dim0; j++){
                gn_weight_perm_temp[j*I*I + i] = gn_weight_temp[i*dim0 + j];
            }
        }

        //mat mult and add 
        float vecmult[8]; //put this on the stack since 3D is max size anyways
        for(int i = 0; i < dim0; i++){
            for(int j = 0; j < num_groups; j++){
                for(int l = 0; l < D; l++){
                    //xn = Wx + b
                    float *W = gn_weight_perm_temp + i*I*I;
                    float *xv = x_norm + i*num_groups*D*I + j*D*I + l*I;
                    float *b = gn_bias_temp + i*I;
                    //(I,I)x(I,1)
                    for(int ii = 0; ii < I; ii++){
                        vecmult[ii] = 0.0f;
                        for(int jj = 0; jj < I; jj++){
                            FLOP(1);
                            FLOP(1);
                            vecmult[ii] += W[ii*I + jj] * xv[jj];
                        }
                        FLOP(1);
                        vecmult[ii] += b[ii];
                    }
                    for(int ii = 0; ii < I; ii++){
                        xv[ii] = vecmult[ii];
                    }
                }
            }
        }
    }

    return;
}


static void optimized_groupnorm_v3(
    const float *x,
    int           B,
    int           C,
    int           D,
    int           I,
    int           num_groups,
    bool          running,
    float        *running_mean_orig,
    float        *running_cov_orig,
    bool          scaling,
    const float  *weight_orig,
    const float  *bias_orig,
    bool          training,
    float         momentum,
    float         eps,
    float        *x_norm
) {

    // Track bytes for input data
    size_t input_bytes = B * C * D * I * sizeof(float);
    BYTES_READ(input_bytes);
    
    // Track bytes for running mean and covariance if used
    if (running) {
        size_t running_mean_bytes = I * (C / num_groups) * sizeof(float);
        size_t running_cov_bytes = I * I * (C / num_groups) * sizeof(float);
        BYTES_READ(running_mean_bytes);
        BYTES_READ(running_cov_bytes);
        BYTES_WRITTEN(running_mean_bytes);
        BYTES_WRITTEN(running_cov_bytes);
    }
    
    // Track bytes for weight and bias if used
    if (scaling) {
        if (weight_orig) {
            size_t weight_bytes = I * (C / num_groups) * sizeof(float);
            BYTES_READ(weight_bytes);
        }
        if (bias_orig) {
            size_t bias_bytes = I * (C / num_groups) * sizeof(float);
            BYTES_READ(bias_bytes);
        }
    }
    
    // Track bytes for output data
    size_t output_bytes = B * C * D * I * sizeof(float);
    BYTES_WRITTEN(output_bytes);

    int dim0 = (B*C)/num_groups;
    int group_size = C/num_groups;
    
    assert(I == 8); //optimized for 3D
    assert(dim0 >= 8); //use this different now so needs to be at least 8 long

    //mean
   __m256 mean_scale = _mm256_set1_ps(1.0 / ((float)(D*num_groups)));
    for(int i = 0; i < dim0; i++){
        //calc mean
        __m256 accum = _mm256_setzero_ps();
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){
                __m256 row = _mm256_loadu_ps(&x[i*num_groups*D*I + j*D*I + k*I]);
                FLOP(8);
                accum = _mm256_add_ps(accum, row);
            }
        }
        FLOP(8);
        accum = _mm256_mul_ps(accum, mean_scale);
        //center around mean
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){
                __m256 row = _mm256_loadu_ps(&x[i*num_groups*D*I + j*D*I + k*I]);
                FLOP(8);
                row = _mm256_sub_ps(row, accum);
                _mm256_storeu_ps(&x_norm[i*num_groups*D*I + j*D*I + k*I], row);
            }
        }
    }

    //calc cov
    __m256 cov_scale = _mm256_set1_ps(1.0 / ((float)(D*num_groups)));
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < I; j++){ //row selector
            __m256 accum = _mm256_setzero_ps();
            for(int k = 0; k < num_groups*D; k++){
                __m256 j_part = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + k*I + j]);
                __m256 l_part = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + k*I]);
                FLOP(16);
                accum = _mm256_fmadd_ps(j_part, l_part, accum);
            }
            FLOP(8);
            accum = _mm256_mul_ps(accum, cov_scale);
            _mm256_storeu_ps(&gn_cov_temp[i*I*I + j*I], accum);
        }
    }

    for(int i = 0; i < dim0; i++){
        __m256 maxv = _mm256_loadu_ps(&gn_cov_temp[i*I*I]);
        //float max = gn_cov_temp[i*I*I];
        for(int j = 1; j < I; j++){
            __m256 row = _mm256_loadu_ps(&gn_cov_temp[i*I*I + j*I]);
            maxv = _mm256_max_ps(maxv, row);
            //max = fmaxf(max, gn_cov_temp[i*I*I + j]);
        }
        _mm256_storeu_ps(gn_max_temp, maxv);
        float max = gn_max_temp[0];
        for(int j = 1; j < I; j++){
            max = gn_max_temp[j] > max ? gn_max_temp[j] : max;
        }
        FLOP(1);
        max = max*eps;
        for(int j = 0; j < I; j++){
            FLOP(1);
            gn_cov_temp[i*I*I + j*I + j] += max;
        }
    }
    
    /*
    //get max for each groups cov matrix
    for(int i = 0; i < dim0; i++){
        gn_max_temp[i] = gn_cov_temp[i*I*I];
        for(int j = 1; j < I*I; j++){
            gn_max_temp[i] = gn_cov_temp[i*I*I + j] > gn_max_temp[i] ? gn_cov_temp[i*I*I + j] : gn_max_temp[i];
        }
    }

    //add pertubation to diagonal
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < I; j++){
            FLOP(2);
            gn_cov_temp[i*I*I + j*I + j] += gn_max_temp[i]*eps;
        }
    }
    */

    //cholesky decomp
    //Src: https://github.com/mdwarfgeek/lib/blob/master/cholesky.c
    /* In-place Cholesky decomposition for L in A = L L^T where L is a
    lower triangular matrix.  Uses the Cholesky-Crout algorithm,
    based on Stoer & Bulirsch "Introduction to numerical analysis"
    Sect. 4.3, but with the loops for i = j and i != j separated and
    changed to access only the lower triangular part of the matrix.
    Output is computed column by column so the accesses in the
    innermost loop (indexed by k) are sequential in memory. */
    for(int i = 0; i < dim0; i++){
        int n = I;
        float *a = &gn_cov_temp[i*I*I];
        float x, r;
        int i, j, k;
        /* Loop over columns */
        for(j = 0; j < n; j++) {
            /* i = j */
            x = a[j*n+j];  /* A_jj */

            for(k = 0; k < j; k++){
                FLOP(2);
                x -= a[j*n+k] * a[j*n+k];  /* L_jk L_jk */
            }

            if(x < 0)
                printf("clofford_groupnorm: Cholesky decomposition x was negative\n");

            FLOP(30);
            x = sqrt(x);

            a[j*n+j] = x;  /* L_jj */
            FLOP(1);
            r = 1.0 / x;

            /* i != j */
            for(i = j+1; i < n; i++) {
                x = a[i*n+j];  /* A_ij */

                for(k = 0; k < j; k++){
                    FLOP(2);
                    x -= a[i*n+k] * a[j*n+k];  /* L_ik L_ij */
                }
                FLOP(1);
                a[i*n+j] = x * r;  /* L_ij = x / L_jj */
            }
        }
        //baseline just transpose back inplace 
        for(i = 0; i < n; i++){
            for(j = i; j < n; j++){
                x = a[i*n +j];
                a[i*n +j] = a[j*n+i];
                a[j*n+i] = x;
            }
        }
    }

    //invert the diagaonal first then
    for(int i = 0; i < dim0; i++){
        for(int d = 0; d < I; d++){
            gn_cov_temp[i*I*I + d*I + d] = 1.0f / gn_cov_temp[i*I*I + d*I + d];
            FLOP(1);
        }
    }

    //solve triangular with the already inverted diagonal
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){ //cost: BCDI(I-1)/2 mults and adds, BCDI div
                    float *mat = gn_cov_temp + (i*I*I);
                    float *vec = x_norm + (i*num_groups*D*I + j*D*I + k*I);
                    for(int r = I-1; r >= 0; r--){//from last row to first
                        float rowsum = 0.0f;
                        for(int c = I-1; c > r; c--){//do the partioal vec row mult
                            FLOP(1);
                            FLOP(1);
                            rowsum += mat[r*I + c]*vec[c];
                        }
                        FLOP(1);
                        rowsum = vec[r] - rowsum;
                        FLOP(1);
                        vec[r] = rowsum * mat[r*I + r];
                    }
            }
        }
    }

    //blow up weight and bias
    if ( scaling ){
        for(int i = 0; i < I; i++){
            for(int ii = 0; ii < I; ii++){
                for(int j = 0; j < group_size; j++){
                    for(int l = 0; l < B; l++){
                        gn_weight_temp[i*I*group_size*B + ii*group_size*B + j*B + l] = weight_orig[i*I*group_size + ii*group_size + ((j*B + l)%group_size)];
                    }
                }
            }
        }
        
        for(int i = 0; i < I; i++){
            for(int j = 0; j < group_size; j++){
                for(int l = 0; l < B; l++){
                    gn_bias_temp[i*group_size*B + j*B + l] = bias_orig[i*group_size + ((j*B + l)%group_size)];
                }
            }
        }
    
        //scaling or just copy over
        //permute weight from (I0, I1, dim0) to (dim0, I0, I1)
        //think their code is wrong since they dont permute the bias and just reshape it...
        for(int i = 0; i < I*I; i++){
            for(int j = 0; j < dim0; j++){
                gn_weight_perm_temp[j*I*I + i] = gn_weight_temp[i*dim0 + j];
            }
        }

        //mat mult and add 
        float vecmult[8]; //put this on the stack since 3D is max size anyways
        for(int i = 0; i < dim0; i++){
            for(int j = 0; j < num_groups; j++){
                for(int l = 0; l < D; l++){
                    //xn = Wx + b
                    float *W = gn_weight_perm_temp + i*I*I;
                    float *xv = x_norm + i*num_groups*D*I + j*D*I + l*I;
                    float *b = gn_bias_temp + i*I;
                    //(I,I)x(I,1)
                    for(int ii = 0; ii < I; ii++){
                        vecmult[ii] = 0.0f;
                        for(int jj = 0; jj < I; jj++){
                            FLOP(1);
                            FLOP(1);
                            vecmult[ii] += W[ii*I + jj] * xv[jj];
                        }
                        FLOP(1);
                        vecmult[ii] += b[ii];
                    }
                    for(int ii = 0; ii < I; ii++){
                        xv[ii] = vecmult[ii];
                    }
                }
            }
        }
    }

    return;
}


static void optimized_groupnorm_v4(
    const float *x,
    int           B,
    int           C,
    int           D,
    int           I,
    int           num_groups,
    bool          running,
    float        *running_mean_orig,
    float        *running_cov_orig,
    bool          scaling,
    const float  *weight_orig,
    const float  *bias_orig,
    bool          training,
    float         momentum,
    float         eps,
    float        *x_norm
) {

    // Track bytes for input data
    size_t input_bytes = B * C * D * I * sizeof(float);
    BYTES_READ(input_bytes);
    
    // Track bytes for running mean and covariance if used
    if (running) {
        size_t running_mean_bytes = I * (C / num_groups) * sizeof(float);
        size_t running_cov_bytes = I * I * (C / num_groups) * sizeof(float);
        BYTES_READ(running_mean_bytes);
        BYTES_READ(running_cov_bytes);
        BYTES_WRITTEN(running_mean_bytes);
        BYTES_WRITTEN(running_cov_bytes);
    }
    
    // Track bytes for weight and bias if used
    if (scaling) {
        if (weight_orig) {
            size_t weight_bytes = I * (C / num_groups) * sizeof(float);
            BYTES_READ(weight_bytes);
        }
        if (bias_orig) {
            size_t bias_bytes = I * (C / num_groups) * sizeof(float);
            BYTES_READ(bias_bytes);
        }
    }
    
    // Track bytes for output data
    size_t output_bytes = B * C * D * I * sizeof(float);
    BYTES_WRITTEN(output_bytes);

    int dim0 = (B*C)/num_groups;
    int group_size = C/num_groups;
    
    assert(I == 8); //optimized for 3D
    assert(dim0 >= 8); //use this different now so needs to be at least 8 long
    assert((D*num_groups)%8 == 0); //needs to be divisible for backsub vectorization

    //mean
   __m256 mean_scale = _mm256_set1_ps(1.0 / ((float)(D*num_groups)));
    for(int i = 0; i < dim0; i++){
        //calc mean
        __m256 accum = _mm256_setzero_ps();
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){
                __m256 row = _mm256_loadu_ps(&x[i*num_groups*D*I + j*D*I + k*I]);
                FLOP(8);
                accum = _mm256_add_ps(accum, row);
            }
        }
        FLOP(8);
        accum = _mm256_mul_ps(accum, mean_scale);
        //center around mean
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){
                __m256 row = _mm256_loadu_ps(&x[i*num_groups*D*I + j*D*I + k*I]);
                FLOP(8);
                row = _mm256_sub_ps(row, accum);
                _mm256_storeu_ps(&x_norm[i*num_groups*D*I + j*D*I + k*I], row);
            }
        }
    }

    //calc cov
    __m256 cov_scale = _mm256_set1_ps(1.0 / ((float)(D*num_groups)));
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < I; j++){ //row selector
            __m256 accum = _mm256_setzero_ps();
            for(int k = 0; k < num_groups*D; k++){
                __m256 j_part = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + k*I + j]);
                __m256 l_part = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + k*I]);
                FLOP(16);
                accum = _mm256_fmadd_ps(j_part, l_part, accum);
            }
            FLOP(8);
            accum = _mm256_mul_ps(accum, cov_scale);
            _mm256_storeu_ps(&gn_cov_temp[i*I*I + j*I], accum);
        }
    }

    for(int i = 0; i < dim0; i++){
        __m256 maxv = _mm256_loadu_ps(&gn_cov_temp[i*I*I]);
        //float max = gn_cov_temp[i*I*I];
        for(int j = 1; j < I; j++){
            __m256 row = _mm256_loadu_ps(&gn_cov_temp[i*I*I + j*I]);
            maxv = _mm256_max_ps(maxv, row);
            //max = fmaxf(max, gn_cov_temp[i*I*I + j]);
        }
        _mm256_storeu_ps(gn_max_temp, maxv);
        float max = gn_max_temp[0];
        for(int j = 1; j < I; j++){
            max = gn_max_temp[j] > max ? gn_max_temp[j] : max;
        }
        FLOP(1);
        max = max*eps;
        for(int j = 0; j < I; j++){
            FLOP(1);
            gn_cov_temp[i*I*I + j*I + j] += max;
        }
    }
    
    /*
    //get max for each groups cov matrix
    for(int i = 0; i < dim0; i++){
        gn_max_temp[i] = gn_cov_temp[i*I*I];
        for(int j = 1; j < I*I; j++){
            gn_max_temp[i] = gn_cov_temp[i*I*I + j] > gn_max_temp[i] ? gn_cov_temp[i*I*I + j] : gn_max_temp[i];
        }
    }

    //add pertubation to diagonal
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < I; j++){
            FLOP(2);
            gn_cov_temp[i*I*I + j*I + j] += gn_max_temp[i]*eps;
        }
    }
    */

    //cholesky decomp
    //Src: https://github.com/mdwarfgeek/lib/blob/master/cholesky.c
    /* In-place Cholesky decomposition for L in A = L L^T where L is a
    lower triangular matrix.  Uses the Cholesky-Crout algorithm,
    based on Stoer & Bulirsch "Introduction to numerical analysis"
    Sect. 4.3, but with the loops for i = j and i != j separated and
    changed to access only the lower triangular part of the matrix.
    Output is computed column by column so the accesses in the
    innermost loop (indexed by k) are sequential in memory. */
    for(int i = 0; i < dim0; i++){
        int n = I;
        float *a = &gn_cov_temp[i*I*I];
        float x, r;
        int i, j, k;
        /* Loop over columns */
        for(j = 0; j < n; j++) {
            /* i = j */
            x = a[j*n+j];  /* A_jj */

            for(k = 0; k < j; k++){
                FLOP(2);
                x -= a[j*n+k] * a[j*n+k];  /* L_jk L_jk */
            }

            if(x < 0)
                printf("clofford_groupnorm: Cholesky decomposition x was negative\n");

            FLOP(30);
            x = sqrt(x);

            a[j*n+j] = x;  /* L_jj */
            FLOP(1);
            r = 1.0 / x;

            /* i != j */
            for(i = j+1; i < n; i++) {
                x = a[i*n+j];  /* A_ij */

                for(k = 0; k < j; k++){
                    FLOP(2);
                    x -= a[i*n+k] * a[j*n+k];  /* L_ik L_ij */
                }
                FLOP(1);
                a[i*n+j] = x * r;  /* L_ij = x / L_jj */
            }
        }
        //baseline just transpose back inplace 
        for(i = 0; i < n; i++){
            for(j = i; j < n; j++){
                x = a[i*n +j];
                a[i*n +j] = a[j*n+i];
                a[j*n+i] = x;
            }
        }
    }

    //invert the diagaonal first then
    for(int i = 0; i < dim0; i++){
        for(int d = 0; d < I; d++){
            gn_cov_temp[i*I*I + d*I + d] = 1.0f / gn_cov_temp[i*I*I + d*I + d];
            FLOP(1);
        }
    }

    
    //reshape
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups*D; j += 8){
            for (int k = 0; k < 8; k++){
                for(int l = 0; l < I; l++){
                    gn_X_temp[i*num_groups*D*I + j*I + l*I + k] = x_norm[i*num_groups*D*I + j*I + k*I + l];
                }
            }
        }
    }
    
    //vectorized backsubstitution
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups*D; j += 8){
            float *mat = gn_cov_temp + (i*I*I);
            float *vec = gn_X_temp + (i*num_groups*D*I + j*I); //8xI vector
            for(int r = I-1; r >= 0; r--){//from last row to first
                //float rowsum = 0.0f;
                __m256 rowsum = _mm256_setzero_ps();
                for(int c = I-1; c > r; c--){//do the partioal vec row mult
                    //FLOP(1);
                    //FLOP(1);
                    //rowsum += mat[r*I + c]*vec[c];
                    __m256 vec_elem = _mm256_loadu_ps(&vec[c*8]);
                    __m256 mat_elem = _mm256_broadcast_ss(&mat[r*I + c]);
                    FLOP(16);
                    rowsum = _mm256_fmadd_ps(mat_elem, vec_elem, rowsum);
                }
                //FLOP(1);
                //rowsum = vec[r] - rowsum;
                __m256 vec_elem = _mm256_loadu_ps(&vec[r*8]);
                FLOP(8);
                rowsum = _mm256_sub_ps(vec_elem, rowsum);
                //FLOP(1);
                //vec[r] = rowsum * mat[r*I + r];
                __m256 mat_elem = _mm256_broadcast_ss(&mat[r*I + r]);
                FLOP(8);
                __m256 res = _mm256_mul_ps(rowsum, mat_elem);
                _mm256_storeu_ps(&vec[r*8], res);
            }
        }
    }
    
    //reshape
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups*D; j += 8){
            for (int k = 0; k < 8; k++){
                for(int l = 0; l < I; l++){
                    x_norm[i*num_groups*D*I + j*I + k*I + l] = gn_X_temp[i*num_groups*D*I + j*I + l*I + k];
                }
            }
        }
    }
    
    /*
    //solve triangular with the already inverted diagonal
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){ //cost: BCDI(I-1)/2 mults and adds, BCDI div
                    float *mat = gn_cov_temp + (i*I*I);
                    float *vec = x_norm + (i*num_groups*D*I + j*D*I + k*I);
                    for(int r = I-1; r >= 0; r--){//from last row to first
                        float rowsum = 0.0f;
                        for(int c = I-1; c > r; c--){//do the partioal vec row mult
                            FLOP(1);
                            FLOP(1);
                            rowsum += mat[r*I + c]*vec[c];
                        }
                        FLOP(1);
                        rowsum = vec[r] - rowsum;
                        FLOP(1);
                        vec[r] = rowsum * mat[r*I + r];
                    }
            }
        }
    }
    */
    
    //blow up weight and bias
    if ( scaling ){
        for(int i = 0; i < I; i++){
            for(int ii = 0; ii < I; ii++){
                for(int j = 0; j < group_size; j++){
                    for(int l = 0; l < B; l++){
                        gn_weight_temp[i*I*group_size*B + ii*group_size*B + j*B + l] = weight_orig[i*I*group_size + ii*group_size + ((j*B + l)%group_size)];
                    }
                }
            }
        }
        
        for(int i = 0; i < I; i++){
            for(int j = 0; j < group_size; j++){
                for(int l = 0; l < B; l++){
                    gn_bias_temp[i*group_size*B + j*B + l] = bias_orig[i*group_size + ((j*B + l)%group_size)];
                }
            }
        }
    
        //scaling or just copy over
        //permute weight from (I0, I1, dim0) to (dim0, I0, I1)
        //think their code is wrong since they dont permute the bias and just reshape it...
        for(int i = 0; i < I*I; i++){
            for(int j = 0; j < dim0; j++){
                gn_weight_perm_temp[j*I*I + i] = gn_weight_temp[i*dim0 + j];
            }
        }

        //mat mult and add
        float vecmult[8]; //put this on the stack since 3D is max size anyways
        for(int i = 0; i < dim0; i++){
            for(int j = 0; j < num_groups; j++){
                for(int l = 0; l < D; l++){
                    //xn = Wx + b
                    float *W = gn_weight_perm_temp + i*I*I;
                    float *xv = x_norm + i*num_groups*D*I + j*D*I + l*I;
                    float *b = gn_bias_temp + i*I;
                    //(I,I)x(I,1)
                    for(int ii = 0; ii < I; ii++){
                        vecmult[ii] = 0.0f;
                        for(int jj = 0; jj < I; jj++){
                            FLOP(1);
                            FLOP(1);
                            vecmult[ii] += W[ii*I + jj] * xv[jj];
                        }
                        FLOP(1);
                        vecmult[ii] += b[ii];
                    }
                    for(int ii = 0; ii < I; ii++){
                        xv[ii] = vecmult[ii];
                    }
                }
            }
        }
    }

    return;
}

static void optimized_groupnorm_v5(
    const float *x,
    int           B,
    int           C,
    int           D,
    int           I,
    int           num_groups,
    bool          running,
    float        *running_mean_orig,
    float        *running_cov_orig,
    bool          scaling,
    const float  *weight_orig,
    const float  *bias_orig,
    bool          training,
    float         momentum,
    float         eps,
    float        *x_norm
) {

    // Track bytes for input data
    size_t input_bytes = B * C * D * I * sizeof(float);
    BYTES_READ(input_bytes);
    
    // Track bytes for running mean and covariance if used
    if (running) {
        size_t running_mean_bytes = I * (C / num_groups) * sizeof(float);
        size_t running_cov_bytes = I * I * (C / num_groups) * sizeof(float);
        BYTES_READ(running_mean_bytes);
        BYTES_READ(running_cov_bytes);
        BYTES_WRITTEN(running_mean_bytes);
        BYTES_WRITTEN(running_cov_bytes);
    }
    
    // Track bytes for weight and bias if used
    if (scaling) {
        if (weight_orig) {
            size_t weight_bytes = I * (C / num_groups) * sizeof(float);
            BYTES_READ(weight_bytes);
        }
        if (bias_orig) {
            size_t bias_bytes = I * (C / num_groups) * sizeof(float);
            BYTES_READ(bias_bytes);
        }
    }
    
    // Track bytes for output data
    size_t output_bytes = B * C * D * I * sizeof(float);
    BYTES_WRITTEN(output_bytes);

    int dim0 = (B*C)/num_groups;
    int group_size = C/num_groups;
    
    assert(I == 8); //optimized for 3D
    assert(dim0 >= 8); //use this different now so needs to be at least 8 long
    assert((D*num_groups)%8 == 0); //needs to be divisible for backsub vectorization

    //mean
   __m256 mean_scale = _mm256_set1_ps(1.0 / ((float)(D*num_groups)));
    for(int i = 0; i < dim0; i++){
        //calc mean
        __m256 accum = _mm256_setzero_ps();
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){
                __m256 row = _mm256_loadu_ps(&x[i*num_groups*D*I + j*D*I + k*I]);
                FLOP(8);
                accum = _mm256_add_ps(accum, row);
            }
        }
        FLOP(8);
        accum = _mm256_mul_ps(accum, mean_scale);
        //center around mean
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){
                __m256 row = _mm256_loadu_ps(&x[i*num_groups*D*I + j*D*I + k*I]);
                FLOP(8);
                row = _mm256_sub_ps(row, accum);
                _mm256_storeu_ps(&x_norm[i*num_groups*D*I + j*D*I + k*I], row);
            }
        }
    }

    //calc cov
    __m256 cov_scale = _mm256_set1_ps(1.0 / ((float)(D*num_groups)));
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < I; j++){ //row selector
            __m256 accum = _mm256_setzero_ps();
            for(int k = 0; k < num_groups*D; k++){
                __m256 j_part = _mm256_broadcast_ss(&x_norm[i*num_groups*D*I + k*I + j]);
                __m256 l_part = _mm256_loadu_ps(&x_norm[i*num_groups*D*I + k*I]);
                FLOP(16);
                accum = _mm256_fmadd_ps(j_part, l_part, accum);
            }
            FLOP(8);
            accum = _mm256_mul_ps(accum, cov_scale);
            _mm256_storeu_ps(&gn_cov_temp[i*I*I + j*I], accum);
        }
    }

    for(int i = 0; i < dim0; i++){
        __m256 maxv = _mm256_loadu_ps(&gn_cov_temp[i*I*I]);
        //float max = gn_cov_temp[i*I*I];
        for(int j = 1; j < I; j++){
            __m256 row = _mm256_loadu_ps(&gn_cov_temp[i*I*I + j*I]);
            maxv = _mm256_max_ps(maxv, row);
            //max = fmaxf(max, gn_cov_temp[i*I*I + j]);
        }
        _mm256_storeu_ps(gn_max_temp, maxv);
        float max = gn_max_temp[0];
        for(int j = 1; j < I; j++){
            max = gn_max_temp[j] > max ? gn_max_temp[j] : max;
        }
        FLOP(1);
        max = max*eps;
        for(int j = 0; j < I; j++){
            FLOP(1);
            gn_cov_temp[i*I*I + j*I + j] += max;
        }
    }
    
    /*
    //get max for each groups cov matrix
    for(int i = 0; i < dim0; i++){
        gn_max_temp[i] = gn_cov_temp[i*I*I];
        for(int j = 1; j < I*I; j++){
            gn_max_temp[i] = gn_cov_temp[i*I*I + j] > gn_max_temp[i] ? gn_cov_temp[i*I*I + j] : gn_max_temp[i];
        }
    }

    //add pertubation to diagonal
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < I; j++){
            FLOP(2);
            gn_cov_temp[i*I*I + j*I + j] += gn_max_temp[i]*eps;
        }
    }
    */

    //cholesky decomp
    //Src: https://github.com/mdwarfgeek/lib/blob/master/cholesky.c
    /* In-place Cholesky decomposition for L in A = L L^T where L is a
    lower triangular matrix.  Uses the Cholesky-Crout algorithm,
    based on Stoer & Bulirsch "Introduction to numerical analysis"
    Sect. 4.3, but with the loops for i = j and i != j separated and
    changed to access only the lower triangular part of the matrix.
    Output is computed column by column so the accesses in the
    innermost loop (indexed by k) are sequential in memory. */
    for(int i = 0; i < dim0; i++){
        int n = I;
        float *a = &gn_cov_temp[i*I*I];
        float x, r;
        int i, j, k;
        /* Loop over columns */
        for(j = 0; j < n; j++) {
            /* i = j */
            x = a[j*n+j];  /* A_jj */

            for(k = 0; k < j; k++){
                FLOP(2);
                x -= a[j*n+k] * a[j*n+k];  /* L_jk L_jk */
            }

            if(x < 0)
                printf("clofford_groupnorm: Cholesky decomposition x was negative\n");

            FLOP(30);
            x = sqrt(x);

            a[j*n+j] = x;  /* L_jj */
            FLOP(1);
            r = 1.0 / x;

            /* i != j */
            for(i = j+1; i < n; i++) {
                x = a[i*n+j];  /* A_ij */

                for(k = 0; k < j; k++){
                    FLOP(2);
                    x -= a[i*n+k] * a[j*n+k];  /* L_ik L_ij */
                }
                FLOP(1);
                a[i*n+j] = x * r;  /* L_ij = x / L_jj */
            }
        }
        /*
        //baseline just transpose back inplace 
        for(i = 0; i < n; i++){
            for(j = i; j < n; j++){
                x = a[i*n +j];
                a[i*n +j] = a[j*n+i];
                a[j*n+i] = x;
            }
        }
        */
    }

    //invert the diagaonal first then
    for(int i = 0; i < dim0; i++){
        for(int d = 0; d < I; d++){
            gn_cov_temp[i*I*I + d*I + d] = 1.0f / gn_cov_temp[i*I*I + d*I + d];
            FLOP(1);
        }
    }

    
    //reshape
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups*D; j += 8){
            for (int k = 0; k < 8; k++){
                for(int l = 0; l < I; l++){
                    gn_X_temp[i*num_groups*D*I + j*I + l*I + k] = x_norm[i*num_groups*D*I + j*I + k*I + l];
                }
            }
        }
    }
    
    //vectorized backsubstitution
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups*D; j += 8){
            float *mat = gn_cov_temp + (i*I*I);
            float *vec = gn_X_temp + (i*num_groups*D*I + j*I); //8xI vector
            for(int r = I-1; r >= 0; r--){//from last row to first
                //float rowsum = 0.0f;
                __m256 rowsum = _mm256_setzero_ps();
                for(int c = I-1; c > r; c--){//do the partioal vec row mult
                    __m256 vec_elem = _mm256_loadu_ps(&vec[c*8]);
                    __m256 mat_elem = _mm256_broadcast_ss(&mat[c*I + r]);
                    FLOP(16);
                    rowsum = _mm256_fmadd_ps(mat_elem, vec_elem, rowsum);
                }
                __m256 vec_elem = _mm256_loadu_ps(&vec[r*8]);
                FLOP(8);
                rowsum = _mm256_sub_ps(vec_elem, rowsum);
                __m256 mat_elem = _mm256_broadcast_ss(&mat[r*I + r]);
                FLOP(8);
                __m256 res = _mm256_mul_ps(rowsum, mat_elem);
                _mm256_storeu_ps(&vec[r*8], res);
            }
        }
    }
    
    //reshape
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups*D; j += 8){
            for (int k = 0; k < 8; k++){
                for(int l = 0; l < I; l++){
                    x_norm[i*num_groups*D*I + j*I + k*I + l] = gn_X_temp[i*num_groups*D*I + j*I + l*I + k];
                }
            }
        }
    }
    
    //blow up weight and bias
    if ( scaling ){
        for(int i = 0; i < I; i++){
            for(int ii = 0; ii < I; ii++){
                for(int j = 0; j < group_size; j++){
                    for(int l = 0; l < B; l++){
                        gn_weight_temp[i*I*group_size*B + ii*group_size*B + j*B + l] = weight_orig[i*I*group_size + ii*group_size + ((j*B + l)%group_size)];
                    }
                }
            }
        }
        
        for(int i = 0; i < I; i++){
            for(int j = 0; j < group_size; j++){
                for(int l = 0; l < B; l++){
                    gn_bias_temp[i*group_size*B + j*B + l] = bias_orig[i*group_size + ((j*B + l)%group_size)];
                }
            }
        }
    
        //scaling or just copy over
        //permute weight from (I0, I1, dim0) to (dim0, I0, I1)
        //think their code is wrong since they dont permute the bias and just reshape it...
        for(int i = 0; i < I*I; i++){
            for(int j = 0; j < dim0; j++){
                gn_weight_perm_temp[j*I*I + i] = gn_weight_temp[i*dim0 + j];
            }
        }

        //mat mult and add
        float vecmult[8]; //put this on the stack since 3D is max size anyways
        for(int i = 0; i < dim0; i++){
            for(int j = 0; j < num_groups; j++){
                for(int l = 0; l < D; l++){
                    //xn = Wx + b
                    float *W = gn_weight_perm_temp + i*I*I;
                    float *xv = x_norm + i*num_groups*D*I + j*D*I + l*I;
                    float *b = gn_bias_temp + i*I;
                    //(I,I)x(I,1)
                    for(int ii = 0; ii < I; ii++){
                        vecmult[ii] = 0.0f;
                        for(int jj = 0; jj < I; jj++){
                            FLOP(1);
                            FLOP(1);
                            vecmult[ii] += W[ii*I + jj] * xv[jj];
                        }
                        FLOP(1);
                        vecmult[ii] += b[ii];
                    }
                    for(int ii = 0; ii < I; ii++){
                        xv[ii] = vecmult[ii];
                    }
                }
            }
        }
    }

    return;
}


// ——— Register them in the order you want them tested ———
void register_groupnorm_functions(void) {
    add_groupnorm_function(baseline_groupnorm,      "baseline_groupnorm");
    add_groupnorm_function(optimized_groupnorm_v0,  "optimized_groupnorm_v0"); //invert diag back sub
    add_groupnorm_function(optimized_groupnorm_v1,  "optimized_groupnorm_v1"); //mean vectorized
    add_groupnorm_function(optimized_groupnorm_v2,  "optimized_groupnorm_v2"); //cov vectorized
    add_groupnorm_function(optimized_groupnorm_v3,  "optimized_groupnorm_v3"); //max permute temp vars
    add_groupnorm_function(optimized_groupnorm_v4,  "optimized_groupnorm_v4"); //vectorize backsubstitution
    add_groupnorm_function(optimized_groupnorm_v5,  "optimized_groupnorm_v5"); //dont transpose cholesky decomp
}
