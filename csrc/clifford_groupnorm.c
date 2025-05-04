#include "clifford_groupnorm.h"

/*
Assumption we run in training mode

Base:
    Adds: 2BCDI + BCDII + dim0*I + dim0(I^3)/6 + BCDI(I-1)/2 = BCDI(2 + I + (I-1)/2) + dim0I(1+II/6)
    Mults: BCDII + dim0*I + dim0(I^3)/6 + BCDI(I-1)/2 = BCDI(I + (I-1)/2) + dim0I(1+II/6)
    Divs: dim0*I + dim0*I*I + BCDI = dim0I(1+I) + BCDI
    Total: BCDI(2 + 2I + I - 1 + 1) + dim0I(2 + II/3 + 1 + I) = BCDI(3I + 2) + dim0I(II/3 + I + 3)
Running Mean:
    Adds: 2dim0*I + 2dim0*I*I + I*group_size*B + I*I*group_size*B = 3dim0I(I+1)
    Mults: dim0*I + dim0*I*I = dim0I(I+1)
    Divs: I*group_size + I*I*group_size = IIC/num_groups
Weight and Bias:
    Adds: BCDII + BCDI = BCDI(1+I)
    Mults: BCDII
    Divs:


Ideas for Optimization:
- Less memory allocation
- Avoid back transposition of X -> XT = x
- Make sure BLAS/LAPCKE routines are very optimized
- Note the expensive things often have I dimesntions, the only I dimension we need to support are 2,4,8... use super optimized verions for this?
- On the back subsititution we could first norm the diagonal since we do backsub many vectors with one mat
- Generally use vectorization (especially if we have the I dimension)
*/
void clifford_groupnorm_notemp( //wrapper around with memory allocation around clifford_groupnorm_temp
    const float *x,
    int B,
    int C,
    int D,
    int _I,
    int num_groups,
    bool running,
    float *running_mean_orig,
    float *running_cov_orig,
    bool scaling,
    const float *weight_orig,
    const float *bias_orig,
    bool training,
    float momentum,
    float eps,
    float *x_norm){

    //basic constants
    int dim0 = (B*C)/num_groups;
    int group_size = C/num_groups;

    //allocate temp memory (zeroed!!!)
    float * running_mean_temp = NULL;
    float * running_cov_temp = NULL;
    if (running){
        size_t running_mean_size = _I * dim0;
        running_mean_temp = calloc(running_mean_size, sizeof(float));
        size_t running_cov_size = _I* _I * dim0;
        running_cov_temp = calloc(running_cov_size, sizeof(float));
    }
    float *weight_temp = NULL;
    float *bias_temp = NULL;
    float *weight_perm_temp = NULL;
    if (scaling){
        size_t weight_size = _I* _I * dim0;
        weight_temp = calloc(weight_size, sizeof(float));
        size_t bias_size = _I * dim0;
        bias_temp = calloc(bias_size, sizeof(float));
        weight_perm_temp = calloc(weight_size, sizeof(float));
    }
    size_t x_size = dim0 * num_groups * D * _I;
    float * X_temp = calloc(x_size, sizeof(float));
    size_t mean_size = dim0 * _I;
    float * mean_temp = calloc(mean_size, sizeof(float));
    size_t cov_size = dim0*_I*_I;
    float * cov_temp = calloc(cov_size, sizeof(float));
    size_t max_size = dim0;
    float * max_temp = calloc(max_size, sizeof(float));


    //compute
    clifford_groupnorm_temp(
        x,
        B,
        C,
        D,
        _I,
        num_groups,
        running,
        running_mean_orig,
        running_mean_temp,
        running_cov_orig,
        running_cov_temp,
        scaling,
        weight_orig,
        weight_temp,
        weight_perm_temp,
        bias_orig,
        bias_temp,
        mean_temp,
        X_temp,
        cov_temp,
        max_temp,
        training,
        momentum,
        eps,
        x_norm);


    //free temp memory
    if (running){
        free(running_mean_temp);
        free(running_cov_temp);
    }
    if (scaling){
        free(weight_temp);
        free(bias_temp);
        free(weight_perm_temp);
    }
    //free all the temp arrays
    free(max_temp);
    free(cov_temp);
    free(X_temp);
    free(mean_temp);
    return;
}

void clifford_groupnorm_temp( //wrapper around clifford_groupnorm_part0, clifford_groupnorm_lapacke, clifford_groupnorm_part1
    const float *x,
    int B,
    int C,
    int D,
    int _I,
    int num_groups,
    bool running,
    float *running_mean_orig,
    float *running_mean_temp,
    float *running_cov_orig,
    float *running_cov_temp,
    bool scaling,
    const float *weight_orig,
    float *weight_temp,
    float *weight_perm_temp,
    const float *bias_orig,
    float *bias_temp,
    float *mean_temp,
    float *X_temp,
    float *cov_temp,
    float *max_temp,
    bool training,
    float momentum,
    float eps,
    float *x_norm){

    clifford_groupnorm_part0(
        x,
        B,
        C,
        D,
        _I,
        num_groups,
        running,
        running_mean_orig,
        running_mean_temp,
        running_cov_orig,
        running_cov_temp,
        mean_temp,
        X_temp,
        cov_temp,
        max_temp,
        training,
        momentum,
        eps,
        x_norm);

    clifford_groupnorm_lapacke(
        x,
        B,
        C,
        D,
        _I,
        num_groups,
        cov_temp);
    
    clifford_groupnorm_part1(
        x,
        B,
        C,
        D,
        _I,
        num_groups,
        scaling,
        weight_orig,
        weight_temp,
        weight_perm_temp,
        bias_orig,
        bias_temp,
        cov_temp,
        training,
        momentum,
        eps,
        x_norm);

    return;
    }
    

void clifford_groupnorm_part0(
    const float *x,
    int B,
    int C,
    int D,
    int _I,
    int num_groups,
    bool running,
    float *running_mean_orig,
    float *running_mean_temp,
    float *running_cov_orig,
    float *running_cov_temp,
    float *mean_temp,
    float *X_temp,
    float *cov_temp,
    float *max_temp,
    bool training,
    float momentum,
    float eps,
    float *x_norm){

    //basic constants
    int dim0 = (B*C)/num_groups;
    int group_size = C/num_groups;

    //blow up running mean and cov along the batch dim
    if ( running ){
        for(int i = 0; i < _I; i++){ //COST: 0 flops
            for(int j = 0; j < group_size; j++){
                for(int l = 0; l < B; l++){
                    running_mean_temp[i*group_size*B + j*B + l] = running_mean_orig[i*group_size + ((j*B + l)%group_size)];
                }
            }
        }
        for(int i = 0; i < _I; i++){
            for(int ii = 0; ii < _I; ii++){
                for(int j = 0; j < group_size; j++){
                    for(int l = 0; l < B; l++){
                        running_cov_temp[i*_I*group_size*B + ii*group_size*B + j*B + l] = running_cov_orig[i*_I*group_size + ii*group_size + ((j*B + l)%group_size)];
                    }
                }
            }
        }
    }

    //mean
    if (training || (!running)) {
        for(int i = 0; i < dim0; i++){
            for(int j = 0; j < num_groups; j++){
                for(int k = 0; k < D; k++){
                    for(int l = 0; l < _I; l++){ //COST: dim0*num_groups*D*I = B*C*D*I flops
                        mean_temp[i*_I + l] += x[i*num_groups*D*_I + j*D*_I + k*_I + l];
                    }
                }
            }
        }
        for(int i = 0; i < dim0; i++){
            for(int l = 0; l < _I; l++){ //COST: dim0*I
                mean_temp[i*_I + l] /= (D*num_groups);
            }
        }
        if (running){ //update running mean
            for(int i = 0; i < _I; i++){
                for(int j = 0; j < dim0; j++){
                        running_mean_temp[i*dim0 + j] += momentum *(mean_temp[j*_I + i] - running_mean_temp[i*dim0 + j]);
                }
            }
        }
    } else { //if not training use running mean
        for(int i = 0; i < _I; i++){
            for(int j = 0; j < dim0; j++){
                    mean_temp[j*_I + i] = running_mean_temp[i*dim0 + j];
            }
        }
    }

    //subtract mean form x
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){
                for(int l = 0; l < _I; l++){
                    x_norm[i*num_groups*D*_I + j*D*_I + k*_I + l] = x[i*num_groups*D*_I + j*D*_I + k*_I + l] - mean_temp[i*_I + l];
                }
            }
        }
    }

    //permute: dim0, num_groups, D, _I -> dim0, _I, num_groups, D
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){
                for(int l = 0; l < _I; l++){
                    X_temp[i*_I*num_groups*D + l*num_groups*D + j*D +k] = x_norm[i*num_groups*D*_I + j*D*_I + k*_I + l];
                }
            }
        }
    }

    //calc cov
    if (training || (!running)){
        for(int i = 0; i < dim0; i++){
            for(int j = 0; j < _I; j++){ //row selector
                for(int l = 0; l < _I; l++){ //col selector
                    for(int k = 0; k < num_groups*D; k++){
                        cov_temp[i*_I*_I + j*_I + l] += X_temp[i*_I*num_groups*D + j*num_groups*D + k]* x_norm[i*num_groups*D*_I + k*_I + l];
        }}}}
        //scale all of cov
        size_t cov_size = dim0*_I*_I;
        for(int i = 0; i < cov_size; i++){
            cov_temp[i] /= (num_groups * D);
        }
        if (running) { //upadate running cov
            for(int i = 0; i < _I; i++){            
                for(int ii = 0; ii < _I; ii++){
                    for(int j = 0; j < dim0; j++){
                        running_cov_temp[i*_I*dim0 + ii*dim0 + j] += momentum *(cov_temp[j*_I*_I + i*_I + ii] - running_cov_temp[i*_I*dim0 + ii*dim0 + j]);
                    }
                }
            }
        }
    } else {
        for(int i = 0; i < _I; i++){ //if not training use running cov
            for(int ii = 0; ii < _I; ii++){
                for(int j = 0; j < dim0; j++){
                    cov_temp[j*_I*_I + i*_I + ii] = running_cov_temp[i*_I*dim0 + ii*dim0 + j];
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
                    running_mean_orig[i*group_size + j] += running_mean_temp[i*group_size*B + l*group_size + j];
                }
            }
        }
        for(int i = 0; i < _I*group_size; i++){
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
                        running_cov_orig[i*_I*group_size + ii*group_size + j] += running_cov_temp[i*_I*group_size*B + ii*group_size*B + l*group_size + j];
                    }
                }
            }
        }
        for(int i = 0; i < _I*_I*group_size; i++){
            running_cov_orig[i] /= B;
        }

    }

    //get max for each groups cov matrix
    for(int i = 0; i < dim0; i++){
        max_temp[i] = cov_temp[i*_I*_I];
        for(int j = 1; j < _I*_I; j++){
            max_temp[i] = cov_temp[i*_I*_I + j] > max_temp[i] ? cov_temp[i*_I*_I + j] : max_temp[i];
        }
    }

    //add pertubation to diagonal
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < _I; j++){
            cov_temp[i*_I*_I + j*_I + j] += max_temp[i]*eps;
    }}
    return;
}

void clifford_groupnorm_lapacke(
    const float *x,
    int B,
    int C,
    int D,
    int _I,
    int num_groups,
    float *cov_temp){

    //basic constants
    int dim0 = (B*C)/num_groups;
    int group_size = C/num_groups;
        
    //cholesky decomp need to pass U (cov) and result
    int info;
    for(int i = 0; i < dim0; i++){
        //COST: https://en.wikipedia.org/wiki/Cholesky_decomposition (I^3)/6 adds and mults
        info = LAPACKE_spotrf(LAPACK_ROW_MAJOR, 'U', _I, &cov_temp[i*_I*_I], _I); //does not mask of lower half....
        if (info > 0) {printf("The matrix is not positive definite.\n");return;}
        if (info < 0) {printf("Illegal value in argument %d\n", -info);return;}
    }
    return;
}

void clifford_groupnorm_part1(
    const float *x,
    int B,
    int C,
    int D,
    int _I,
    int num_groups,
    bool scaling,
    const float *weight_orig,
    float *weight_temp,
    float *weight_perm_temp,
    const float *bias_orig,
    float *bias_temp,
    float *cov_temp,
    bool training,
    float momentum,
    float eps,
    float *x_norm){

    //basic constants
    int dim0 = (B*C)/num_groups;
    int group_size = C/num_groups;

    //solve triangular
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){ //cost: BCDI(I-1)/2 mults and adds, BCDI div
                    float *mat = cov_temp + (i*_I*_I);
                    float *vec = x_norm + (i*num_groups*D*_I + j*D*_I + k*_I);
                    for(int r = _I-1; r >= 0; r--){//from last row to first
                        float rowsum = 0.0f;
                        for(int c = _I-1; c > r; c--){//do the partioal vec row mult
                            rowsum += mat[r*_I + c]*vec[c];
                        }
                        rowsum = vec[r] - rowsum;
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
                        weight_temp[i*_I*group_size*B + ii*group_size*B + j*B + l] = weight_orig[i*_I*group_size + ii*group_size + ((j*B + l)%group_size)];
                    }
                }
            }
        }
        
        for(int i = 0; i < _I; i++){
            for(int j = 0; j < group_size; j++){
                for(int l = 0; l < B; l++){
                    bias_temp[i*group_size*B + j*B + l] = bias_orig[i*group_size + ((j*B + l)%group_size)];
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
                weight_perm_temp[j*_I*_I + i] = weight_temp[i*dim0 + j];
            }
        }

        //mat mult and add 
        float vecmult[8]; //put this on the stack since 3D is max size anyways
        for(int i = 0; i < dim0; i++){
            for(int j = 0; j < num_groups; j++){
                for(int l = 0; l < D; l++){
                    //xn = Wx + b
                    float *W = weight_perm_temp + i*_I*_I;
                    float *xv = x_norm + i*num_groups*D*_I + j*D*_I + l*_I;
                    float *b = bias_temp + i*_I;
                    //(I,I)x(I,1)
                    for(int ii = 0; ii < _I; ii++){
                        vecmult[ii] = 0.0f;
                        for(int jj = 0; jj < _I; jj++){
                            vecmult[ii] += W[ii*_I + jj] * xv[jj];
                        }
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