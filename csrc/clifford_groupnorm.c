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


void clifford_groupnorm_forward(
    const float *x,
    int B,
    int C,
    int D,
    int _I, //cant use I since it is a macro in linpacke....
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

    //blow up running mean and cov along the batch dim
    float * running_mean;
    float * running_cov;
    if ( running ){
        size_t running_mean_size = _I * dim0;
        running_mean = calloc(running_mean_size, sizeof(float));
        for(int i = 0; i < _I; i++){ //COST: 0 flops
            for(int j = 0; j < group_size; j++){
                for(int l = 0; l < B; l++){
                    running_mean[i*group_size*B + j*B + l] = running_mean_orig[i*group_size + ((j*B + l)%group_size)];
                }
            }
        }
        size_t running_cov_size = _I* _I * dim0;
        running_cov = calloc(running_cov_size, sizeof(float)); //(_I,_I,dim0)
        for(int i = 0; i < _I; i++){
            for(int ii = 0; ii < _I; ii++){
                for(int j = 0; j < group_size; j++){
                    for(int l = 0; l < B; l++){
                        running_cov[i*_I*group_size*B + ii*group_size*B + j*B + l] = running_cov_orig[i*_I*group_size + ii*group_size + ((j*B + l)%group_size)];
                    }
                }
            }
        }
    }

    //mean
    size_t mean_size = dim0 * _I;
    float * mean = calloc(mean_size, sizeof(float));
    if (training || (!running)) {
        for(int i = 0; i < dim0; i++){
            for(int j = 0; j < num_groups; j++){
                for(int k = 0; k < D; k++){
                    for(int l = 0; l < _I; l++){ //COST: dim0*num_groups*D*I = B*C*D*I flops
                        mean[i*_I + l] += x[i*num_groups*D*_I + j*D*_I + k*_I + l];
                    }
                }
            }
        }
        for(int i = 0; i < dim0; i++){
            for(int l = 0; l < _I; l++){ //COST: dim0*I
                mean[i*_I + l] /= (D*num_groups);
            }
        }
        if (running){ //update running mean
            for(int i = 0; i < _I; i++){
                for(int j = 0; j < dim0; j++){
                        running_mean[i*dim0 + j] += momentum *(mean[j*_I + i] - running_mean[i*dim0 + j]);
                }
            }
        }
    } else { //if not training use running mean
        for(int i = 0; i < _I; i++){
            for(int j = 0; j < dim0; j++){
                    mean[j*_I + i] = running_mean[i*dim0 + j];
            }
        }
    }

    //subtract mean form x
    size_t x_size = dim0 * num_groups * D * _I;
    float * x_centered = calloc(x_size, sizeof(float)); //could write this into the result
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){
                for(int l = 0; l < _I; l++){
                    x_centered[i*num_groups*D*_I + j*D*_I + k*_I + l] = x[i*num_groups*D*_I + j*D*_I + k*_I + l] - mean[i*_I + l];
                }
            }
        }
    }

    //permute: dim0, num_groups, D, _I -> dim0, _I, num_groups, D
    float * X = calloc(x_size, sizeof(float));
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){
                for(int l = 0; l < _I; l++){
                    X[i*_I*num_groups*D + l*num_groups*D + j*D +k] = x_centered[i*num_groups*D*_I + j*D*_I + k*_I + l];
                }
            }
        }
    }

    //calc cov
    size_t cov_size = dim0*_I*_I;
    float * cov = calloc(cov_size, sizeof(float));
    if (training || (!running)){
        for(int i = 0; i < dim0; i++){
            for(int j = 0; j < _I; j++){ //row selector
                for(int l = 0; l < _I; l++){ //col selector
                    for(int k = 0; k < num_groups*D; k++){
                        cov[i*_I*_I + j*_I + l] += X[i*_I*num_groups*D + j*num_groups*D + k]* x_centered[i*num_groups*D*_I + k*_I + l];
        }}}}
        //scale all of cov
        for(int i = 0; i < cov_size; i++){
            cov[i] /= (num_groups * D);
        }
        if (running) { //upadate running cov
            for(int i = 0; i < _I; i++){            
                for(int ii = 0; ii < _I; ii++){
                    for(int j = 0; j < dim0; j++){
                        running_cov[i*_I*dim0 + ii*dim0 + j] += momentum *(cov[j*_I*_I + i*_I + ii] - running_cov[i*_I*dim0 + ii*dim0 + j]);
                    }
                }
            }
        }
    } else {
        for(int i = 0; i < _I; i++){ //if not training use running cov
            for(int ii = 0; ii < _I; ii++){
                for(int j = 0; j < dim0; j++){
                    cov[j*_I*_I + i*_I + ii] = running_cov[i*_I*dim0 + ii*dim0 + j];
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
                    running_mean_orig[i*group_size + j] += running_mean[i*group_size*B + l*group_size + j];
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
                        running_cov_orig[i*_I*group_size + ii*group_size + j] += running_cov[i*_I*group_size*B + ii*group_size*B + l*group_size + j];
                    }
                }
            }
        }
        for(int i = 0; i < _I*_I*group_size; i++){
            running_cov_orig[i] /= B;
        }
        free(running_mean);
        free(running_cov);
    }

    //get max for each groups cov matrix
    size_t max_values_size = dim0;
    float * max_values = calloc(max_values_size, sizeof(float));
    for(int i = 0; i < dim0; i++){
        max_values[i] = cov[i*_I*_I];
        for(int j = 1; j < _I*_I; j++){
            max_values[i] = cov[i*_I*_I + j] > max_values[i] ? cov[i*_I*_I + j] : max_values[i];
        }
    }

    //add pertubation to diagonal
    float * cov_eye = calloc(cov_size, sizeof(float)); //overwrite
    for(int i = 0; i < cov_size; i++){
        cov_eye[i] = cov[i];
    }
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < _I; j++){
                    cov_eye[i*_I*_I + j*_I + j] += max_values[i]*eps;
    }}

    //cholesky decomp //need to pass U (cov) and result
    float * U = calloc(cov_size, sizeof(float)); //just use cov
    for(int i = 0; i < cov_size; i++){ //copy since cholseky is inplace
        U[i] = cov_eye[i];
    }
    int info;
    for(int i = 0; i < dim0; i++){
        //COST: https://en.wikipedia.org/wiki/Cholesky_decomposition (I^3)/6 adds and mults
        info = LAPACKE_spotrf(LAPACK_ROW_MAJOR, 'U', _I, &U[i*_I*_I], _I); //does not mask of lower half....
        if (info > 0) {printf("The matrix is not positive definite.\n");return;}
        if (info < 0) {printf("Illegal value in argument %d\n", -info);return;}
    }


    //solve triangular
    float * x_norm_here = calloc(x_size, sizeof(float)); //use result
    for(int i = 0; i < x_size; i++){ //copy as it will be overwritten by blas function
        x_norm_here[i] = x_centered[i];
    }
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){ //cost: BCDI(I-1)/2 mults and adds, BCDI div
                    float *mat = U + (i*_I*_I);
                    float *vec = x_norm_here + (i*num_groups*D*_I + j*D*_I + k*_I);
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
    float * weight;
    float * bias;
    if ( scaling ){
        size_t weight_size = _I* _I * dim0;
        weight = calloc(weight_size, sizeof(float)); // (_I,_I,dim0)
        for(int i = 0; i < _I; i++){
            for(int ii = 0; ii < _I; ii++){
                for(int j = 0; j < group_size; j++){
                    for(int l = 0; l < B; l++){
                        weight[i*_I*group_size*B + ii*group_size*B + j*B + l] = weight_orig[i*_I*group_size + ii*group_size + ((j*B + l)%group_size)];
                    }
                }
            }
        }
        size_t bias_size = _I * dim0;
        bias = calloc(bias_size, sizeof(float));
        for(int i = 0; i < _I; i++){
            for(int j = 0; j < group_size; j++){
                for(int l = 0; l < B; l++){
                    bias[i*group_size*B + j*B + l] = bias_orig[i*group_size + ((j*B + l)%group_size)];
                }
            }
        }
    }

    //scaling or just copy over
    if ( scaling ){
        //permute weight from (_I0, _I1, dim0) to (dim0, _I0, _I1)
        //think their code is wrong since they dont permute the bias and just reshape it...
        size_t weight_size = _I* _I * dim0;
        float *weight_perm = calloc(weight_size, sizeof(float));
        for(int i = 0; i < _I*_I; i++){
            for(int j = 0; j < dim0; j++){
                weight_perm[j*_I*_I + i] = weight[i*dim0 + j];
            }
        }

        //mat mult and add 
        for(int i = 0; i < x_size; i++){
            x_norm[i] = 0;
        }
        for(int i = 0; i < dim0; i++){
            for(int j = 0; j < num_groups; j++){
                for(int l = 0; l < D; l++){
                    //xn = Wx + b
                    float *W = weight_perm + i*_I*_I;
                    float *x = x_norm_here + i*num_groups*D*_I + j*D*_I + l*_I;
                    float *xn = x_norm + + i*num_groups*D*_I + j*D*_I + l*_I;
                    float *b = bias + i*_I;
                    //(I,I)x(I,1)
                    for(int ii = 0; ii < _I; ii++){
                        for(int jj = 0; jj < _I; jj++){
                            xn[ii] += W[ii*_I + jj] * x[jj];
                        }
                        xn[ii] += b[ii];
                    }
                }
            }
        }

        free(weight);
        free(bias);
        free(weight_perm);
    } else {
        //copy result over without scaling
        for(int i = 0; i < x_size; i++){
            x_norm[i] = x_norm_here[i];
        }
    }
    
    //free all the temp arrays
    free(x_norm_here);
    free(U);
    free(cov_eye);
    free(max_values);
    free(cov);
    free(X);
    free(x_centered);
    free(mean);

    return;
}