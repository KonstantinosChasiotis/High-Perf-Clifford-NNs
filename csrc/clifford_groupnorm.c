#include "clifford_groupnorm.h"

void clifford_groupnorm_forward(
    const float *x,
    int B,
    int C,
    int D,
    int _I, //cant use I since it is a macro in linpacke....
    int num_groups,
    float eps,
    float *x_norm){

    //reshape
    int dim0 = (B*C)/num_groups;

    //mean
    size_t mean_size = dim0 * _I;
    float * mean = calloc(mean_size, sizeof(float));
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){
                for(int l = 0; l < _I; l++){
                    mean[i*_I + l] += x[i*num_groups*D*_I + j*D*_I + k*_I + l];
                }
            }
        }
    }
    for(int i = 0; i < dim0; i++){
        for(int l = 0; l < _I; l++){
            mean[i*_I + l] /= (D*num_groups);
        }
    }

    //subtract mean form x
    size_t x_size = dim0 * num_groups * D * _I;
    float * x_centered = calloc(x_size, sizeof(float));
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){
                for(int l = 0; l < _I; l++){
                    x_centered[i*num_groups*D*_I + j*D*_I + k*_I + l] = x[i*num_groups*D*_I + j*D*_I + k*_I + l] - mean[i*_I + l];
                }
            }
        }
    }

    //permute: dim0, *(num_groups,D), _I -> dim0, _I, (here is empty dim in python) ,num_groups,D
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

    //transpose X (think this is the same as x_centered .... ups nice)
    float * XT = calloc(x_size, sizeof(float));
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < _I; j++){
            for(int k = 0; k < num_groups * D; k++){
                    XT[i*_I*num_groups*D + k*_I + j] = X[i*_I*num_groups*D + j*num_groups*D + k];
            }
        }
    }

    //matmult to get cov
    size_t cov_size = dim0*_I*_I;
    float * cov = calloc(cov_size, sizeof(float));
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < _I; j++){ //row selector
            for(int l = 0; l < _I; l++){ //col selector
                for(int k = 0; k < num_groups*D; k++){
                    cov[i*_I*_I + j*_I + l] += X[i*_I*num_groups*D + j*num_groups*D + k]* XT[i*num_groups*D*_I + k*_I + l];
    }}}}
    //scale all of cov
    for(int i = 0; i < cov_size; i++){
        cov[i] /= (num_groups * D);
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
    float * cov_eye = calloc(cov_size, sizeof(float));
    for(int i = 0; i < cov_size; i++){
        cov_eye[i] = cov[i];
    }
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < _I; j++){
                    cov_eye[i*_I*_I + j*_I + j] += max_values[i]*eps;
    }}

    //cholesky decomp
    float * U = calloc(cov_size, sizeof(float));
    for(int i = 0; i < cov_size; i++){ // copy since cholseky is inplace
        U[i] = cov_eye[i];
    }
    int info;
    for(int i = 0; i < dim0; i++){
        info = LAPACKE_spotrf(LAPACK_ROW_MAJOR, 'U', _I, &U[i*_I*_I], _I); //does not mask of lower half....
        // Check for convergence
        if (info > 0) {printf("The matrix is not positive definite.\n");return;}
        if (info < 0) {printf("Illegal value in argument %d\n", -info);return;}
    }


    //solve triangular
    float * x_norm_here = calloc(x_size, sizeof(float));
    for(int i = 0; i < x_size; i++){//copy as it will be overwritten by lapacke function
        x_norm_here[i] = x_centered[i];
    }
    for(int i = 0; i < dim0; i++){
        for(int j = 0; j < num_groups; j++){
            for(int k = 0; k < D; k++){
                    cblas_strsv(CblasRowMajor, CblasUpper, CblasNoTrans ,CblasNonUnit, _I, &U[i*_I*_I], _I, &x_norm_here[i*num_groups*D*_I + j*D*_I + k*_I], 1);
            }
        }
    }

    //copy result over
    for(int i = 0; i < x_size; i++){
        x_norm[i] = x_norm_here[i];
    }

    //free all the temp arrays
    free(x_norm_here);
    free(U);
    free(cov_eye);
    free(max_values);
    free(cov);
    free(XT);
    free(X);
    free(x_centered);
    free(mean);

    return;
}