#include "clifford_sigmoid.h"
#include <math.h>

// Sigmoid function: 1 / (1 + exp(-x))
static float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

void clifford_sigmoid_forward(
    const float *in,
    float *out,
    int B,
    int C,
    int D,
    int n_blades,
    const float *weight,
    const float *bias
    )
{
    float *temp = (float *)malloc(B * C * D * n_blades * sizeof(float));
    if (!temp) {
        // Handle allocation failure
        return;
    }

    // Iterate batch
    for (int b = 0; b < B; b++){
        // Iterate channels
        for (int c = 0; c < C; c++){ //TODO: Do the channels represent the blades or is there dedicated blade dim?
            // Iterate pos
            for (int blade = 0; blade < D; blade++){ //Blade dim (after embedding)
                // Iterate blades & accumulate. Each blade once (groups=channels)
                float sum = 0.0f;
    
                // Compute each blade convolutions independently
                for (int i = 0; i < n_blades; i++){
                    sum += weight[blade*n_blades + i] * in[blade*n_blades + i];
                }

                // Add the bias term
                sum += bias[blade];
                int index = b * (C * D) + c * (D) + blade;
                temp[index] = sum;
            }
        }
    }

    // Apply the sigmoid function
    for (int b = 0; b < B; b++){
        for (int c = 0; c < C; c++){
            for (int blade = 0; blade < D; blade++){ // Blade dim (after embedding)
                // Apply sigmoid
                int index = b * (C * D) + c * (D) + blade;
                temp[index] = sigmoid(temp[index]);
            }
        }
    }

    // Scale the function by vector v
    for (int b = 0; b < B; b++){
        for (int c = 0; c < C; c++){
            for (int blade = 0; blade < D; blade++){ //Blade dim (after embedding)
                // Scale and return v
                int index = b * (C * D) + c * (D) + blade;
                out[index] = in[index] * temp[index];
            }
        }
    }

    // Free allocated memory
    free(temp);
    // TODO: Implement the embed/get functions scatter steps
} 