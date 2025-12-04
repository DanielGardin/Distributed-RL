#pragma once

#include "activations.h"

typedef struct LinearLayer {
    int input_size;
    int output_size;
    float *weights;        // 2D array [output_size, input_size]
    float *biases;         // 1D array [output_size]
    Activation activation;

    // gradients
    float *weights_grad;  // Same size as weights
    float *biases_grad;   // Same size as biases
} LinearLayer;

LinearLayer create_linear(
    int input_size,
    int output_size,
    Activation activation
);

typedef struct LinearCache {
    int batch_size;
    float *layer_inputs;     // [batch_size, input_size]
    float *pre_activations;  // [batch_size, output_size]
} LinearCache;


void kaiming_init(LinearLayer *linear);

void linear_forward(
    const LinearLayer *linear,
    const float *input,
    int batch_size,
    float *out,
    float *pre_activation      // NULL when in inference mode
);

void linear_backward(
    LinearLayer *linear,
    const LinearCache *cache,
    const float *out_grad,      // Indicates ∂f/∂x_{out} of size [batch_size, output_size]
    float *in_grad              // Outputs ∂f/∂x_{in} of size [batch_size, input_size]
);

void linear_zero_grad(LinearLayer *linear);

void free_linear(LinearLayer* linear);

LinearCache create_linear_cache(const LinearLayer *linear, int batch_size);

void free_linear_cache(LinearCache *cache);
