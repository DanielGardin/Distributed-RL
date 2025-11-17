#pragma once

#include "activations.h"

typedef struct LinearLayer {
    int input_size;
    int output_size;
    float *weights; // Flattened 2D array: weights[output_size][input_size]
    float *biases;  // 1D array: biases[output_size]
    Activation activation;

    // gradients
    float *weights_grad;
    float *biases_grad;
} LinearLayer;

LinearLayer create_linear(
    int input_size,
    int output_size,
    Activation activation
);

void kaiming_init(LinearLayer *linear);

void linear_forward(const LinearLayer* linear, const float* input, int batch_size, float* out);

void free_linear(LinearLayer* linear);

typedef struct MLP {
    LinearLayer *layers;
    int num_layers;
} MLP;

MLP create_mlp(
    int* input_sizes,
    int output_size,
    int num_layers,
    Activation *activations
);

void mlp_forward(const MLP* mlp, const float* input, int batch_size, float* out);

void free_mlp(MLP* mlp);
