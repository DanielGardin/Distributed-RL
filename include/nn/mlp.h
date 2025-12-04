#pragma once

#include "linear.h"

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

typedef struct MLPCache {
    int num_layers, batch_size;
    LinearCache *layer_caches;
} MLPCache;

void mlp_forward(const MLP* mlp, const float* input, int batch_size, float* out, MLPCache *cache);

void mlp_backward(MLP *mlp, const MLPCache *cache, const float *out_grad, float  *in_grad);

void mlp_zero_grad(MLP *mlp);

int save_mlp_weights(MLP *mlp, char *path);

void free_mlp(MLP* mlp);

MLPCache create_mlp_cache(const MLP *mlp, int batch_size);

void free_mlp_cache(MLPCache *cache);