#pragma once

#include "linear.h"

typedef struct MLP {
    LinearLayer *layers;
    int num_layers;
    int input_size;
    int output_size;
} MLP;

MLP create_mlp(
    int* input_sizes,
    int output_size,
    int num_layers,
    Activation *activations
);

typedef struct MLPCache {
    int num_layers, size, capacity;
    LinearCache *layer_caches;
    float *output;
} MLPCache;

void kaiming_mlp_init(MLP *mlp);

void mlp_forward(const MLP* mlp, const float* input, int batch_size, float* out, MLPCache *cache);

void mlp_backward(MLP *mlp, const MLPCache *cache, const float *out_grad, float  *in_grad);

void mlp_zero_grad(MLP *mlp);

int save_mlp_weights(MLP *mlp, char *path);

int load_mlp_weights(MLP *mlp, const char *path);

void free_mlp(MLP* mlp);

int get_num_params(MLP *mlp);

MLPCache create_mlp_cache(const MLP *mlp, int capacity);

void empty_mlp_cache(MLPCache *cache);

void free_mlp_cache(MLPCache *cache);
