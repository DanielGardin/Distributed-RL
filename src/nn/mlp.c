#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "rng.h"
#include "nn/mlp.h"

MLP create_mlp(
    int* input_sizes,
    int output_size,
    int num_layers,
    Activation *activations
) {
    LinearLayer *layers = malloc(num_layers * sizeof(LinearLayer));

    for (int i=0; i < num_layers-1; i++) {
        layers[i] = create_linear(
            input_sizes[i],
            input_sizes[i+1],
            activations[i]
        );
    }

    layers[num_layers-1] = create_linear(
        input_sizes[num_layers-1],
        output_size,
        activations[num_layers-1]
    );

    return (MLP){
        .layers=layers,
        .num_layers=num_layers,
        .input_size=input_sizes[0],
        .output_size=output_size
    };
}

void mlp_forward(const MLP* mlp, const float* input, int batch_size, float* out, MLPCache *cache) {
    const float *current_input = input;
    float *output;

    if (cache) memcpy(
        cache->layer_caches[0].layer_inputs,
        input,
        batch_size * mlp->input_size * sizeof(float)
    );

    const LinearLayer *layer;
    for (int l = 0; l < mlp->num_layers; l++) {
        layer = &mlp->layers[l];

        if (l == mlp->num_layers - 1) output = out;
        else if (cache) output = cache->layer_caches[l+1].layer_inputs;
        else output = malloc(batch_size * layer->output_size * sizeof(float));

        linear_forward(layer, current_input, batch_size, output, cache ? &cache->layer_caches[l] : NULL);

        if (l > 0 && cache == NULL)
            free((float *)current_input);

        current_input = output;
    }
}

void mlp_backward(MLP *mlp, const MLPCache *cache, const float *out_grad, float *input_gradient) {
    int num_layers = mlp->num_layers;
    int batch_size = cache->batch_size;

    const float *current_grad = out_grad;
    float *next_grad = NULL;

    for (int l = num_layers - 1; l >= 0; l--) {
        LinearLayer *layer = &mlp->layers[l];

        if (l == 0)
            next_grad = input_gradient;
        else
            next_grad = malloc(batch_size * layer->input_size * sizeof(float));

        linear_backward(layer, &cache->layer_caches[l], current_grad, next_grad);

        if (current_grad != out_grad)
            free((float *)current_grad);

        current_grad = next_grad;
    }
}

void free_mlp(MLP* mlp) {
    for (int layer=0; layer<mlp->num_layers; layer++)
        free_linear(&mlp->layers[layer]);

    free(mlp->layers);
}

void mlp_zero_grad(MLP *mlp) {
    for (int i = 0; i < mlp->num_layers; i++)
        linear_zero_grad(&mlp->layers[i]);
}

int save_mlp_weights(MLP *mlp, char *path) {
    FILE *file = fopen(path, "wb");

    if (!file) {
        printf("Error opening file %s.", path);
        return 0;
    }

    LinearLayer *layer;
    int in, out;
    for (int l=0; l<mlp->num_layers; l++) {
        layer = &mlp->layers[l];

        in = layer->input_size;
        out = layer->output_size;

        fprintf(file, "Layer %d %d %d", l, in, out);
        fwrite(layer->weights, sizeof(float), in*out, file);
        fwrite(layer->biases, sizeof(float), out, file);
    }

    fclose(file);

    return 0;
}

MLPCache create_mlp_cache(const MLP *mlp, int batch_size) {
    MLPCache cache;

    cache.num_layers = mlp->num_layers;
    cache.batch_size = batch_size;
    cache.layer_caches = malloc(mlp->num_layers * sizeof(LinearCache));

    for (int l = 0; l < mlp->num_layers; l++) {
        cache.layer_caches[l] = create_linear_cache(
            &mlp->layers[l], batch_size
        );
    }

    return cache;
}

void free_mlp_cache(MLPCache *cache) {
    for (int l = 0; l < cache->num_layers; l++)
        free_linear_cache(&cache->layer_caches[l]);

    free(cache->layer_caches);
}
