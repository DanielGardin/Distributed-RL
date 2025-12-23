#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

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

void kaiming_mlp_init(MLP *mlp) {
    for (int l=0; l<mlp->num_layers; l++)
        kaiming_linear_init(&mlp->layers[l]);
}


void mlp_forward(const MLP* mlp, const float* input, int batch_size, float* out, MLPCache *cache) {
    const float *current_input = input;
    float *output;

    int in_size = mlp->input_size;

    if (cache) {
        if (cache->size + batch_size > cache->capacity) {
            fprintf(
                stderr, "WARNING: The current batch overflows the cache capacity. Results by this call are not being cached. "
                "Currently using %d out of %d cache capacity.\n",
                cache->size, cache->capacity
            );

            cache = NULL;
        } else {
            memcpy(
                cache->layer_caches[0].layer_inputs + cache->size * in_size,
                input,
                batch_size * in_size * sizeof(float)
            );
        }
    }
    
    const LinearLayer *layer;
    for (int l = 0; l < mlp->num_layers; l++) {
        layer = &mlp->layers[l];

        if (l == mlp->num_layers - 1) {
            if (out) output = out;
            else output = cache->output + cache->size * mlp->output_size;
        } else if (cache) {
            // Write the current layer's output directly into the next layer's input buffer
            output = cache->layer_caches[l+1].layer_inputs + cache->size * layer->output_size;
        } else output = malloc(batch_size * layer->output_size * sizeof(float));

        linear_forward(layer, current_input, batch_size, output, cache ? &cache->layer_caches[l] : NULL);

        if (l > 0 && !cache) free((float *)current_input);

        current_input = output;
    }

    if (cache) {
        if (out) {
            memcpy(
                cache->output + cache->size * mlp->output_size,
                out,
                batch_size * mlp->output_size * sizeof(float)
            );
        }

        cache->size += batch_size;
    }
}

void mlp_backward(MLP *mlp, const MLPCache *cache, const float *out_grad, float *input_gradient) {
    int num_layers = mlp->num_layers;
    int batch_size = cache->size;

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

int get_num_params(MLP *mlp) {
    int total = 0;
    for (int i = 0; i < mlp->num_layers; i++) {
        LinearLayer *layer = &mlp->layers[i];
        total += (layer->input_size + 1) * layer->output_size;
    }

    return total;
}

void mlp_zero_grad(MLP *mlp) {
    for (int i = 0; i < mlp->num_layers; i++)
        linear_zero_grad(&mlp->layers[i]);
}

int save_mlp_weights(MLP *mlp, char *path) {
    FILE *file = fopen(path, "wb");

    if (!file) {
        fprintf(stderr, "Error opening file %s.\n", path);
        return 0;
    }

    const uint32_t magic = 0x4D4C5057; /* 'MLPW' */
    const uint32_t version = 1;
    uint32_t num_layers = (uint32_t)mlp->num_layers;

    if (fwrite(&magic, sizeof(magic), 1, file) != 1 ||
        fwrite(&version, sizeof(version), 1, file) != 1 ||
        fwrite(&num_layers, sizeof(num_layers), 1, file) != 1) {
        fprintf(stderr, "Error writing header to %s.\n", path);
        fclose(file);
        return 0;
    }

    for (int l = 0; l < mlp->num_layers; l++) {
        LinearLayer *layer = &mlp->layers[l];
        uint32_t in = (uint32_t)layer->input_size;
        uint32_t out = (uint32_t)layer->output_size;

        if (fwrite(&in, sizeof(in), 1, file) != 1 ||
            fwrite(&out, sizeof(out), 1, file) != 1) {
            fprintf(stderr, "Error writing layer sizes to %s.\n", path);
            fclose(file);
            return 0;
        }

        size_t wcount = (size_t)in * (size_t)out;
        if (fwrite(layer->weights, sizeof(float), wcount, file) != wcount) {
            fprintf(stderr, "Error writing weights for layer %d to %s.\n", l, path);
            fclose(file);
            return 0;
        }

        if (fwrite(layer->biases, sizeof(float), out, file) != out) {
            fprintf(stderr, "Error writing biases for layer %d to %s.\n", l, path);
            fclose(file);
            return 0;
        }
    }

    fclose(file);
    return 1;
}

int load_mlp_weights(MLP *mlp, const char *path) {
    FILE *file = fopen(path, "rb");
    if (!file) {
        fprintf(stderr, "Error opening file %s.\n", path);
        return 0;
    }

    uint32_t magic = 0, version = 0, num_layers = 0;
    if (fread(&magic, sizeof(magic), 1, file) != 1 ||
        fread(&version, sizeof(version), 1, file) != 1 ||
        fread(&num_layers, sizeof(num_layers), 1, file) != 1) {
        fprintf(stderr, "Error reading header from %s.\n", path);
        fclose(file);
        return 0;
    }

    if (magic != 0x4D4C5057 || version != 1) {
        fprintf(stderr, "Invalid file format or version in %s.\n", path);
        fclose(file);
        return 0;
    }

    if ((int)num_layers != mlp->num_layers) {
        fprintf(stderr, "Layer count mismatch: file=%u, mlp=%d.\n", num_layers, mlp->num_layers);
        fclose(file);
        return 0;
    }

    for (int l = 0; l < mlp->num_layers; l++) {
        uint32_t in = 0, out = 0;
        if (fread(&in, sizeof(in), 1, file) != 1 ||
            fread(&out, sizeof(out), 1, file) != 1) {
            fprintf(stderr, "Error reading layer sizes from %s.\n", path);
            fclose(file);
            return 0;
        }

        LinearLayer *layer = &mlp->layers[l];
        if ((int)in != layer->input_size || (int)out != layer->output_size) {
            fprintf(stderr, "Layer %d size mismatch: file=(%u,%u) mlp=(%d,%d).\n", l, in, out, layer->input_size, layer->output_size);
            fclose(file);
            return 0;
        }

        size_t wcount = (size_t)in * (size_t)out;
        if (fread(layer->weights, sizeof(float), wcount, file) != wcount) {
            fprintf(stderr, "Error reading weights for layer %d from %s.\n", l, path);
            fclose(file);
            return 0;
        }

        if (fread(layer->biases, sizeof(float), out, file) != out) {
            fprintf(stderr, "Error reading biases for layer %d from %s.\n", l, path);
            fclose(file);
            return 0;
        }
    }

    fclose(file);
    return 1;
}

MLPCache create_mlp_cache(const MLP *mlp, int capacity) {
    MLPCache cache;

    cache.size = 0;
    cache.capacity = capacity;
    cache.num_layers = mlp->num_layers;
    cache.layer_caches = malloc(mlp->num_layers * sizeof(LinearCache));
    cache.output = malloc(capacity * mlp->output_size *  sizeof(float));

    for (int l = 0; l < mlp->num_layers; l++) {
        cache.layer_caches[l] = create_linear_cache(
            &mlp->layers[l], capacity
        );
    }

    return cache;
}

void empty_mlp_cache(MLPCache *cache) {
    cache->size = 0;
    for (int l = 0; l < cache->num_layers; l++)
        empty_linear_cache(&cache->layer_caches[l]);
}

void free_mlp_cache(MLPCache *cache) {
    for (int l = 0; l < cache->num_layers; l++)
        free_linear_cache(&cache->layer_caches[l]);

    free(cache->layer_caches);
    free(cache->output);
}
