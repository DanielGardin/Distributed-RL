#include "nn/optimizers.h"

void gd_step(MLP *mlp, float lr) {
    LinearLayer *layer;

    for (int l=0; l<mlp->num_layers; l++) {
        layer = &mlp->layers[l];

        int in = layer->input_size;
        int out = layer->output_size;

        for (int i=0; i<in*out; i++) {
            layer->weights[i] -= lr * layer->weights_grad[i];
        }

        for (int i=0; i<out; i++) {
            layer->biases[i] -= lr * layer->biases_grad[i];
        }
    }
}
