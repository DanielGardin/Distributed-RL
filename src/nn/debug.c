#include <stdio.h>
#include <stdlib.h>

#include "nn/debug.h"

void print_mlp(MLP *mlp) {
    fprintf(stderr, "INFO: Network parameters:\n");
    float *row;


    for (int l=0; l<mlp->num_layers; l++) {
        LinearLayer *layer = &mlp->layers[l];
        fprintf(stderr, "Layer %d (Weights) %d x %d\n", l, layer->input_size, layer->output_size);
    
        row = layer->weights;
        for (int i=0; i<layer->input_size; i++) {
            fprintf(stderr, "    ");

            for (int j=0; j<layer->output_size; j++) {
                fprintf(stderr, "%.3f ", row[j]);
            };
            fprintf(stderr, "\n");

            row += layer->output_size;
        };
        fprintf(stderr, "\nLayer %d (Bias) %d\n", l, layer->output_size);

        fprintf(stderr, "    ");
        for (int i=0; i<layer->output_size; i++) {
            fprintf(stderr, "%.3f ", layer->biases[i]);
        };
        fprintf(stderr, "\n\n"); 
    };
}

void print_mlp_grad(MLP *mlp) {
    fprintf(stderr, "INFO: Network parameters' gradients:\n");
    float *row;


    for (int l=0; l<mlp->num_layers; l++) {
        LinearLayer *layer = &mlp->layers[l];
        fprintf(stderr, "Layer %d (Weights) %d x %d\n", l, layer->input_size, layer->output_size);
    
        row = layer->weights_grad;
        for (int i=0; i<layer->input_size; i++) {
            fprintf(stderr, "    ");

            for (int j=0; j<layer->output_size; j++) {
                fprintf(stderr, "%.3f ", row[j]);
            };
            fprintf(stderr, "\n");

            row += layer->output_size;
        };
        fprintf(stderr, "\nLayer %d (Bias) %d\n", l, layer->output_size);

        fprintf(stderr, "    ");
        for (int i=0; i<layer->output_size; i++) {
            fprintf(stderr, "%.3f ", layer->biases_grad[i]);
        };
        fprintf(stderr, "\n\n"); 
    };
}