#include <stdlib.h>
#include <cblas.h>
#include <math.h>

#include "rng.h"
#include "nn.h"

LinearLayer create_linear(
    int input_size,
    int output_size,
    Activation activation
) {
    int matsize = input_size*output_size;
    float *weights = calloc(matsize, sizeof(float));
    float *biases = calloc(output_size, sizeof(float));

    float *weights_grad = calloc(matsize, sizeof(float));
    float *biases_grad = calloc(output_size, sizeof(float));

    return (LinearLayer){
        .input_size=input_size,
        .output_size=output_size,
        .weights=weights,
        .biases=biases,
        .activation=activation,
        .weights_grad=weights_grad,
        .biases_grad=biases_grad
    };
};

void kaiming_init(LinearLayer *linear) {
    float limit = sqrtf(6.0f / linear->input_size);

    for (int i = 0; i<linear->input_size*linear->output_size; i++)
        linear->weights[i] = rand_uniform(-limit, limit);
};

void linear_forward(const LinearLayer* linear, const float* input, int batch_size, float* out) {
    int insize = linear->input_size;
    int outsize = linear->output_size;

    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        batch_size, outsize, insize,
        1.0f,
        input, insize,
        linear->weights, outsize,
        0.0f,
        out, outsize
    );

    float *out_row = out;

    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < outsize; ++j) {
            float v = out_row[j] + linear->biases[j];
            out_row[j] = linear->activation.fn(v);
        }

        out_row += outsize;
    }

};

void free_linear(LinearLayer* linear) {
    free(linear->weights);
    free(linear->biases);
    free(linear->weights_grad);
    free(linear->biases_grad);
};

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

    return (MLP){.layers=layers, .num_layers=num_layers};
};

void mlp_forward(const MLP* mlp, const float* input, int batch_size, float* out) {
    const float *inmat = input;
    float *outmat;

    const LinearLayer *layer;
    for (int l = 0; l < mlp->num_layers; l++) {
        layer = &mlp->layers[l];
        int outdim = layer->output_size;

        outmat = l == mlp->num_layers - 1 ? out : calloc(batch_size * outdim, sizeof(float));

        linear_forward(layer, inmat, batch_size, outmat);

        if (l > 0 && inmat != input)
            free((void*)inmat);

        inmat = outmat;
    }
};

void free_mlp(MLP* mlp) {
    for (int layer=0; layer<mlp->num_layers; layer++) {
        free_linear(&mlp->layers[layer]);
    };
};
