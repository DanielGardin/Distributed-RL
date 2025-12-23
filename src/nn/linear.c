#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <cblas.h>
#include <math.h>

#include "rng.h"
#include "nn/linear.h"

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
        .biases_grad=biases_grad,
    };
}

void kaiming_linear_init(LinearLayer *linear) {
    float limit = sqrtf(6.0f / linear->input_size);

    for (int i = 0; i<linear->input_size*linear->output_size; i++)
        linear->weights[i] = rand_uniform(-limit, limit);
};

void linear_forward(
    const LinearLayer *linear,
    const float *input,
    int batch_size,
    float *out,
    LinearCache *cache
) {
    int insize = linear->input_size;
    int outsize = linear->output_size;

    // Z = X W^T
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans,
        batch_size, outsize, insize,
        1.0f,
        input, insize,
        linear->weights, insize,
        0.0f,
        out, outsize
    );

    // O = σ(Z)
    if (cache) {
        float *pre_activations = cache->pre_activations + cache->size * outsize;

        for (int idx = 0; idx < batch_size * outsize; ++idx) {
            const int j = idx % outsize;
            float z = out[idx] + linear->biases[j];
            pre_activations[idx] = z;
            out[idx] = linear->activation.fn(z);
        }

        cache->size += batch_size;
    } else {
        for (int idx = 0; idx < batch_size * linear->output_size; ++idx) {
            const int j = idx % linear->output_size;
            out[idx] = linear->activation.fn(out[idx] + linear->biases[j]);
        }
    }
}

void linear_backward(
    LinearLayer *linear,
    const LinearCache *cache,
    const float *out_grad,      // Indicates ∂f/∂x_{out} of size [batch_size, output_size]
    float *in_grad              // Outputs ∂f/∂x_{in} of size [batch_size, input_size]
) {
    int in_size = linear->input_size;
    int out_size = linear->output_size;
    int batch_size = cache->size;

    float *grad_pre = malloc(batch_size * out_size * sizeof(float));  // ∂f/∂z of size [batch_size, output_size]
    for (int i = 0; i < batch_size * out_size; i++) {
        grad_pre[i] = out_grad[i] * linear->activation.dfn(cache->pre_activations[i]);
    }

    // Weight gradient ∂f/∂W = (dz/dW)^T (∂f/∂z)
    cblas_sgemm(
        CblasRowMajor, CblasTrans, CblasNoTrans,
        out_size, in_size, batch_size,
        1.0f,
        grad_pre, out_size,
        cache->layer_inputs, in_size,
        1.0f,
        linear->weights_grad, in_size
    );

    for (int b = 0; b < batch_size; b++) {
        for (int o = 0; o < out_size; o++) {
            linear->biases_grad[o] += grad_pre[b * out_size + o];
        }
    }

    if (in_grad) {
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            batch_size, in_size, out_size,
            1.0f,
            grad_pre, out_size,
            linear->weights, in_size,
            0.0f,
            in_grad, in_size
        );
    }

    free(grad_pre);
}

void linear_zero_grad(LinearLayer *linear) {
    memset(linear->weights_grad, 0, linear->output_size * linear->input_size * sizeof(float));
    memset(linear->biases_grad, 0, linear->output_size * sizeof(float));
}

void free_linear(LinearLayer* linear) {
    free(linear->weights);
    free(linear->biases);
    free(linear->weights_grad);
    free(linear->biases_grad);
}

LinearCache create_linear_cache(const LinearLayer *linear, int capacity) {
    return (LinearCache) {
        .size=0,
        .capacity = capacity,
        .layer_inputs = malloc(capacity * linear->input_size * sizeof(float)),
        .pre_activations = malloc(capacity * linear->output_size * sizeof(float))
    };
}

void empty_linear_cache(LinearCache *cache) {
    cache->size = 0;
}

void free_linear_cache(LinearCache *cache) {
    free(cache->layer_inputs);
    free(cache->pre_activations);
}
