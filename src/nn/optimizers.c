#include "nn/optimizers.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct GDState {
    float lr;
} GDState;

void gd_step(GDState *state, MLP *mlp, MLPCache *cache) {
    float lr = state->lr;

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

Optimizer make_gd(float lr) {
    GDState *state = malloc(sizeof(GDState));

    state->lr = lr;

    return (Optimizer) {
        .state=state,
        .step=(void (*)(void *, MLP *, MLPCache *))gd_step
    };
}

typedef struct AdamState {
    float lr;
    float beta1;
    float beta2;
    float epsilon;
    int num_layers;
    long t;
    float **weights_m;
    float **weights_v;
    float **biases_m;
    float **biases_v;
} AdamState;

void adam_step(AdamState *state, MLP *mlp, MLPCache *cache) {
    float lr = state->lr;
    float beta1 = state->beta1;
    float beta2 = state->beta2;
    float epsilon = state->epsilon;

    state->t++;
    LinearLayer *layer;
    float bias_correction1 = 1.0f - powf(beta1, state->t);
    float bias_correction2 = 1.0f - powf(beta2, state->t);

    for (int l = 0; l < mlp->num_layers; l++) {
        layer = &mlp->layers[l];

        int in = layer->input_size;
        int out = layer->output_size;

        for (int i = 0; i < in * out; i++) {
            state->weights_m[l][i] = beta1 * state->weights_m[l][i] + (1.0f - beta1) * layer->weights_grad[i];
            state->weights_v[l][i] = beta2 * state->weights_v[l][i] + (1.0f - beta2) * layer->weights_grad[i] * layer->weights_grad[i];
            float m_hat = state->weights_m[l][i] / bias_correction1;
            float v_hat = state->weights_v[l][i] / bias_correction2;
            layer->weights[i] -= lr * m_hat / (sqrtf(v_hat) + epsilon);
        }

        for (int i = 0; i < out; i++) {
            state->biases_m[l][i] = beta1 * state->biases_m[l][i] + (1.0f - beta1) * layer->biases_grad[i];
            state->biases_v[l][i] = beta2 * state->biases_v[l][i] + (1.0f - beta2) * layer->biases_grad[i] * layer->biases_grad[i];
            float m_hat = state->biases_m[l][i] / bias_correction1;
            float v_hat = state->biases_v[l][i] / bias_correction2;
            layer->biases[i] -= lr * m_hat / (sqrtf(v_hat) + epsilon);
        }
    }
}

void free_adam_state(AdamState *state) {
    for (int l = 0; l < state->num_layers; l++) {
        free(state->weights_m[l]);
        free(state->weights_v[l]);
        free(state->biases_m[l]);
        free(state->biases_v[l]);
    }
    free(state->weights_m);
    free(state->weights_v);
    free(state->biases_m);
    free(state->biases_v);
}

Optimizer make_adam(
    MLP *mlp,
    float lr,
    float beta1,
    float beta2,
    float epsilon
) {
    AdamState *state = malloc(sizeof(AdamState));

    state->lr=lr;
    state->beta1=beta1;
    state->beta2=beta2;
    state->epsilon=epsilon;

    state->t = 0;
    state->weights_m = malloc(mlp->num_layers * sizeof(float*));
    state->weights_v = malloc(mlp->num_layers * sizeof(float*));
    state->biases_m = malloc(mlp->num_layers * sizeof(float*));
    state->biases_v = malloc(mlp->num_layers * sizeof(float*));

    state->num_layers = mlp->num_layers;
    for (int l = 0; l < mlp->num_layers; l++) {
        int weight_size = mlp->layers[l].input_size * mlp->layers[l].output_size;
        int bias_size = mlp->layers[l].output_size;

        state->weights_m[l] = calloc(weight_size, sizeof(float));
        state->weights_v[l] = calloc(weight_size, sizeof(float));
        state->biases_m[l] = calloc(bias_size, sizeof(float));
        state->biases_v[l] = calloc(bias_size, sizeof(float));
    }

    return (Optimizer){
        .state=state,
        .step=(void (*)(void *, MLP *, MLPCache *))adam_step,
        .destroy=(void (*)(void *))free_adam_state
    };
}
