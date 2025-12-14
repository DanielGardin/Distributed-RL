#pragma once

#include "mlp.h"

typedef struct AdamState {
    int num_layers;
    long t;
    float **weights_m;
    float **weights_v;
    float **biases_m;
    float **biases_v;
} AdamState;

AdamState create_adam_state(const MLP *mlp);


void free_adam_state(AdamState *state);

void gd_step(MLP *mlp, float lr);

void adam_step(MLP *mlp, AdamState *state, float lr, float beta1, float beta2, float epsilon);
