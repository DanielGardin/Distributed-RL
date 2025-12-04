#pragma once

typedef struct Activation {
    float (*fn)(float);
    float (*dfn)(float);
} Activation;

float relu_fn(float x);
float relu_dfn(float x);

float sigmoid_fn(float x);
float sigmoid_dfn(float x);

float softplus_fn(float x);
float softplus_dfn(float x);

float logsigmoid_fn(float x);
float logsigmoid_dfn(float x);

extern Activation relu;
extern Activation sigmoid;
extern Activation softplus;
extern Activation logsigmoid;
extern Activation identity;
