#include <math.h>
#include "activations.h"

float relu_fn(float x)    { return x > 0.0f ? x : 0.0f; }
float relu_dfn(float x)   { return x > 0.0f ? 1.0f : 0.0f; }

float sigmoid_fn(float x) {
    return 1.0f / (1.0f + expf(-x));
}
float sigmoid_dfn(float x) {
    float s = sigmoid_fn(x);
    return s * (1.0f - s);
}

float softplus_fn(float x) {
    return log1pf(expf(x));
}
float softplus_dfn(float x) {
    return 1.0f / (1.0f + expf(-x));
}

float logsigmoid_fn(float x) {
    return -log1pf(expf(-x));
}
float logsigmoid_dfn(float x) {
    return 1.0f - sigmoid_fn(x);
}

Activation relu       = { relu_fn,       relu_dfn };
Activation sigmoid    = { sigmoid_fn,    sigmoid_dfn };
Activation softplus   = { softplus_fn,   softplus_dfn };
Activation logsigmoid = { logsigmoid_fn, logsigmoid_dfn };
