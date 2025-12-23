#pragma once

#include "stdlib.h"
#include "mlp.h"

typedef struct Optimizer {
    void *state;
    void (*step)(void *, MLP *, MLPCache *);
    void (*destroy)(void *);
} Optimizer;

static inline void optimizer_step(Optimizer *opt, MLP *mlp, MLPCache *cache) {
    opt->step(opt->state, mlp, cache);
    empty_mlp_cache(cache);
}

static inline void free_optimizer(Optimizer *opt) {
    if (opt->destroy) opt->destroy(opt->state);
    free(opt->state);
}

Optimizer make_gd(float lr);

Optimizer make_adam(
    MLP *mlp,
    float lr,
    float beta1,
    float beta2,
    float epsilon
);

