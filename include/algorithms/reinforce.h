#pragma once

#include "algorithms/common.h"
#include "environments/common.h"
#include "nn/mlp.h"
#include "rng.h"

// Calculate policy gradient
void discrete_reinforce_step(
    Env *env,
    Policy *policy,
    int n_episodes,
    int max_steps_per_episode,
    float gamma
);