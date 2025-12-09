#pragma once

#include "policy.h"

typedef enum {
    NoBaseline,
    MeanBaseline
} Baseline;


void binary_policy_gradient(
    Env *env,
    Policy *policy,
    int n_episodes,
    int max_steps_per_episode,
    float gamma,
    Baseline baseline
);

void discrete_policy_gradient(
    Env *env,
    Policy *policy,
    int n_episodes,
    int max_steps_per_episode,
    float gamma
);
