#pragma once

#include "policy.h"

typedef struct EpisodeStatistics {
    float episode_return;
    int total_steps;
    float mean_entropy;
} EpisodeStatistics;


/**
 * Executes a rollout in the environment using the given policy.
 *
 * The arrays must be pre-allocated by the caller.
 *
 * Inputs:
 *  - env           : environment instance (mutated)
 *  - policy        : policy used for action selection
 *  - n_steps       : maximum number of environment steps
 *
 * Outputs (length = n_steps unless terminated early):
 *  - observations  : [t][obs_dim]
 *  - actions       : [t][act_dim] or scalar
 *  - rewards       : [t]
 *  - dones         : [t] = 1 if terminal at step t
 *
 * Returns:
 *  - Number of executed steps (≤ n_steps)
 *
 * Side effects:
 *  - Mutates environment state
 *
 * Thread-safety:
 *  - Not thread-safe (env is stateful)
 */
EpisodeStatistics policy_rollout(
    Env *env,
    const Policy *policy,
    int n_steps,
    float *observations,
    float *actions,
    float *rewards,
    bool *dones
);

void discounted_cumsum_inplace(float *r, bool *dones, int size, float gamma);