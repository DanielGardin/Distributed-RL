#pragma once

#include "environments/common.h"
#include "nn/mlp.h"

/**
 * Policy interface.
 *
 * The policy is a thin wrapper around an MLP with two function pointers:
 *  - sample:    draws an action from π(a | s)
 *  - log_prob:  computes log π(a | s) and (optionally) ∇z log π(a | s), where z is the network output
 *
 * Ownership:
 *  - The Policy does NOT own the MLP. The caller is responsible for its lifetime.
 *
 */
typedef struct Policy {
    /** Pointer to the underlying neural network (NOT owned). */
    MLP *mlp;

    /**
     * Samples an action from the policy.
     *
     * Inputs:
     *  - mlp        : network used to compute π(a | s)
     *  - obs        : pointer to observation vector, shape = (batch_size, obs_dim)
     *  - batch_size : batch size to compute
     *
     * Outputs:
     *  - action     : sampled action, shape = (batch_size, act_dim,)
     *  - log_prob   : log π(a | s) of the sampled actions, shape = (batch_size,) (optional)
     *
     * Preconditions:
     *  - obs, action, log_prob are valid pointers
     */
    void (*sample) (
        const MLP *mlp,
        const float *obs,
        int batch_size,
        float *action,
        float *log_prob
    );

    void (*log_prob) (
        const float *logits,
        const float *actions,
        int batch_size,
        float *log_prob,
        float *grad_out
    );
} Policy;

/**
 * Samples one action from the policy given an observation.
 *
 * Wrapper around Policy.sample.
 *
 * Inputs:
 *  - policy   : initialized Policy
 *  - obs      : observation vector, shape = (batch_size, obs_dim)
 *
 * Outputs:
 *  - action   : sampled actions
 *  - log_prob : log π(a | s)
 *
 * Preconditions:
 *  - All pointers provided are valid
 */
void policy_sample_action(const Policy *policy, const float *obs, int batch_size, float *action, float *log_prob);

void policy_log_prob_from_logits(
    const Policy *policy,
    const float *logits,
    const float *actions,
    int batch_size,
    float *log_prob,      // Can be NULL
    float *grad_out       // Can be NULL
);

void policy_log_prob(
    const Policy *policy,
    const float *obs,
    const float *actions,
    int batch_size,
    float *log_prob,
    float *grad_out,
    MLPCache *cache
);

/**
 * Creates a discrete-action policy wrapper.
 *
 * The output distribution is a categorical distribution produced by the MLP.
 *
 * Arguments:
 *  - mlp : network producing logits over discrete actions
 *
 * Returns:
 *  - Initialized Policy (does not allocate or copy mlp)
 */
Policy create_discrete_policy(MLP *mlp);

/**
 * Creates a binary-action policy wrapper.
 *
 * The output distribution is Bernoulli (sigmoid on single output neuron).
 *
 * Arguments:
 *  - mlp : network producing a single logit
 *
 * Returns:
 *  - Initialized Policy (does not allocate or copy mlp)
 */
Policy create_binary_policy(MLP *mlp);

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
int policy_rollout(
    Env *env,
    const Policy *policy,
    int n_steps,
    float *observations,
    float *actions,
    float *rewards,
    bool *dones
);

/**
 * In-place discounted inverse cumulative sum.
 *
 * Definition:
 *   r[t] ← ∑_{k=0}^{T-t-1} γ^k r[t+k]
 *
 * Used to compute Monte-Carlo returns for vanilla REINFORCE.
 *
 * Inputs:
 *  - r     : array of length T (overwritten)
 *  - T     : number of timesteps
 *  - gamma : discount factor, 0 ≤ γ ≤ 1
 *
 * Preconditions:
 *  - r points to at least T floats
 *  - T > 0
 */
void discounted_cumsum_inplace(float *r, int T, float gamma);
