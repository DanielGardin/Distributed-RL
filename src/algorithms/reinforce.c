#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "algorithms/reinforce.h"

void discrete_reinforce_step(
    Env *env,
    Policy *policy,
    int n_episodes,
    int max_steps_per_episode,
    float gamma
) {
    float *observations = malloc(max_steps_per_episode * env->obs_size * sizeof(float));
    float *actions = malloc(max_steps_per_episode * env->act_size * sizeof(float));
    float *rewards = malloc(max_steps_per_episode * sizeof(float));
    bool *dones = malloc(max_steps_per_episode * sizeof(bool));

    int total_steps = policy_rollout(
        env,
        policy,
        max_steps_per_episode,
        observations,
        actions,
        rewards,
        dones
    );

    discounted_cumsum_inplace(rewards, total_steps, gamma);

    MLPCache cache = create_mlp_cache(policy->mlp, total_steps);
    float *logits = malloc(total_steps * env->act_size * sizeof(float));

    mlp_forward(policy->mlp, observations, total_steps, logits, &cache);
    policy_log_prob_from_logits(policy, logits, actions, total_steps, NULL, logits);

    for (int i = 0; i < total_steps; i++) {
        logits[i] *= rewards[i];
    }

    mlp_backward(policy->mlp, &cache, logits, NULL);

    free(observations);
    free(actions);
    free(rewards);
    free(dones);

    free(logits);
    free_mlp_cache(&cache);
}