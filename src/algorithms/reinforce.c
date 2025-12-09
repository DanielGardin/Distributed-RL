#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include "algorithms/utils.h"
#include "algorithms/reinforce.h"

void binary_policy_gradient(
    Env *env,
    Policy *policy,
    int n_episodes,
    int max_steps_per_episode,
    float gamma,
    Baseline baseline
) {
    int buffer_size = n_episodes * max_steps_per_episode;

    float *observations = malloc(buffer_size * env->obs_size * sizeof(float));
    float *actions = malloc(buffer_size * sizeof(float));
    float *rewards = malloc(buffer_size * sizeof(float));
    bool *dones = malloc(buffer_size * sizeof(bool));

    int offset = 0;
    float mean_return = 0;
    for (int i = 0; i < n_episodes; i++) {
        EpisodeStatistics stats = policy_rollout(
            env,
            policy,
            max_steps_per_episode,
            observations + offset,
            actions + offset,
            rewards + offset,
            dones + offset
        );

        offset += stats.total_steps;
        mean_return += stats.episode_return;
    }

    discounted_cumsum_inplace(rewards, dones, offset, gamma);

    MLPCache cache = create_mlp_cache(policy->mlp, offset);
    float *logits = malloc(offset * sizeof(float));
    mlp_forward(policy->mlp, observations, offset, logits, &cache);

    float *dlogp  = malloc(offset * sizeof(float));
    policy_log_prob_from_logits(policy, logits, actions, offset, NULL, dlogp);

    float baseline_value = 0.0f;
    if (baseline == MeanBaseline) baseline_value = mean_return;

    for (int t = 0; t < offset; t++)
        dlogp[t] *= -(rewards[t] - baseline_value);

    mlp_backward(policy->mlp, &cache, dlogp, NULL);

    free(logits);
    free(dlogp);
    free_mlp_cache(&cache);

    free(observations);
    free(actions);
    free(rewards);
    free(dones);
}


// void discrete_policy_gradient(
//     Env *env,
//     Policy *policy,
//     int n_episodes,
//     int max_steps_per_episode,
//     float gamma
// ) {
//     float *observations = malloc(max_steps_per_episode * env->obs_size * sizeof(float));
//     float *actions = malloc(max_steps_per_episode * env->act_size * sizeof(float));
//     float *rewards = malloc(max_steps_per_episode * sizeof(float));
//     bool *dones = malloc(max_steps_per_episode * sizeof(bool));

//     for (int i = 0; i < n_episodes; i++) {
//         int total_steps = policy_rollout(
//             env,
//             policy,
//             max_steps_per_episode,
//             observations,
//             actions,
//             rewards,
//             dones
//         );

//         discounted_cumsum_inplace(rewards, total_steps, gamma);

//         MLPCache cache = create_mlp_cache(policy->mlp, total_steps);
//         float *logits = malloc(total_steps * env->act_size * sizeof(float));
//         mlp_forward(policy->mlp, observations, total_steps, logits, &cache);

//         float *dlogp  = malloc(total_steps * sizeof(float));
//         policy_log_prob_from_logits(policy, logits, actions, total_steps, NULL, dlogp);

//         for (int t = 0; t < total_steps; t++)
//             dlogp[t] *= -rewards[t];

//         mlp_backward(policy->mlp, &cache, dlogp, NULL);

//         free(logits);
//         free(dlogp);
//         free_mlp_cache(&cache);
//     }

//     free(observations);
//     free(actions);
//     free(rewards);
//     free(dones);
// }
