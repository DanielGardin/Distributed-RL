/***************************
 *    Algorithms' utils    *
 ***************************/
#include <stdio.h>
#include "algorithms/utils.h"

void discounted_cumsum_inplace(float *r, bool *dones, int size, float gamma) {
    float running = 0.0f;
    
    for (int t = size - 1; t >= 0; --t) {
        if (dones[t]) running = 0.0f;

        running = r[t] + gamma * running;
        r[t] = running;
    }
}

EpisodeStatistics policy_rollout(
    Env *env,
    const Policy *policy,
    int n_steps,
    float *observations,
    float *actions,
    float *rewards,
    bool *dones
) {
    float *cur_obs = observations;
    float *next_obs;
    float *last_obs_buffer = malloc(env->obs_size * sizeof(float));

    float *cur_act = actions;
    float *cur_rew = rewards;
    bool *cur_done = dones;

    env_reset(env, cur_obs);

    int step_count = 0;
    float cum_reward = 0.0f;
    float entropy, total_entropy = 0.0f;
    bool done = false;
    for (; step_count < n_steps && !done; step_count++) {
        policy_sample_action(policy, cur_obs, 1, cur_act, NULL, &entropy);
        total_entropy += entropy;

        next_obs = (step_count + 1 < n_steps)
                    ? cur_obs + env->obs_size
                    : last_obs_buffer;

        env_step(env, cur_act, next_obs, cur_rew, &done);
        cum_reward += *cur_rew;
        *cur_done = done;

        cur_obs = next_obs;
        cur_act += env->act_size;
        cur_rew += 1;
        cur_done += 1;

        if (done) break;
    };

    free(last_obs_buffer);

    return (EpisodeStatistics) {
        .episode_return = cum_reward,
        .total_steps=step_count,
        .mean_entropy=total_entropy/step_count
    };
}

void print_array(float *array, int size) {
    fprintf(stderr, "{ %.3f", array[0]);
    for (int i = 1; i<size; i++)
        fprintf(stderr, ", %.3f", array[i]);

    fprintf(stderr, " }");
}

// EpisodeStatistics evaluate_policy(Env *env, const Policy *policy, int max_steps, int repeats) {
//     float *obs = malloc(env->obs_size * sizeof(float));
//     float *act = malloc(env->act_size * sizeof(float));

//     float reward;
//     bool done = false;

//     env_reset(env, obs);

//     int step_count = 0;
//     for (; !done && step_count < max_steps; step_count++) {
//         env_render(env);

//         print_array(obs, env->obs_size);
//         fprintf(stderr, "\n");

//         policy_sample_action(policy, obs, 1, act, NULL);

//         env_step(env, act, obs, &reward, &done);

//         print_array(act, env->act_size);
//         fprintf(stderr, "\n");
//     };

//     free(obs);
//     free(act);
// }
