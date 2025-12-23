/***************************
 *    Algorithms' utils    *
 ***************************/
#include <stdio.h>
#include "algorithms/utils.h"

ExperienceBuffer create_buffer(
    int capacity,
    int obs_size,
    int act_size
) {
    float *observations = malloc(capacity * obs_size * sizeof(float));
    float *actions = malloc(capacity * act_size * sizeof(float));
    float *rewards = malloc(capacity * sizeof(float));
    bool *dones = malloc(capacity * sizeof(bool));

    return (ExperienceBuffer) {
        .capacity=capacity,
        .size=0,
        .observations=observations,
        .actions=actions,
        .rewards=rewards,
        .dones=dones
    };
}

void free_buffer(ExperienceBuffer *buffer) {
    free(buffer->observations);
    free(buffer->actions);
    free(buffer->rewards);
    free(buffer->dones);
}

void policy_rollout(
    Env *env,
    const Policy *policy,
    int n_steps,
    int n_episodes,
    ExperienceBuffer *buffer,
    MLPCache *cache
) {
    float *last_obs_buffer = malloc(env->obs_size * sizeof(float));
    float *logits = malloc(policy->mlp->output_size * sizeof(float));
    
    buffer->size = 0;
    float *cur_obs = buffer->observations;
    float *cur_act = buffer->actions;
    float *cur_rew = buffer->rewards;
    bool *cur_done = buffer->dones;
    float *next_obs;
    
    int step_count;
    for (int i=0; i < n_episodes; i++) {
        bool done = false;
        env_reset(env, cur_obs);

        for (step_count=0; !done; step_count++) {
            mlp_forward(policy->mlp, cur_obs, 1, logits, cache);
            policy_sample_action_from_logits(policy, logits, 1, cur_act);
        
            next_obs = (buffer->size+1 < buffer->capacity)
                       ? cur_obs + env->obs_size
                       : last_obs_buffer;

            env_step(env, cur_act, next_obs, cur_rew, &done);
            done = done || step_count+1>=n_steps;

            *cur_done = done;
            buffer->size++;

            cur_obs = next_obs;
            cur_act += env->act_size;
            cur_rew += 1;
            cur_done += 1;
        };
    }

    free(logits);
    free(last_obs_buffer);
}

float mean_return(ExperienceBuffer *buffer) {
    float total_return = 0.0f;
    int n_episodes = 0;

    float cum_reward = 0.0f;
    for (int t = 0; t < buffer->size; t++) {
        cum_reward += buffer->rewards[t];

        if (buffer->dones[t]) {
            total_return += cum_reward;
            cum_reward = 0.0f;
            n_episodes++;
        }
    }

    if (n_episodes == 0) {
        total_return = cum_reward;
        n_episodes++;
    }

    return total_return / n_episodes;
}

void discounted_cumsum(ExperienceBuffer *buffer, float gamma, float *returns) {
    float running = 0.0f;

    for (int t = buffer->size - 1; t >= 0; --t) {
        if (buffer->dones[t]) running = 0.0f;

        running = buffer->rewards[t] + gamma * running;
        returns[t] = running;   
    }
}
