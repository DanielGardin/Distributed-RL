#pragma once

#include "policy.h"

typedef struct ExperienceBuffer {
    int capacity, size;
    float *observations;
    float *actions;
    float *rewards;
    bool *dones;
} ExperienceBuffer;

ExperienceBuffer create_buffer(
    int capacity,
    int obs_size,
    int act_size
);

void free_buffer(ExperienceBuffer *buffer);

void policy_rollout(
    Env *env,
    const Policy *policy,
    int n_steps,
    int n_episodes,
    ExperienceBuffer *buffer,
    MLPCache *cache
);

float mean_return(ExperienceBuffer *buffer);

void discounted_cumsum(ExperienceBuffer *buffer, float gamma, float *returns);
