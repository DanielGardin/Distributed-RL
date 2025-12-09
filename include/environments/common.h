#pragma once

#include <stdbool.h>
#include <stdlib.h>

#define ENV_INLINE static inline

typedef struct Env {
    void *ptr;
    int obs_size;
    int act_size;
    int act_space;
    void (*reset)(void* env, float* obs_buf);
    void (*step)(void* env, const float* action, float* obs_buf, float* reward_buf, bool* done_buf);
    void (*destroy)(void* env);
    void (*render)(void* env);
} Env;

ENV_INLINE void env_reset(Env *env, float*obs_buf) {
    return env->reset(env->ptr, obs_buf);
};

ENV_INLINE void env_step(
    Env *env, const float *action, float *obs_buf, float *reward_buf, bool *done_buf
) {
    env->step(env->ptr, action, obs_buf, reward_buf, done_buf);
};

ENV_INLINE void env_destroy(Env *env) {
    env->destroy(env->ptr);
};

ENV_INLINE void env_render(Env *env) {
    env->render(env->ptr);
};
