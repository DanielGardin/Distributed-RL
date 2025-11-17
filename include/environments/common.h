#pragma once

#include <stdbool.h>
#include <stdlib.h>

#define ENV_INLINE static inline

typedef struct Env {
    void *ptr;
    size_t obs_size;
    size_t act_size;
    void (*reset)(void*, float*);
    void (*step)(void*, float*, float*, float*, bool*);
    void (*destroy)(void*);
    void (*render)(void*);
} Env;

ENV_INLINE void env_reset(Env *e, float*obs_buf) {
    return e->reset(e->ptr, obs_buf);
};

ENV_INLINE void env_step(
    Env *e, float *action, float *obs_buf, float *reward_buf, bool *done_buf
) {
    e->step(e->ptr, action, obs_buf, reward_buf, done_buf);
};

ENV_INLINE void env_destroy(Env *e) {
    e->destroy(e->ptr);
};

ENV_INLINE void env_render(Env *e) {
    e->render(e->ptr);
};
