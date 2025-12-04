#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "raylib.h"

#include "algorithms/reinforce.h"
#include "algorithms/common.h"
#include "environments/cartpole.h"
#include "nn/mlp.h"
#include "nn/optimizers.h"
#include "nn/debug.h"
#include "rng.h"

#define WIDTH 600
#define HEIGHT 200

void render_episode(Env *env, Policy *policy);

int main() {
    rng_seed(1);

    Env env = make_cartpole_env(1.0f, true);

    int input_size[2] = {env.obs_size, 1};
    int output_size = env.act_size;
    Activation activations[2] = {relu, identity};

    MLP policynet = create_mlp(
        input_size,
        output_size,
        2,
        activations
    );
    
    for (int l=0; l<policynet.num_layers; l++)
    kaiming_init(&policynet.layers[l]);
    
    Policy policy = create_binary_policy(&policynet);
    
    for (int n_grad_steps = 0; n_grad_steps < 1000; n_grad_steps++) {
        discrete_reinforce_step(&env, &policy, 1, 100, 0.99);
        gd_step(&policynet, 1e-4);
        mlp_zero_grad(&policynet);
    }

    print_mlp(&policynet);    
    // save_mlp_weights(&policy, "weights.bin");

    render_episode(&env, &policy);

    free_mlp(&policynet);
    env_destroy(&env);

    return 0;
}

void print_array(float *array, int size) {
    fprintf(stderr, "{ %.3f", array[0]);
    for (int i = 1; i<size; i++)
        fprintf(stderr, ", %.3f", array[i]);

    fprintf(stderr, " }");
}

void render_episode(Env *env, Policy *policy) {
    InitWindow(WIDTH, HEIGHT, "puffer Cartpole");
    SetTargetFPS(20);

    float *obs = malloc(env->obs_size * sizeof(float));
    float *act = malloc(env->act_size * sizeof(float));

    float reward;
    bool done = false;

    env_reset(env, obs);

    int step_count = 0;
    for (; !done && !WindowShouldClose(); step_count++) {
        env_render(env);

        fprintf(stderr, "INFO: Observation ");
        print_array(obs, env->obs_size);
        fprintf(stderr, "\n");

        policy_sample_action(policy, obs, 1, act, NULL);

        env_step(env, act, obs, &reward, &done);

        fprintf(stderr, "INFO: Action      ");
        print_array(act, env->act_size);
        fprintf(stderr, "\n");

        fprintf(stderr, "INFO: Reward        %.3f\n", reward);
    };
    CloseWindow();

    fprintf(stderr, "INFO: Finished simulation at step %d\n", step_count);

    free(obs);
    free(act);
}