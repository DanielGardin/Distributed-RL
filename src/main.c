#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "raylib.h"

#include "environments/cartpole.h"
#include "nn.h"
#include "rng.h"


#define WIDTH 600
#define HEIGHT 200

int main() {
    InitWindow(WIDTH, HEIGHT, "puffer Cartpole");
    SetTargetFPS(20);

    rng_seed(1);

    Env env = make_cartpole_env(1.0f, true);

    int input_size[2] = {4, 32};
    int output_size = 1;
    Activation activations[2] = {relu, logsigmoid};

    MLP policy = create_mlp(
        input_size,
        output_size,
        2,
        activations
    );

    for (int l=0; l<policy.num_layers; l++)
        kaiming_init(&policy.layers[l]);

    // for (int l=0; l<policy.num_layers; l++) {
    //     LinearLayer *layer = &policy.layers[l];

    //     printf("Layer %d (Weights) %d x %d\n", l, layer->input_size, layer->output_size);
    //     float *row = layer->weights;
    //     for (int i=0; i<layer->input_size; i++) {

    //         for (int j=0; j<layer->output_size; j++) {
    //             printf("%.3f ", row[j]);
    //         };
    //         printf("\n");
    //         row += layer->output_size;
    //     };
    // };

    float *obs = malloc(env.obs_size * sizeof(float));
    float log_prob, action;
    float reward;
    bool done = false;
    env_reset(&env, obs);
    while (!done && !WindowShouldClose()) {
        env_render(&env);

        fprintf(
            stderr, "INFO: Observation {%f, %f, %f, %f}\n",
            obs[0], obs[1], obs[2], obs[3]
        );

        mlp_forward(&policy, obs, 1, &log_prob);
        action = expf(log_prob);

        env_step(&env, &action, obs, &reward, &done);

        fprintf(stderr, "INFO: Action      {%f}\n", action);
        fprintf(stderr, "INFO: Reward       %d\n", (int)reward);
    };
    CloseWindow();

    fprintf(stderr, "Finished simulation at step %d\n", ((CartpoleState*)env.ptr)->step_count);
    env_destroy(&env);
    free(obs);

    return 0;
};