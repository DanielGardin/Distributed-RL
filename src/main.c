#include <stdio.h>
#include <stdlib.h>
#include "environments/cartpole.h"
#include "raylib.h"

#define WIDTH 600
#define HEIGHT 200

int main() {
    InitWindow(WIDTH, HEIGHT, "puffer Cartpole");
    SetTargetFPS(50);

    Env env = make_cartpole_env(1.0f, true);

    float *obs = malloc(env.obs_size * sizeof(float));
    float action;
    float reward;
    bool done = false;
    env_reset(&env, obs);

    int i = 0;
    while (!done && !WindowShouldClose()) {
        printf("INFO: Starting render step %d\n", i);
        // BeginDrawing();
        env_render(&env);
        // EndDrawing();

        printf("Observation: %f, %f, %f, %f\n", 
        obs[0], obs[1], obs[2], obs[3]);

        action = ++i % 2;

        printf("Action: %f", action);

        env_step(&env, &action, obs, &reward, &done);

        printf("Got reward: %d\n", (int)reward);
    };
    CloseWindow();

    printf("Finished simulation at step %d\n", i);
    env_destroy(&env);
    free(obs);

    return 0;
};