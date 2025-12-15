#include <stdlib.h>
#include <math.h>

#include "raylib.h"
#include "environments/cartpole.h"
#include "rng.h"

#define X_THRESHOLD 2.4f
#define THETA_THRESHOLD_RADIANS (12 * 2 * M_PI / 360)
#define CART_MASS 1.0f
#define POLE_MASS 0.1f
#define TOTAL_MASS  (CART_MASS+POLE_MASS)
#define POLE_LENGTH 0.5f
#define GRAVITY 9.8f
#define TAU 0.02f

#define WIDTH 600
#define HEIGHT 200
#define SCALE 100

const Color BACKGROUND_COLOR = (Color){ 24, 32, 38, 255};
const Color POLE_COLOR = (Color){140, 170, 200, 255};
const Color CART_COLOR = (Color){180, 140, 140, 255};
const Color TEXT_COLOR = (Color){225, 230, 235, 255};

void cartpole_reset(CartpoleState *state, float *obs_buf) {
    state->step_count=0;
    state->x = rand_uniform(-0.05f, 0.05f);
    state->x_dot=rand_uniform(-0.05f, 0.05f);
    state->theta=rand_uniform(-0.05f, 0.05f);
    state->theta_dot=rand_uniform(-0.05f, 0.05f);

    obs_buf[0] = state->x;
    obs_buf[1] = state->x_dot;
    obs_buf[2] = state->theta;
    obs_buf[3] = state->theta_dot;
}

void cartpole_step(CartpoleState *state, const float *action, float *obs_buf, float *reward_buf, bool *done_buf) {
    float force = state->continuous ? *(action) * state->force_magnitude
                                  : (*(action) > 0.5f ? state->force_magnitude: -state->force_magnitude);

    float costheta = cosf(state->theta);
    float sintheta = sinf(state->theta);

    float tmp = (
        force + POLE_MASS * POLE_LENGTH * state->theta_dot * state->theta_dot * sintheta
    ) / TOTAL_MASS;

    float thetaacc = (GRAVITY * sintheta - costheta * tmp) / (
            POLE_LENGTH
            * (4.0f / 3.0f - POLE_MASS * costheta * costheta / TOTAL_MASS)
        );

    float xacc = tmp - POLE_LENGTH * POLE_MASS * thetaacc * costheta / TOTAL_MASS;

    // Semi-implicit Euler
    state->x_dot     += TAU * xacc;
    state->x         += TAU * state->x_dot;
    state->theta_dot += TAU * thetaacc;
    state->theta     += TAU * state->theta_dot;

    state->step_count++;

    obs_buf[0] = state->x;
    obs_buf[1] = state->x_dot;
    obs_buf[2] = state->theta;
    obs_buf[3] = state->theta_dot;

    *done_buf = (
        state->x < -X_THRESHOLD || state->x > X_THRESHOLD ||
        state->theta < -THETA_THRESHOLD_RADIANS || state->theta > THETA_THRESHOLD_RADIANS
    );
    *reward_buf = 1.0f;
}

void cartpole_destroy(CartpoleState *state) {
    free(state);
}

void cartpole_render(CartpoleState *state) {
    BeginDrawing();
    ClearBackground(BACKGROUND_COLOR);

    DrawLine(0, HEIGHT / 1.5, WIDTH, HEIGHT / 1.5, POLE_COLOR);

    float cart_x = WIDTH / 2 + state->x * SCALE;
    float cart_y = HEIGHT / 1.6;

    DrawRectangle((int)(cart_x - 20), (int)(cart_y - 10), 40, 20, POLE_COLOR);

    float pole_length = 2.0f * POLE_LENGTH * SCALE;
    float pole_x2 = cart_x + sinf(state->theta) * pole_length;
    float pole_y2 = cart_y - cosf(state->theta) * pole_length;

    DrawLineEx((Vector2){cart_x, cart_y}, (Vector2){pole_x2, pole_y2}, 5, CART_COLOR);
    DrawText(TextFormat("Steps: %i", state->step_count), 10, 10, 20, TEXT_COLOR);
    DrawText(TextFormat("Cart Position: %.2f", state->x), 10, 40, 20, TEXT_COLOR);
    DrawText(TextFormat("Pole Angle: %.2f", state->theta * 180.0f / PI), 10, 70, 20, TEXT_COLOR);
    EndDrawing();
}

Env make_cartpole_env(float force_magnitude, bool continuous) {
    CartpoleState *state = malloc(sizeof(CartpoleState));

    state->force_magnitude=force_magnitude;
    state->continuous=continuous;

    static int act_space[1] = {2};

    return (Env){
        .ptr = state,
        .name = "Cartpole",
        .obs_size = 4,
        .act_size = 1,
        .act_space = act_space,
        .reset = (void (*)(void*, float*))cartpole_reset,
        .step = (void (*)(void*, const float*, float*, float*, bool*))cartpole_step,
        .destroy = (void (*)(void*))cartpole_destroy,
        .render = (void (*)(void*))cartpole_render
    };
}
