#pragma once
#include <stdbool.h>

#include "environments/common.h"

/* Cartpole environment 

Cartpole is a classic control problem where the goal is to balance a pole on a moving cart
by applying horizontal forces to the cart.
This environment corresponds to the version of the cart-pole problem described by Barto, Sutton,
and Anderson in https://ieeexplore.ieee.org/document/6313077.

Action Space:
- int: Push cart to the left (0) or right (1)

Observation Space:
- float[4]:
    - Cart Position: x
    - Cart Velocity: x_dot
    - Pole Angle: theta
    - Pole Angular Velocity: theta_dot

Reward Structure:
- +1 for every time step the pole remains upright

Termination Conditions:
- Pole Angle exceeds ±12 degrees
- Cart Position exceeds ±2.4 units
- Episode length reaches `max_steps` (default 200)

C implementation heavly inspired by https://github.com/PufferAI/PufferLib
*/

typedef struct CartpoleState {
    float x;
    float x_dot;
    float theta;
    float theta_dot;
    int step_count;
    float force_magnitude;
    bool continuous;
} CartpoleState;

Env make_cartpole_env(float force_magnitude, bool continuous);
