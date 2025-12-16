# Distributed-RL

A simple, high-performance, distributed reinforcement learning (RL) project in C using MPI, OpenBLAS, and raylib. The main demo trains a policy on the CartPole environment with synchronous model broadcast and gradient aggregation across MPI processes. The repository includes a small neural network library, a REINFORCE-style policy gradient loop, performance metrics, and unit tests.

## Features
- Distributed training via MPI (model broadcast + gradient reduction)
- Lightweight MLP, activations, and Adam optimizer in C
- CartPole environment with optional raylib rendering
- Detailed performance and learning metrics exported to CSV
- CMake build with test targets and `ctest` integration

## Project Structure
- `src/`: C source files
  - `main.c`: CartPole distributed training demo and CLI
  - `algorithms/`: policy gradient utilities
  - `nn/`: MLP, activations, optimizers, caches, debug helpers
  - `environments/`: CartPole (and placeholders for others)
  - `distributed/`: MPI helpers (init, broadcast, reduce)
  - `metrics.c`: metrics tracking, CSV output, MPI reduction
- `include/`: public headers mirroring the `src/` layout
- `test/`: unit tests (`test_mlp`, `test_gradient`, `test_overfitting`, `test_utils`)
- `external/`: vendored `raylib-5.5_linux_amd64` (headers + libs)
- `build/`: CMake build directory (generated)

## Dependencies
- CMake >= 3.10
- C compiler with C11 support (e.g., GCC)
- MPI (OpenMPI or MPICH)
- BLAS (OpenBLAS recommended)
- raylib (bundled under `external/raylib-5.5_linux_amd64` or system-installed)

## Build
Use an out-of-source build:
```bash
cmake -S . -B build
cmake --build build -j
```
Artifacts:
- Demo executable: `build/bin/reinforce`
- Tests: `build/test/{test_mlp,test_gradient,test_overfitting}`

## Run
The demo is MPI-parallel. Example:
```bash
mpirun -np 4 ./build/bin/reinforce -n 16 -e 4 -o results
```
CLI options:
- `Environment` (positional): environment name, default `cartpole`
- `-s <int>`: RNG seed (default: 1)
- `-n <int>`: NN hidden size (default: 16)
- `-e <int>`: episodes per gradient step (batch size, default: 1)
- `-m <int>`: max steps per episode (default: 500)
- `-y <float>`: discount factor gamma (default: 0.99)
- `-k <int>`: number of gradient steps (default: 2500)
- `-l <float>`: learning rate (default: 1e-2)
- `-o <path>`: output directory for CSVs and weights (default: disabled)
- `-r`: render an episode using the trained policy (raylib window)
- `-h`: print help

Examples:
```bash
# Short run with output
mpirun -np 4 ./build/bin/reinforce -n 16 -e 2 -k 100 -o results

# Render an episode (run on rank 0)
mpirun -np 1 ./build/bin/reinforce -r -k 100 -n 16
```

## Outputs
When `-o <path>` is provided:
- `training_results.csv`: per (grad_step, episode) rows with `returns,steps,loss`
- `training_timeline_rank{r}.csv`: per-rank timeline with phases and durations
  - Columns: `rank,update,phase,start,duration`, where `phase ∈ {step,comm,rollout,forward,backward,update}`
- `weights.bin`: serialized MLP weights from rank 0 after training

Console summary (rank 0) includes:
- Environment, MPI processes, gradient steps, episodes per step
- Wall time breakdown (training vs total)
- Training phase times: communication, rollout, forward, backward, optimizer
- Throughput: total episodes/steps, episodes/sec, steps/sec, avg episode length
- Learning: avg/min/max return, return std dev
- Scalability: comm/compute ratio, parallel efficiency (based on compute ratio)

## Tests
Run tests after building:
```bash
ctest --test-dir build --output-on-failure
```
