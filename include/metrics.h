#pragma once

#include "distributed/mpi_utils.h"

typedef struct TrainingMetrics {
    int updates_capacity;
    int num_episodes;

    double wall_time_train;
    double wall_time_total;

    // Size [updates * episodes]
    float *returns;
    int *steps;
    float *loss;

    // Size [updates]

    // Durations (seconds)
    double *step_times;
    double *comm_times;
    double *rollout_times;
    double *forward_times;
    double *backward_times;
    double *update_times;

    // Start timestamps (seconds, MPI_Wtime())
    double *step_starts;
    double *comm_starts;
    double *rollout_starts;
    double *forward_starts;
    double *backward_starts;
    double *update_starts;
} TrainingMetrics;

TrainingMetrics create_metrics(int grad_steps, int n_episodes);

void free_metrics(TrainingMetrics *metrics);

void reduce_metrics(TrainingMetrics *metrics, const MPIContext *mpi_ctx, int root_rank);

void write_metrics_timeline_csv(const TrainingMetrics *metrics, const MPIContext *mpi_ctx, const char *filepath);

void write_metrics_results_csv(const TrainingMetrics *metrics, const MPIContext *mpi_ctx, const char *filepath);
