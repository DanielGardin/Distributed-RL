#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "metrics.h"

TrainingMetrics create_metrics(int grad_steps, int n_episodes) {
    TrainingMetrics metrics = {0};

    metrics.updates_capacity = grad_steps;
    metrics.num_episodes = n_episodes;

    metrics.returns = calloc(grad_steps * n_episodes, sizeof(float));
    metrics.steps = calloc(grad_steps * n_episodes, sizeof(int));
    metrics.loss = calloc(grad_steps * n_episodes, sizeof(float));

    // Size [updates]

    metrics.step_times = calloc(grad_steps, sizeof(double));
    metrics.comm_times = calloc(grad_steps, sizeof(double));
    metrics.rollout_times = calloc(grad_steps, sizeof(double));
    metrics.forward_times = calloc(grad_steps, sizeof(double));
    metrics.backward_times = calloc(grad_steps, sizeof(double));
    metrics.update_times = calloc(grad_steps, sizeof(double));

    // Starts
    metrics.step_starts = calloc(grad_steps, sizeof(double));
    metrics.comm_starts = calloc(grad_steps, sizeof(double));
    metrics.rollout_starts = calloc(grad_steps, sizeof(double));
    metrics.forward_starts = calloc(grad_steps, sizeof(double));
    metrics.backward_starts = calloc(grad_steps, sizeof(double));
    metrics.update_starts = calloc(grad_steps, sizeof(double));

    return metrics;
}

void write_metrics_timeline_csv(const TrainingMetrics *metrics, const MPIContext *mpi_ctx, const char *filepath) {
    FILE *f = fopen(filepath, "w");
    if (!f) return;

    fprintf(f, "rank,update,phase,start,duration\n");
    int updates = metrics->updates_capacity;
    int rank = mpi_ctx ? mpi_ctx->rank : 0;

    for (int i = 0; i < updates; i++) {
        // step
        fprintf(f, "%d,%d,step,%.9f,%.9f\n", rank, i, metrics->step_starts[i], metrics->step_times[i]);
        // comm (combined broadcast + reduce)
        fprintf(f, "%d,%d,comm,%.9f,%.9f\n", rank, i, metrics->comm_starts[i], metrics->comm_times[i]);
        // rollout/forward/backward/update (first-episode starts; durations accumulated)
        fprintf(f, "%d,%d,rollout,%.9f,%.9f\n", rank, i, metrics->rollout_starts[i], metrics->rollout_times[i]);
        fprintf(f, "%d,%d,forward,%.9f,%.9f\n", rank, i, metrics->forward_starts[i], metrics->forward_times[i]);
        fprintf(f, "%d,%d,backward,%.9f,%.9f\n", rank, i, metrics->backward_starts[i], metrics->backward_times[i]);
        fprintf(f, "%d,%d,update,%.9f,%.9f\n", rank, i, metrics->update_starts[i], metrics->update_times[i]);
    }

    fclose(f);
}

void write_metrics_results_csv(const TrainingMetrics *metrics, const MPIContext *mpi_ctx, const char *filepath) {
    FILE *f = fopen(filepath, "w");
    if (!f) return;

    int updates = metrics->updates_capacity;
    int episodes = metrics->num_episodes;
    (void)mpi_ctx; // unused for averages removal

    fprintf(f, "grad_step,episode,returns,steps,loss\n");

    for (int gs = 0; gs < updates; gs++) {
        for (int ep = 0; ep < episodes; ep++) {
            int idx = gs * episodes + ep;
            double ret_val = metrics->returns[idx];
            int steps_val = metrics->steps[idx];
            double loss_val = metrics->loss[idx];
            fprintf(f, "%d,%d,%.9f,%d,%.9f\n",
                    gs, ep, ret_val, steps_val, loss_val);
        }
    }

    fclose(f);
}

void free_metrics(TrainingMetrics *metrics) {
    free(metrics->returns);
    free(metrics->steps);
    free(metrics->loss);
    free(metrics->step_times);
    free(metrics->comm_times);
    free(metrics->rollout_times);
    free(metrics->forward_times);
    free(metrics->backward_times);
    free(metrics->update_times);

    free(metrics->step_starts);
    free(metrics->comm_starts);
    free(metrics->rollout_starts);
    free(metrics->forward_starts);
    free(metrics->backward_starts);
    free(metrics->update_starts);
}

void reduce_metrics(TrainingMetrics *metrics, const MPIContext *mpi_ctx, int root_rank) {
    int episode_elems = metrics->updates_capacity * metrics->num_episodes;
    int update_elems  = metrics->updates_capacity;

    MPI_Reduce(mpi_ctx->rank == root_rank ? MPI_IN_PLACE : metrics->returns,
               metrics->returns,
               episode_elems, MPI_FLOAT, MPI_SUM,
               root_rank, mpi_ctx->comm);

    MPI_Reduce(mpi_ctx->rank == root_rank ? MPI_IN_PLACE : metrics->steps,
               metrics->steps,
               episode_elems, MPI_INT, MPI_SUM,
               root_rank, mpi_ctx->comm);

    MPI_Reduce(mpi_ctx->rank == root_rank ? MPI_IN_PLACE : metrics->loss,
               metrics->loss,
               episode_elems, MPI_FLOAT, MPI_SUM,
               root_rank, mpi_ctx->comm);

    MPI_Reduce(mpi_ctx->rank == root_rank ? MPI_IN_PLACE : metrics->step_times,
               metrics->step_times,
               update_elems, MPI_DOUBLE, MPI_SUM,
               root_rank, mpi_ctx->comm);

    MPI_Reduce(mpi_ctx->rank == root_rank ? MPI_IN_PLACE : metrics->rollout_times,
               metrics->rollout_times,
               update_elems, MPI_DOUBLE, MPI_SUM,
               root_rank, mpi_ctx->comm);

    MPI_Reduce(mpi_ctx->rank == root_rank ? MPI_IN_PLACE : metrics->forward_times,
               metrics->forward_times,
               update_elems, MPI_DOUBLE, MPI_SUM,
               root_rank, mpi_ctx->comm);

    MPI_Reduce(mpi_ctx->rank == root_rank ? MPI_IN_PLACE : metrics->backward_times,
               metrics->backward_times,
               update_elems, MPI_DOUBLE, MPI_SUM,
               root_rank, mpi_ctx->comm);

    MPI_Reduce(mpi_ctx->rank == root_rank ? MPI_IN_PLACE : metrics->update_times,
               metrics->update_times,
               update_elems, MPI_DOUBLE, MPI_SUM,
               root_rank, mpi_ctx->comm);

    MPI_Reduce(mpi_ctx->rank == root_rank ? MPI_IN_PLACE : metrics->comm_times,
               metrics->comm_times,
               update_elems, MPI_DOUBLE, MPI_SUM,
               root_rank, mpi_ctx->comm);

    MPI_Reduce(mpi_ctx->rank == root_rank ? MPI_IN_PLACE : &metrics->wall_time_train,
               &metrics->wall_time_train,
               1, MPI_DOUBLE, MPI_SUM,
               root_rank, mpi_ctx->comm);

    MPI_Reduce(mpi_ctx->rank == root_rank ? MPI_IN_PLACE : &metrics->wall_time_total,
               &metrics->wall_time_total,
               1, MPI_DOUBLE, MPI_SUM,
               root_rank, mpi_ctx->comm);

    if (mpi_ctx->rank == root_rank) {
        for (int i = 0; i < episode_elems; i++) {
            metrics->returns[i] /= mpi_ctx->world_size;
            metrics->loss[i] /= mpi_ctx->world_size;
        }
    }
}