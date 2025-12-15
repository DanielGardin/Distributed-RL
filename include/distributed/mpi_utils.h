#pragma once

#include <mpi.h>

typedef struct {
    int rank;
    int world_size;
    MPI_Comm comm;
} MPIContext;

MPIContext mpi_init_context(int *argc, char ***argv);

void main_printf(const MPIContext *ctx, const char *format, ...);

void mpi_finalize();

double get_time();
