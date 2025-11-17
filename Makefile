CC := gcc
CFLAGS := -Iinclude -Wall -Wextra -O2 -g
RAYLIB_DIR := external/raylib-5.5_linux_amd64
RAYLIB_INCLUDE := -I$(RAYLIB_DIR)/include
# Link directly against the shipped shared object file. We also add an rpath
# so the final binary can find the shared library relative to its location.
RAYLIB_LIB := $(RAYLIB_DIR)/lib/libraylib.so.550

# BLAS linking: default to OpenBLAS. Override via environment, e.g.
#    make BLAS_LIB="-lopenblas"
BLAS_LIB ?= -lopenblas
# Optional include path for cblas.h (set if your distro installs headers in a non-standard location)
BLAS_INC ?=

LDFLAGS := -lm $(RAYLIB_LIB) $(BLAS_LIB) -Wl,-rpath,'$$ORIGIN/../$(RAYLIB_DIR)/lib'

CFLAGS := $(CFLAGS) $(RAYLIB_INCLUDE) $(BLAS_INC)

SRCS := $(shell find src -type f -name '*.c')

OBJS := $(SRCS:.c=.o)

TARGET := bin/cartpole_demo

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
