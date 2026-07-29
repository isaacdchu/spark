SHELL := /bin/bash
CXX = g++-15
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	OPENBLAS_DIR := $(shell brew --prefix openblas)
	OPENBLAS_INCLUDE := -I$(OPENBLAS_DIR)/include
	OPENBLAS_LIB := -L$(OPENBLAS_DIR)/lib -lopenblas
else
	OPENBLAS_INCLUDE := $(shell pkg-config --cflags openblas 2>/dev/null)
	OPENBLAS_LIB := $(shell pkg-config --libs openblas 2>/dev/null || echo -lopenblas)
endif
# tiny-dnn's vectorized convolution kernel is compiled out unless CNN_USE_AVX is
# defined, and backend.h #errors when it is defined without AVX available. Probe
# the compiler under the same -march=native we build with, so this stays off on
# arm64 (macOS) and on pre-AVX x86. Note this only speeds up 5x5 convolutions:
# conv2d_op_avx.h dispatches to the scalar conv2d_op_internal for every other
# kernel size, so 3x3 is unaffected.
TINYDNN_AVX := $(shell echo | $(CXX) -march=native -dM -E - 2>/dev/null | grep -q '__AVX__' && echo -DCNN_USE_AVX)
CXXFLAGS = -std=c++23 -Wall -Werror -Wextra -Wno-unused -Wno-psabi -O3 -march=native -fopenmp -DCNN_NO_SERIALIZATION -DCNN_USE_OMP $(TINYDNN_AVX)
UNAME_M := $(shell uname -m)
PLATFORM_DIR := $(shell echo $(UNAME_S) | tr '[:upper:]' '[:lower:]')-$(UNAME_M)
XNNPACK_LIBS = $(wildcard include/xnnpack/lib/$(PLATFORM_DIR)/*.a)
ifeq ($(UNAME_S),Darwin)
	LDFLAGS = $(OPENBLAS_LIB) $(XNNPACK_LIBS)
else
	# GNU ld resolves archive symbols in a single left-to-right pass, so the
	# circular refs between libXNNPACK.a/libcpuinfo.a/libpthreadpool.a/the
	# microkernel libs (fine for macOS's linker) need --start-group/--end-group.
	LDFLAGS = $(OPENBLAS_LIB) -Wl,--start-group $(XNNPACK_LIBS) -Wl,--end-group
endif
SOURCE = src
BUILD = build
INCLUDE = -Iinclude/taskflow -Iinclude/xsimd -Iinclude/xnnpack -isystem include $(OPENBLAS_INCLUDE)
# Where the Python venv lives. Overridable so it can sit outside the checkout:
# under WSL2 the repo is typically on /mnt/c, where a multi-gigabyte torch
# install is slow to reach over 9p and gets picked up by folder sync. Pair with
# UV_PROJECT_ENVIRONMENT so uv builds it in the same place, e.g.
#   export UV_PROJECT_ENVIRONMENT=$HOME/venvs/w8
#   make VENV=$HOME/venvs/w8 tests
VENV ?= .venv
ACTIVATE = source $(VENV)/bin/activate

$(BUILD)/main: main.cpp $(SOURCE)/*.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDE) -I$(SOURCE) main.cpp -o $(BUILD)/main $(LDFLAGS)

run: $(BUILD)/main
	./$(BUILD)/main

mprof: $(BUILD)/main
	$(ACTIVATE) && mprof run ./$(BUILD)/main

tfprof: $(BUILD)/main
	TF_ENABLE_PROFILER=result.tfp ./$(BUILD)/main

$(BUILD)/tests: tests/tests.cpp $(SOURCE)/*.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDE) -I$(SOURCE) tests/tests.cpp -o $(BUILD)/tests $(LDFLAGS)

tests: $(BUILD)/tests
	./$(BUILD)/tests tests/actual.txt
	$(ACTIVATE) && python tests/tests.py tests/expected.txt && python tests/compare.py tests/actual.txt tests/expected.txt

# All benchmarking is launched from the dashboard. Start it with `make dashboard`,
# then choose parameters and run sweeps from the Run panel. The dashboard invokes
# benchmark/run_sweep.sh (which compiles benchmark/benchmark.cpp per block config)
# under the hood, so there is no standalone benchmark run target.
dashboard:
	$(ACTIVATE) && python benchmark/benchmark.py

clean:
	rm -rf $(BUILD)/*

.PHONY: clean run mprof tfprof tests dashboard
