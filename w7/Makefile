CXX = g++-15
OPENBLAS_DIR = $(shell brew --prefix openblas)
CXXFLAGS = -std=c++23 -Wall -Werror -Wextra -Wno-unused -Wno-psabi -O3 -march=native -fopenmp
XNNPACK_LIBS = $(wildcard include/xnnpack/lib/*.a)
LDFLAGS = -L$(OPENBLAS_DIR)/lib -lopenblas $(XNNPACK_LIBS)
SOURCE = src
BUILD = build
INCLUDE = -Iinclude/taskflow -Iinclude/xsimd -Iinclude/xnnpack -I$(OPENBLAS_DIR)/include

$(BUILD)/main: main.cpp $(SOURCE)/*.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDE) $(LDFLAGS) -I$(SOURCE) main.cpp -o $(BUILD)/main

run: $(BUILD)/main
	./$(BUILD)/main

mprof: $(BUILD)/main
	mprof run ./$(BUILD)/main

tfprof: $(BUILD)/main
	TF_ENABLE_PROFILER=result.tfp ./$(BUILD)/main

$(BUILD)/tests: tests/tests.cpp $(SOURCE)/*.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDE) $(LDFLAGS) -I$(SOURCE) tests/tests.cpp -o $(BUILD)/tests

tests: $(BUILD)/tests
	./$(BUILD)/tests tests/actual.txt
	source .venv/bin/activate && python tests/tests.py tests/expected.txt && python tests/compare.py tests/actual.txt tests/expected.txt

# All benchmarking is launched from the dashboard. Start it with `make dashboard`,
# then choose parameters and run sweeps from the Run panel. The dashboard invokes
# benchmark/run_sweep.sh (which compiles benchmark/benchmark.cpp per block config)
# under the hood, so there is no standalone benchmark run target.
dashboard:
	source .venv/bin/activate && python benchmark/benchmark.py

clean:
	rm -rf $(BUILD)/*

.PHONY: clean run mprof tfprof tests dashboard