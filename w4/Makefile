CXX = g++-15
OPENBLAS_DIR = $(shell brew --prefix openblas)
XNNPACK_DIR = XNNPACK
XNNPACK_BUILD = $(XNNPACK_DIR)/build
CXXFLAGS = -std=c++23 -Wall -Werror -Wextra -Wno-unused -O2 -fopenmp
LDFLAGS = -L$(OPENBLAS_DIR)/lib -lopenblas \
	-L$(XNNPACK_BUILD) -lXNNPACK -lxnnpack-microkernels-prod \
	-L$(XNNPACK_BUILD)/pthreadpool -lpthreadpool \
	-L$(XNNPACK_BUILD)/cpuinfo -lcpuinfo \
	-L$(XNNPACK_BUILD)/kleidiai -lkleidiai
SOURCE = src
BUILD = build
INCLUDE = -Iinclude -I$(OPENBLAS_DIR)/include \
	-I$(XNNPACK_DIR)/include \
	-I$(XNNPACK_BUILD)/pthreadpool-source/include

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

clean:
	rm -rf $(BUILD)/*

.PHONY: clean run mprof tfprof tests