CXX = g++-15
OPENBLAS_DIR = $(shell brew --prefix openblas)
CXXFLAGS = -std=c++23 -Wall -Werror -Wextra -Wno-unused -O2 -fopenmp
LDFLAGS = -L$(OPENBLAS_DIR)/lib -lopenblas
SOURCE = src
BUILD = build
INCLUDE = -Iinclude -I$(OPENBLAS_DIR)/include

$(BUILD)/main: main.cpp $(SOURCE)/*.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDE) $(LDFLAGS) -I$(SOURCE) main.cpp -o $(BUILD)/main

run: $(BUILD)/main
	OMP_NUM_THREADS=11 OPENBLAS_NUM_THREADS=11 ./$(BUILD)/main

$(BUILD)/tests: tests/tests.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE) $(LDFLAGS) -I$(SOURCE) tests/tests.cpp -o $(BUILD)/tests

tests: $(BUILD)/tests
	./$(BUILD)/tests tests/actual.txt
	source .venv/bin/activate && python tests/tests.py tests/expected.txt && python tests/compare.py tests/actual.txt tests/expected.txt

clean:
	rm -rf $(BUILD)/*

.PHONY: clean run tests