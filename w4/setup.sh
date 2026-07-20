#!/bin/bash

DIR="$(dirname "$0")"

mkdir -p "$DIR"/build
mkdir -p "$DIR"/include

if [ -d "$DIR"/include/taskflow/ ]; then
    echo "Taskflow already exists, skipping clone."
else
    echo "Cloning taskflow..."
    mkdir -p "$DIR"/include/taskflow
    cd "$DIR" && git clone "https://github.com/taskflow/taskflow.git"
    echo "Copying taskflow headers..."
    cp -r taskflow/taskflow/ include/
    echo "Cleaning up taskflow clone..."
    rm -rf taskflow/
fi

if [ -d "$DIR"/XNNPACK/ ]; then
    echo "XNNPACK already exists, skipping clone."
else
    echo "Cloning and building XNNPACK..."
    cd "$DIR" && git clone "https://github.com/google/XNNPACK.git"
    cd XNNPACK
    mkdir -p build && cd build
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DXNNPACK_BUILD_TESTS=OFF \
        -DXNNPACK_BUILD_BENCHMARKS=OFF \
        -DXNNPACK_BUILD_LIBRARY_TYPE=static
    cmake --build . -j"$(sysctl -n hw.ncpu)"
    cd "$DIR"
fi

echo "Setup complete."
