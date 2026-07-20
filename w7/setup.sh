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
    cp -r taskflow/taskflow/ include/taskflow/
    echo "Cleaning up taskflow clone..."
    rm -rf taskflow/
fi

if [ -d "$DIR"/include/xsimd/ ]; then
    echo "xsimd already exists, skipping clone."
else
    echo "Cloning xsimd..."
    mkdir -p "$DIR"/include/xsimd
    cd "$DIR" && git clone "https://github.com/xtensor-stack/xsimd.git"
    echo "Copying xsimd headers..."
    cp -r xsimd/include/xsimd/ include/xsimd/
    echo "Cleaning up xsimd clone..."
    rm -rf xsimd/
fi

if [ -d "$DIR"/include/xnnpack/ ]; then
    echo "XNNPACK already exists, skipping clone."
else
    echo "Cloning XNNPACK..."
    cd "$DIR" && git clone "https://github.com/google/XNNPACK.git"
    echo "Building XNNPACK (this can take a while)..."
    cmake -S XNNPACK -B XNNPACK/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DXNNPACK_BUILD_TESTS=OFF \
        -DXNNPACK_BUILD_BENCHMARKS=OFF
    cmake --build XNNPACK/build --config Release -j
    echo "Staging XNNPACK headers and libraries..."
    mkdir -p "$DIR"/include/xnnpack/lib
    # Public header (include/xnnpack.h) plus the pthreadpool header it pulls in.
    cp -r XNNPACK/include/. include/xnnpack/
    find XNNPACK/build -name "pthreadpool.h" -exec cp {} include/xnnpack/ \;
    find XNNPACK/build -name "cpuinfo.h" -exec cp {} include/xnnpack/ \;
    # Static libraries: libXNNPACK.a and its bundled deps (pthreadpool, cpuinfo, microkernels).
    find XNNPACK/build -name "*.a" -exec cp {} include/xnnpack/lib/ \;
    echo "Cleaning up XNNPACK clone..."
    rm -rf XNNPACK/
fi

echo "Setup complete."
