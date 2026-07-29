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

# Static libs are platform/arch-specific (Mach-O on macOS, ELF on Linux), so
# they're staged per-platform and can coexist in the same checkout, e.g. one
# contributor on a Mac and another on WSL2/Linux without clobbering each other.
PLATFORM_DIR="$(uname -s | tr '[:upper:]' '[:lower:]')-$(uname -m)"

# Guard on the actual built library rather than just the directory: a build that
# crashed midway (XNNPACK's microkernel compile is memory-heavy and can OOM-kill
# WSL2) leaves an empty platform dir behind, and skipping on that would wedge the
# setup. Checking for libXNNPACK.a makes a failed run safely retryable.
if [ -f "$DIR"/include/xnnpack/lib/"$PLATFORM_DIR"/libXNNPACK.a ]; then
    echo "XNNPACK ($PLATFORM_DIR) already exists, skipping clone."
else
    if [ ! -d "$DIR"/XNNPACK ]; then
        echo "Cloning XNNPACK..."
        cd "$DIR" && git clone "https://github.com/google/XNNPACK.git"
    else
        echo "Reusing existing XNNPACK clone."
        cd "$DIR"
    fi
    echo "Building XNNPACK (this can take a while)..."
    # Bound parallelism to avoid OOM: the microkernel translation units are large,
    # so cap jobs at roughly one per 2 GB of RAM (WSL2 defaults to ~50% of host
    # RAM). Override with JOBS=N. This is what crashed WSL2 with an unbounded -j.
    if [ -z "${JOBS:-}" ]; then
        mem_gb=$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo 2>/dev/null || echo 4)
        JOBS=$(( mem_gb / 2 ))
        [ "$JOBS" -lt 1 ] && JOBS=1
        ncpu=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 1)
        [ "$JOBS" -gt "$ncpu" ] && JOBS=$ncpu
    fi
    echo "Using $JOBS parallel build job(s)."
    cmake -S XNNPACK -B XNNPACK/build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
        -DCMAKE_C_COMPILER="${CC:-gcc-15}" \
        -DCMAKE_CXX_COMPILER="${CXX:-g++-15}" \
        -DXNNPACK_BUILD_TESTS=OFF \
        -DXNNPACK_BUILD_BENCHMARKS=OFF
    cmake --build XNNPACK/build --config Release -j "$JOBS"
    echo "Staging XNNPACK headers and libraries..."
    mkdir -p "$DIR"/include/xnnpack/lib/"$PLATFORM_DIR"
    # Public header (include/xnnpack.h) plus the pthreadpool header it pulls in.
    # Only seed these on a fresh checkout. The vendored copies are pinned to the
    # revision the conv2d_xnnpack backend was written against; overwriting them
    # with whatever upstream HEAD happens to be can silently break that API.
    if [ ! -f "$DIR"/include/xnnpack/xnnpack.h ]; then
        cp -r XNNPACK/include/. include/xnnpack/
        find XNNPACK/build -name "pthreadpool.h" -exec cp {} include/xnnpack/ \;
        find XNNPACK/build -name "cpuinfo.h" -exec cp {} include/xnnpack/ \;
    else
        echo "Reusing existing pinned XNNPACK headers."
    fi
    # Static libraries: libXNNPACK.a and its bundled deps (pthreadpool, cpuinfo, microkernels).
    find XNNPACK/build -name "*.a" -exec cp {} include/xnnpack/lib/"$PLATFORM_DIR"/ \;
    echo "Cleaning up XNNPACK clone..."
    rm -rf XNNPACK/
fi

if [ -d "$DIR"/include/tiny_dnn/ ]; then
    echo "tiny-dnn already exists, skipping clone."
else
    echo "Cloning tiny-dnn..."
    cd "$DIR" && git clone "https://github.com/tiny-dnn/tiny-dnn.git"
    echo "Copying tiny-dnn headers..."
    cp -r tiny-dnn/tiny_dnn "$DIR"/include/
    echo "Patching tiny-dnn for g++-15/C++23..."
    # util.h defines to_xtensor/from_xtensor, which force an xt::xarray
    # instantiation from tiny-dnn's bundled xtensor that does not compile under
    # g++-15/C++23. They are unused by the conv/network path, so strip them.
    # Fail loudly if the pattern stops matching upstream: a silent no-op here
    # reports success and then buries you in xtensor template errors at compile
    # time, a long way from the actual cause.
    python3 - "$DIR"/include/tiny_dnn/util/util.h <<'PYEOF' || exit 1
import re, sys
path = sys.argv[1]
with open(path) as f:
    text = f.read()
text, n = re.subn(
    r"inline xt::xarray<float_t> to_xtensor.*?\ninline tensor_t from_xtensor.*?\n\}\n",
    "// to_xtensor/from_xtensor removed by setup.sh (xtensor/C++23 incompatibility)\n",
    text,
    flags=re.S,
)
if n != 1:
    sys.exit(f"ERROR: expected 1 to_xtensor/from_xtensor block in {path}, patched {n}")
with open(path, "w") as f:
    f.write(text)
PYEOF
    echo "Cleaning up tiny-dnn clone..."
    rm -rf tiny-dnn/
fi

echo "Setup complete."
