#!/usr/bin/env bash
#
# run_sweep.sh -- two-level benchmark sweep launcher for the scone benchmark.
#
# The benchmark sweeps over two kinds of values:
#   1. BLOCK SIZES  -- compile-time constexpr register-tile sizes. Each distinct
#      block configuration requires RECOMPILING the benchmark binary with a
#      different set of -D compile defines.
#   2. THREAD COUNT -- a runtime value read from the SCONE_NUM_THREADS env var.
#      Changing it does NOT require a recompile; just relaunch the same binary.
#
# So the sweep is: for each block config -> compile once -> for each thread
# count -> run once. Every run appends its rows into ONE master CSV.
#
# This script is dashboard-driven: the sweep space and problem shape come from
# env vars (with defaults that preserve the old hardcoded behavior), it writes a
# key=value STATUS file the dashboard polls for progress, and it tee's the
# binary's stdout into a LOG file the dashboard can tail.
#
# Invoked from the repo root as:  bash benchmark/run_sweep.sh
# (the `make benchmark-sweep` target does exactly this).

set -euo pipefail

# ----------------------------------------------------------------------------
# Sweep space -- defaults (used only when the matching env var is unset)
# ----------------------------------------------------------------------------

# Each entry is one block configuration, given as two space-separated ints
# interpreted in this exact order:
#
#     SCONE_BLOCK_RSCK_Q  SCONE_BLOCK_RSCK_KV
#
#   - SCONE_BLOCK_RSCK_Q / SCONE_BLOCK_RSCK_KV -> conv2d_block_rsck tile (defaults 6, 2)
#
# To add more configurations here, add more "Q KV" strings to the array.
# At runtime the dashboard overrides this via SCONE_BLOCK_CONFIGS (see below).
BLOCK_CONFIGS=(
  "6 2"    # default configuration
  "8 2"    # wider Q tiles
  "6 4"    # wider KV tiles
)

# Runtime thread counts to sweep for each compiled binary (no recompile needed).
# Overridden at runtime by SCONE_THREAD_COUNTS.
THREAD_COUNTS=(1 2 4 8)

# ----------------------------------------------------------------------------
# Env-driven overrides of the sweep space
# ----------------------------------------------------------------------------

# SCONE_BLOCK_CONFIGS: configs separated by ';', each two space-separated ints
# "RSCK_Q RSCK_KV". Example: "6 2;8 2".
if [[ -n "${SCONE_BLOCK_CONFIGS:-}" ]]; then
  BLOCK_CONFIGS=()
  IFS=';' read -r -a _raw_block_configs <<< "${SCONE_BLOCK_CONFIGS}"
  for _cfg in "${_raw_block_configs[@]}"; do
    # Trim leading and trailing whitespace.
    _cfg="${_cfg#"${_cfg%%[![:space:]]*}"}"
    _cfg="${_cfg%"${_cfg##*[![:space:]]}"}"
    [[ -z "${_cfg}" ]] && continue   # ignore empty entries (e.g. trailing ';')
    BLOCK_CONFIGS+=("${_cfg}")
  done
fi

# SCONE_THREAD_COUNTS: comma- OR space-separated ints. Example: "1,2,4" or "1 2 4".
if [[ -n "${SCONE_THREAD_COUNTS:-}" ]]; then
  read -r -a THREAD_COUNTS <<< "${SCONE_THREAD_COUNTS//,/ }"
fi

if [[ "${#BLOCK_CONFIGS[@]}" -eq 0 ]]; then
  echo "ERROR: no block configs to sweep (SCONE_BLOCK_CONFIGS parsed empty)." >&2
  exit 1
fi
if [[ "${#THREAD_COUNTS[@]}" -eq 0 ]]; then
  echo "ERROR: no thread counts to sweep (SCONE_THREAD_COUNTS parsed empty)." >&2
  exit 1
fi

# ----------------------------------------------------------------------------
# Paths and build flags (mirrors the Makefile `$(BUILD)/benchmark` rule)
# ----------------------------------------------------------------------------

# Resolve the repo root from this script's location so it works regardless of
# the caller's current directory, then operate from there.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CXX="${CXX:-g++-15}"

# OpenBLAS discovery: Homebrew on macOS, pkg-config (falling back to a bare
# -lopenblas, which the multiarch-aware Linux linker/GCC search paths pick up)
# everywhere else.
if [[ "$(uname -s)" == "Darwin" ]]; then
  OPENBLAS_DIR="$(brew --prefix openblas)"
  OPENBLAS_INCLUDE=(-I"${OPENBLAS_DIR}/include")
  OPENBLAS_LIB=(-L"${OPENBLAS_DIR}/lib" -lopenblas)
else
  if pkg-config --exists openblas 2>/dev/null; then
    read -r -a OPENBLAS_INCLUDE <<< "$(pkg-config --cflags openblas)"
    read -r -a OPENBLAS_LIB <<< "$(pkg-config --libs openblas)"
  else
    OPENBLAS_INCLUDE=()
    OPENBLAS_LIB=(-lopenblas)
  fi
fi

# These mirror CXXFLAGS / INCLUDE / LDFLAGS from the Makefile exactly, so the
# sweep binary is built with the same flags as `make benchmark`.
CXXFLAGS=(-std=c++23 -Wall -Werror -Wextra -Wno-unused -Wno-psabi -O3 -march=native -fopenmp)
INCLUDE=(-Iinclude/taskflow -Iinclude/xsimd -Iinclude/xnnpack "${OPENBLAS_INCLUDE[@]}")
# XNNPACK static libs are staged per-platform by setup.sh (Mach-O on macOS,
# ELF on Linux can't share a directory), mirroring the Makefile.
PLATFORM_DIR="$(uname -s | tr '[:upper:]' '[:lower:]')-$(uname -m)"
XNNPACK_LIBS=(include/xnnpack/lib/"${PLATFORM_DIR}"/*.a)
if [[ "$(uname -s)" == "Darwin" ]]; then
  LDFLAGS=("${OPENBLAS_LIB[@]}" "${XNNPACK_LIBS[@]}")
else
  # GNU ld needs --start-group/--end-group for the circular refs between
  # libXNNPACK.a/libcpuinfo.a/libpthreadpool.a/the microkernel libs (see Makefile).
  LDFLAGS=("${OPENBLAS_LIB[@]}" -Wl,--start-group "${XNNPACK_LIBS[@]}" -Wl,--end-group)
fi

SOURCE="src"
BUILD="build"
BENCH_SRC="benchmark/benchmark.cpp"
BENCH_BIN="${BUILD}/benchmark_sweep"

# Master CSV that all runs aggregate into. Overridable via env.
MASTER_CSV="${SCONE_BENCH_OUTPUT:-benchmark/benchmark_results.csv}"

# Progress + log files the dashboard polls/tails. Overridable via env.
STATUS_FILE="${SCONE_BENCH_STATUS:-benchmark/.sweep_status}"
LOG_FILE="${SCONE_BENCH_LOG:-benchmark/.sweep_log}"

# Allow overriding the command that is run for each thread count. This exists
# so the sweep can be dry-run validated against a mock binary. Defaults to the
# real compiled benchmark binary.
BENCH_CMD="${SCONE_BENCH_CMD:-./${BENCH_BIN}}"

mkdir -p "${BUILD}"
mkdir -p "$(dirname "${STATUS_FILE}")" "$(dirname "${LOG_FILE}")" "$(dirname "${MASTER_CSV}")"

# NOTE on passthrough: the problem-shape / backend env vars the C++ binary reads
# (SCONE_BENCH_BACKENDS, SCONE_BENCH_BATCH_SIZES, SCONE_BENCH_INPUT_SIZES,
# SCONE_BENCH_KERNEL_SIZES, SCONE_BENCH_INPUT_CHANNELS, SCONE_BENCH_OUTPUT_CHANNELS,
# SCONE_BENCH_STRIDES, SCONE_BENCH_PADDINGS) are NOT consumed here. We only set
# SCONE_NUM_THREADS / SCONE_BENCH_OUTPUT / SCONE_BENCH_APPEND inline when invoking
# the binary; every other var in this process's environment is inherited by the
# child unchanged, so those pass straight through to the binary.

# ----------------------------------------------------------------------------
# STATUS file: rewritten (truncated) on every update.
# ----------------------------------------------------------------------------
STATE="starting"
TOTAL=$(( ${#BLOCK_CONFIGS[@]} * ${#THREAD_COUNTS[@]} ))
COMPLETED=0
CONFIG_INDEX=0
CONFIG_TOTAL=${#BLOCK_CONFIGS[@]}
BLOCK_CONFIG=""
THREADS=""
MESSAGE=""
ERRORMSG=""

write_status() {
  {
    echo "state=${STATE}"
    echo "pid=$$"
    echo "total=${TOTAL}"
    echo "completed=${COMPLETED}"
    echo "config_index=${CONFIG_INDEX}"
    echo "config_total=${CONFIG_TOTAL}"
    echo "block_config=${BLOCK_CONFIG}"
    echo "threads=${THREADS}"
    echo "message=${MESSAGE}"
    if [[ "${STATE}" == "error" ]]; then
      echo "error=${ERRORMSG}"
    fi
  } > "${STATUS_FILE}"
}

fail() {
  # Mark the sweep as errored, persist status, and exit non-zero.
  STATE="error"
  ERRORMSG="$1"
  MESSAGE="$1"
  write_status
  echo "ERROR: $1" >&2
  exit 1
}

# Safety net: any unexpected non-zero command (that we did not handle) records
# an error state before the script aborts under set -e.
trap 'rc=$?; if [[ "${STATE}" != "error" ]]; then STATE="error"; ERRORMSG="unexpected failure (exit ${rc})"; MESSAGE="${ERRORMSG}"; write_status; fi' ERR

# ----------------------------------------------------------------------------
# Aggregation + log setup
# ----------------------------------------------------------------------------
# Master CSV protocol given the binary's contract:
#   - FIRST run of the whole sweep: SCONE_BENCH_APPEND unset -> binary truncates
#     the file and writes the header.
#   - ALL subsequent runs: SCONE_BENCH_APPEND=1 -> binary appends rows only.
# Remove any stale master file up front so the first run truly starts clean.
rm -f "${MASTER_CSV}"
FIRST_RUN=1

# Truncate the log at sweep start.
: > "${LOG_FILE}"

MESSAGE="initializing sweep"
write_status

echo "=================================================================="
echo "scone benchmark sweep"
echo "  block configs : ${#BLOCK_CONFIGS[@]}"
echo "  thread counts : ${THREAD_COUNTS[*]}"
echo "  total runs    : ${TOTAL}"
echo "  master csv    : ${MASTER_CSV}"
echo "  status file   : ${STATUS_FILE}"
echo "  log file      : ${LOG_FILE}"
echo "  bench command : ${BENCH_CMD}"
echo "=================================================================="

# ----------------------------------------------------------------------------
# Sweep
# ----------------------------------------------------------------------------
for config in "${BLOCK_CONFIGS[@]}"; do
  CONFIG_INDEX=$((CONFIG_INDEX + 1))
  BLOCK_CONFIG="${config}"

  # Split the "Q KV" tuple into its two values.
  read -r RSCK_Q RSCK_KV <<< "${config}"

  DEFINES=(
    -DSCONE_BLOCK_RSCK_Q="${RSCK_Q}"
    -DSCONE_BLOCK_RSCK_KV="${RSCK_KV}"
  )

  echo ""
  echo "------------------------------------------------------------------"
  echo "[block config ${CONFIG_INDEX}/${CONFIG_TOTAL}] RSCK_Q=${RSCK_Q} RSCK_KV=${RSCK_KV}"
  echo "------------------------------------------------------------------"

  # Only actually compile when running the real binary. When a mock command is
  # supplied via SCONE_BENCH_CMD (dry-run validation), skip compilation.
  if [[ "${BENCH_CMD}" == "./${BENCH_BIN}" ]]; then
    STATE="compiling"
    THREADS=""
    MESSAGE="compiling block config ${CONFIG_INDEX}/${CONFIG_TOTAL}: ${config}"
    write_status

    COMPILE_CMD=("${CXX}" "${CXXFLAGS[@]}" "${DEFINES[@]}" "${INCLUDE[@]}" -I"${SOURCE}" "${BENCH_SRC}" -o "${BENCH_BIN}" "${LDFLAGS[@]}")
    echo "compiling:"
    echo "  ${COMPILE_CMD[*]}"
    if ! "${COMPILE_CMD[@]}"; then
      fail "compilation failed for block config '${config}'"
    fi
  else
    echo "(mock command in use; skipping compilation)"
  fi

  # Inner loop: run once per thread count, no recompile.
  for threads in "${THREAD_COUNTS[@]}"; do
    THREADS="${threads}"
    STATE="running"
    MESSAGE="running block config ${CONFIG_INDEX}/${CONFIG_TOTAL} (${config}) with ${threads} thread(s)"
    write_status

    echo ""
    echo ">>> running: block config ${CONFIG_INDEX}/${CONFIG_TOTAL} (${config}) | SCONE_NUM_THREADS=${threads}"

    # Determine append behavior: first run of the whole sweep writes the header,
    # all later runs append. We keep setting only these three vars inline; every
    # other SCONE_BENCH_* var in the environment is inherited by the binary.
    if [[ "${FIRST_RUN}" -eq 1 ]]; then
      APPEND_FLAG=""
    else
      APPEND_FLAG="1"
    fi

    # Run the binary, tee its stdout+stderr into the log while still showing it
    # in the terminal. Capture the binary's own exit status via PIPESTATUS[0]
    # (tee's success must not mask a failing binary). set +e so the pipeline
    # failing does not abort before we can inspect PIPESTATUS.
    set +e
    if [[ -z "${APPEND_FLAG}" ]]; then
      SCONE_NUM_THREADS="${threads}" \
      SCONE_BENCH_OUTPUT="${MASTER_CSV}" \
        ${BENCH_CMD} 2>&1 | tee -a "${LOG_FILE}"
    else
      SCONE_NUM_THREADS="${threads}" \
      SCONE_BENCH_OUTPUT="${MASTER_CSV}" \
      SCONE_BENCH_APPEND="${APPEND_FLAG}" \
        ${BENCH_CMD} 2>&1 | tee -a "${LOG_FILE}"
    fi
    run_rc=${PIPESTATUS[0]}
    set -e

    if [[ "${run_rc}" -ne 0 ]]; then
      fail "benchmark run failed (exit ${run_rc}) for block config '${config}', threads=${threads}"
    fi

    FIRST_RUN=0
    COMPLETED=$((COMPLETED + 1))
    MESSAGE="completed ${COMPLETED}/${TOTAL}"
    write_status
  done
done

# ----------------------------------------------------------------------------
# Verify aggregation: the master CSV must contain exactly one header line.
# ----------------------------------------------------------------------------
if [[ ! -f "${MASTER_CSV}" ]]; then
  fail "master CSV '${MASTER_CSV}' was not created"
fi

HEADER_LINE="$(head -n 1 "${MASTER_CSV}")"
HEADER_COUNT="$(grep -Fxc "${HEADER_LINE}" "${MASTER_CSV}" || true)"
TOTAL_LINES="$(wc -l < "${MASTER_CSV}" | tr -d ' ')"

echo ""
echo "=================================================================="
echo "sweep complete"
echo "  master csv    : ${MASTER_CSV}"
echo "  total lines   : ${TOTAL_LINES}"
echo "  header line   : ${HEADER_LINE}"
echo "  header count  : ${HEADER_COUNT}"
echo "=================================================================="

if [[ "${HEADER_COUNT}" -ne 1 ]]; then
  fail "expected exactly one header line in '${MASTER_CSV}', found ${HEADER_COUNT}"
fi

STATE="done"
THREADS=""
MESSAGE="sweep complete: ${COMPLETED}/${TOTAL} runs, master CSV has one header"
write_status

echo "OK: master CSV has exactly one header and aggregated all runs."
