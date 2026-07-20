import time
import os
import sys
import contextlib
import torch
import torch.nn.functional as F
from torch import Tensor

# def get_current_rss_kib() -> float:
#     process = psutil.Process(os.getpid())
#     return process.memory_info().rss / 1024
class BenchmarkResult:
    def __init__(self, input_size: int, kernel_size: int, total_ms: float):
        self.input_size = input_size
        self.kernel_size = kernel_size
        self.total_ms = total_ms

results: list[BenchmarkResult] = []

def save_csv(path: str, results: list[BenchmarkResult]):
    with open(path, 'a', encoding='utf-8') as f:
        for r in results:
            f.write(f"PYTORCH,{r.input_size},{r.kernel_size},{r.total_ms}\n")

@contextlib.contextmanager
def suppress_cpp_stderr():
    # Open /dev/null
    with open(os.devnull, "w") as devnull:
        # Save a duplicate copy of the original stderr file descriptor
        old_stderr_fd = os.dup(sys.stderr.fileno())
        try:
            # Replace stderr file descriptor with /dev/null's descriptor
            os.dup2(devnull.fileno(), sys.stderr.fileno())
            yield
        finally:
            # Restore original stderr
            os.dup2(old_stderr_fd, sys.stderr.fileno())
            os.close(old_stderr_fd)

def warmup() -> None:
    for _ in range(10):
        inp_warmup = Tensor([x for x in range(16 * 64 * 64 * 64)]).reshape(16, 64, 64, 64).permute(0, 3, 1, 2).contiguous()
        ker_warmup = torch.ones(64, 64, 3, 3)
        _ = F.conv2d(inp_warmup, ker_warmup, padding=0)

def benchmark(input_shape, kernel_shape) -> tuple[float, float, float]:
    mean_duration: float = 0
    mean_peak_memory: float = 0
    mean_gflops_per_sec: float = 0
    for _ in range(10):
        inp = Tensor(
            [x for x in range(input_shape[0] * input_shape[1] * input_shape[2] * input_shape[3])]
        ).reshape(input_shape).permute(0, 3, 1, 2).contiguous()
        ker = torch.ones(kernel_shape).permute(0, 3, 1, 2).contiguous()
        with suppress_cpp_stderr():
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                profile_memory=True,
                with_flops=True,
                record_shapes=False
            ) as prof:
                # Target operation to isolate
                start = time.perf_counter_ns()
                _ = F.conv2d(inp, ker, padding=1)
                end = time.perf_counter_ns()
        events = prof.key_averages()
        raw_events = prof.events()
        if not raw_events:
            raise RuntimeError("No events recorded by the profiler")
        running_total = 0
        peak_memory_bytes = 0
        for event in sorted(raw_events, key=lambda x: x.time_range.start):
            running_total += event.cpu_memory_usage
            if running_total > peak_memory_bytes:
                peak_memory_bytes = running_total
        cpu_memory_mib = peak_memory_bytes / (1024 * 1024)
        mean_peak_memory += cpu_memory_mib
        flops: int = sum([event.flops for event in events])
        mean_duration += (end - start)
        mean_gflops_per_sec += flops # Total GFLOPs
    mean_duration /= 10
    mean_peak_memory /= 10
    mean_gflops_per_sec /= mean_duration  # Average GFLOPs per second
    mean_gflops_per_sec /= 10
    results.append(BenchmarkResult(input_shape[1], kernel_shape[1], mean_duration / 1_000_000))
    return (mean_duration, mean_peak_memory, mean_gflops_per_sec)

def main() -> None:
    warmup()
    input_sizes = (16, 32, 64, 128)
    for input_size in input_sizes:
        kernel_sizes = sorted(list(set([1, 3, 5, input_size // 4, input_size // 2])))
        for kernel_size in kernel_sizes:
            input_shape = (16, input_size, input_size, 3)
            kernel_shape = (3, kernel_size, kernel_size, 3)
            mean_duration, mean_peak_memory, mean_gflops_per_sec = benchmark(input_shape, kernel_shape)
            print("Input: {} {} {} {}, Kernel: {} {} {} {}, Conv2d time: {:.3f} ms, Peak memory usage: {:.3f} MiB, Mean GFLOPs/s: {:.3f}".format(
                input_shape[0], input_shape[1], input_shape[2], input_shape[3],
                kernel_shape[0], kernel_shape[1], kernel_shape[2], kernel_shape[3],
                (mean_duration / 1_000_000), mean_peak_memory, mean_gflops_per_sec
            ))

    save_csv("benchmark_results.csv", results)

if __name__ == "__main__":
    main()
