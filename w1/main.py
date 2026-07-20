import pathlib
import time
from typing import Callable
import torch
import polars as pl
from benchmarks.direct import conv2d as conv_direct
from benchmarks.fft import conv2d as conv_fft
from benchmarks.matrix import conv2d as conv_matrix
from benchmarks.reference import conv2d as conv_reference

def benchmark(
    conv_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    name: str,
    device: torch.device,
    sync: Callable[[], None]
) -> pl.DataFrame:
    torch.manual_seed(0)  # For reproducibility
    num_runs = 10
    input_channel_size = 3
    kernel_channel_size = 3
    batch_size = 16
    input_sizes = [16, 32, 64]
    df = pl.DataFrame(schema={
            "Name": pl.Utf8,
            "Input Size": pl.Int64,
            "Kernel Size": pl.Int64,
            "Median Elapsed Time": pl.Float64,
            "Max Norm of Difference": pl.Float64
    })
    for input_size in input_sizes:
        kernel_sizes = sorted(list(set([1, 3, 5, input_size // 4 - 1, input_size // 2 - 1, input_size - 1])))
        for kernel_size in kernel_sizes:
            elapsed_time = []
            input_tensor = torch.randn(batch_size, input_channel_size, input_size, input_size).to(device)
            kernel_tensor = torch.randn(kernel_channel_size, input_channel_size, kernel_size, kernel_size).to(device)
            ref_input_tensor = input_tensor.clone()
            ref_kernel_tensor = kernel_tensor.clone()
            start_time = time.perf_counter()
            output_tensor = conv_func(input_tensor, kernel_tensor)
            sync()
            end_time = time.perf_counter()
            elapsed_time.append(end_time - start_time)
            for _ in range(num_runs - 1):
                start_time = time.perf_counter()
                output_tensor = conv_func(input_tensor, kernel_tensor)
                end_time = time.perf_counter()
                sync()
                elapsed_time.append(end_time - start_time)
            median_time = sorted(elapsed_time)[len(elapsed_time) // 2]
            median_time *= 1000  # convert to milliseconds
            # print(f"{input_size} {kernel_size} {elapsed_time:.6f}")
            reference_tensor = torch.nn.functional.conv2d(ref_input_tensor, ref_kernel_tensor, padding='same').reshape_as(output_tensor)
            norm_diff = torch.max(torch.linalg.matrix_norm(output_tensor - reference_tensor)).item()
            # print(f"{input_size} {kernel_size} {elapsed_time:.6f} {norm_diff:.6f}")
            df = df.vstack(pl.DataFrame(
                {
                    "Name": [name],
                    "Input Size": [input_size],
                    "Kernel Size": [kernel_size],
                    "Median Elapsed Time": [median_time],
                    "Max Norm of Difference": [norm_diff]
                }, schema=df.schema, orient="row"
            ))
    # pathlib.Path("results").mkdir(exist_ok=True)
    df.write_csv(pathlib.Path(f"results/{name.strip().lower().replace(' ', '_')}.csv"))
    return df

def main():
    sync = lambda: None
    if torch.cuda.is_available():
        device = torch.device("cuda")
        sync = lambda: torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        sync = lambda: torch.mps.synchronize()
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    df = pl.DataFrame(schema={
            "Name": pl.Utf8,
            "Input Size": pl.Int64,
            "Kernel Size": pl.Int64,
            "Median Elapsed Time": pl.Float64,
            "Max Norm of Difference": pl.Float64
    })
    print("Benchmarking direct convolution...")
    df = df.vstack(benchmark(conv_direct, "Direct", device, sync))
    print("Benchmarking matrix-based convolution...")
    df = df.vstack(benchmark(conv_matrix, "Matrix", device, sync))
    print("Benchmarking FFT-based convolution...")
    df = df.vstack(benchmark(conv_fft, "FFT", device, sync))
    print("Benchmarking reference convolution...")
    df = df.vstack(benchmark(conv_reference, "Reference", device, sync))
    pathlib.Path("results").mkdir(exist_ok=True)
    df.write_csv(pathlib.Path("results/benchmark_results.csv"))
    print("Benchmarks completed. Results saved to the 'results' directory.")

if __name__ == "__main__":
    main()
