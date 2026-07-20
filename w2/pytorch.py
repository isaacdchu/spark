import torch
import torch.nn.functional as F
import time

input_warmup: torch.Tensor = torch.Tensor([x for x in range(16 * 3 * 256 * 256)]).reshape(16, 3, 256, 256)
kernel_warmup: torch.Tensor = torch.ones(3, 3, 5, 5)
for i in range(5):
    output_warmup = F.conv2d(input_warmup, kernel_warmup, padding='valid')

input_tensor: torch.Tensor = torch.Tensor([x for x in range(16 * 3 * 1920 * 1080)]).reshape(16, 3, 1920, 1080)
kernel_tensor: torch.Tensor = torch.ones(3, 3, 5, 5)
print(f"Device: {input_tensor.device}")
inp = input_tensor.cpu()
ker = kernel_tensor.cpu()
times = []
output_tensor: torch.Tensor = torch.zeros(16, 3, 1916, 1076)
with torch.no_grad():
    for _ in range(32):
        t0 = time.perf_counter()
        output_tensor = F.conv2d(inp, ker, stride=1, padding='valid', dilation=1)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
print(f"Conv2d elapsed: {torch.tensor(times).mean().item():.6f}ms")
