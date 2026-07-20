import torch
import warnings

def conv2d(input_tensor: torch.Tensor, kernel_tensor: torch.Tensor) -> torch.Tensor:
    """
    FFT-based convolution (valid equivalent to `torch.nn.functional.conv2d` with
    symmetric padding). Returns output shaped (B, C_out, H, W) matching input spatial
    size by extracting the central region after full convolution.
    """
    B, C_in, H, W = input_tensor.shape
    C_out, C_in_k, kH, kW = kernel_tensor.shape
    if C_in != C_in_k:
        raise ValueError("Input and kernel channel dimensions do not match")

    # full convolution size
    out_H = H + kH - 1
    out_W = W + kW - 1

    # ensure everything is on the same device/dtype, flip kernel to match conv2d
    device = input_tensor.device
    dtype = input_tensor.dtype
    kernel_flipped = kernel_tensor.flip(-2, -1).to(device=device, dtype=dtype)

    # pad input and kernel to full convolution size (left,right,top,bottom)
    pad_input = (0, out_W - W, 0, out_H - H)
    pad_kernel = (0, out_W - kW, 0, out_H - kH)
    input_padded = torch.nn.functional.pad(input_tensor, pad_input).to(device=device, dtype=dtype).contiguous()
    kernel_padded = torch.nn.functional.pad(kernel_flipped, pad_kernel).to(device=device, dtype=dtype).contiguous()

    input_fft: torch.Tensor = torch.fft.rfft2(input_padded)     # shape: [B, C_in, Hf, Wfr]
    kernel_fft: torch.Tensor = torch.fft.rfft2(kernel_padded)   # shape: [C_out, C_in, Hf, Wfr]

    # Multiply and sum over input channels with proper broadcasting:
    # make shapes [B, 1, C_in, Hf, Wfr] and [1, C_out, C_in, Hf, Wfr]
    input_fft = input_fft.unsqueeze(1)
    kernel_fft = kernel_fft.unsqueeze(0)
    output_fft: torch.Tensor = (input_fft * kernel_fft).sum(dim=2)  # sum over C_in -> [B, C_out, Hf, Wfr]

    # inverse FFT to spatial domain (size = full conv size)
    output = torch.fft.irfft2(output_fft, s=(out_H, out_W), dim=(-2, -1))

    # extract central region so result matches conv2d with padding=k//2 (same spatial size)
    # For even and odd kernel sizes, use floor division to choose the same
    # centering that `torch.nn.functional.conv2d(..., padding=kernel_size//2)` uses.
    start_h = kH // 2
    start_w = kW // 2
    return output[..., start_h:start_h + H, start_w:start_w + W]
