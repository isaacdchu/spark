import torch

def conv2d(input_tensor: torch.Tensor, kernel_tensor: torch.Tensor) -> torch.Tensor:
    # input is (B, C_in, H, W), kernel is (C_out, C_in, kH, kW)
    # manual convolution with padding='same' (zero-padding of kernel_size//2 on each side)
    B, C_in, H, W = input_tensor.shape
    C_out, C_in_k, kH, kW = kernel_tensor.shape
    if C_in != C_in_k:
        raise ValueError("Input and kernel channel dimensions do not match")

    padding = kH // 2
    input_padded = torch.nn.functional.pad(input_tensor, (padding, padding, padding, padding))

    # output shape should be (B, C_out, H, W)
    output = torch.zeros((B, C_out, H, W), dtype=input_tensor.dtype, device=input_tensor.device)

    # Vectorized per-offset accumulation: for each (i,j) take the HxW patch and
    # accumulate using an einsum to multiply the input channels with kernel channels
    for i in range(kH):
        for j in range(kW):
            # slice of shape [B, C_in, H, W]
            inp_patch = input_padded[:, :, i:i + H, j:j + W]
            # kernel slice [C_out, C_in]
            kern_slice = kernel_tensor[:, :, i, j]
            # accumulate: out[b,o,h,w] += sum_c inp_patch[b,c,h,w] * kern_slice[o,c]
            output += torch.einsum('bchw,oc->bohw', inp_patch, kern_slice)

    return output
