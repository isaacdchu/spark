import torch

def conv2d(input_tensor: torch.Tensor, kernel_tensor: torch.Tensor) -> torch.Tensor:
    # im2col
    input_matrix = torch.nn.functional.unfold(input_tensor, kernel_tensor.shape[2:], padding=kernel_tensor.shape[2]//2)
    # reshape kernel tensor to (out_channels, in_channels * kernel_height * kernel_width)
    kernel_matrix = kernel_tensor.reshape(kernel_tensor.shape[0], -1)
    return torch.matmul(kernel_matrix, input_matrix)
