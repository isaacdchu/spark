import torch

def conv2d(input_tensor: torch.Tensor, kernel_tensor: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.conv2d(input_tensor, kernel_tensor, padding='same')
