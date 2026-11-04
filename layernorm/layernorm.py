import torch
import triton
import triton.language as tl

@triton.jit
def layernorm_kernel( # yea i went for a row by row approach ik
    input_ptr, 
    weight_ptr, 
    bias_ptr, 
    mean_ptr, 
    variance_ptr, 
    N, 
    C, 
    eps, 
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    if row_idx >= N:
        return

    row_start_ptr = input_ptr + row_idx * C

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < C

    x = tl.load(row_start_ptr + cols, mask=mask, other=0.0)

    mean = tl.sum(x, axis=0) / C
    tl.store(mean_ptr + row_idx, mean)

    x_centered = tl.where(mask, x - mean, 0.0)
    variance = tl.sum(x_centered * x_centered, axis=0) / C
    tl.store(variance_ptr + row_idx, variance)

    w = tl.load(weight_ptr + cols, mask=mask, other=1.0)
    b = tl.load(bias_ptr + cols, mask=mask, other=0.0)

    normalized_x = x_centered / tl.sqrt(variance + eps)
    output_x = normalized_x * w + b

    tl.store(row_start_ptr + cols, output_x, mask=mask)

def solve(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    output: torch.Tensor,
    N: int,
    C: int,
    eps: float,
):
    mean = torch.zeros(N, device=input.device, dtype=torch.float32)
    variance = torch.zeros(N, device=input.device, dtype=torch.float32)

    BLOCK_SIZE = triton.next_power_of_2(C)

    grid = (N,)

    layernorm_kernel[grid](
        input, 
        weight, 
        bias, 
        mean, 
        variance, 
        N, 
        C, 
        eps, 
        BLOCK_SIZE=BLOCK_SIZE
    )

    return input, mean, variance