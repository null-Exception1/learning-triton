import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def relu_kernel(input, output, n_elements, BLOCK_SIZE: tl.constexpr):
    
    pid = tl.program_id(0)

    start = pid * BLOCK_SIZE 

    offsets = start + tl.arange(0,BLOCK_SIZE)

    mask = (offsets < n_elements)
    input_nums = tl.load(input+offsets, mask=mask)
    mask2 = input_nums > 0
    processed = tl.where(mask2, input_nums, 0.0)

    tl.store(output+offsets, processed, mask=mask)


def solve(input: torch.Tensor, N: int):
    output = torch.empty_like(input)
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    relu_kernel[grid](input, output, N, BLOCK_SIZE)
    return output

a = torch.randn(1024, device=DEVICE)
print(a.device)
output = solve(a,4)

print(a)
print(output)


