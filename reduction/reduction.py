import torch
import triton
import triton.language as tl

@triton.jit
def reduction(input,output,N,BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)

    block_offsets = pid*BLOCK_SIZE + tl.arange(0,BLOCK_SIZE)

    block = tl.load(input + block_offsets,mask=block_offsets < N)

    tl.atomic_add(output,tl.sum(block))

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    BLOCK_SIZE = 64
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    reduction[grid](
        input,output,N,BLOCK_SIZE=BLOCK_SIZE
    )
