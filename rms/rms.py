import torch
import triton
import triton.language as tl

@triton.jit
def sq(input_ptr, input_sq_ptr, N, BLOCK_SIZE: tl.constexpr):
    
    pid = tl.program_id(0)

    offsets = pid*BLOCK_SIZE + tl.arange(0,BLOCK_SIZE)

    block = tl.load(input_ptr + offsets,mask=offsets < N,other=0.0)

    block_sq = tl.sum(block*block)

    tl.atomic_add(input_sq_ptr,block_sq, sem="relaxed")

@triton.jit
def normalized_output(input_ptr, output_ptr, x_sq_ptr, N, gamma, beta,eps, BLOCK_SIZE: tl.constexpr):
    
    pid = tl.program_id(0)

    offsets = pid*BLOCK_SIZE + tl.arange(0,BLOCK_SIZE)

    mask = offsets<N

    block = tl.load(input_ptr + offsets,mask=offsets<N,other=0.0)

    x_sq = tl.load(x_sq_ptr)

    tl.store(output_ptr + offsets, gamma*(block*tl.rsqrt((x_sq/N) + eps)) + beta, mask=mask)
    
# input, output are tensors on the GPU
def solve(input: torch.Tensor, gamma: float, beta: float, output: torch.Tensor, N: int, eps: float):
    BLOCK_SIZE = 2048
    grid = (triton.cdiv(N,BLOCK_SIZE),)

    x_sq = input.new_zeros((1, ))

    sq[grid](input,x_sq,N,BLOCK_SIZE)


    normalized_output[grid](input,output,x_sq,N,gamma,beta,eps,BLOCK_SIZE)

