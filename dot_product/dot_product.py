import torch
import triton
import triton.language as tl

@triton.jit 
def dot_product_kernel(a_ptr,b_ptr,result_ptr,n,BLOCK_SIZE: tl.constexpr):

    pid = tl.program_id(0)
    offsets = pid*BLOCK_SIZE + tl.arange(0,BLOCK_SIZE)

    a_nums = tl.load(a_ptr + offsets,mask = offsets < n, other=0.0)
    b_nums = tl.load(b_ptr + offsets,mask = offsets < n, other=0.0)
    
    tl.atomic_add(result_ptr, tl.sum(a_nums*b_nums))

# a, b, result are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, result: torch.Tensor, n: int):
    
    BLOCK_SIZE = 64

    grid = (triton.cdiv(n,BLOCK_SIZE),)

    dot_product_kernel[grid](a,b,result,n,BLOCK_SIZE)
