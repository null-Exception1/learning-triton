import torch
import triton
import triton.language as tl

@triton.jit
def max_x(x, maximum_x, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    x_offsets = pid*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    x_nums = tl.load(x + x_offsets,mask=x_offsets < N, other=-float('inf'))
    
    max_block = tl.max(x_nums)

    tl.atomic_max(maximum_x,max_block)

@triton.jit
def compute_denominator(x_ptr, exp_ptr, max_x_ptr, denominator_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    x_nums = tl.load(x_ptr + offsets, mask=offsets < N, other=-float('inf'))
    maximum = tl.load(max_x_ptr)

    deno_block = tl.exp(x_nums-maximum)

    tl.store(exp_ptr+offsets,deno_block,mask=offsets < N)

    #mask = offsets > N
    #tl.where(offsets > N, deno_block, 0.0)
    
    sum_deno_block = tl.sum(deno_block)

    tl.atomic_add(denominator_ptr, sum_deno_block)

@triton.jit
def softmax(exp_ptr, output_ptr, denominator_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    exp_nums = tl.load(exp_ptr + offsets)
    deno = tl.load(denominator_ptr)

    tl.store(output_ptr+offsets, exp_nums/deno, mask=offsets < N)

def solve(x: torch.Tensor, output:torch.Tensor, N: int):
    exp = torch.empty_like(x)
    denominator = torch.tensor(0.0,device="cuda")
    maximum_x = torch.tensor(-float('inf'),device="cuda")
    BLOCK_SIZE = 256
    
    grid = (triton.cdiv(N,BLOCK_SIZE),)
    
    max_x[grid](
        x, maximum_x, N, BLOCK_SIZE=BLOCK_SIZE
    )
    #print(maximum_x)
    compute_denominator[grid](
        x, exp, maximum_x, denominator, N, BLOCK_SIZE=BLOCK_SIZE
    )
    #print(x,exp,denominator)
    softmax[grid](
        exp, output, denominator, N, BLOCK_SIZE=BLOCK_SIZE
    )

    return output
