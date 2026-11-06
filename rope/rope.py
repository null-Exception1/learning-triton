import torch
import triton
import triton.language as tl
import pdb

@triton.jit
def rope(Q_ptr,cos_ptr,sin_ptr,output_ptr,M,D, BLOCK_SIZE: tl.constexpr):
    
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)
    offsets =  col_idx*BLOCK_SIZE + tl.arange(0,BLOCK_SIZE)
    mask=offsets<D

    Q_cos = tl.load(row_idx*D + Q_ptr + offsets,mask=mask)
    cos = tl.load(row_idx*D + cos_ptr + offsets,mask=mask)
    sin = tl.load(row_idx*D + sin_ptr + offsets,mask=mask)

    output_cos = Q_cos * cos 

    first_half = (D//2) + col_idx*BLOCK_SIZE + tl.arange(0,BLOCK_SIZE) 
    sin_mask_first = first_half < D
    first_half = tl.where(sin_mask_first, first_half, 0)

    second_half = col_idx*BLOCK_SIZE + tl.arange(0,BLOCK_SIZE) - (D//2)
    sin_mask_second = (second_half >= 0) & (second_half < (D//2))
    second_half = tl.where(sin_mask_second, second_half, 0)

    Q_sin = tl.load(row_idx*D + Q_ptr + first_half + second_half, mask = first_half + second_half < D, other=0.0) 
    Q_sin = tl.where(sin_mask_first,-Q_sin,Q_sin)

    output_sin = Q_sin * sin
    full_output = output_cos + output_sin

    tl.store(row_idx*D + output_ptr + offsets, full_output, mask=mask)

    #pdb.set_trace()

# Q, cos, sin, output are tensors on the GPU
def solve(
    Q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, output: torch.Tensor, M: int, D: int
):  
    BLOCK_SIZE = 1024
    grid = (M,triton.cdiv(D,BLOCK_SIZE))
    
    rope[grid](Q,cos,sin,output,M,D, BLOCK_SIZE)

Q   = torch.tensor([[1.0, 2.0, 3.0, 4.0],
               [1.0, 1.0, 1.0, 1.0]],device='cuda')
Cos = torch.tensor([[1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0]],device='cuda')
Sin = torch.tensor([[0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0]],device='cuda')

Output = torch.zeros(2,4,device='cuda')

solve(Q, Cos, Sin, Output, 2, 4)

print(Output)