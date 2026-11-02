import torch
import triton
import triton.language as tl


@triton.jit
def matrix_transpose_kernel(input, output, rows, cols, stride_ir, stride_ic, stride_or, stride_oc, BLOCK_SIZE: tl.constexpr):
    
    input_ptr = input.to(tl.pointer_type(tl.float32))
    output_ptr = output.to(tl.pointer_type(tl.float32))

    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    offs_r = pid_row*BLOCK_SIZE + tl.arange(0,BLOCK_SIZE)
    offs_c = pid_col*BLOCK_SIZE + tl.arange(0,BLOCK_SIZE)

    global_input_ptrs = input_ptr + offs_r[:,None]*stride_ir + offs_c[None,:]*stride_ic

    mask = (offs_r[:,None] < rows) & (offs_c[None,:] < cols)

    input_values = tl.load(global_input_ptrs,mask=mask,other=0)

    transposed = tl.trans(input_values)

    global_output_ptrs = output_ptr + offs_c[:,None]*rows + offs_r[None,:]

    tl.store(global_output_ptrs, transposed, mask=mask.T)
    

# input, output are tensors on the GPU
def solve(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
    stride_ir, stride_ic = cols, 1
    stride_or, stride_oc = rows, 1

    grid = (rows, cols)
    matrix_transpose_kernel[grid](
        input, output, rows, cols, stride_ir, stride_ic, stride_or, stride_oc, BLOCK_SIZE=16)
