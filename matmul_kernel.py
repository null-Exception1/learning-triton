import torch
import triton
import triton.language as tl
import math
import pdb

@triton.jit
def matrix_multiplication_kernel(
    a, b, c, M, N, K, stride_am, stride_an, stride_bn, stride_bk, stride_cm, stride_ck, BLOCK_SIZE: tl.constexpr
):
    a_ptr = a.to(tl.pointer_type(tl.float32))
    b_ptr = b.to(tl.pointer_type(tl.float32))
    c_ptr = c.to(tl.pointer_type(tl.float32))

    pid_m = tl.program_id(0) # this is the one guiding our outer dim
    pid_k = tl.program_id(1)

    off_h = tl.arange(0,BLOCK_SIZE)[None,:]
    off_v = tl.arange(0,BLOCK_SIZE)[:,None]
    
    c_accum = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    for block in range(0, tl.cdiv(N,BLOCK_SIZE) ):

        horizontal_a = off_h + block*BLOCK_SIZE
        vertical_a = (off_v + pid_m*BLOCK_SIZE)*N

        mask_a = (horizontal_a < N) & (vertical_a < M*N)
        
        a_block_offsets = a_ptr + (horizontal_a + vertical_a)

        horizontal_b = off_h + pid_k*BLOCK_SIZE
        vertical_b = (off_v + block*BLOCK_SIZE)*K

        mask_b = (horizontal_b < K) & (vertical_b < N*K)
        b_block_offsets = b_ptr + (horizontal_b + vertical_b)

        pdb.set_trace()
        a_mult = tl.load(a_block_offsets,mask=mask_a)
        b_mult = tl.load(b_block_offsets,mask=mask_b)
        

        c_mult = tl.dot(a_mult,b_mult)
        
        c_accum += c_mult


    ###
    horizontal_c = off_h + pid_k*BLOCK_SIZE
    vertical_c = (off_v + pid_m*BLOCK_SIZE)*K
    
    mask_c = (horizontal_c < K) & (vertical_c < M*K)
    c_block_offsets = c_ptr + (horizontal_c + vertical_c)

    tl.store(c_block_offsets, c_accum, mask=mask_c)

    ### 


# a, b, c are tensors on the GPU
def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, M: int, N: int, K: int):
    stride_am, stride_an = N, 1
    stride_bn, stride_bk = K, 1
    stride_cm, stride_ck = K, 1
    BLOCK_SIZE = 16
    grid = (triton.cdiv(M, BLOCK_SIZE),triton.cdiv(K, BLOCK_SIZE))
    matrix_multiplication_kernel[grid](
        a, b, c, M, N, K, stride_am, stride_an, stride_bn, stride_bk, stride_cm, stride_ck, BLOCK_SIZE=BLOCK_SIZE
    )
    return c

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

m,n,k = 20,30,40

x = torch.full((m, n), fill_value=1,device=DEVICE,dtype=torch.float32)
y = torch.full((n, k), fill_value=2, device=DEVICE,dtype=torch.float32)

print("X")
print(x)
print("Y")
print(y)

z2 = torch.zeros(m, k, device=DEVICE,dtype=torch.float32) 

z = x @ y
import numpy

solve(x, y, z2, m, n, k)

print("PyTorch result sample:\n",z)
print("Triton result sample:\n", z2)

if torch.allclose(z, z2, atol=1e-4):
    print("Test Passed! Results match PyTorch.")
else:
    print("Test Failed! Results mismatch.")