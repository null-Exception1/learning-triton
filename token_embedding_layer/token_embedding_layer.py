import torch
import triton
import triton.language as tl

import pdb

@triton.jit 
def TEL(token_ids_ptr,position_ids_ptr,token_embeddings_ptr,position_embeddings_ptr,output_ptr,gamma_ptr,beta_ptr, B,T,D,BLOCK_SIZE: tl.constexpr,eps):
    batch = tl.program_id(0) # B
    index = tl.program_id(1) # T
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < D
    gamma = tl.load(gamma_ptr + cols,mask=mask)
    beta = tl.load(beta_ptr + cols,mask=mask)

    token_id = tl.load(token_ids_ptr+batch*T+index) 
    position_id = tl.load(position_ids_ptr+index) 

    token_embedding = tl.load(token_embeddings_ptr + token_id*D + cols,mask=mask,other=0.0)
    position_embedding = tl.load(position_embeddings_ptr + position_id*D + cols,mask=mask,other=0.0)

    #pdb.set_trace()

    s_b_t = token_embedding + position_embedding

    x = s_b_t

    mean = tl.sum(x, axis=0) / D
    x_centered = x - mean

    x_centered = tl.where(mask,x_centered,0.0)

    variance = tl.sum(x_centered * x_centered, axis=0) / D

    normalized_x = x_centered / tl.sqrt(variance + eps)
    output_at_t = normalized_x * gamma + beta
    #full_output_ptr = tl.load(output_ptr + tl.arange(0,B*T*D))
    tl.store(output_ptr + batch*T*D + index*D + cols, output_at_t,mask=mask)
    #pdb.set_trace()
def solve(
    token_ids: torch.Tensor,
    position_ids: torch.Tensor,
    token_embeddings: torch.Tensor,
    position_embeddings: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    output: torch.Tensor,
    B: int,
    T: int,
    V: int,
    P: int,
    D: int,
    eps: float,
):
    grid = (B,T)
    BLOCK_SIZE = triton.next_power_of_2(D) 
    print("BLOCK SIZE : ", BLOCK_SIZE)
    TEL[grid](token_ids,position_ids,token_embeddings,position_embeddings,output,gamma,beta,B,T,D,BLOCK_SIZE,eps)

    #print(output)

token_ids        = torch.tensor([[2, 0]],device='cuda')
position_ids = torch.tensor([0,1],device='cuda')
token_embeddings = torch.tensor([[ 1.0,  2.0,  3.0,  4.0],
                            [ 0.0,  1.0,  0.0, -1.0],
                            [ 2.0,  0.0, -2.0,  0.0]],device='cuda')
position_embeddings = torch.tensor([[ 0.0,  0.0,  0.0,  0.0],
                               [ 1.0,  0.0, -1.0,  0.0]],device='cuda')
gamma = torch.tensor([1.0, 1.0, 1.0, 1.0],device='cuda')
beta  = torch.tensor([0.0, 0.0, 0.0, 0.0],device='cuda')
B,T,D = 1,2,4
eps = 1e-5
output = torch.zeros(B,T,D,device='cuda')
solve(token_ids,position_ids,token_embeddings,position_embeddings,gamma,beta,output,B,T,0,0,D,eps,)