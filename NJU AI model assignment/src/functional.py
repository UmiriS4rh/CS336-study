import torch
import torch.nn as nn

def matmul_with_multi_head(
    A1: torch.Tensor,
    W1: torch.Tensor,
    num_heads: int,
) -> torch.Tensor:
    head_dim=A1.shape[-1]//num_heads
    A2=A1.reshape(A1.shape[0], A1.shape[1], num_heads, head_dim).transpose(1, 2)
    W2=W1.reshape(num_heads, head_dim, -1)
    O2=torch.matmul(A2, W2)
    O2=O2.transpose(1, 2)
    print(O2.shape)
    return O2
"""
    result=[]
    for i in range(A1.shape[0]):
        O1=A1@W1
        result.append(O1)
    print(O1.shape)  朴素写法
"""




def matmul_with_importance(
    A1: torch.Tensor,
    W1: torch.Tensor,
    P: torch.Tensor,
    num_heads: int,
    top_p: float = 1.0,
    top_k: int | None = None,
    grad_output: torch.Tensor | None = None,
):

    b,s,h=A1.shape
    e=W1.shape[1]  
    head_dim=h//num_heads
    if top_k is None:
        top_k=s
    topk_indices=torch.topk(P, k=top_k, dim=1).indices  
    topk_mask=torch.zeros_like(P, dtype=torch.bool)
    topk_mask.scatter_(1, topk_indices, True)
    important_mask=topk_mask & (P>=top_p)
    A3=A1[important_mask]
    t=A3.shape[0]
    A3=A3.reshape(t,num_heads,head_dim).transpose(0,1)
    W2=W1.reshape(num_heads, head_dim, e)
    O3=torch.matmul(A3, W2)
    O3=O3.transpose(0,1)
    print(O3.shape)
    if grad_output is None:
        dA1=None
        dW1=None
    else:
        grad_heads = grad_output.transpose(0, 1)
        dA3=torch.matmul(grad_heads,W2.transpose(1,2))
        dA3=dA3.transpose(0,1).reshape(t, h)
        dA1=torch.zeros_like(A1)
        dA1[important_mask]=dA3
        dW2=torch.matmul(A3.transpose(1,2),grad_heads)
        dW1=dW2.reshape(h,e)
    return O3, dA1, dW1





batch_size=2
seq_len=5
hidden_dim=10
embbed_dim=10
num_heads=2
P=torch.randn(batch_size, seq_len)
top_k=3
top_p=0.5
A1=torch.randn(batch_size, seq_len, hidden_dim)
W1=torch.randn(hidden_dim, embbed_dim)
matmul_with_importance(A1, W1, P, num_heads, top_p=top_p, top_k=top_k)
