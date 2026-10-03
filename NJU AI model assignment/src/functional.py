import torch


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
    """
    Task 2 + Task 3
    """

    # =========================
    # Task 2
    # =========================

    # TODO


    # =========================
    # Task 3
    # =========================

    # TODO

    raise NotImplementedError

batch_size=2
seq_len=5
hidden_dim=10
embbed_dim=10
num_heads=2
head_dim=hidden_dim//num_heads
A1=torch.randn(batch_size, seq_len, hidden_dim)
W1=torch.randn(hidden_dim, embbed_dim)
matmul_with_multi_head(A1, W1, num_heads)
