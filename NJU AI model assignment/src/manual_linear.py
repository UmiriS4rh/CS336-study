import torch
class ManualLinear:
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        device=None,
        dtype=torch.float32,
    ):
        self.W=torch.randn(in_dim, out_dim, device=device, dtype=dtype)
        self.W_grad = torch.zeros_like(self.W)

        pass

    
    def forward(self, x: torch.Tensor):

        self.input=x
        return x@self.W


    def backward(self, grad_output: torch.Tensor):
        b,s,d=self.input.shape
        x_flat=self.input.reshape(-1,d)
        grad_out_flat=grad_output.reshape(-1,self.W.shape[1])
        self.W_grad=x_flat.T@grad_out_flat #决定该层参数
        grad_input=grad_output@self.W.T  #决定上一层的梯度
        return grad_input

        

        