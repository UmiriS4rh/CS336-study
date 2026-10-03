import torch

from src.manual_linear import ManualLinear


# 固定随机数，方便每次运行得到相同结果
torch.manual_seed(0)


# =========================
# 1. 设置测试尺寸
# =========================

batch_size = 2
seq_len = 3
in_dim = 4
out_dim = 5


# =========================
# 2. 创建输入
# =========================

x = torch.randn(
    batch_size,
    seq_len,
    in_dim,
)

print("x.shape =", x.shape)


# =========================
# 3. 创建 ManualLinear
# =========================

layer = ManualLinear(
    in_dim=in_dim,
    out_dim=out_dim,
)


# =========================
# 4. 测试 forward
# =========================

y = layer.forward(x)

print("y.shape =", y.shape)

assert y.shape == (
    batch_size,
    seq_len,
    out_dim,
)

print("Forward shape test passed!")


# =========================
# 5. 构造一个简单 loss
# =========================

loss = (y ** 2).sum()

print("loss =", loss.item())


# 对于 loss = sum(y^2)
# 这里先给你 grad_output
grad_output = 2 * y

print("grad_output.shape =", grad_output.shape)


# =========================
# 6. 测试 backward
# =========================

grad_input = layer.backward(grad_output)

print("grad_input.shape =", grad_input.shape)
print("W_grad.shape =", layer.W_grad.shape)


assert grad_input.shape == x.shape
assert layer.W_grad.shape == layer.W.shape

print("Backward shape test passed!")