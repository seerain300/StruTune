import math
import torch
import triton
import triton.language as tl


@triton.jit
def linear_3d_constK(
    X, W, BIAS, Y,
    B: tl.constexpr, L: tl.constexpr, D: tl.constexpr, K: tl.constexpr, BLOCK_D: tl.constexpr,
    stride_x_b, stride_x_l, stride_x_d,
    stride_w_k, stride_w_d,
    stride_y_b, stride_y_l, stride_y_k,
    stride_b_k,
):
    """
    Compute Y[b, l, o] = sum_{d=0..D-1} X[b, l, d] * W[o, d] + BIAS[o]
    Launch as grid = (B, L, K). Each program handles one output channel o for a given (b, l).
    Loop over D in tiles to compute the dot product.
    """
    b = tl.program_id(0)
    l = tl.program_id(1)
    o = tl.program_id(2)

    base_x = b * stride_x_b + l * stride_x_l
    base_y = b * stride_y_b + l * stride_y_l

    acc = 0.0
    for d0 in range(0, D, BLOCK_D):
        d = d0 + tl.arange(0, BLOCK_D)
        mask = d < D
        x = tl.load(X + base_x + d * stride_x_d, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + o * stride_w_k + d * stride_w_d, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(x * w, axis=0)

    b_bias = tl.load(BIAS + o * stride_b_k, other=0.0).to(tl.float32)
    tl.store(Y + base_y + o * stride_y_k, acc + b_bias)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.d_model = 256
        self.order = 2
        self.layer_norm_eps = 1e-5

    def forward(self, hidden_states: torch.Tensor,
                norm1_weight: torch.Tensor, norm1_bias: torch.Tensor,
                norm2_weight: torch.Tensor, norm2_bias: torch.Tensor,
                in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor,
                mlp_fc1_weight: torch.Tensor, mlp_fc1_bias: torch.Tensor,
                mlp_fc2_weight: torch.Tensor, mlp_fc2_bias: torch.Tensor):
        """
        Triton-powered forward:
        - First LayerNorm over last dim (affine) on hidden_states using PyTorch for correctness.
        - All linear matvecs: in-projection, out-projection, and two MLP layers via Triton kernel.
        Returns final tensor of shape [B, L, d_model].
        """
        device = hidden_states.device
        dtype = torch.float32
        B, L, D = hidden_states.shape
        assert D == self.d_model, "Expected hidden_states with last dim d_model=256"

        # First residual: ensure float32 and contiguous
        residual = hidden_states.to(dtype).contiguous()

        # First LayerNorm (affine) with eps using PyTorch
        # This matches the original exactly and ensures correctness.
        normed = torch.layer_norm(
            residual,
            normalized_shape=(D,),
            weight=norm1_weight,
            bias=norm1_bias,
            eps=self.layer_norm_eps,
        )

        # Helper to run Triton linear_3d_constK
        def triton_linear(x, w, bias):
            K = w.shape[0]
            out = torch.empty((B, L, K), device=device, dtype=dtype)
            grid = (B, L, K)
            linear_3d_constK[grid](
                x, w, bias, out,
                B=B, L=L, D=x.shape[2], K=K, BLOCK_D=128,
                stride_x_b=x.stride(0), stride_x_l=x.stride(1), stride_x_d=x.stride(2),
                stride_w_k=w.stride(0), stride_w_d=w.stride(1),
                stride_y_b=out.stride(0), stride_y_l=out.stride(1), stride_y_k=out.stride(2),
                stride_b_k=bias.stride(0),
                num_warps=4, num_stages=2
            )
            return out

        # In-projection: [B, L, D] -> [B, L, inner_width]
        inner_width = self.d_model * (self.order + 1)
        u = triton_linear(normed, in_proj_weight, in_proj_bias)  # [B, L, inner_width]

        # Out-projection: [B, L, d_model]
        y_out = triton_linear(u, out_proj_weight, out_proj_bias)  # [B, L, D]

        # MLP layers: [B, L, d_model] -> [B, L, d_model]
        y_mlp1 = triton_linear(y_out, mlp_fc1_weight, mlp_fc1_bias)  # [B, L, D]
        y_mlp2 = triton_linear(y_mlp1, mlp_fc2_weight, mlp_fc2_bias)  # [B, L, D]

        return y_mlp2


def run(*args):
    return ModelNew()(*args)
