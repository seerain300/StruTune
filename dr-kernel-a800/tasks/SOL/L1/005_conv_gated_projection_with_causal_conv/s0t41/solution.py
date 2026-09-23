import torch
import triton
import triton.language as tl


# Triton kernel: elementwise multiply (B, M, N) = A * B (elementwise)
@triton.jit
def elementwise_mul_kernel(
    A_ptr, B_ptr, C_ptr,
    B: tl.constexpr, M: tl.constexpr, N: tl.constexpr, BLOCK_M: tl.constexpr
):
    b_id = tl.program_id(0)
    m_id = tl.program_id(1)
    n_id = tl.program_id(2)
    off = b_id * (M * N) + m_id * N + n_id
    a = tl.load(A_ptr + off)
    b = tl.load(B_ptr + off)
    c = a * b
    tl.store(C_ptr + off, c)


# Triton kernel: left-pad along last dim by PAD for tensors of shape (B, M, N)
# Produces (B, M, N + PAD) where padded positions are zeros.
@triton.jit
def pad_left_kernel_3d(
    inp_ptr, out_ptr,
    B: tl.constexpr, M: tl.constexpr, N: tl.constexpr, PAD: tl.constexpr, BLOCK_M: tl.constexpr
):
    b_id = tl.program_id(0)
    m_id = tl.program_id(1)
    n_out = tl.program_id(2)
    n_src = n_out - PAD
    valid = n_out >= PAD
    off_src = b_id * (M * N) + m_id * N + n_src
    off_dst = b_id * (M * (N + PAD)) + m_id * (N + PAD) + n_out
    v = tl.load(inp_ptr + off_src, mask=valid, other=0.0)
    tl.store(out_ptr + off_dst, v)


# Triton kernel: grouped 1D conv with groups=B and kernel_size=K=4
# Input: X (B, S_padded, H), Weight (H, 4), Bias (H,), Output (B, H, S_padded)
@triton.jit
def conv1d_groupsB_kernel(
    X_ptr,         # *f32, (B, S_padded, H)
    W_ptr,         # *f32, (H, 4)
    BIAS_ptr,      # *f32, (H,)
    OUT_ptr,       # *f32, (B, H, S_padded)
    B: tl.constexpr, H: tl.constexpr, S_padded: tl.constexpr, K: tl.constexpr
):
    b_id = tl.program_id(0)  # batch
    h_id = tl.program_id(1)  # channel in groups (here corresponds to input channel)
    t_tile = tl.program_id(2)
    T0 = t_tile * 64  # process 64 time positions per program
    for t in range(0, 64):
        t_off = T0 + t
        valid_t = t_off < S_padded

        acc = 0.0
        for k in range(0, K):  # K=4
            src_t = t_off + k
            valid_k = src_t < (S_padded - K + 1)  # ensure within bounds
            x_off = b_id * (S_padded * H) + src_t * H + h_id
            val = tl.load(X_ptr + x_off, mask=valid_t & valid_k, other=0.0)
            w_off = h_id * 4 + k
            w_val = tl.load(W_ptr + w_off)
            acc += val * w_val

        bias = tl.load(BIAS_ptr + h_id)
        acc += bias

        out_off = b_id * (H * S_padded) + h_id * S_padded + t_off
        tl.store(OUT_ptr + out_off, acc, mask=valid_t)


class ModelNew(torch.nn.Module):
    def forward(self, x: torch.Tensor, in_proj_weight: torch.Tensor, in_proj_bias: torch.Tensor,
                conv_weight: torch.Tensor, conv_bias: torch.Tensor, out_proj_weight: torch.Tensor, out_proj_bias: torch.Tensor):
        device = x.device

        # Cast and make contiguous (compute in fp32)
        x = x.to(torch.float32).contiguous()
        in_proj_weight = in_proj_weight.to(torch.float32).contiguous()
        in_proj_bias = in_proj_bias.to(torch.float32).contiguous()
        conv_weight = conv_weight.to(torch.float32).contiguous()
        conv_bias = conv_bias.to(torch.float32).contiguous()
        out_proj_weight = out_proj_weight.to(torch.float32).contiguous()
        out_proj_bias = out_proj_bias.to(torch.float32).contiguous()

        B, S, H = x.shape
        K = conv_weight.shape[2]  # kernel_size, given as 4 in the reference
        pad_left = K - 1

        # 1) in_proj via PyTorch F.linear: BCx (B, S, 3H)
        # Note: the original code uses F.linear(x, in_proj_weight, in_proj_bias)
        # Here we rely on torch to ensure correctness, then we process the chunks in Triton.
        BCx = torch.nn.functional.linear(x, in_proj_weight, in_proj_bias)  # (B, S, 3H)

        # 2) Transpose to (B, 3H, S) and chunk along dim=1 to get (B, H, S)
        # Ensure contiguous for Triton pointer math
        BCx_T = BCx.transpose(-1, -2).contiguous()  # (B, 3H, S)
        # B_t, C_t, x_proj each: (B, H, S)
        B_t = BCx_T[:, :H, :]                      # (B, H, S)
        C_t = BCx_T[:, H:2*H, :]                  # (B, H, S)
        x_proj = BCx_T[:, 2*H:, :]                # (B, H, S)

        # 3) Element-wise gating: Bx = B_t * x_proj
        Bx = torch.empty((B, H, S), device=device, dtype=torch.float32)

        BLOCK_M_gate = 128
        grid_gate = (B, H, triton.cdiv(S, BLOCK_M_gate))
        elementwise_mul_kernel[grid_gate](
            B_t, x_proj, Bx,
            B=B, M=H, N=S, BLOCK_M=BLOCK_M_gate,
            num_warps=4, num_stages=2
        )

        # 4) Left-pad along sequence for causal conv
        Bx_padded = torch.empty((B, S + pad_left, H), device=device, dtype=torch.float32)

        BLOCK_M_pad = 128
        grid_pad = (B, H, S + pad_left)
        pad_left_kernel_3d[grid_pad](
            Bx, Bx_padded,
            B=B, M=H, N=S, PAD=pad_left, BLOCK_M=BLOCK_M_pad,
            num_warps=4, num_stages=2
        )

        # 5) Grouped 1D conv with groups=B (per-input-channel), kernel_size=4, bias per channel
        # conv_weight is (H, 1, 4); we reshape to (H, 4)
        conv_weight_K = conv_weight[:, 0, :].contiguous()  # (H, 4)
        conv_out = torch.empty((B, H, S), device=device, dtype=torch.float32)

        BLOCK_T_conv = 64
        grid_conv = (B, H, triton.cdiv(S, BLOCK_T_conv))
        conv1d_groupsB_kernel[grid_conv](
            Bx_padded, conv_weight_K, conv_bias,
            conv_out,
            B=B, H=H, S_padded=S + pad_left, K=K
        )

        # 6) Output gating: y = C_t * conv_out, elementwise
        y = C_t * conv_out  # (B, H, S)

        # 7) Final linear projection: y -> output (B, S, H)
        output = torch.nn.functional.linear(y, out_proj_weight, out_proj_bias)  # (B, S, H)

        return output


def run(*args):
    return ModelNew()(*args)
