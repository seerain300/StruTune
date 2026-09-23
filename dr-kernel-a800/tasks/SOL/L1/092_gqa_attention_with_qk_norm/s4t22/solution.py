import torch
import triton
import triton.language as tl


# Kernel 1: Linear with bias: X @ W.T + b
# X: [B, S, H_in], W: [H_out, H_in], b: [H_out]
# Out: [B, S, H_out]
@triton.jit
def linear_bias_kernel(
    X_ptr, W_ptr, BIAS_ptr, OUT_ptr,
    Bsz, Ssz, H_in, H_out,
    stride_x_b, stride_x_s, stride_x_h,
    stride_w_o, stride_w_i,
    stride_out_b, stride_out_s, stride_out_h,
    BLOCK_K: tl.constexpr,
):
    # Grid: (Bsz, Ssz, H_out)
    b = tl.program_id(0)
    s = tl.program_id(1)
    o = tl.program_id(2)

    acc = 0.0
    for k in range(0, H_in, BLOCK_K):
        k_idx = k + tl.arange(0, BLOCK_K)
        mask_k = k_idx < H_in

        x_off = b * stride_x_b + s * stride_x_s + k_idx * stride_x_h
        x_vec = tl.load(X_ptr + x_off, mask=mask_k, other=0.0)

        w_off = o * stride_w_o + k_idx * stride_w_i
        w_vec = tl.load(W_ptr + w_off, mask=mask_k, other=0.0)

        acc += tl.sum(x_vec * w_vec, axis=0)

    bias_val = tl.load(BIAS_ptr + o)
    acc = acc + bias_val

    out_off = b * stride_out_b + s * stride_out_s + o * stride_out_h
    tl.store(OUT_ptr + out_off, acc)


# Kernel 2: RMSNorm over last dim (size = D, here 128): y = x * rsqrt(mean(x^2) + eps), then scale by weight
# Input x: [B, S, D], weight: [D] -> Output y: [B, S, D]
# Note: eps is set to 0.0 to match original code behavior.
@triton.jit
def rmsnorm_kernel(
    X_ptr, Weight_ptr, Out_ptr,
    Bsz, Ssz, D,
    stride_x_b, stride_x_s, stride_x_d,
    stride_w_d, stride_out_b, stride_out_s, stride_out_d,
    BLOCK: tl.constexpr,
):
    # Grid: (Bsz, Ssz, D)
    b = tl.program_id(0)
    s = tl.program_id(1)
    d = tl.program_id(2)  # d ranges over 0..D-1

    row_off = b * stride_x_b + s * stride_x_s

    sum_sq = 0.0
    for i in range(0, BLOCK):
        off = row_off + (d + i) * stride_x_d
        x_val = tl.load(X_ptr + off)
        sum_sq += x_val * x_val

    mean = sum_sq / D
    inv_rms = tl.rsqrt(mean)  # eps = 0.0
    weight_val = tl.load(Weight_ptr + (d + 0) * stride_w_d)

    y = 0.0
    for i in range(0, BLOCK):
        off = row_off + (d + i) * stride_x_d
        x_val = tl.load(X_ptr + off)
        y += x_val * inv_rms * weight_val

    out_off = b * stride_out_b + s * stride_out_s + (d + 0) * stride_out_d
    tl.store(Out_ptr + out_off, y)


# Kernel 3: Rotate half of the last 64 dims: q1, q2 = q[:64], q[64:], q_rot = q1*cos - q2*sin
# Apply rotation to vectors of length 128.
@triton.jit
def rotate_half_kernel(
    Q_ptr, Sin_ptr, Cos_ptr, Out_ptr,
    Bsz, Ssz, D,  # D should be 128
    stride_q_b, stride_q_s, stride_q_d,
    stride_sin_d, stride_cos_d, stride_out_b, stride_out_s, stride_out_d,
    BLOCK: tl.constexpr,
):
    # Grid: (Bsz, Ssz, D)
    b = tl.program_id(0)
    s = tl.program_id(1)
    d = tl.program_id(2)

    q_off = b * stride_q_b + s * stride_q_s + d * stride_q_d
    sin_off = d * stride_sin_d
    cos_off = d * stride_cos_d

    q0 = tl.load(Q_ptr + q_off)
    sin_val = tl.load(Sin_ptr + sin_off)
    cos_val = tl.load(Cos_ptr + cos_off)

    q1 = tl.load(Q_ptr + q_off + 64 * stride_q_d)
    # Rotate: new_q = [q0*cos - q1*sin, q0*sin + q1*cos] for the last half
    new_q0 = q0 * cos_val - q1 * sin_val
    q2 = tl.load(Q_ptr + q_off + 64 * stride_q_d + 64 * stride_q_d)  # read q2 from original second half
    new_q1 = q0 * sin_val + q1 * cos_val

    # Store rotated halves back
    tl.store(Out_ptr + b * stride_out_b + s * stride_out_s + (d + 0) * stride_out_d, new_q0)
    tl.store(Out_ptr + b * stride_out_b + s * stride_out_s + (d + 64) * stride_out_d, new_q1)


# Kernel 4: Linear without bias: X @ W.T
# X: [B, S, H_in], W: [H_out, H_in] -> Out: [B, S, H_out]
@triton.jit
def linear_nobias_kernel(
    X_ptr, W_ptr, OUT_ptr,
    Bsz, Ssz, H_in, H_out,
    stride_x_b, stride_x_s, stride_x_h,
    stride_w_o, stride_w_i,
    stride_out_b, stride_out_s, stride_out_h,
    BLOCK_K: tl.constexpr,
):
    # Grid: (Bsz, Ssz, H_out)
    b = tl.program_id(0)
    s = tl.program_id(1)
    o = tl.program_id(2)

    acc = 0.0
    for k in range(0, H_in, BLOCK_K):
        k_idx = k + tl.arange(0, BLOCK_K)
        mask_k = k_idx < H_in

        x_off = b * stride_x_b + s * stride_x_s + k_idx * stride_x_h
        x_vec = tl.load(X_ptr + x_off, mask=mask_k, other=0.0)

        w_off = o * stride_w_o + k_idx * stride_w_i
        w_vec = tl.load(W_ptr + w_off, mask=mask_k, other=0.0)

        acc += tl.sum(x_vec * w_vec, axis=0)

    out_off = b * stride_out_b + s * stride_out_s + o * stride_out_h
    tl.store(OUT_ptr + out_off, acc)


class ModelNew(torch.nn.Module):
    def __init__(self, batch_size: int, seq_len: int):
        super().__init__()
        # Allocate random hidden_states like the original. We don't use torch ops in forward after init.
        self.hidden_states = torch.randn(batch_size, seq_len, 768, device='cuda')

    def forward(
        self,
        q_proj_weight: torch.Tensor,   # [128, 768]
        q_proj_bias: torch.Tensor,     # [128]
        k_proj_weight: torch.Tensor,   # [128, 768]
        k_proj_bias: torch.Tensor,     # [128]
        v_proj_weight: torch.Tensor,   # [128, 768]
        v_proj_bias: torch.Tensor,     # [128]
        o_proj_weight: torch.Tensor,   # [11008, 128]
        q_norm_weight: torch.Tensor,   # [128]
        k_norm_weight: torch.Tensor,   # [128]
        cos: torch.Tensor,             # [128]
        sin: torch.Tensor,             # [128]
    ):
        # Shapes:
        Bsz = self.hidden_states.shape[0]
        Ssz = self.hidden_states.shape[1]
        H_in = self.hidden_states.shape[2]  # 768
        H_out = 128  # Q, K, V projection dims

        # 1) Q projection (linear with bias) -> [Bsz, Ssz, 128]
        Q = torch.empty((Bsz, Ssz, H_out), device='cuda', dtype=self.hidden_states.dtype)
        grid = (Bsz, Ssz, H_out)
        stride_x_b = H_in * Ssz
        stride_x_s = H_in
        stride_x_h = 1
        stride_w_o = H_in
        stride_w_i = 1
        stride_out_b = H_out * Ssz
        stride_out_s = H_out
        stride_out_h = 1
        linear_bias_kernel[grid](
            self.hidden_states, q_proj_weight, q_proj_bias, Q,
            Bsz, Ssz, H_in, H_out,
            stride_x_b, stride_x_s, stride_x_h,
            stride_w_o, stride_w_i,
            stride_out_b, stride_out_s, stride_out_h,
            BLOCK_K=64,
        )

        # 2) RMSNorm for Q (head_dim=128) -> [Bsz, Ssz, 128]
        D = 128
        Q_norm = torch.empty_like(Q)
        stride_q_b = Q.shape[0] * Q.shape[1]
        stride_q_s = Q.shape[1]
        stride_q_d = 1
        stride_w_d = D
        stride_out_b_q = Q_norm.shape[0] * Q_norm.shape[1]
        stride_out_s_q = Q_norm.shape[1]
        stride_out_d_q = 1
        grid_rms = (Bsz, Ssz, D)
        rmsnorm_kernel[grid_rms](
            Q, q_norm_weight, Q_norm,
            Bsz, Ssz, D,
            stride_q_b, stride_q_s, stride_q_d,
            stride_w_d, stride_out_b_q, stride_out_s_q, stride_out_d_q,
            BLOCK=128,
        )

        # 3) Rotate Q halves (RoPE)
        Q_rot = torch.empty_like(Q)
        stride_q_b_rot = Q.shape[0] * Q.shape[1]
        stride_q_s_rot = Q.shape[1]
        stride_q_d_rot = 1
        stride_sin_d = D
        stride_cos_d = D
        stride_out_b_rot = Q_rot.shape[0] * Q_rot.shape[1]
        stride_out_s_rot = Q_rot.shape[1]
        stride_out_d_rot = 1
        grid_rot = (Bsz, Ssz, D)
        rotate_half_kernel[grid_rot](
            Q_norm, cos, sin, Q_rot,
            Bsz, Ssz, D,
            stride_q_b_rot, stride_q_s_rot, stride_q_d_rot,
            stride_sin_d, stride_cos_d, stride_out_b_rot, stride_out_s_rot, stride_out_d_rot,
            BLOCK=128,
        )

        # 4) Final output projection: Q_rot @ o_proj_weight.T (no bias) -> [Bsz, Ssz, 11008]
        Out = torch.empty((Bsz, Ssz, o_proj_weight.shape[0]), device='cuda', dtype=Q_rot.dtype)

        prod = Bsz * Ssz
        H_out_final = o_proj_weight.shape[0]  # 11008
        # Flatten Q_rot to [prod, D]


def run(*args):
    return ModelNew()(*args)
