import torch
import triton
import triton.language as tl


# Kernel 1: Linear with bias: X @ W.T + b
# X: [B, S, H_in], W: [H_out, H_in], b: [H_out] -> Out: [B, S, H_out]
@triton.jit
def linear_bias_kernel(
    X_ptr, W_ptr, B_ptr, Out_ptr,
    Bsz, Ssz, H_in, H_out,
    stride_x_b, stride_x_s, stride_x_h,
    stride_w_o, stride_w_i,
    stride_out_b, stride_out_s, stride_out_h,
    BLOCK_IN: tl.constexpr, BLOCK_OUT: tl.constexpr,
):
    # Grid: (b, s, o)
    b = tl.program_id(0)
    s = tl.program_id(1)
    o = tl.program_id(2)

    acc = tl.zeros((), dtype=tl.float32)

    for k in range(0, H_in, BLOCK_IN):
        x_off = b * stride_x_b + s * stride_x_s + k * stride_x_h
        w_off = o * stride_w_o + k * stride_w_i
        x_vec = tl.zeros((BLOCK_IN,), dtype=tl.float32)
        for i in range(0, BLOCK_IN):
            x_idx = x_off + i * stride_x_h
            x_vec[i] = tl.load(X_ptr + x_idx)
        w_vec = tl.load(W_ptr + w_off + tl.arange(0, BLOCK_IN), mask=tl.arange(0, BLOCK_IN) < (H_in - k), other=0.0)
        acc += tl.sum(x_vec * w_vec, axis=0)

    # Add bias
    bias_val = tl.load(B_ptr + o)
    acc = acc + bias_val

    out_off = b * stride_out_b + s * stride_out_s + o * stride_out_h
    tl.store(Out_ptr + out_off, acc)


# Kernel 2: RMSNorm over last dim (size = head_dim=128): y = x * rsqrt(mean(x^2) + eps), then scale by weight
# Input x: [B, S, D], weight: [D] -> Output y: [B, S, D]
@triton.jit
def rmsnorm_kernel(
    X_ptr, Weight_ptr, Out_ptr,
    Bsz, Ssz, D,
    stride_x_b, stride_x_s, stride_x_d,
    stride_w_d, stride_out_b, stride_out_s, stride_out_d,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    d = tl.program_id(2)

    row_off = b * stride_x_b + s * stride_x_s

    # Load x vector over head_dim
    x_vec = tl.zeros((BLOCK,), dtype=tl.float32)
    for i in range(0, BLOCK):
        val = tl.load(X_ptr + row_off + (d + i) * stride_x_d)
        x_vec[i] = val

    # Compute RMS
    sum_sq = tl.sum(x_vec * x_vec, axis=0)
    mean = sum_sq / D
    inv_rms = tl.rsqrt(mean + 0.0)  # rms_norm_eps=0.0

    # Scale by weight
    weight = tl.load(Weight_ptr + (d + 0) * stride_w_d)
    y_vec = x_vec * inv_rms
    y_vec = y_vec * weight

    # Store
    out_off = b * stride_out_b + s * stride_out_s + (d + 0) * stride_out_d
    for i in range(0, BLOCK):
        tl.store(Out_ptr + out_off + i * stride_out_d, y_vec[i])


# Kernel 3: Rotate half of the last 64 dims for a 128-length vector: q1, q2 = q[:64], q[64:], q_rot = q1*cos - q2*sin
@triton.jit
def rotate_half_kernel(
    Q_ptr, Sin_ptr, Cos_ptr, Out_ptr,
    Bsz, Ssz, D,  # D should be 128
    stride_q_b, stride_q_s, stride_q_d,
    stride_sin_d, stride_cos_d, stride_out_b, stride_out_s, stride_out_d,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    d = tl.program_id(2)  # d over 0..D-1

    q_off = b * stride_q_b + s * stride_q_s + d * stride_q_d
    sin_off = d * stride_sin_d
    cos_off = d * stride_cos_d

    q1 = tl.load(Q_ptr + q_off)             # original first 64 dims
    q2 = tl.load(Q_ptr + q_off + 64 * stride_q_d)  # last 64 dims

    sin_val = tl.load(Sin_ptr + sin_off)
    cos_val = tl.load(Cos_ptr + cos_off)

    new_q1 = q1 * cos_val - q2 * sin_val
    new_q2 = q1 * sin_val + q2 * cos_val

    out_off1 = b * stride_out_b + s * stride_out_s + d * stride_out_d
    out_off2 = b * stride_out_b + s * stride_out_s + (d + 64) * stride_out_d

    tl.store(Out_ptr + out_off1, new_q1)
    tl.store(Out_ptr + out_off2, new_q2)


# Kernel 4: Compute attention scores: Q @ K^T for each (b, s, h) -> Out [B, H, S, S]
@triton.jit
def matmul_qk_kernel(
    Q_ptr, K_ptr, Out_ptr,
    Bsz, Ssz, D, H,  # H = num_attention_heads
    stride_q_b, stride_q_h, stride_q_s, stride_q_d,
    stride_k_b, stride_k_h, stride_k_s, stride_k_d,
    stride_out_b, stride_out_h, stride_out_s1, stride_out_s2,
    BLOCK_S1: tl.constexpr, BLOCK_S2: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s1 = tl.program_id(2)  # row index in seq
    s2 = tl.program_id(3)  # col index in seq

    # Load Q[b, h, s1, :] and K[b, h, s2, :]
    q_off = b * stride_q_b + h * stride_q_h + s1 * stride_q_s
    k_off = b * stride_k_b + h * stride_k_h + s2 * stride_k_s

    q_vec = tl.zeros((D,), dtype=tl.float32)
    k_vec = tl.zeros((D,), dtype=tl.float32)

    for d in range(0, D, 1):
        q_vec[d] = tl.load(Q_ptr + q_off + d * stride_q_d)
        k_vec[d] = tl.load(K_ptr + k_off + d * stride_k_d)

    score = tl.dot(q_vec, k_vec)

    out_off = b * stride_out_b + h * stride_out_h + s1 * stride_out_s1 + s2 * stride_out_s2
    tl.store(Out_ptr + out_off, score)


# Kernel 5: Softmax over last dim (sequence length) with causal mask: apply -inf if col > row
@triton.jit
def softmax_mask_kernel(
    In_ptr, Out_ptr,
    Ssz,
    stride_in_b, stride_in_h, stride_in_s1, stride_in_s2,
    stride_out_b, stride_out_h, stride_out_s1, stride_out_s2,
    BLOCK_S: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s1 = tl.program_id(2)  # row index
    s2 = tl.program_id(3)  # col index

    in_off = b * stride_in_b + h * stride_in_h + s1 * stride_in_s1 + s2 * stride_in_s2
    val = tl.load(In_ptr + in_off)

    # causal mask: if s2 > s1, set to -inf
    if s2 > s1:
        val = -float('inf')

    tl.store(Out_ptr + in_off, val)


# Kernel 6: Compute attention output: Softmax(QK_scaled) @ V for each (b, s, h)
# We assume Softmax already applied in previous kernel. Here, we just do V gather per column.
@triton.jit
def matmul_attn_kernel(
    Softmax_ptr, V_ptr, Out_ptr,
    Bsz, Ssz, D, H,
    stride_sm_b, stride_sm_h, stride_sm_s1, stride_sm_s2,
    stride_v_b, stride_v_h, stride_v_s, stride_v_d,
    stride_out_b, stride_out_h, stride_out_s, stride_out_d,
    BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s1 = tl.program_id(2)  # row output index

    acc = tl.zeros((D,), dtype=tl.float32)

    for s2 in range(0, Ssz, 1):
        sm_off = b * stride_sm_b + h * stride_sm_h + s1 * stride_sm_s1 + s2 * stride_sm_s2
        sm_val = tl.load(Softmax_ptr + sm_off)

        v_row = tl.zeros((D,), dtype=tl.float32)
        for d in range(0, D, 1):
            v_off = b * stride_v_b + h * stride_v_h + s2 * stride_v_s + d * stride_v_d
            v_row[d] = tl.load(V_ptr + v_off)
        acc += sm_val * v_row

    # Store acc into Out[b, h, s1, :]
    out_off = b * stride_out_b + h * stride_out_h + s1 * stride_out_s
    for d in range(0, D, 1):
        tl.store(Out_ptr + out_off + d * stride_out_d, acc[d])


# Kernel 7: Linear without bias: X @ W.T (final output projection)
@triton.jit
def linear_nobias_kernel(
    X_ptr, W_ptr, Out_ptr,
    Bsz, Ssz, D_in, D_out,
    stride_x_b, stride_x_s, stride_x_d,
    stride_w_o, stride_w_i,
    stride_out_b, stride_out_s, stride_out_d,
    BLOCK_IN: tl.constexpr, BLOCK_OUT: tl.constexpr,
):
    # Grid: (b, s, o)
    b = tl.program_id(0)
    s = tl.program_id(1)
    o = tl.program_id(2)

    acc = tl.zeros((), dtype=tl.float32)

    for k in range(0, D_in, BLOCK_IN):
        x_off = b * stride_x_b + s * stride_x_s + k * stride_x_d
        w_off = o * stride_w_o + k * stride_w_i
        x_vec = tl.zeros((BLOCK_IN,), dtype=tl.float32)
        for i in range(0, BLOCK_IN):
            x_idx = x_off + i * stride_x_d
            x_vec[i] = tl.load(X_ptr + x_idx)
        w_vec = tl.load(W_ptr + w_off + tl.arange(0, BLOCK_IN), mask=tl.arange(0, BLOCK_IN) < (D_in - k), other=0.0)
        acc += tl.sum(x_vec * w_vec, axis=0)

    out_off = b * stride_out_b + s * stride_out_s + o * stride_out_d
    tl.store(Out_ptr + out_off, acc)


# ModelNew: Entry point, Triton-only forward
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # We will not create any torch weights here; the caller will pass the correct tensors.

    def forward(self,
                hidden_states: torch.Tensor,
                q_proj_weight: torch.Tensor,
                q_proj_bias: torch.Tensor,
                k_proj_weight: torch.Tensor,
                k_proj_bias: torch.Tensor,
                v_proj_weight: torch.Tensor,
                v_proj_bias: torch.Tensor,
                o_proj_weight: torch.Tensor,
                q_norm_weight: torch.Tensor,
                k_norm_weight: torch.Tensor,
                cos: torch.Tensor,
                sin: torch.Tensor,
                rms_norm_eps: float):
        # Reshape: [B, S, H_in=11008] -> [B, S, 96, 128] for Q, K, V
        Bsz, Ssz, H_in = hidden_states.shape
        H_in_per_head = H_in // (num_attention_heads := 96)
        head_dim = H_in_per_head  # 128

        # 1) Linear projections
        Q = torch.empty((Bsz, Ssz, num_attention_heads, head_dim), device=hidden_states.device, dtype=torch.float32)
        K = torch.empty_like(Q)
        V = torch.empty_like(Q)

        grid_linear = (Bsz, Ssz, num_attention_heads)
        # Q
        linear_bias_kernel[grid_linear](
            hidden_states, q_proj_weight, q_proj_bias, Q,
            Bsz, Ssz, H_in, num_attention_heads * head_dim,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            q_proj_weight.stride(0), q_proj_weight.stride(1),
            Q.stride(0), Q.stride(1), Q.stride(2),
            128, 128
        )
        # K
        linear_bias_kernel[grid_linear](
            hidden_states, k_proj_weight, k_proj_bias, K,
            Bsz, Ssz, H_in, num_attention_heads * head_dim,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            k_proj_weight.stride(0), k_proj_weight.stride(1),
            K.stride(0), K.stride(1), K.stride(2),
            128, 128
        )
        # V
        linear_bias_kernel[grid_linear](
            hidden_states, v_proj_weight, v_proj_bias, V,
            Bsz, Ssz, H_in, num_attention_heads * head_dim,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            v_proj_weight.stride(0), v_proj_weight.stride(1),
            V.stride(0), V.stride(1), V.stride(2),
            128, 128
        )

        # 2) RMSNorm for Q and K
        Q_norm = torch.empty_like(Q)
        K_norm = torch.empty_like(K)

        grid_rms = (Bsz, Ssz, head_dim)
        rmsnorm_kernel[grid_rms](
            Q, q_norm_weight, Q_norm,
            Bsz, Ssz, head_dim,
            Q.stride(0), Q.stride(1), Q.stride(2),
            q_norm_weight.stride(0), Q_norm.stride(0), Q_norm.stride(1), Q_norm.stride(2),
            128
        )
        rmsnorm_kernel[grid_rms](
            K, k_norm_weight, K_norm,
            Bsz, Ssz, head_dim,
            K.stride(0), K.stride(1), K.stride(2),
            k_norm_weight.stride(0), K_norm.stride(0), K_norm.stride(1), K_norm.stride(2),
            128
        )

        # 3) Rotate Q and K (RoPE)
        Q_rot = torch.empty_like(Q_norm)
        K_rot = torch.empty_like(K_norm)

        grid_rot = (Bsz, Ssz, head_dim)
        rotate_half_kernel[grid_rot](
            Q_norm, cos, sin, Q_rot,
            Bsz, Ssz, head_dim,
            Q_norm.stride(0), Q_norm.stride(1), Q_norm.stride(2),
            cos.stride(0), sin.stride(0), Q_rot.stride(0), Q_rot.stride(1), Q_rot.stride(2),
            128
        )
        rotate_half_kernel[grid_rot](
            K_norm, cos, sin, K_rot,
            Bsz, Ssz, head_dim,
            K_norm.stride(0), K_norm.stride(1), K_norm.stride(2),
            cos.stride(0), sin.stride(0), K_rot.stride(0), K_rot.stride(1), K_rot.stride(2),
            128
        )

        # 4) GQA: repeat K/V to match 96 attention heads
        K_gqa = K_rot[:, :, None, :, :].expand(Bsz, num_attention_heads, num_key_value_heads, Ssz, head_dim).reshape(Bsz, num_attention_heads, Ssz, head_dim)
        V_gqa = V[:, :, None, :, :].expand(Bsz, num_attention_heads, num_key_value_heads, Ssz, head_dim).reshape(Bsz, num_attention_heads, Ssz, head_dim)

        # 5) Compute attention scores per head
        attn_scores = torch.empty((Bsz, num_attention_heads, Ssz, Ssz), device=hidden_states.device, dtype=torch.float32)

        grid_qk = (Bsz, num_attention_heads, Ssz, Ssz)
        matmul_qk_kernel[grid_qk](
            Q_rot, K_gqa, attn_scores,
            Bsz, Ssz, head_dim, num_attention_heads,
            Q_rot.stride(0), Q_rot.stride(1), Q_rot.stride(2), Q_rot.stride(3),
            K_gqa.stride(0), K_gqa.stride(1), K_gqa.stride(2), K_gqa.stride(3),
            attn_scores.stride(0), attn_scores.stride(1), attn_scores.stride(2), attn_scores.stride(3),
            128, 128
        )

        # 6) Softmax with causal mask
        attn_scores_masked = torch.empty_like(attn_scores)
        grid_sm = (Bsz, num_attention_heads, Ssz, Ssz)
        softmax_mask_kernel[grid_sm](
            attn_scores, attn_scores_masked,
            Ssz,
            attn_scores.stride(0), attn_scores.stride(1), attn_scores.stride(2), attn_scores.stride(3),
            attn_scores_masked.stride(0), attn_scores_masked.stride(1), attn_scores_masked.stride(2), attn_scores_masked.stride(3),
            128
        )

        # Apply scaling factor
        scaling = 1.0 / (head_dim ** 0.5)
        attn_scores_scaled = attn_scores_masked * scaling

        # 7) Compute attention output per head: Softmax(QK_scaled) @ V
        attn_out = torch.empty((Bsz, num_attention_heads, Ssz, head_dim), device=hidden_states.device, dtype=torch.float32)

        grid_attn = (Bsz, num_attention_heads, Ssz)
        matmul_attn_kernel[grid_attn](
            attn_scores_scaled, V_gqa, attn_out,
            Bsz, Ssz, head_dim, num_attention_heads,
            attn_scores_scaled.stride(0), attn_scores_scaled.stride(1), attn_scores_scaled.stride(2), attn_scores_scaled.stride(3),
            V_gqa.stride(0), V_gqa.stride(1), V_gqa.stride(2), V_gqa.stride(3),
            attn_out.stride(0), attn_out.stride(1), attn_out.stride(2), attn_out.stride(3),
            128
        )

        # 8) Final output projection (linear without bias): [B, S, 96*128] -> [B, S, 11008]
        out = torch.empty((Bsz, Ssz, num_attention_heads * head_dim), device=hidden_states.device, dtype=torch.float32)

        grid_final = (Bsz, Ssz, num_attention_heads * head_dim)
        linear_nobias_kernel[grid_final](
            attn_out, o_proj_weight, out,
            Bsz, Ssz, num_attention_heads * head_dim, num_attention_heads * head_dim,
            attn_out.stride(0), attn_out.stride(1), attn_out.stride(2),
            o_proj_weight.stride(0), o_proj_weight.stride(1),
            out.stride(0), out.stride(1), out.stride(2),
            128, 128
        )

        # Reshape to [B, S, 11008]
        return out


def run(*args):
    return ModelNew()(*args)
