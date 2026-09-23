import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False

# Constants from the original code (fixed in this implementation)
HEAD_DIM = 128
NUM_ATTENTION_HEADS = 96
NUM_KEY_VALUE_HEADS = 8
NUM_KEY_VALUE_GROUPS = 12
NUM_ATTENTION_DIM = NUM_ATTENTION_HEADS * HEAD_DIM  # 12288
NUM_KEY_VALUE_DIM = NUM_KEY_VALUE_HEADS * HEAD_DIM  # 1024

# Triton kernel: dense linear Y = X @ W^T + B
# X: [B, S, Din], W: [Dout, Din], B: [Dout], Y: [B, S, Dout]
@triton.jit
def linear_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    Bsz, Ssz, Din, Dout,
    stride_xb, stride_xs, stride_xd,
    stride_w0, stride_w1,  # W strides for (Dout, Din)
    stride_yb, stride_ys, stride_yd,
):
    pid_m = tl.program_id(axis=0)  # over B*S
    pid_n = tl.program_id(axis=1)  # over Dout tiles

    b = pid_m // Ssz
    s = pid_m % Ssz

    d_out_offsets = pid_n * 128 + tl.arange(0, 128)
    acc = tl.zeros([128], dtype=tl.float32)

    # Loop over Din in tiles of 128
    for d_in_start in range(0, Din, 128):
        d_in_offsets = d_in_start + tl.arange(0, 128)
        # Load X[b, s, d_in_offsets]
        x = tl.load(
            X_ptr + b * stride_xb + s * stride_xs + d_in_offsets * stride_xd,
            mask=d_in_offsets < Din,
            other=0.0
        ).to(tl.float32)  # [128]
        # Load W[d_out_offsets, d_in_offsets] -> shape [128, 128]
        w = tl.load(
            W_ptr + d_out_offsets[:, None] * stride_w0 + d_in_offsets[None, :] * stride_w1,
            mask=(d_out_offsets[:, None] < Dout) & (d_in_offsets[None, :] < Din),
            other=0.0
        ).to(tl.float32)  # [128, 128]
        # Accumulate: acc += x[None, :] @ w  -> x[None, 128] * w[128, 128] -> [128]
        acc += tl.sum(w * x[None, :], axis=1)
    # Add bias
    bias = tl.load(B_ptr + d_out_offsets, mask=d_out_offsets < Dout, other=0.0).to(tl.float32)
    acc += bias
    # Store Y[b, s, d_out_offsets]
    tl.store(
        Y_ptr + b * stride_yb + s * stride_ys + d_out_offsets * stride_yd,
        acc,
        mask=d_out_offsets < Dout
    )


# Triton kernel: RMSNorm per (b, h, s, d)
# X: [B, H, S, D], W: [D], Y: [B, H, S, D]
@triton.jit
def rmsnorm_kernel(
    X_ptr, W_ptr, Y_ptr,
    Bsz, Hsz, Ssz, D,
    stride_xb, stride_xh, stride_xs, stride_xd,
    stride_yb, stride_yh, stride_ys, stride_yd,
    eps,
):
    b = tl.program_id(axis=0)
    h = tl.program_id(axis=1)
    s = tl.program_id(axis=2)
    d_offsets = tl.arange(0, D)
    x = tl.load(
        X_ptr + b * stride_xb + h * stride_xh + s * stride_xs + d_offsets * stride_xd,
        mask=d_offsets < D,
        other=0.0
    ).to(tl.float32)  # [D]
    mean_sq = tl.sum(x * x, axis=0) / D
    inv_rms = tl.rsqrt(mean_sq + eps)
    w = tl.load(W_ptr + d_offsets, mask=d_offsets < D, other=1.0).to(tl.float32)
    y = (x * inv_rms) * w
    tl.store(
        Y_ptr + b * stride_yb + h * stride_yh + s * stride_ys + d_offsets * stride_yd,
        y,
        mask=d_offsets < D
    )


# Triton kernel: apply rotation (RoPE) for last dim D=128, split into two halves (64,64)
# X: [B, H, S, D], C: [S, D/2], S: [S, D/2], Y: [B, H, S, D]
@triton.jit
def rotate_half_kernel(
    X_ptr, C_ptr, S_ptr, Y_ptr,
    Bsz, Hsz, Ssz, D,
    stride_xb, stride_xh, stride_xs, stride_xd,
    stride_yb, stride_yh, stride_ys, stride_yd,
    stride_c0, stride_c1,
    stride_s0, stride_s1,
):
    b = tl.program_id(axis=0)
    h = tl.program_id(axis=1)
    s = tl.program_id(axis=2)
    d_offsets = tl.arange(0, D)
    x = tl.load(
        X_ptr + b * stride_xb + h * stride_xh + s * stride_xs + d_offsets * stride_xd,
        mask=d_offsets < D,
        other=0.0
    ).to(tl.float32)  # [D]
    # Split
    q1 = x[0:64]
    q2 = x[64:128]
    rotated_half = tl.cat((-q2, q1), axis=0)  # [128]
    cos_vec = tl.load(C_ptr + s * stride_c0 + tl.arange(0, 64) * stride_c1).to(tl.float32)
    sin_vec = tl.load(S_ptr + s * stride_s0 + tl.arange(0, 64) * stride_s1).to(tl.float32)
    y = x * cos_vec + rotated_half * sin_vec
    tl.store(
        Y_ptr + b * stride_yb + h * stride_yh + s * stride_ys + d_offsets * stride_yd,
        y,
        mask=d_offsets < D
    )


# Triton kernel: compute attention scores Y[b, h, :, :] = Q[b,h,:] @ K[b,h,:]^T * scaling
# Inputs: Q: [B, H, S, D], K: [B, H, S, D], Y: [B, H, S, S]
@triton.jit
def attention_scores_kernel(
    Q_ptr, K_ptr, Y_ptr,
    Bsz, Hsz, Ssz, D,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_yb, stride_yh, stride_ys, stride_yt,
    scaling,
):
    b = tl.program_id(axis=0)
    h = tl.program_id(axis=1)
    # We iterate over s_out and s_key to form SxS matrix
    # Triton loop over S
    for i in range(0, Ssz):
        for j in range(0, Ssz):
            # Load Q[b,h,i,:] and K[b,h,j,:]
            q = tl.load(
                Q_ptr + b * stride_qb + h * stride_qh + i * stride_qs + tl.arange(0, D) * stride_qd,
                mask=tl.arange(0, D) < D,
                other=0.0
            ).to(tl.float32)  # [D]
            k = tl.load(
                K_ptr + b * stride_kb + h * stride_kh + j * stride_ks + tl.arange(0, D) * stride_kd,
                mask=tl.arange(0, D) < D,
                other=0.0
            ).to(tl.float32)  # [D]
            score = tl.dot(q, k) * scaling  # scalar
            # Store to Y[b,h,i,j]
            tl.store(
                Y_ptr + b * stride_yb + h * stride_yh + i * stride_ys + j * stride_yt,
                score,
                mask=(i < Ssz) & (j < Ssz)
            )


# Triton kernel: softmax over sequence dimension Y[b, h, :, ] = softmax(AttentionScores)
@triton.jit
def softmax_row_kernel(
    X_ptr, Y_ptr,
    Ssz,
    stride_xb, stride_xh, stride_xs,
    stride_yb, stride_yh, stride_ys,
):
    b = tl.program_id(axis=0)
    h = tl.program_id(axis=1)
    # Compute row max for numerical stability
    row_max = -float('inf')
    for i in range(0, Ssz):
        val = tl.load(X_ptr + b * stride_xb + h * stride_xh + i * stride_xs).to(tl.float32)
        if val > row_max:
            row_max = val
    # Compute exp and sum
    row_sum = 0.0
    for i in range(0, Ssz):
        val = tl.load(X_ptr + b * stride_xb + h * stride_xh + i * stride_xs).to(tl.float32)
        exp_val = tl.exp(val - row_max)
        tl.store(Y_ptr + b * stride_yb + h * stride_yh + i * stride_ys, exp_val)
        row_sum += exp_val
    # Normalize
    for i in range(0, Ssz):
        val = tl.load(Y_ptr + b * stride_yb + h * stride_yh + i * stride_ys).to(tl.float32)
        tl.store(Y_ptr + b * stride_yb + h * stride_yh + i * stride_ys, val / row_sum)


# Triton kernel: Y[b, h, s] = Soft[b,h,:] @ V[b,h,s,:]
# Inputs: Soft: [B, H, S], V: [B, H, S, D]
@triton.jit
def matmul_softmax_v_kernel(
    Soft_ptr, V_ptr, Y_ptr,
    Bsz, Hsz, Ssz, D,
    stride_sb, stride_sh, stride_ss,  # Soft strides
    stride_vb, stride_vh, stride_vs, stride_vd,  # V strides
    stride_yb, stride_yh, stride_ys, stride_yd,   # Y strides
):
    b = tl.program_id(axis=0)
    h = tl.program_id(axis=1)
    s = tl.program_id(axis=2)
    acc = tl.zeros([D], dtype=tl.float32)
    for i in range(0, Ssz):
        soft_i = tl.load(Soft_ptr + b * stride_sb + h * stride_sh + i * stride_ss).to(tl.float32)
        v_row = tl.load(
            V_ptr + b * stride_vb + h * stride_vh + s * stride_vs + tl.arange(0, D) * stride_vd,
            mask=tl.arange(0, D) < D,
            other=0.0
        ).to(tl.float32)
        acc += soft_i * v_row
    tl.store(
        Y_ptr + b * stride_yb + h * stride_yh + s * stride_ys + tl.arange(0, D) * stride_yd,
        acc,
        mask=tl.arange(0, D) < D
    )


def run_triton_only(
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
    rms_norm_eps: float,
    batch_size: int, seq_length: int,
):
    # Ensure CUDA tensors and float32 compute
    assert hidden_states.is_cuda, "hidden_states must be CUDA"
    # 1) Linear layers: Q, K, V
    # Prepare X = hidden_states: [B, S, hidden_size]
    Bsz, Ssz, hidden = hidden_states.shape
    Din = hidden

    # Q = hidden @ q_proj_weight^T + q_proj_bias
    Q = torch.empty((Bsz, Ssz, Din), device=hidden_states.device, dtype=torch.float32)
    Wq = q_proj_weight
    Bq = q_proj_bias if q_proj_bias is not None else torch.zeros(Wq.shape[0], device=hidden_states.device, dtype=torch.float32)
    grid_q = (Bsz * Ssz, (Din + 127) // 128)
    linear_kernel[grid_q](
        hidden_states, Wq, Bq, Q,
        Bsz, Ssz, Din, Din,
        hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
        Wq.stride(0), Wq.stride(1),
        Q.stride(0), Q.stride(1), Q.stride(2),
        num_warps=4
    )

    # K = hidden @ k_proj_weight^T + k_proj_bias
    K = torch.empty((Bsz, Ssz, Din), device=hidden_states.device, dtype=torch.float32)
    Wk = k_proj_weight
    Bk = k_proj_bias if k_proj_bias is not None else torch.zeros(Wk.shape[0], device=hidden_states.device, dtype=torch.float32)
    grid_k = (Bsz * Ssz, (Din + 127) // 128)
    linear_kernel[grid_k](
        hidden_states, Wk, Bk, K,
        Bsz, Ssz, Din, Din,
        hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
        Wk.stride(0), Wk.stride(1),
        K.stride(0), K.stride(1), K.stride(2),
        num_warps=4
    )

    # V = hidden @ v_proj_weight^T + v_proj_bias
    V = torch.empty((Bsz, Ssz, Din), device=hidden_states.device, dtype=torch.float32)
    Wv = v_proj_weight
    Bv = v_proj_bias if v_proj_bias is not None else torch.zeros(Wv.shape[0], device=hidden_states.device, dtype=torch.float32)
    grid_v = (Bsz * Ssz, (Din + 127) // 128)
    linear_kernel[grid_v](
        hidden_states, Wv, Bv, V,
        Bsz, Ssz, Din, Din,
        hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
        Wv.stride(0), Wv.stride(1),
        V.stride(0), V.stride(1), V.stride(2),
        num_warps=4
    )

    # 2) RMSNorm for Q and K: learnable per head per dim
    # Q_norm: [H, D] learnable, K_norm: [H, D]
    Q_norm = torch.empty((Bsz, NUM_ATTENTION_HEADS, Ssz, HEAD_DIM), device=hidden_states.device, dtype=torch.float32)
    # Launch RMSNorm kernel over (B, H, S)
    grid_rmsq = (Bsz, NUM_ATTENTION_HEADS, Ssz)
    rmsnorm_kernel[grid_rmsq](
        Q, q_norm_weight, Q_norm,
        Bsz, NUM_ATTENTION_HEADS, Ssz, HEAD_DIM,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        Q_norm.stride(0), Q_norm.stride(1), Q_norm.stride(2), Q_norm.stride(3),
        rms_norm_eps,
        num_warps=4
    )

    K_norm = torch.empty((Bsz, NUM_ATTENTION_HEADS, Ssz, HEAD_DIM), device=hidden_states.device, dtype=torch.float32)
    grid_rmsk = (Bsz, NUM_ATTENTION_HEADS, Ssz)
    rmsnorm_kernel[grid_rmsk](
        K, k_norm_weight, K_norm,
        Bsz, NUM_ATTENTION_HEADS, Ssz, HEAD_DIM,
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        K_norm.stride(0), K_norm.stride(1), K_norm.stride(2), K_norm.stride(3),
        rms_norm_eps,
        num_warps=4
    )

    # 3) Apply rotation (RoPE) for Q and K
    Q_rot = torch.empty((Bsz, NUM_ATTENTION_HEADS, Ssz, HEAD_DIM), device=hidden_states.device, dtype=torch.float32)
    K_rot = torch.empty((Bsz, NUM_ATTENTION_HEADS, Ssz, HEAD_DIM), device=hidden_states.device, dtype=torch.float32)
    # cos/sin are [S, 64]
    grid_rot = (Bsz, NUM_ATTENTION_HEADS, Ssz)
    rotate_half_kernel[grid_rot](
        Q_norm, cos, sin, Q_rot,
        Bsz, NUM_ATTENTION_HEADS, Ssz, HEAD_DIM,
        Q_norm.stride(0), Q_norm.stride(1), Q_norm.stride(2), Q_norm.stride(3),
        Q_rot.stride(0), Q_rot.stride(1), Q_rot.stride(2), Q_rot.stride(3),
        cos.stride(0), cos.stride(1),
        sin.stride(0), sin.stride(1),
        num_warps=4
    )
    rotate_half_kernel[grid_rot](
        K_norm, cos, sin, K_rot,
        Bsz, NUM_ATTENTION_HEADS, Ssz, HEAD_DIM,
        K_norm.stride(0), K_norm.stride(1), K_norm.stride(2), K_norm.stride(3),
        K_rot.stride(0), K_rot.stride(1), K_rot.stride(2), K_rot.stride(3),
        cos.stride(0), cos.stride(1),
        sin.stride(0), sin.stride(1),
        num_warps=4
    )

    # 4) Compute attention scores: S[b,h,i,j] = Q_rot[b,h,i,:] @ K_rot[b,h,j,:] * scaling
    # We need Q_rot and K_rot to have shape [B, H, S, D]; above produced exactly that.
    AttnScores = torch.empty((Bsz, NUM_ATTENTION_HEADS, Ssz, Ssz), device=hidden_states.device, dtype=torch.float32)
    # Launch attention_scores_kernel
    grid_as = (Bsz, NUM_ATTENTION_HEADS)
    attention_scores_kernel[grid_as](
        Q_rot, K_rot, AttnScores,
        Bsz, NUM_ATTENTION_HEADS, Ssz, HEAD_DIM,
        Q_rot.stride(0), Q_rot.stride(1), Q_rot.stride(2), Q_rot.stride(3),
        K_rot.stride(0), K_rot.stride(1), K_rot.stride(2), K_rot.stride(3),
        AttnScores.stride(0), AttnScores.stride(1), AttnScores.stride(2), AttnScores.stride(3),
        1.0 / (HEAD_DIM ** 0.5),
        num_warps=1  # scalar loop; can be small
    )

    # 5) Softmax over sequence dimension per (b,h)
    Soft = torch.empty((Bsz, NUM_ATTENTION_HEADS, Ssz), device=hidden_states.device, dtype=torch.float32)
    # softmax_row_kernel: process per (b,h)
    grid_sm = (Bsz, NUM_ATTENTION_HEADS)
    softmax_row_kernel[grid_sm](
        AttnScores, Soft,
        Ssz,
        AttnScores.stride(0), AttnScores.stride(1), AttnScores.stride(2),
        Soft.stride(0), Soft.stride(1), Soft.stride(2),
        num_warps=1
    )

    # 6) Compute attention output Y[b,h,s,:] = Soft[b,h,s] @ V[b,h,s,:]
    Y_attn = torch.empty((Bsz, NUM_ATTENTION_HEADS, Ssz, HEAD_DIM), device=hidden_states.device, dtype=torch.float32)
    grid_mmv = (Bsz, NUM_ATTENTION_HEADS, Ssz)
    matmul_softmax_v_kernel[grid_mmv](
        Soft, V, Y_attn,
        Bsz, NUM_ATTENTION_HEADS, Ssz, HEAD_DIM,
        Soft.stride(0), Soft.stride(1), Soft.stride(2),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        Y_attn.stride(0), Y_attn.stride(1), Y_attn.stride(2), Y_attn.stride(3),
        num_warps=4
    )

    # 7) Reshape attn_output to [B, S, H*HEAD_DIM] and final output projection
    # Flatten heads: [B, S, H*HEAD_DIM]
    attn_output_flat = Y_attn.view(Bsz, Ssz, NUM_ATTENTION_DIM)

    # Final output projection O = attn_output_flat @ o_proj_weight^T
    # o_proj_weight: [NUM_ATTENTION_DIM, NUM_ATTENTION_DIM], bias=None
    O_final = torch.empty((Bsz, Ssz, NUM_ATTENTION_DIM), device=hidden_states.device, dtype=torch.float32)
    Wo = o_proj_weight
    B_o = torch.zeros(Wo.shape[0], device=hidden_states.device, dtype=torch.float32)
    grid_o = (Bsz * Ssz, (NUM_ATTENTION_DIM + 127) // 128)
    linear_kernel[grid_o](
        attn_output_flat, Wo, B_o, O_final,
        Bsz, Ssz, NUM_ATTENTION_DIM, NUM_ATTENTION_DIM,
        attn_output_flat.stride(0), attn_output_flat.stride(1), attn_output_flat.stride(2),
        Wo.stride(0), Wo.stride(1),
        O_final.stride(0), O_final.stride(1), O_final.stride(2),
        num_warps=4
    )

    return O_final


class ModelNew(nn.Module):
    def forward(self, hidden_states, q_proj_weight, q_proj_bias, k_proj_weight, k_proj_bias,
                v_proj_weight, v_proj_bias, o_proj_weight, q_norm_weight, k_norm_weight,
                cos, sin, rms_norm_eps):
        # Only Triton kernels; no torch matmul/softmax in forward
        Bsz = hidden_states.shape[0]
        Ssz = hidden_states.shape[1]
        # Ensure CUDA tensors
        hidden = hidden_states.to(torch.float32).contiguous()
        q_proj_weight = q_proj_weight.to(torch.float32).contiguous()
        k_proj_weight = k_proj_weight.to(torch.float32).contiguous()
        v_proj_weight = v_proj_weight.to(torch.float32).contiguous()
        o_proj_weight = o_proj_weight.to(torch.float32).contiguous()
        q_norm_weight = q_norm_weight.to(torch.float32).contiguous()
        k_norm_weight = k_norm_weight.to(torch.float32).contiguous()
        cos = cos.to(torch.float32).contiguous()  # [S, 64]
        sin = sin.to(torch.float32).contiguous()  # [S, 64]
        # Run Triton-only pipeline
        output = run_triton_only(
            hidden, q_proj_weight, q_proj_bias, k_proj_weight, k_proj_bias,
            v_proj_weight, v_proj_bias, o_proj_weight, q_norm_weight, k_norm_weight,
            cos, sin, rms_norm_eps, Bsz, Ssz
        )
        return output


def run(*args):
    return ModelNew()(*args)
