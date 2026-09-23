import torch
import triton
import triton.language as tl


# Linear projection: Y[M, N] = X[M, K] @ W[N, K]^T + bias[N]
@triton.jit
def linear_fwd_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wm, stride_wk,
    stride_b,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    num_warps: tl.constexpr, num_stages: tl.constexpr,
):
    pid_m = tl.program_id(0)  # tile over M
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(
            X_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk),
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )
        w = tl.load(
            W_ptr + (offs_n[None, :] * stride_wm + offs_k[:, None] * stride_wk),
            mask=(offs_n[None, :] < N) & (offs_k[:, None] < K),
            other=0.0,
        )
        acc += tl.dot(x, w)  # [BLOCK_M, BLOCK_N]

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=(offs_n < N), other=0.0)  # [BLOCK_N]
    acc += bias[None, :]  # broadcast over rows

    # Store
    tl.store(
        Y_ptr + (offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn),
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


# RMSNorm per row across last dimension: Y[i, :] = X[i, :] * rsqrt(mean(X[i, :]^2) + eps)
@triton.jit
def rmsnorm_kernel(
    X_ptr, W_ptr, Y_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_w,
    stride_ym, stride_yn,
    eps,  # float scalar
    BLOCK_N: tl.constexpr,
):
    i = tl.program_id(0)  # one program per row
    offs = tl.arange(0, BLOCK_N)
    x = tl.load(X_ptr + i * stride_xm + offs * stride_xn, mask=offs < N, other=0.0)
    x = x.to(tl.float32)
    mean_sq = tl.sum(x * x) / N
    scale = 1.0 / tl.sqrt(mean_sq + eps)
    y = x * scale  # apply per-row weight is identity (no weight provided in original)
    tl.store(Y_ptr + i * stride_ym + offs * stride_yn, y, mask=offs < N)


# Apply half rotation: split last 64 dims, rotate (q1, q2) -> (q2, -q1) with cos/sin on q2
@triton.jit
def apply_half_rotation_kernel(
    X_ptr, Cos_ptr, Sin_ptr, Y_ptr,
    M, N,  # N must be 128 here
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)  # N=128

    # Load X
    x = tl.load(X_ptr + offs_m[:, None] * stride_xm + offs_n[None, :] * stride_xn,
                mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0)

    # First 64 and second 64
    q1 = x[:, :64]
    q2 = x[:, 64:128]

    # Load cos/sin
    c = tl.load(Cos_ptr + tl.arange(0, 64), mask=None, other=0.0)  # [64]
    s = tl.load(Sin_ptr + tl.arange(0, 64), mask=None, other=0.0)  # [64]

    # Rotate q2: (q2, -q1) with c/s applied on q2
    # q2_rot = c*q2 - s*q1  (for rotation combining halves)
    q2_rot = q2 * c[None, :] - q1 * s[None, :]
    q1_rot = q2 * s[None, :] + q1 * c[None, :]

    y = tl.concatenate([q2_rot, q1_rot], axis=1)
    tl.store(Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
             y, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# Grouped Query Attention: expand KV heads to match Q heads (GQA mapping)
def gqa_expand(kv_states: torch.Tensor, num_key_value_groups: int) -> torch.Tensor:
    """
    kv_states: [B, S, H_k, D]
    Returns: [B, H_q, S, D] where H_q = num_attention_heads, each kv head is repeated
             to its corresponding q head group.
    """
    B, S, H_k, D = kv_states.shape
    H_q = kv_states.shape[2]  # In the given model, this is 96
    # kv_h = q_h // num_key_value_groups
    # Expand to Q heads and reshape
    kv_heads = kv_states.view(B, S, H_k, 1, D)  # [B, S, H_k, 1, D]
    # Repeat along H_q dimension
    # We need to broadcast kv_heads to [B, S, H_q, 1, D] where each kv head is repeated
    # Create index for repetition
    # torch.repeat_interleave does this efficiently
    kv_repeated = kv_heads.repeat_interleave(num_key_value_groups, dim=2)  # [B, S, H_q, 1, D]
    kv_q = kv_repeated.view(B, S, H_q, D)  # [B, S, H_q, D]
    return kv_q


# Attention kernel: for each (batch, query i), compute scores across all j, softmax, and accumulate with V
# Y_all[B, M, D] where M = S * H_q, D = head_dim
@triton.jit
def attention_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    B, M, D,
    stride_qb, stride_qm, stride_qd,
    stride_kb, stride_km, stride_kd,
    stride_vb, stride_vm, stride_vd,
    stride_ob, stride_om, stride_od,
    BLOCK_M: tl.constexpr,  # tile over query rows (M)
    BLOCK_D: tl.constexpr,  # tile over dim
    BLOCK_K: tl.constexpr,  # reduction tile over K dimension
    num_warps: tl.constexpr, num_stages: tl.constexpr,
):
    # One program per (batch, query position) pair
    pid_b = tl.program_id(0)  # batch index
    pid_m = tl.program_id(1)  # query row index in M
    i = pid_m  # query position within this batch's sequence
    # Compute base pointers for this batch
    Q_base = Q_ptr + pid_b * stride_qb
    K_base = K_ptr + pid_b * stride_kb
    V_base = V_ptr + pid_b * stride_vb
    O_base = Out_ptr + pid_b * stride_ob

    # Initialize logits [D]
    logits = tl.zeros((D,), dtype=tl.float32)
    # Compute Q row i across all columns (K dimension == D)
    for k0 in range(0, D, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        q = tl.load(Q_base + i * stride_qm + offs_k * stride_qd,
                    mask=offs_k < D, other=0.0)  # [BLOCK_K]
        # k_cols are all j positions; load K rows j
        for j0 in range(0, M, BLOCK_M):
            offs_j = j0 + tl.arange(0, BLOCK_M)
            k = tl.load(K_base + offs_j * stride_km + offs_k * stride_kd,
                        mask=(offs_j < M) & (offs_k < D), other=0.0)  # [BLOCK_M, BLOCK_K]
            # logits += sum over BLOCK_K dims of q[k] * k[j] along k axis
            # We need to make q a [1, BLOCK_K] and dot with k to get [BLOCK_M]
            q_ext = q[None, :]  # [1, BLOCK_K]
            contrib = tl.dot(q_ext, k)[0, :]  # [BLOCK_M]
            logits += contrib  # accumulate over k0

    scaling = 1.0 / tl.sqrt(D)
    logits = logits * scaling

    # Apply causal mask: j >= i -> -inf
    for j0 in range(0, M, BLOCK_M):
        offs_j = j0 + tl.arange(0, BLOCK_M)
        # Compute pairwise logits for this tile
        soft = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for k0 in range(0, D, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            q = tl.load(Q_base + i * stride_qm + offs_k * stride_qd,
                        mask=offs_k < D, other=0.0)  # [BLOCK_K]
            k = tl.load(K_base + offs_j * stride_km + offs_k * stride_kd,
                        mask=(offs_j[None, :] < M) & (offs_k[:, None] < D), other=0.0)  # [BLOCK_M, BLOCK_K]
            q_ext = q[None, :]  # [1, BLOCK_K]
            contrib = tl.dot(q_ext, k)[0, :]  # [BLOCK_M]
            soft += contrib

        # Softmax across offs_j
        soft = soft * scaling
        maxv = tl.max(soft, axis=0)
        soft = soft - maxv
        soft = tl.exp(soft)
        sumv = tl.sum(soft, axis=0)
        soft = soft / sumv

        # Apply causal
        causal = (offs_j[None, :] >= i)  # [1, BLOCK_M] broadcast
        soft = tl.where(causal, soft, -float('inf'))

        # Accumulate output with V
        for k0 in range(0, D, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            v = tl.load(V_base + offs_j * stride_vm + offs_k * stride_vd,
                        mask=(offs_j < M) & (offs_k < D), other=0.0)  # [BLOCK_M, BLOCK_K]
            # out[i] += sum_j soft[j] * v[j, :]
            # soft is [BLOCK_M], v is [BLOCK_M, BLOCK_K]
            # compute outer sum per block
            out_block = tl.zeros((BLOCK_K,), dtype=tl.float32)
            for j in range(0, BLOCK_M):
                # soft[j] * v[j, :]
                # v[j, :] is [BLOCK_K]
                out_block += soft[j] * v[j, :]
            # store into Out at row i
            tl.store(O_base + i * stride_om + offs_k * stride_od,
                     out_block, mask=offs_k < D)


# Final output projection: Y[M, OUT_N] = Attn[M, K] @ OUT_W[OUT_N, K]^T (no bias)
@triton.jit
def linear_out_kernel(
    Attn_ptr, OUT_W_ptr, OUT_ptr,
    M, IN_N, OUT_N,
    stride_am, stride_an,
    stride_wm, stride_wn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    num_warps: tl.constexpr, num_stages: tl.constexpr,
):
    pid_m = tl.program_id(0)  # tile over M
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, IN_N, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(
            Attn_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_an),
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < IN_N),
            other=0.0,
        )
        w = tl.load(
            OUT_W_ptr + (offs_n[None, :] * stride_wm + offs_k[:, None] * stride_wn),
            mask=(offs_n[None, :] < OUT_N) & (offs_k[:, None] < IN_N),
            other=0.0,
        )
        acc += tl.dot(a, w)  # [BLOCK_M, BLOCK_N]

    tl.store(
        OUT_ptr + (offs_m[:, None] * stride_om + offs_n[None, :] * stride_on),
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < OUT_N),
    )


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, q_proj_weight: torch.Tensor, q_proj_bias: torch.Tensor,
                k_proj_weight: torch.Tensor, k_proj_bias: torch.Tensor, v_proj_weight: torch.Tensor, v_proj_bias: torch.Tensor,
                o_proj_weight: torch.Tensor, q_norm_weight: torch.Tensor, k_norm_weight: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor, rms_norm_eps: float):
        # Ensure CUDA and float32 for Triton kernels (can cast inputs if needed)
        device = hidden_states.device
        # 1) Q, K, V via linear
        B, S, D = hidden_states.shape  # S is seq_len, D is head_dim
        assert D == 128, "Head dim must be 128"
        # Q
        q = torch.empty((B, S, D), device=device, dtype=torch.float32)
        linear_fwd_kernel[(B, S)](
            hidden_states, q_proj_weight, q_proj_bias, q,
            B, D, D,  # M=S, N=D, K=D
            hidden_states.stride(0), hidden_states.stride(2),
            q_proj_weight.stride(0), q_proj_weight.stride(1),
            0,  # bias stride is 1 element; we pass bias directly
            q.stride(0), q.stride(2),
            64, 64, 64,
            4, 3,
        )
        # K
        k = torch.empty((B, S, D), device=device, dtype=torch.float32)
        linear_fwd_kernel[(B, S)](
            hidden_states, k_proj_weight, k_proj_bias, k,
            B, D, D,
            hidden_states.stride(0), hidden_states.stride(2),
            k_proj_weight.stride(0), k_proj_weight.stride(1),
            0,
            k.stride(0), k.stride(2),
            64, 64, 64,
            4, 3,
        )
        # V
        v = torch.empty((B, S, D), device=device, dtype=torch.float32)
        linear_fwd_kernel[(B, S)](
            hidden_states, v_proj_weight, v_proj_bias, v,
            B, D, D,
            hidden_states.stride(0), hidden_states.stride(2),
            v_proj_weight.stride(0), v_proj_weight.stride(1),
            0,
            v.stride(0), v.stride(2),
            64, 64, 64,
            4, 3,
        )

        # 2) RMSNorm for Q and K
        q_norm = torch.empty_like(q)
        rmsnorm_kernel[(B * S,)](
            q, q_norm_weight, q_norm,
            B * S, D,
            q.stride(0), D,  # q.stride(2) is 1 for [B,S,D] contiguous; using D stride for row is fine because we access row by i=row index
            0,  # q_norm_weight is 1D
            q_norm.stride(0), D,
            rms_norm_eps,
            128,
        )
        k_norm = torch.empty_like(k)
        rmsnorm_kernel[(B * S,)](
            k, k_norm_weight, k_norm,
            B * S, D,
            k.stride(0), D,
            0,
            k_norm.stride(0), D,
            rms_norm_eps,
            128,
        )

        # 3) Half rotation on Q and K
        q_rot = torch.empty_like(q_norm)
        apply_half_rotation_kernel[(B * S,)](
            q_norm, cos, sin, q_rot,
            B * S, D,
            D, D,  # both strides over last dim
            q_rot.stride(0), q_rot.stride(2),
            64, 128,
            4, 3,
        )
        k_rot = torch.empty_like(k_norm)
        apply_half_rotation_kernel[(B * S,)](
            k_norm, cos, sin, k_rot,
            B * S, D,
            D, D,
            k_rot.stride(0), k_rot.stride(2),
            64, 128,
            4, 3,
        )

        # Reshape for attention: [B, H, S, D]
        # Since num_key_value_heads=8 and num_attention_heads=96, we expand K/V to 96 heads
        # Original code uses num_key_value_groups=12 -> H_q = 96, H_k = 8
        H_q = 96
        H_k = 8
        k_rot_exp = gqa_expand(k_rot, num_key_value_groups=12)  # [B, S, 96, D]
        v_exp = gqa_expand(v, num_key_value_groups=12)          # [B, S, 96, D]

        # For Q and K rotation, the reference code rotates only Q and K. V is not rotated in the original code.
        # We proceed to compute attention using q_rot and k_rot (as rotated Q/K) and v_exp as the expanded V.

        # 4) Prepare for attention: out shape [B, M, D], where M = S * H_q
        M = S * H_q
        q_rot_trans = q_rot.view(B, S, H_q, D)                  # [B, S, 96, D]
        k_rot_exp = k_rot_exp                                  # [B, S, 96, D]
        v_exp = v_exp                                          # [B, S, 96, D]

        # Flatten to [B, M, D] for kernel: we need contiguous [B, M, D]
        q_flat = q_rot_trans.view(B, M, D)
        k_flat = k_rot_exp.view(B, M, D)
        v_flat = v_exp.view(B, M, D)

        # Allocate output attention [B, M, D]
        attn_out = torch.empty((B, M, D), device=device, dtype=torch.float32)

        # Launch attention kernel: one program per (batch, row)
        attention_kernel[(B, M)](
            q_flat, k_flat, v_flat, attn_out,
            B, M, D,
            q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
            k_flat.stride(0), k_flat.stride(1), k_flat.stride(2),
            v_flat.stride(0), v_flat.stride(1), v_flat.stride(2),
            attn_out.stride(0), attn_out.stride(1), attn_out.stride(2),
            1,  # BLOCK_M = 1
            128,  # BLOCK_D = 128
            32,  # BLOCK_K = 32 (tile over D for reduction)
            4, 3,
        )

        # 5) Final output projection: attn_out[M, D] @ o_proj_weight[OUT_N, D]^T
        OUT_N = D * H_q  # original code multiplies by head_dim; here H_q*D == 96*128=12288
        out = torch.empty((B, M, OUT_N), device=device, dtype=torch.float32)

        # We need to reshape attn_out to [M, D] where M=B*S*H_q
        attn_mat = attn_out.reshape(M, D)  # [M, D]
        OUT_W = o_proj_weight  # [OUT_N, D]
        linear_out_kernel[(1,)](
            attn_mat, OUT_W, out,
            M, D, OUT_N,
            attn_mat.stride(0), D,
            OUT_W.stride(0), OUT_W.stride(1),
            out.stride(0), OUT_N,
            64, 128, 64,
            4, 3,
        )

        # Reshape to original [B, S, H_q*D]
        out = out.view(B, S, H_q * D)
        return out


def run(*args):
    return ModelNew()(*args)
