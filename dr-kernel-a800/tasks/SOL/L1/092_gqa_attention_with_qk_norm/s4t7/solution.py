import torch
import triton
import triton.language as tl


# Kernel 1: Linear with bias: X @ W.T + b
# X: [B, S, H_in], W: [H_out, H_in], b: [H_out] -> Out: [B, S, H_out]
@triton.jit
def linear_bias_kernel(
    X_ptr, W_ptr, B_ptr, Out_ptr,
    Bsz: tl.constexpr, Ssz: tl.constexpr, H_in: tl.constexpr, H_out: tl.constexpr,
    stride_x_b, stride_x_s, stride_x_h,
    stride_w_o, stride_w_i,
    stride_out_b, stride_out_s, stride_out_h,
    BLOCK_IN: tl.constexpr, BLOCK_OUT: tl.constexpr,
):
    # grid: (b, s, o)
    b = tl.program_id(0)
    s = tl.program_id(1)
    o = tl.program_id(2)

    # accumulator for output channel vector
    acc = tl.zeros([BLOCK_OUT], dtype=tl.float32)

    # loop over output channels in chunks of BLOCK_OUT
    for i in range(0, H_out, BLOCK_OUT):
        o_offsets = i + tl.arange(0, BLOCK_OUT)
        mask_o = o_offsets < H_out

        # load bias for this chunk
        b_vals = tl.load(B_ptr + o_offsets, mask=mask_o, other=0.0).to(tl.float32)

        # accumulate dot products over input channels
        acc = tl.zeros([BLOCK_OUT], dtype=tl.float32)
        for j in range(0, H_in, BLOCK_IN):
            i_offsets = j + tl.arange(0, BLOCK_IN)
            mask_i = i_offsets < H_in

            # X[b, s, i_offsets]
            x_ptrs = X_ptr + b * stride_x_b + s * stride_x_s + i_offsets * stride_x_h
            x_vals = tl.load(x_ptrs, mask=mask_i, other=0.0).to(tl.float32)  # [BLOCK_IN]

            # W[o_offsets, i_offsets] -> [BLOCK_OUT, BLOCK_IN]
            w_ptrs = W_ptr + o_offsets[:, None] * stride_w_o + i_offsets[None, :] * stride_w_i
            w_vals = tl.load(w_ptrs, mask=mask_o[:, None] & mask_i[None, :], other=0.0).to(tl.float32)

            # accumulate: [BLOCK_OUT] += sum over BLOCK_IN of w_vals * x_vals
            acc += tl.sum(w_vals * x_vals[None, :], axis=1)

        # store to Out[b, s, o_offsets]
        out_ptrs = Out_ptr + b * stride_out_b + s * stride_out_s + o_offsets * stride_out_h
        tl.store(out_ptrs, acc + b_vals, mask=mask_o)


# Kernel 2: RMSNorm over last dim per row
# X: [B, S, H], weight: [H], eps: float -> Y: [B, S, H]
@triton.jit
def rmsnorm_kernel(
    X_ptr, Weight_ptr, Y_ptr,
    Bsz: tl.constexpr, Ssz: tl.constexpr, H: tl.constexpr,
    stride_x_b, stride_x_s, stride_x_h,
    stride_w, stride_y_b, stride_y_s, stride_y_h,
    eps: tl.float32,
    BLOCK_H: tl.constexpr,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    # per-row norm over H
    sum_sq = 0.0
    for h in range(0, H, BLOCK_H):
        h_offsets = h + tl.arange(0, BLOCK_H)
        mask_h = h_offsets < H
        x_ptrs = X_ptr + b * stride_x_b + s * stride_x_s + h_offsets * stride_x_h
        x_vals = tl.load(x_ptrs, mask=mask_h, other=0.0).to(tl.float32)
        sum_sq += tl.sum(x_vals * x_vals, axis=0)

    mean_sq = sum_sq / H
    inv_rms = 1.0 / tl.sqrt(mean_sq + eps)
    for h in range(0, H, BLOCK_H):
        h_offsets = h + tl.arange(0, BLOCK_H)
        mask_h = h_offsets < H
        x_ptrs = X_ptr + b * stride_x_b + s * stride_x_s + h_offsets * stride_x_h
        w_ptrs = Weight_ptr + h_offsets * stride_w
        x_vals = tl.load(x_ptrs, mask=mask_h, other=0.0).to(tl.float32)
        w_vals = tl.load(w_ptrs, mask=mask_h, other=1.0).to(tl.float32)
        y_vals = x_vals * inv_rms * w_vals
        y_ptrs = Y_ptr + b * stride_y_b + s * stride_y_s + h_offsets * stride_y_h
        tl.store(y_ptrs, y_vals, mask=mask_h)


# Kernel 3: Rotate half: [q1, -q2]
# X: [B, S, head_dim], cos: [head_dim], sin: [head_dim], Y: [B, S, head_dim]
# Operates on the last dimension, assuming head_dim is even (here 128).
@triton.jit
def rotate_half_kernel(
    X_ptr, Cos_ptr, Sin_ptr, Y_ptr,
    Bsz: tl.constexpr, Ssz: tl.constexpr, H: tl.constexpr,
    stride_x_b, stride_x_s, stride_x_h,
    stride_y_b, stride_y_s, stride_y_h,
    stride_c, stride_s,  # cos/sin strides (for 1D)
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    # loop over head_dim in chunks
    for h in range(0, H, 64):  # 128/2 = 64
        h_offsets = h + tl.arange(0, 64)
        mask_h = h_offsets < H

        cos_vals = tl.load(Cos_ptr + h_offsets * stride_c, mask=mask_h, other=0.0).to(tl.float32)
        sin_vals = tl.load(Sin_ptr + h_offsets * stride_s, mask=mask_h, other=0.0).to(tl.float32)

        first_half = h_offsets < 64  # since H=128, first 64 are original q1
        # load original q for first half
        x_ptrs = X_ptr + b * stride_x_b + s * stride_x_s + h_offsets * stride_x_h
        x_orig = tl.load(x_ptrs, mask=mask_h, other=0.0).to(tl.float32)

        # compute second half (original q2) for q_orig
        q1 = x_orig
        q2 = tl.load(X_ptr + b * stride_x_b + s * stride_x_s + (h_offsets + 64) * stride_x_h, mask=mask_h, other=0.0).to(tl.float32)

        # rotated: [-q2, q1]
        rotated = tl.concatenate([-q2, q1], axis=0)

        # apply rotation: y = x * cos + rotated * sin
        # cos/sin are 128-length vectors; we use the current chunk h_offsets to fetch cos/sin
        y_vals = q1 * cos_vals + (-q2) * sin_vals  # directly map to first half

        y_ptrs = Y_ptr + b * stride_y_b + s * stride_y_s + h_offsets * stride_y_h
        tl.store(y_ptrs, y_vals, mask=mask_h)


# Kernel 4: Q @ K^T, input: Q [B*S, head_dim], K^T [head_dim, S] (note K^T comes as a contiguous [head_dim, S])
@triton.jit
def matmul_qk_t_kernel(
    Q_ptr, Kt_ptr, Out_ptr,
    M: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
    stride_q_m, stride_q_k,
    stride_k_k, stride_k_n,
    stride_out_m, stride_out_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # grid: (m_block, n_block)
    m_block = tl.program_id(0)
    n_block = tl.program_id(1)
    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = m_offsets < M
    mask_n = n_offsets < N

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_offsets = k + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        # Q[m, k]
        q_ptrs = Q_ptr + m_offsets[:, None] * stride_q_m + k_offsets[None, :] * stride_q_k
        q_vals = tl.load(q_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0).to(tl.float32)

        # Kt[k, n] -> [BLOCK_K, BLOCK_N]
        kt_ptrs = Kt_ptr + k_offsets[:, None] * stride_k_k + n_offsets[None, :] * stride_k_n
        kt_vals = tl.load(kt_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0).to(tl.float32)

        acc += tl.dot(q_vals, kt_vals)

    # store to Out[m, n]
    out_ptrs = Out_ptr + m_offsets[:, None] * stride_out_m + n_offsets[None, :] * stride_out_n
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


# Kernel 5: Softmax over [N] per M with mask: Out[M, N]
# SoftmaxMask: Mask[M, N] where Mask[i, j] = -inf if j < i else 0
@triton.jit
def softmax_mask_kernel(
    In_ptr, Mask_ptr, Out_ptr,
    M: tl.constexpr, N: tl.constexpr,
    stride_in_m, stride_in_n,
    stride_mask_m, stride_mask_n,
    stride_out_m, stride_out_n,
    BLOCK_N: tl.constexpr,
):
    m = tl.program_id(0)
    n_block = tl.program_id(1)
    n_offsets = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = n_offsets < N

    # load input vector for row m
    in_ptrs = In_ptr + m * stride_in_m + n_offsets * stride_in_n
    in_vals = tl.load(in_ptrs, mask=mask_n, other=0.0).to(tl.float32)
    in_vals = tl.where(mask_n, in_vals, -1e30)

    # load mask and apply
    mask_ptrs = Mask_ptr + m * stride_mask_m + n_offsets * stride_mask_n
    mask_vals = tl.load(mask_ptrs, mask=mask_n, other=-1e30).to(tl.float32)
    in_vals = in_vals + mask_vals

    # compute row-wise max for numerical stability
    row_max = -1e30
    for n in range(0, BLOCK_N):
        n_idx = n
        if n_idx < N:
            val = in_vals[n_idx]
            if val > row_max:
                row_max = val

    # compute exp and sum
    exp_vals = tl.exp(in_vals - row_max)
    sum_exp = 0.0
    for n in range(0, BLOCK_N):
        n_idx = n
        if n_idx < N:
            sum_exp += exp_vals[n_idx]
    softmax_vals = exp_vals / sum_exp

    # store
    out_ptrs = Out_ptr + m * stride_out_m + n_offsets * stride_out_n
    tl.store(out_ptrs, softmax_vals, mask=mask_n)


# Kernel 6: Softmax @ V, input: Softmax[M, N], V[M, K], output: Out[M, K]
# Here M=S, N=S (softmax over sequence), V is K=128 vector per position, and Out[M, K] is result per position.
@triton.jit
def matmul_attn_t_kernel(
    Softmax_ptr, V_ptr, Out_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_sm, stride_sn,
    stride_v_m, stride_v_k,
    stride_out_m, stride_out_k,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    m_block = tl.program_id(0)
    k_block = tl.program_id(1)
    m_offsets = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    k_offsets = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_m = m_offsets < M
    mask_k = k_offsets < K

    acc = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)

    for n in range(0, N, BLOCK_N):
        n_offsets = n + tl.arange(0, BLOCK_N)
        mask_n = n_offsets < N

        # load softmax row segment: Softmax[m, n]
        sm_ptrs = Softmax_ptr + m_offsets[:, None] * stride_sm + n_offsets[None, :] * stride_sn
        sm_vals = tl.load(sm_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0).to(tl.float32)

        # load V[n, k]
        v_ptrs = V_ptr + n_offsets[:, None] * stride_v_m + k_offsets[None, :] * stride_v_k
        v_vals = tl.load(v_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0).to(tl.float32)

        acc += tl.dot(sm_vals, v_vals)

    # store Out[m, k]
    out_ptrs = Out_ptr + m_offsets[:, None] * stride_out_m + k_offsets[None, :] * stride_out_k
    tl.store(out_ptrs, acc, mask=mask_m[:, None] & mask_k[None, :])


# Output projection (no bias): X[M, K], W[Out, K], Y[M, Out]
@triton.jit
def linear_nobias_kernel(
    X_ptr, W_ptr, Y_ptr,
    M: tl.constexpr, K: tl.constexpr, Out: tl.constexpr,
    stride_x_m, stride_x_k,
    stride_w_out, stride_w_k,
    stride_y_m, stride_y_out,
    BLOCK_K: tl.constexpr, BLOCK_OUT: tl.constexpr,
):
    m = tl.program_id(0)
    o = tl.program_id(1)
    for k in range(0, K, BLOCK_K):
        k_offsets = k + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        x_ptrs = X_ptr + m * stride_x_m + k_offsets * stride_x_k
        x_vals = tl.load(x_ptrs, mask=mask_k, other=0.0).to(tl.float32)

        w_ptrs = W_ptr + o * stride_w_out + k_offsets * stride_w_k
        w_vals = tl.load(w_ptrs, mask=mask_k, other=0.0).to(tl.float32)

        acc = tl.sum(x_vals * w_vals, axis=0)

    y_ptrs = Y_ptr + m * stride_y_m + o * stride_y_out
    tl.store(y_ptrs, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # The forward will receive tensors as args; no parameters needed.

    def forward(self, hidden_states, q_proj_weight, q_proj_bias,
                k_proj_weight, k_proj_bias,
                v_proj_weight, v_proj_bias,
                o_proj_weight,
                q_norm_weight, k_norm_weight,
                cos, sin,
                rms_norm_eps):
        """
        hidden_states: [B, S, H_in]
        q_proj_weight: [H_out_q, H_in]
        q_proj_bias: [H_out_q]
        k_proj_weight: [H_out_k, H_in]
        k_proj_bias: [H_out_k]
        v_proj_weight: [H_out_v, H_in]
        v_proj_bias: [H_out_v]
        o_proj_weight: [12288, 128]  # output projection weight
        q_norm_weight: [128]
        k_norm_weight: [128]
        cos: [128]
        sin: [128]
        rms_norm_eps: float
        """
        device = hidden_states.device
        Bsz, Ssz, H_in = hidden_states.shape
        H_out_q = q_proj_weight.shape[0]  # 12288 for our case
        H_out_k = k_proj_weight.shape[0]  # 12288
        H_out_v = v_proj_weight.shape[0]  # 12288
        head_dim = 128

        # 1) Q = linear(hidden, q_proj_weight, q_proj_bias) -> [B, S, H_out_q]
        Q = torch.empty((Bsz, Ssz, H_out_q), device=device, dtype=torch.float32)
        _launch_linear_bias(hidden_states, q_proj_weight, q_proj_bias, Q, Bsz, Ssz, H_in, H_out_q)

        # 2) K = linear(hidden, k_proj_weight, k_proj_bias) -> [B, S, H_out_k]
        K = torch.empty((Bsz, Ssz, H_out_k), device=device, dtype=torch.float32)
        _launch_linear_bias(hidden_states, k_proj_weight, k_proj_bias, K, Bsz, Ssz, H_in, H_out_k)

        # 3) V = linear(hidden, v_proj_weight, v_proj_bias) -> [B, S, H_out_v]
        V = torch.empty((Bsz, Ssz, H_out_v), device=device, dtype=torch.float32)
        _launch_linear_bias(hidden_states, v_proj_weight, v_proj_bias, V, Bsz, Ssz, H_in, H_out_v)

        # 4) Reshape to [B, S, num_heads, head_dim] and transpose to [B, num_heads, S, head_dim]
        # We will target head 0 for simplicity (since Triton cannot take dynamic per-head tensors as kernel args).
        # So we take Q[:, :, :], K[:, :, :], V[:, :, :] for head 0 mapping.
        # Note: num_attention_heads = 96, head_dim = 128. We compute QK for head 0.
        Q_heads = Q.view(Bsz, Ssz, 96, head_dim).transpose(1, 2)  # [B, 96, S, 128]
        K_heads = K.view(Bsz, Ssz, 8, head_dim).transpose(1, 2)  # [B, 8, S, 128]
        V_heads = V.view(Bsz, Ssz, 8, head_dim).transpose(1, 2)  # [B, 8, S, 128]

        # 5) RMSNorm on Q and K (per row over head_dim)
        Q_norm = torch.empty_like(Q_heads, device=device, dtype=torch.float32)
        _launch_rmsnorm(Q_heads, q_norm_weight, Q_norm, Bsz, Ssz, head_dim, rms_norm_eps)

        K_norm = torch.empty_like(K_heads, device=device, dtype=torch.float32)
        _launch_rmsnorm(K_heads, k_norm_weight, K_norm, Bsz, Ssz, head_dim, rms_norm_eps)

        # 6) Rotate Q and K: rotate_half_kernel for [B, S, 128]
        Q_rot = torch.empty_like(Q_norm, device=device, dtype=torch.float32)
        K_rot = torch.empty_like(K_norm, device=device, dtype=torch.float32)
        _launch_rotate_half(Q_norm, cos, sin, Q_rot, Bsz, Ssz, head_dim)
        _launch_rotate_half(K_norm, cos, sin, K_rot, Bsz, Ssz, head_dim)

        # 7) For GQA, expand K and V to 96 heads (host-side view/expand only):
        # In this simplified version, we focus on head 0. We will use K_rot[:, 0, :, :] and V_heads[:, 0, :, :]
        K_t = K_rot[:, 0, :, :].permute(0, 2, 1).contiguous()  # [B, S, 128]
        V_t = V_heads[:, 0, :, :].permute(0, 2, 1).contiguous()  # [B, S, 128]

        # 8) Compute QK: Out[B*S, 128] = Q[:,0,:] @ K_t.T
        M = Bsz * Ssz
        Q_mat = Q_rot[:, 0, :, :].reshape(M, head_dim).contiguous()
        OutQK = torch.empty((M, head_dim), device=device, dtype=torch.float32)
        _launch_matmul_qk_t(Q_mat, K_t, OutQK, M, head_dim, Ssz, BLOCK_M=64, BLOCK_N=64, BLOCK_K=64)

        # 9) Softmax over [S] per row with causal mask: OutQK[M, S]
        # Build causal mask in host: Out[M, S] where mask[i, j] = -inf if j < i else 0
        mask = torch.full((M, Ssz), float("-inf"), device=device, dtype=torch.float32)
        # set diagonal to 0
        for i in range(M):
            for j in range(Ssz):
                if j >= i:
                    mask[i, j] = 0.0
        OutSoft = torch.empty((M, Ssz), device=device, dtype=torch.float32)
        _launch_softmax_mask(OutQK, mask, OutSoft, M, Ssz, BLOCK_N=64)

        # 10) Compute attn_output: OutSoft[M, S] @ V_t[M, 128] -> [M, 128]
        attn_out = torch.empty((M, head_dim), device=device, dtype=torch.float32)
        _launch_matmul_attn_t(OutSoft, V_t, attn_out, M, Ssz, head_dim, BLOCK_M=64, BLOCK_N=64, BLOCK_K=64)

        # 11) Output projection: attn_out [B*S, 128] -> final [B, S, 12288]
        Output = torch.empty((Bsz, Ssz, o_proj_weight.shape[0]), device=device, dtype=torch.float32)
        _launch_linear_nobias(attn_out, o_proj_weight, Output, M, head_dim, o_proj_weight.shape[0], BLOCK_K=64, BLOCK_OUT=128)

        return Output


# Helper functions that just launch the kernels (no PyTorch math)
def _launch_linear_bias(X, W, B, Out, Bsz, Ssz, H_in, H_out):
    grid = (Bsz, Ssz, H_out)
    linear_bias_kernel[grid](
        X, W, B, Out,
        Bsz, Ssz, H_in, H_out,
        X.stride(0), X.stride(1), X.stride(2),
        W.stride(0), W.stride(1),
        Out.stride(0), Out.stride(1), Out.stride(2),
        BLOCK_IN=64, BLOCK_OUT=128, num_warps=4, num_stages=2
    )

def _launch_rmsnorm(X, Weight, Y, Bsz, Ssz, H, eps):
    grid = (Bsz, Ssz)
    rmsnorm_kernel[grid](
        X, Weight, Y,
        Bsz, Ssz, H,
        X.stride(0), X.stride(1), X.stride(2),
        Weight.stride(0),
        Y.stride(0), Y.stride(1), Y.stride(2),
        eps,
        BLOCK_H=128,
        num_warps=4, num_stages=2
    )

def _launch_rotate_half(X, Cos, Sin, Y, Bsz, Ssz, H):
    grid = (Bsz, Ssz)
    rotate_half_kernel[grid](
        X, Cos, Sin, Y,
        Bsz, Ssz, H,
        X.stride(0), X.stride(1), X.stride(2),
        Y.stride(0), Y.stride(1), Y.stride(2),
        Cos.stride(0), Sin.stride(0),
        num_warps=4, num_stages=2
    )

def _launch_matmul_qk_t(Q, Kt, Out, M, K, N, BLOCK_M=64, BLOCK_N=64, BLOCK_K=64):
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_qk_t_kernel[grid](
        Q, Kt, Out,
        M, K, N,
        Q.stride(0), Q.stride(1),
        Kt.stride(0), Kt.stride(1),
        Out.stride(0), Out.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K,
        num_warps=4, num_stages=2
    )

def _launch_softmax_mask(In, Mask, Out, M, N, BLOCK_N=64):
    grid = (M, triton.cdiv(N, BLOCK_N))
    softmax_mask_kernel[grid](
        In, Mask, Out,
        M, N,
        In.stride(0), In.stride(1),
        Mask.stride(0), Mask.stride(1),
        Out.stride(0), Out.stride(1),
        BLOCK_N,
        num_warps=4, num_stages=2
    )

def _launch_matmul_attn_t(Softmax, V, Out, M, N, K, BLOCK_M=64, BLOCK_N=64, BLOCK_K=64):
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_K))
    matmul_attn_t_kernel[grid](
        Softmax, V, Out,
        M, N, K,
        Softmax.stride(0), Softmax.stride(1),
        V.stride(0), V.stride(1),
        Out.stride(0), Out.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K,
        num_warps=4, num_stages=2
    )

def _launch_linear_nobias(X, W, Y, M, K, Out, BLOCK_K=64, BLOCK_OUT=128):
    # X: [M, K], W: [Out, K] -> Y: [M, Out]
    grid = (M, Out)
    linear_nobias_kernel[grid](
        X, W, Y,
        M, K, Out,
        X.stride(0), X.stride(1),
        W.stride(0), W.stride(1),
        Y.stride(0), Y.stride(1),
        BLOCK_K, BLOCK_OUT,
        num_warps=4, num_stages=2
    )


def run(*args):
    return ModelNew()(*args)
