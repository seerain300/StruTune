import torch
import triton
import triton.language as tl


# Triton GEMM for F.linear without bias: Y[M, N] = X[M, K] @ W[N, K]^T
# Here X has shape [B*S, H], W has shape [N, K] where N is output features (H), K is input features (H).
@triton.jit
def linear_no_bias_kernel(
    X_ptr, W_ptr, Y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Load A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = X_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)

        # Load B tile as [BLOCK_K, BLOCK_N] to use tl.dot: B is W with shape [N, K]
        b_ptrs = W_ptr + (offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk)
        b = tl.load(b_ptrs, mask=(offs_n[None, :] < N) & (offs_k[:, None] < K), other=0.0)

        # Accumulate
        acc += tl.dot(a, b)

    # Store result
    y_ptrs = Y_ptr + (offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn)
    tl.store(y_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# Triton per-head RMSNorm: normalize over last dimension (D) and scale by per-head weight
# Input: X [B, S, D], weight [H, D], output [B, S, D], epsilon scalar
@triton.jit
def rmsnorm_heads_kernel(
    X_ptr, weight_ptr, Y_ptr,
    B, S, D, H,
    stride_xb, stride_xs, stride_xd,
    stride_w_h, stride_w_d,
    stride_yb, stride_ys, stride_yd,
    eps,  # float32
    BLOCK_D: tl.constexpr,
):
    # One program per (b, s)
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S

    if b >= B or s >= S:
        return

    # Loop over heads; for each head h, normalize and scale
    for h in range(0, H):
        # Accumulate sum of squares over D
        sumsq = 0.0
        for d0 in range(0, D, BLOCK_D):
            offs_d = d0 + tl.arange(0, BLOCK_D)
            x = tl.load(X_ptr + b * stride_xb + s * stride_xs + offs_d * stride_xd,
                        mask=offs_d < D, other=0.0)
            sumsq += tl.sum(x * x, axis=0)

        mean = sumsq / D
        inv_rms = 1.0 / tl.sqrt(mean + eps)
        scale = tl.load(weight_ptr + h * stride_w_h + tl.arange(0, D) * stride_w_d,
                        mask=tl.arange(0, D) < D, other=0.0) * inv_rms

        for d0 in range(0, D, BLOCK_D):
            offs_d = d0 + tl.arange(0, BLOCK_D)
            x = tl.load(X_ptr + b * stride_xb + s * stride_xs + offs_d * stride_xd,
                        mask=offs_d < D, other=0.0)
            y = x * scale[offs_d]
            tl.store(Y_ptr + b * stride_yb + s * stride_ys + offs_d * stride_yd,
                     y, mask=offs_d < D)


# Triton rotation for query/key: split 128 -> 64 halves, rotate and apply cos/sin, write back to output
@triton.jit
def rotate_half_heads_kernel(
    X_ptr, cos_ptr, sin_ptr, Y_ptr,
    B, S, D,
    stride_xb, stride_xs, stride_xd,
    stride_yb, stride_ys, stride_yd,
    BLOCK_D: tl.constexpr,
):
    # Launch per (b, s)
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S

    if b >= B or s >= S:
        return

    # For query/key heads with D=128, split halves
    half = D // 2
    # Load cos and sin vectors
    cos = tl.load(cos_ptr + tl.arange(0, half))
    sin = tl.load(sin_ptr + tl.arange(0, half))

    for d0 in range(0, D, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        mask = offs_d < D
        x = tl.load(X_ptr + b * stride_xb + s * stride_xs + offs_d * stride_xd, mask=mask, other=0.0)
        q1 = x[:half]
        q2 = x[half:]
        # Rotate: q1 <- -q2, q2 <- q1
        q_rot = tl.cat([-q2, q1], axis=0)
        # Apply rotation
        y = q_rot * cos + x * sin
        tl.store(Y_ptr + b * stride_yb + s * stride_ys + offs_d * stride_yd, y, mask=mask)


# Triton kernel to expand key/value from num_key_value_heads to num_attention_heads via replication (num_key_value_groups)
# Input: K [B, KVH, S, D], Output: Ke [B, H, S, D], where KVH=8, H=96, groups=12 (KVH=H/12)
@triton.jit
def expand_kv_heads_kernel(
    K_ptr, Ke_ptr,
    B, S, D, KVH, H, GROUPS,
    stride_kb, stride_kv, stride_ks, stride_kd,
    stride_keb, stride_keh, stride_kes, stride_ked,
    BLOCK_D: tl.constexpr,
):
    # Launch per (b, s)
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S

    if b >= B or s >= S:
        return

    # For each original head k in 0..KVH-1, replicate to H positions: h = k * GROUPS + g, g in 0..GROUPS-1
    for k in range(0, KVH):
        for g in range(0, GROUPS):
            h = k * GROUPS + g
            x = tl.load(K_ptr + b * stride_kb + k * stride_kv + s * stride_ks + tl.arange(0, D) * stride_kd,
                        mask=tl.arange(0, D) < D, other=0.0)
            tl.store(Ke_ptr + b * stride_keb + h * stride_keh + s * stride_kes + tl.arange(0, D) * stride_ked,
                     x, mask=tl.arange(0, D) < D)


# Triton kernel: compute attention scores per (b, h) across sequence axis s for each sequence position
# Inputs:
#   Q [B, H, S, D], Ke [B, H, S, D] (expanded key), scaling = 1/sqrt(D)
# Output:
#   attn_scores [B, H, S, S] (float32, masked), no softmax
@triton.jit
def attn_scores_kernel(
    Q_ptr, Ke_ptr, attn_ptr,
    B, H, S, D,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_keb, stride_keh, stride_kes, stride_ked,
    stride_attnb, stride_attnh, stride_attns, stride_attnj,
    scaling,
    BLOCK_S: tl.constexpr,
):
    # One program per (b, h)
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H
    if b >= B or h >= H:
        return

    # For each query position i (sequence index), compute dot with all keys j
    for i in range(0, S):
        # Initialize acc vector for all j
        acc = tl.zeros((BLOCK_S,), dtype=tl.float32)

        # Loop over j in tiles
        for j0 in range(0, S, BLOCK_S):
            j = j0 + tl.arange(0, BLOCK_S)

            # Load Q[i]
            q = tl.load(Q_ptr + b * stride_qb + h * stride_qh + i * stride_qs + tl.arange(0, D) * stride_qd,
                        mask=tl.arange(0, D) < D, other=0.0)  # [D]

            # Load Ke[j] as [D] and accumulate dot
            for d0 in range(0, D, BLOCK_S):
                offs_d = d0 + tl.arange(0, BLOCK_S)
                mask_d = offs_d < D
                k = tl.load(Ke_ptr + b * stride_keb + h * stride_keh + j * stride_kes + offs_d * stride_ked,
                            mask=mask_d, other=0.0)  # [BLOCK_S]
                # Since k is [D], we need to multiply q[offs_d] * k[offs_d] for d in 0..D-1 and sum.
                # To do this, we multiply elementwise q[offs_d] * k[offs_d] and reduce over D using a loop.
                # However, k is [BLOCK_S], not [D]. We need to load Ke for each j separately as [D] and dot with q.
                # Fix: load Ke[j, d] per d-block and accumulate. Since Triton kernel supports loops, we can compute:
                # For each d in offs_d, load k_j_d = Ke[b,h,j,d] and add q[d] * k_j_d to acc.
                # We will implement this with an inner loop over d block.
                for d_idx in range(0, BLOCK_S):
                    d = d0 + d_idx
                    mask_d_idx = d < D
                    k_j_d = tl.load(Ke_ptr + b * stride_keb + h * stride_keh + j[0] * stride_kes + d * stride_ked,
                                   mask=mask_d_idx, other=0.0)
                    # Broadcast q[d] to scalar
                    q_d = tl.load(Q_ptr + b * stride_qb + h * stride_qh + i * stride_qs + d * stride_qd,
                                  mask=mask_d_idx, other=0.0)
                    acc += q_d * k_j_d

        # Scale
        acc = acc * scaling

        # Apply causal mask: if j > i, set -inf
        # attn[i, j] = -inf if j > i, else acc[j]
        for j_off in range(0, BLOCK_S):
            j_val = j0 + j_off
            if j_val < S:
                if j_val > i:
                    acc[j_off] = -float('inf')

        # Store acc to attn[b, h, i, :]
        attn_ptrs = attn_ptr + b * stride_attnb + h * stride_attnh + i * stride_attns + (j0 + tl.arange(0, BLOCK_S)) * stride_attnj
        tl.store(attn_ptrs, acc, mask=(j0 + tl.arange(0, BLOCK_S) < S))


# Triton kernel: row-wise softmax over the last dimension (sequence axis) for each row (b, h, i)
# Input: attn_scores [B, H, S, S], Output: attn_probs [B, H, S, S]
@triton.jit
def softmax_row_kernel(
    attn_ptr, out_ptr,
    B, H, S,
    stride_absb, stride_absh, stride_absi, stride_absj,
    stride_outb, stride_outh, stride_outi, stride_outj,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // (H * S)
    rem = pid % (H * S)
    h = rem // S
    i = rem % S

    # Load row vector across j
    max_val = -float('inf')
    row = tl.zeros((BLOCK_S,), dtype=tl.float32)
    for j0 in range(0, S, BLOCK_S):
        j = j0 + tl.arange(0, BLOCK_S)
        ptrs = attn_ptr + b * stride_absb + h * stride_absh + i * stride_absi + j * stride_absj
        vals = tl.load(ptrs, mask=j < S, other=-float('inf'))
        row = vals
        # Find max
        block_max = tl.max(vals, axis=0)
        max_val = tl.maximum(max_val, block_max)

    # Compute exp and sum
    sum_exp = 0.0
    for j0 in range(0, S, BLOCK_S):
        j = j0 + tl.arange(0, BLOCK_S)
        ptrs = attn_ptr + b * stride_absb + h * stride_absh + i * stride_absi + j * stride_absj
        vals = tl.load(ptrs, mask=j < S, other=-float('inf'))
        exps = tl.exp(vals - max_val)
        sum_exp += tl.sum(exps, axis=0)

    # Store normalized probs
    for j0 in range(0, S, BLOCK_S):
        j = j0 + tl.arange(0, BLOCK_S)
        ptrs = attn_ptr + b * stride_absb + h * stride_absh + i * stride_absi + j * stride_absj
        vals = tl.load(ptrs, mask=j < S, other=-float('inf'))
        probs = tl.exp(vals - max_val) / sum_exp
        out_ptrs = out_ptr + b * stride_outb + h * stride_outh + i * stride_outi + j * stride_outj
        tl.store(out_ptrs, probs, mask=j < S)


# Triton GEMM for output projection: Y[M, N] = A[M, K] @ W[N, K]^T (no bias)
@triton.jit
def output_proj_kernel(
    A_ptr, W_ptr, Y_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)

        b_ptrs = W_ptr + (offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk)
        b = tl.load(b_ptrs, mask=(offs_n[None, :] < N) & (offs_k[:, None] < K), other=0.0)

        acc += tl.dot(a, b)

    y_ptrs = Y_ptr + (offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn)
    tl.store(y_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# Entry point model
class ModelNew(torch.nn.Module):
    def __init__(
        self,
        num_attention_heads=96,
        head_dim=128,
        num_key_value_heads=8,
        num_key_value_groups=12,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_key_value_groups
        self.scaling = 1.0 / math.sqrt(head_dim)

    def forward(
        self,
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
    ):
        # hidden_states: [B, S, H], H = num_attention_heads * head_dim
        B, S, H = hidden_states.shape
        D = self.head_dim  # 128
        Hq = self.num_attention_heads  # 96
        KVH = self.num_key_value_heads  # 8
        GROUPS = self.num_key_value_groups  # 12

        # Ensure device and dtype
        device = hidden_states.device
        dtype = hidden_states.dtype
        assert dtype == torch.float32, "This Triton implementation expects float32 tensors."

        # 1) Linear projections (no bias): F.linear -> Triton GEMM
        # Prepare X_flat for query: [B*S, H]
        Xq = hidden_states.reshape(B * S, H).contiguous()
        # Wq: [H, H] for no bias
        # Note: For no bias, we can pass Wq as is. K=M=H, N=H
        Yq_flat = torch.empty((B * S, H), dtype=torch.float32, device=device)
        linear_no_bias_kernel[(B, S), (H, H)](
            Xq, q_proj_weight, Yq_flat,
            B * S, H, H,
            Xq.stride(0), H,  # stride_xm = H, stride_xk = 1 for contiguous
            q_proj_weight.stride(0), q_proj_weight.stride(1),
            Yq_flat.stride(0), Yq_flat.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
        )
        query_states = Yq_flat.view(B, S, H)

        Xk = hidden_states.reshape(B * S, H).contiguous()
        Yk_flat = torch.empty((B * S, H), dtype=torch.float32, device=device)
        linear_no_bias_kernel[(B, S), (H, H)](
            Xk, k_proj_weight, Yk_flat,
            B * S, H, H,
            Xk.stride(0), H,
            k_proj_weight.stride(0), k_proj_weight.stride(1),
            Yk_flat.stride(0), Yk_flat.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
        )
        key_states = Yk_flat.view(B, S, H)

        Xv = hidden_states.reshape(B * S, H).contiguous()
        Yv_flat = torch.empty((B * S, H), dtype=torch.float32, device=device)
        linear_no_bias_kernel[(B, S), (H, H)](
            Xv, v_proj_weight, Yv_flat,
            B * S, H, H,
            Xv.stride(0), H,
            v_proj_weight.stride(0), v_proj_weight.stride(1),
            Yv_flat.stride(0), Yv_flat.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
        )
        value_states = Yv_flat.view(B, S, H)

        # 2) Reshape to head form
        # query: [B, S, 96, 128]
        query_heads = query_states.view(B, S, Hq, D)
        # key/value: [B, S, 8, 128]
        key_heads = key_states.view(B, S, KVH, D)
        value_heads = value_states.view(B, S, KVH, D)

        # 3) RMSNorm per head (query and key), with per-head learned scale
        # Initialize outputs
        q_norm_out = torch.empty_like(query_heads)
        k_norm_out = torch.empty_like(key_heads)

        # q norm: weight shape [Hq, D]
        # Launch kernel: one program per (b, s)
        for b in range(B):
            for s in range(S):
                # Query RMSNorm
                rmsnorm_heads_kernel[(1,), (1,)](
                    query_heads[b, s].contiguous(), q_norm_weight, q_norm_out[b, s],
                    1, 1, D, Hq,
                    query_heads[b, s].stride(0), query_heads[b, s].stride(1),
                    q_norm_weight.stride(0), q_norm_weight.stride(1),
                    q_norm_out[b, s].stride(0), q_norm_out[b, s].stride(1),
                    rms_norm_eps,
                    BLOCK_D=D,
                    num_warps=4,
                )
                # Key RMSNorm
                rmsnorm_heads_kernel[(1,), (1,)](
                    key_heads[b, s].contiguous(), k_norm_weight, k_norm_out[b, s],
                    1, 1, D, KVH,
                    key_heads[b, s].stride(0), key_heads[b, s].stride(1),
                    k_norm_weight.stride(0), k_norm_weight.stride(1),
                    k_norm_out[b, s].stride(0), k_norm_out[b, s].stride(1),
                    rms_norm_eps,
                    BLOCK_D=D,
                    num_warps=4,
                )

        # 4) Rotate query and key (RoPE)
        # Prepare per-head rotations; here we assume cos/sin are [D]
        q_rot = torch.empty_like(q_norm_out)
        k_rot = torch.empty_like(k_norm_out)

        for b in range(B):
            for s in range(S):
                # Query rotation
                rotate_half_heads_kernel[(1,), (1,)](
                    q_norm_out[b, s], cos, sin, q_rot[b, s],
                    1, 1, D,
                    q_norm_out[b, s].stride(0), q_norm_out[b, s].stride(1),
                    q_rot[b, s].stride(0), q_rot[b, s].stride(1),
                    BLOCK_D=D,
                    num_warps=4,
                )
                # Key rotation
                rotate_half_heads_kernel[(1,), (1,)](
                    k_norm_out[b, s], cos, sin, k_rot[b, s],
                    1, 1, D,
                    k_norm_out[b, s].stride(0), k_norm_out[b, s].stride(1),
                    k_rot[b, s].stride(0), k_rot[b, s].stride(1),
                    BLOCK_D=D,
                    num_warps=4,
                )

        # 5) GQA: expand key/value from KVH to Hq by replication
        # Output shapes: [B, Hq, S, D]
        key_expanded = torch.empty((B, Hq, S, D), dtype=torch.float32, device=device)
        value_expanded = torch.empty((B, Hq, S, D), dtype=torch.float32, device=device)

        expand_kv_heads_kernel[(B, S), (1,)](
            k_rot, key_expanded,
            B, S, D, KVH, Hq, GROUPS,
            k_rot.stride(0), k_rot.stride(1), k_rot.stride(2), k_rot.stride(3),
            key_expanded.stride(0), key_expanded.stride(1), key_expanded.stride(2), key_expanded.stride(3),
            BLOCK_D=D,
            num_warps=4,
        )
        expand_kv_heads_kernel[(B, S), (1,)](
            value_expanded, value_expanded,
            B, S, D, KVH, Hq, GROUPS,
            value_heads.stride(0), value_heads.stride(1), value_heads.stride(2), value_heads.stride(3),
            value_expanded.stride(0), value_expanded.stride(1), value_expanded.stride(2), value_expanded.stride(3),
            BLOCK_D=D,
            num_warps=4,
        )

        # 6) Attention scores per (b, h) across all s: attn[b, h, s, :] = q[b,h,s,:] @ k_expanded[b,h,: ,:]^T * scaling
        # We will compute this in Triton with per (b,h) program, tiled across sequence positions.
        attn_scores = torch.empty((B, Hq, S, S), dtype=torch.float32, device=device)

        # Launch per (b, h)
        for b in range(B):
            for h in range(Hq):
                attn_scores_kernel[(1,), (1,)](
                    q_rot[b, h], key_expanded[b, h], attn_scores[b, h],
                    1, 1, S, D,
                    q_rot[b, h].stride(0), q_rot[b, h].stride(1), q_rot[b, h].stride(2), q_rot[b, h].stride(3),
                    key_expanded[b, h].stride(0), key_expanded[b, h].stride(1), key_expanded[b, h].stride(2), key_expanded[b, h].stride(3),
                    attn_scores[b, h].stride(0), attn_scores[b, h].stride(1), attn_scores[b, h].stride(2), attn_scores[b, h].stride(3),
                    self.scaling,
                    BLOCK_S=64,
                    num_warps=4,
                )

        # 7) Apply causal mask and softmax over sequence axis for each (b, h, i)
        attn_probs = torch.empty_like(attn_scores)

        for b in range(B):
            for h in range(Hq):
                softmax_row_kernel[(1,), (1,)](
                    attn_scores[b, h], attn_probs[b, h],
                    1, 1, S,
                    attn_scores[b, h].stride(0), attn_scores[b, h].stride(1), attn_scores[b, h].stride(2), attn_scores[b, h].stride(3),
                    attn_probs[b, h].stride(0), attn_probs[b, h].stride(1), attn_probs[b, h].stride(2), attn_probs[b, h].stride(3),
                    BLOCK_S=128,
                    num_warps=4,
                )

        # 8) Compute attention output: attn_output[b, h, s, :] = attn_probs[b, h, s, :] @ value_expanded[b, h, :, :]
        attn_output_flat = torch.empty((B * S, D), dtype=torch.float32, device=device)
        # We need to compute per (b, h, s):
        for b in range(B):
            for h in range(Hq):
                for s in range(S):
                    # attn_probs[b,h,s,:] is a vector of length S
                    prob = attn_probs[b, h, s]  # shape [S]
                    val = value_expanded[b, h, s]  # shape [D]
                    # outer product: prob[:, None] * val[None, :]
                    attn_vec = torch.matmul(prob.unsqueeze(1), val.unsqueeze(1).transpose(0, 1))  # [S, D]
                    attn_output_flat[b * S + s] = attn_vec.reshape(D)

        # 9) Output projection (no bias): final_linear [B, S, H]
        # attn_output_flat: [B*S, D], o_proj_weight: [H, H], output: [B*S, H]
        final_output_flat = torch.empty((B * S, H), dtype=torch.float32, device=device)
        output_proj_kernel[(B, S), (H, H)](
            attn_output_flat, o_proj_weight, final_output_flat,
            B * S, H, D,
            attn_output_flat.stride(0), D,
            o_proj_weight.stride(0), o_proj_weight.stride(1),
            final_output_flat.stride(0), final_output_flat.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32,
        )

        output = final_output_flat.view(B, S, H)
        return output


# The evaluation environment expects Model to forward via ModelNew
class Model(torch.nn.Module):
    def forward(self, *args):
        return ModelNew(*args)


def run(*args):
    return ModelNew()(*args)
