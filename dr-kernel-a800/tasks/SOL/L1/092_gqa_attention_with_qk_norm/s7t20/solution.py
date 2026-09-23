import torch
import triton
import triton.language as tl

# Triton kernel: Linear over B, S, H dims: out[b, s, h] = sum_k input[b, s, k] * weight[h, k] + bias[h]
# Input shape: [B, S, K]; weight shape: [H, K]; bias shape: [H]; out shape: [B, S, H]
@triton.jit
def triton_linear_bsh(x_ptr, w_ptr, b_ptr, out_ptr,
                      B, S, H, K,
                      x_stride0, x_stride1, x_stride2,  # strides for x: (B, S, K)
                      w_stride0, w_stride1,             # strides for w: (H, K)
                      out_stride0, out_stride1, out_stride2,
                      BLOCK_K: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    base_x = b * x_stride0 + s * x_stride1
    base_out = b * out_stride0 + s * out_stride1 + h * out_stride2

    acc = tl.zeros([1], dtype=tl.float32)
    # Loop over K in tiles
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + tl.arange(0, BLOCK_K)
        mask = kk < K
        x_vec = tl.load(x_ptr + base_x + kk * x_stride2, mask=mask, other=0.0)
        w_vec = tl.load(w_ptr + h * w_stride0 + kk * w_stride1, mask=mask, other=0.0)
        acc += tl.sum(x_vec * w_vec, axis=0)

    # Add bias
    bias = tl.load(b_ptr + h)
    acc = acc + bias

    # Store as float32 (attention math done in fp32)
    tl.store(out_ptr + base_out, acc)


# Triton kernel: RMSNorm per row (b, h) across last dim of size S.
# out[b, :, h] = weight[h] * (x[b, :, h] / sqrt(mean(x^2) + eps))
@triton.jit
def triton_rmsnorm_row(x_ptr, weight_ptr, out_ptr,
                       B, S,
                       x_stride0, x_stride1, x_stride2,
                       out_stride0, out_stride1, out_stride2,
                       eps: tl.constexpr,
                       BLOCK_D: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)

    sum_sq = 0.0
    # compute mean over S
    for d in range(0, S, BLOCK_D):
        offs = d + tl.arange(0, BLOCK_D)
        mask = offs < S
        x = tl.load(x_ptr + b * x_stride0 + offs * x_stride1 + h * x_stride2, mask=mask, other=0.0)
        sum_sq += tl.sum(x * x, axis=0)

    mean = sum_sq / S
    inv_rms = 1.0 / tl.sqrt(mean + eps)
    scale = tl.load(weight_ptr + h) * inv_rms

    # write normalized row
    for d in range(0, S, BLOCK_D):
        offs = d + tl.arange(0, BLOCK_D)
        mask = offs < S
        x = tl.load(x_ptr + b * x_stride0 + offs * x_stride1 + h * x_stride2, mask=mask, other=0.0)
        y = x * scale
        tl.store(out_ptr + b * out_stride0 + offs * out_stride1 + h * out_stride2, y, mask=mask)


# Triton kernel: Apply Rotary Position Embedding per row (b, h, s)
# Rotate 128-dim: split into two 64 halves; use cos/sin vectors of length 128.
# out[:, d] = q[:, d] * cos[d] + (d < 64 ? q[:, 64 + d] : -q[:, d - 64]) * sin[d]
@triton.jit
def triton_rope_row(x_ptr, cos_ptr, sin_ptr, out_ptr,
                    B, H, S, head_dim,
                    x_stride0, x_stride1, x_stride2,
                    cos_stride0, cos_stride1,
                    sin_stride0, sin_stride1,
                    out_stride0, out_stride1, out_stride2,
                    BLOCK_D: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    base_x = b * x_stride0 + s * x_stride1 + h * x_stride2
    base_out = b * out_stride0 + s * out_stride1 + h * out_stride2

    for d0 in range(0, head_dim, BLOCK_D):
        d = d0 + tl.arange(0, BLOCK_D)
        mask = d < head_dim
        q = tl.load(x_ptr + base_x + d, mask=mask, other=0.0)
        cos_vec = tl.load(cos_ptr + d, mask=mask, other=0.0)
        sin_vec = tl.load(sin_ptr + d, mask=mask, other=0.0)

        # Split: first 64 and last 64
        d64 = 64
        q1 = q[:d64]
        q2 = q[d64:]
        rotated = tl.concatenate([-q2, q1], axis=0)
        q_out = q * cos_vec + rotated * sin_vec

        tl.store(out_ptr + base_out + d, q_out, mask=mask)


# Triton kernel: Expand KV heads from KVH to H using groups mapping (H = KVH * GROUPS).
# Copy K[b, kh, j, :] -> K_out[b, h_target, j, :] where h_target = kh * GROUPS + g
# and similarly for V. Grid: (B, KVH, GROUPS, S)
@triton.jit
def triton_expand_kv(K_ptr, V_ptr, K_out_ptr, V_out_ptr,
                     B, S, KVH, KD,  # KD is head_dim, but we assume KD=128 and store at last dim
                     K_stride0, K_stride1, K_stride2, K_stride3,  # strides for K: (B, KVH, S, KD)
                     V_stride0, V_stride1, V_stride2, V_stride3,  # strides for V: (B, KVH, S, KD)
                     Kout_stride0, Kout_stride1, Kout_stride2, Kout_stride3,  # strides for K_out: (B, H, S, KD)
                     Vout_stride0, Vout_stride1, Vout_stride2, Vout_stride3,  # strides for V_out: (B, H, S, KD)
                     GROUPS: tl.constexpr):
    kh = tl.program_id(0)
    b = tl.program_id(1)
    g = tl.program_id(2)
    j = tl.program_id(3)

    h_target = kh * GROUPS + g

    # K
    k_row = tl.load(K_ptr + b * K_stride0 + kh * K_stride1 + j * K_stride2)  # load 128-dim vector
    tl.store(K_out_ptr + b * Kout_stride0 + h_target * Kout_stride1 + j * Kout_stride2, k_row)

    # V
    v_row = tl.load(V_ptr + b * V_stride0 + kh * V_stride1 + j * V_stride2)
    tl.store(V_out_ptr + b * Vout_stride0 + h_target * Vout_stride1 + j * Vout_stride2, v_row)


# Triton kernel: Compute attention output for a given (b, h).
# Q_in: [B, H, S, KD]; K_in_expanded: [B, H, S, KD]; V_expanded: [B, H, S, KD]
# Compute attn_scores[b, i, j] = Q[b, h, i, :] @ K[b, h, j, :], scaled by inv_sqrt(D)
# Apply causal mask: attn_scores[i, j] = -inf if i < j, else normal.
# Softmax along j, then out[b, h, i] = sum_j attn_scores[i, j] * V[b, h, j, :].
@triton.jit
def triton_attention_bh(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    B, H, S, KD,
    Q_stride0, Q_stride1, Q_stride2, Q_stride3,
    K_stride0, K_stride1, K_stride2, K_stride3,
    V_stride0, V_stride1, V_stride2, V_stride3,
    Out_stride0, Out_stride1, Out_stride2,
    BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr, BLOCK_KD: tl.constexpr
):
    b = tl.program_id(0)
    h = tl.program_id(1)

    inv_sqrt = 1.0 / tl.sqrt(float(KD))
    acc = tl.zeros([S], dtype=tl.float32)  # output per i position

    for i0 in range(0, S, BLOCK_I):
        i = i0 + tl.arange(0, BLOCK_I)
        mask_i = i < S

        # Compute attention scores for this i vector
        scores = tl.zeros([BLOCK_I, S], dtype=tl.float32)
        for j0 in range(0, S, BLOCK_J):
            j = j0 + tl.arange(0, BLOCK_J)
            mask_j = j < S

            # accumulate Q[i] @ K[j]
            # loop over KD in tiles
            dot = tl.zeros([BLOCK_I, BLOCK_J], dtype=tl.float32)
            for kd0 in range(0, KD, BLOCK_KD):
                kd = kd0 + tl.arange(0, BLOCK_KD)
                mask_kd = kd < KD

                # load Q[i, kd], shape [BLOCK_I, BLOCK_KD]
                q_ptr = Q_ptr + b * Q_stride0 + h * Q_stride1 + i[:, None] * Q_stride2 + kd[None, :] * Q_stride3
                q = tl.load(q_ptr, mask=mask_i[:, None] & mask_kd[None, :], other=0.0)

                # load K[j, kd], shape [BLOCK_J, BLOCK_KD]
                k_ptr = K_ptr + b * K_stride0 + h * K_stride1 + j[:, None] * K_stride2 + kd[None, :] * K_stride3
                k = tl.load(k_ptr, mask=mask_j[:, None] & mask_kd[None, :], other=0.0)

                # multiply and sum over kd
                dot += tl.sum(q * k, axis=1)  # [BLOCK_I, BLOCK_J]

            # scale
            dot = dot * inv_sqrt

            # apply causal mask: for i<I and j>=I, set to -inf, else keep
            # Create -inf tile
            neg_inf = -float("inf")
            # Broadcast i and j to form mask i<j
            # We need to combine masks: for valid i and j, set -inf when i < j
            # Compute i and j vectors for mask
            i_vec = i[:, None]
            j_vec = j[None, :]
            mask_ij = (i_vec < j_vec) & mask_i[:, None] & mask_j[None, :]
            scores_block = tl.where(mask_ij, neg_inf, dot)
            scores += scores_block

        # softmax along j
        exp_scores = tl.exp(scores)  # scores is [BLOCK_I, S]
        # sum over j
        sums = tl.sum(exp_scores, axis=1)  # [BLOCK_I]
        probs = exp_scores / sums[:, None]  # softmax along j per i

        # output: out[i] = sum_j probs[i, j] * V[b, h, j]
        out_vec = tl.zeros([BLOCK_I], dtype=tl.float32)
        for j0 in range(0, S, BLOCK_J):
            j = j0 + tl.arange(0, BLOCK_J)
            mask_j = j < S
            v_ptr = V_ptr + b * V_stride0 + h * V_stride1 + j * V_stride2
            v = tl.load(v_ptr, mask=mask_j, other=0.0)  # [BLOCK_J]
            # probs[i, j] for current j range
            # probs is [BLOCK_I, S], we need probs[i, j] for this j range
            # Build i index vector
            i_vec = i
            # probs[i, j] is probs[i, j]
            probs_slice = probs[:, j0:j0+BLOCK_J]  # [BLOCK_I, BLOCK_J]
            # multiply and sum over j
            out_vec += tl.sum(probs_slice * v[None, :], axis=1)

        # accumulate
        acc += out_vec

    # store output
    out_ptr_base = Out_ptr + b * Out_stride0 + h * Out_stride1
    for i0 in range(0, S, BLOCK_I):
        i = i0 + tl.arange(0, BLOCK_I)
        mask_i = i < S
        tl.store(out_ptr_base + i * Out_stride2, acc[i0:i0+BLOCK_I], mask=mask_i)


# Triton kernel: Output projection linear_bsh (no bias)
# Input: attn_out [B, S, H]; weight: [H, K]; output: [B, S, H]
@triton.jit
def triton_linear_bsh_out(x_ptr, w_ptr, out_ptr,
                          B, S, H, K,
                          x_stride0, x_stride1, x_stride2,  # (B, S, H)
                          w_stride0, w_stride1,            # (H, K)
                          out_stride0, out_stride1, out_stride2,
                          BLOCK_K: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    base_x = b * x_stride0 + s * x_stride1
    base_out = b * out_stride0 + s * out_stride1 + h * out_stride2

    acc = tl.zeros([1], dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        kk = k0 + tl.arange(0, BLOCK_K)
        mask = kk < K
        x_vec = tl.load(x_ptr + base_x + kk * x_stride2, mask=mask, other=0.0)
        w_vec = tl.load(w_ptr + h * w_stride0 + kk * w_stride1, mask=mask, other=0.0)
        acc += tl.sum(x_vec * w_vec, axis=0)

    tl.store(out_ptr + base_out, acc)


# ----------------- ModelNew -----------------

class ModelNew(torch.nn.Module):
    def __init__(self,
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
                 num_attention_heads: int = 96,
                 num_key_value_heads: int = 8,
                 head_dim: int = 128):
        super().__init__()
        # store parameters as buffers
        self.register_buffer('q_proj_weight', q_proj_weight)
        self.register_buffer('q_proj_bias', q_proj_bias)
        self.register_buffer('k_proj_weight', k_proj_weight)
        self.register_buffer('k_proj_bias', k_proj_bias)
        self.register_buffer('v_proj_weight', v_proj_weight)
        self.register_buffer('v_proj_bias', v_proj_bias)
        self.register_buffer('o_proj_weight', o_proj_weight)
        self.register_buffer('q_norm_weight', q_norm_weight)
        self.register_buffer('k_norm_weight', k_norm_weight)
        self.register_buffer('cos', cos)
        self.register_buffer('sin', sin)
        self.rms_norm_eps = float(rms_norm_eps)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.head_dim = int(head_dim)
        self.groups = self.num_attention_heads // self.num_key_value_heads  # 12

    def forward(self, hidden_states: torch.Tensor):
        # hidden_states: [B, S, K] where K=head_dim=128
        B, S, K = hidden_states.shape
        assert K == self.head_dim, f"hidden_states last dim must be head_dim={self.head_dim}, got {K}"
        H = self.num_attention_heads
        KVH = self.num_key_value_heads
        KD = self.head_dim

        device = hidden_states.device
        dtype = hidden_states.dtype

        # 1) Dense linear for Q, K, V using Triton
        # Prepare inputs for Q, K, V as (B, S, K)
        # Allocate outputs as (B, S, H) in fp32
        Q = torch.empty((B, S, H), dtype=torch.float32, device=device)
        K_in = torch.empty((B, S, H), dtype=torch.float32, device=device)
        V_in = torch.empty((B, S, H), dtype=torch.float32, device=device)

        # Launch Q linear
        grid_Q = (B, H, S)
        triton_linear_bsh(
            hidden_states, self.q_proj_weight, self.q_proj_bias, Q,
            B, S, H, K,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            self.q_proj_weight.stride(0), self.q_proj_weight.stride(1),
            Q.stride(0), Q.stride(1), Q.stride(2),
            BLOCK_K=64,
            num_warps=4, num_stages=2
        )

        # Launch K linear
        grid_K = (B, H, S)
        triton_linear_bsh(
            hidden_states, self.k_proj_weight, self.k_proj_bias, K_in,
            B, S, H, K,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            self.k_proj_weight.stride(0), self.k_proj_weight.stride(1),
            K_in.stride(0), K_in.stride(1), K_in.stride(2),
            BLOCK_K=64,
            num_warps=4, num_stages=2
        )

        # Launch V linear
        grid_V = (B, H, S)
        V_in = torch.empty((B, S, H), dtype=torch.float32, device=device)
        triton_linear_bsh(
            hidden_states, self.v_proj_weight, self.v_proj_bias, V_in,
            B, S, H, K,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            self.v_proj_weight.stride(0), self.v_proj_weight.stride(1),
            V_in.stride(0), V_in.stride(1), V_in.stride(2),
            BLOCK_K=64,
            num_warps=4, num_stages=2
        )

        # 2) RMSNorm for Q and K (no bias in RMSNorm path, but we have q_norm/k_norm weights)
        Q_norm = torch.empty((B, H), dtype=torch.float32, device=device)
        K_norm = torch.empty((B, H), dtype=torch.float32, device=device)

        # Launch RMSNorm for Q: grid (B, H)
        grid_rms_Q = (B, H)
        triton_rmsnorm_row(
            Q, self.q_norm_weight, Q_norm,
            B, S,
            Q.stride(0), Q.stride(1), Q.stride(2),
            Q_norm.stride(0), Q_norm.stride(1), Q_norm.stride(2),
            eps=self.rms_norm_eps,
            BLOCK_D=128,
            num_warps=4, num_stages=2
        )

        # Launch RMSNorm for K: grid (B, H)
        grid_rms_K = (B, H)
        K_norm = torch.empty((B, H), dtype=torch.float32, device=device)
        triton_rmsnorm_row(
            K_in, self.k_norm_weight, K_norm,
            B, S,
            K_in.stride(0), K_in.stride(1), K_in.stride(2),
            K_norm.stride(0), K_norm.stride(1), K_norm.stride(2),
            eps=self.rms_norm_eps,
            BLOCK_D=128,
            num_warps=4, num_stages=2
        )

        # 3) RoPE for Q and K
        Q_rot = torch.empty((B, S, H), dtype=torch.float32, device=device)
        K_rot = torch.empty((B, S, H), dtype=torch.float32, device=device)

        grid_rope = (B, H, S)
        triton_rope_row(
            Q_norm, self.cos, self.sin, Q_rot,
            B, H, S, self.head_dim,
            Q_norm.stride(0), Q_norm.stride(1), Q_norm.stride(2),
            self.cos.stride(0), self.cos.stride(1),
            self.sin.stride(0), self.sin.stride(1),
            Q_rot.stride(0), Q_rot.stride(1), Q_rot.stride(2),
            BLOCK_D=128,
            num_warps=4, num_stages=2
        )

        triton_rope_row(
            K_norm, self.cos, self.sin, K_rot,
            B, H, S, self.head_dim,
            K_norm.stride(0), K_norm.stride(1), K_norm.stride(2),
            self.cos.stride(0), self.cos.stride(1),
            self.sin.stride(0), self.sin.stride(1),
            K_rot.stride(0), K_rot.stride(1), K_rot.stride(2),
            BLOCK_D=128,
            num_warps=4, num_stages=2
        )

        # 4) GQA expansion: expand KV heads from KVH to H via groups=12
        # Allocate expanded K and V: shape [B, H, S, KD]
        K_exp = torch.empty((B, H, S, KD), dtype=torch.float32, device=device)
        V_exp = torch.empty((B, H, S, KD), dtype=torch.float32, device=device)

        # Fill K_exp/V_exp with copies using Triton kernel; we can pre-fill with zeros and then copy rows.
        # Here we will manually copy using grid (B, KVH, GROUPS, S).
        grid_expand = (KVH, B, self.groups, S)
        # Note: Triton kernels expect int program_id per dimension, we map as:
        # program_id(0)=kh, program_id(1)=b, program_id(2)=g, program_id(3)=j
        for kh in range(KVH):
            for g in range(self.groups):
                b = 0  # loop over batch handled by separate grid; but Triton uses program_id(1)=b, so we launch with b in grid.
                # Instead, launch with grid using b dimension:
                pass
        # The above "pass" is a placeholder; Triton requires grid=(B,KVH,GROUPS,S). We launch properly below.
        # We need to launch for each b; Triton allows grid=(B,KVH,GROUPS,S), so we compute grid sizes:
        grid_expand = (KVH, B, self.groups, S)
        triton_expand_kv(
            K_rot, V_rot, K_exp, V_exp,
            B, S, KVH, KD,
            K_rot.stride(0), K_rot.stride(1), K_rot.stride(2), K_rot.stride(3),
            V_rot.stride(0), V_rot.stride(1), V_rot.stride(2), V_rot.stride(3),
            K_exp.stride(0), K_exp.stride(1), K_exp.stride(2), K_exp.stride(3),
            V_exp.stride(0), V_exp.stride(1), V_exp.stride(2), V_exp.stride(3),
            GROUPS=self.groups,
            num_warps=2, num_stages=2
        )
        # Note: V_rot is not defined; we used V_in earlier. Fix: recompute V_rot as RMSNorm(V_in) and then expand.
        # However, the original code RMSNorms Q and K, not V. We should use V_in directly, not RMSNorm V.
        # So we expand K_rot and V_in: need to create V_rot without RMSNorm? The original applies RMSNorm to Q and K only.
        # But for attention, it normalizes Q and K; V is not normalized. So we should use V_in directly for expansion.
        # Let's correct: we need K_rot_expanded and V_in_expanded.
        # We'll re-launch expand with K_rot and V_in.
        # To avoid confusion, we'll recompute expand using K_rot and V_rot (where V_rot is V_in without RMSNorm).

        # Correct: We should not RMSNorm V. We must use V_in for expansion.
        # Re-launch expand with K_rot and V_in (no RMSNorm on V):
        # Triton expects contiguous pointers; ensure V_in is float32. We already computed V_in (dense linear).
        # We need V_in without RMSNorm; let's make V_rot = V_in. We can reuse V_in as is.

        # Define V_rot as V_in (no RMSNorm)
        V_rot = V_in  # [B, S, H] in fp32

        # Re-launch expand with K_rot and V_rot (V_rot=V_in)
        grid_expand = (KVH, B, self.groups, S)
        triton_expand_kv(
            K_rot, V_rot, K_exp, V_exp,
            B, S, KVH, KD,
            K_rot.stride(0), K_rot.stride(1), K_rot.stride(2), K_rot.stride(3),
            V_rot.stride(0), V_rot.stride(1), V_rot.stride(2), V_rot.stride(3),
            K_exp.stride(0), K_exp.stride(1), K_exp.stride(2), K_exp.stride(3),
            V_exp.stride(0), V_exp.stride(1), V_exp.stride(2), V_exp.stride(3),
            GROUPS=self.groups,
            num_warps=2, num_stages=2
        )

        # 5) Attention compute: Triton kernel over (B, H), tiled over i/j
        # Q_rot, K_exp, V_exp are all [B, H, S, KD] float32. We need to pass as such.
        # However, our attention kernel expects (Q,K,V) as [B, H, S, KD]. We already have them expanded.
        AttnOut = torch.empty((B, S, H), dtype=torch.float32, device=device)

        grid_attention = (B, H)
        triton_attention_bh(
            Q_rot, K_exp, V_exp, AttnOut,
            B, H, S, KD,
            Q_rot.stride(0), Q_rot.stride(1), Q_rot.stride(2), Q_rot.stride(3),
            K_exp.stride(0), K_exp.stride(1), K_exp.stride(2), K_exp.stride(3),
            V_exp.stride(0), V_exp.stride(1), V_exp.stride(2), V_exp.stride(3),
            AttnOut.stride(0), AttnOut.stride(1), AttnOut.stride(2),
            BLOCK_I=32, BLOCK_J=64, BLOCK_KD=32,
            num_warps=4, num_stages=2
        )

        # 6) Output projection linear (no bias)
        Out = torch.empty((B, S, H), dtype=torch.float32, device=device)
        grid_output = (B, H, S)
        triton_linear_bsh_out(
            AttnOut, self.o_proj_weight, Out,
            B, S, H, H,  # note: o_proj_weight shape is [H, H] (self.num_attention_heads == H)
            AttnOut.stride(0), AttnOut.stride(1), AttnOut.stride(2),
            self.o_proj_weight.stride(0), self.o_proj_weight.stride(1),
            Out.stride(0), Out.stride(1), Out.stride(2),
            BLOCK_K=64,
            num_warps=4, num_stages=2
        )

        # Return output tensor
        return Out


def run(*args):
    return ModelNew()(*args)
