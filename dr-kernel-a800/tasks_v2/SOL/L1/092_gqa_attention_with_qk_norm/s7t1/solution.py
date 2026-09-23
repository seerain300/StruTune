import torch
import torch.nn as nn
import triton
import triton.language as tl

# Kernel 1: Row-wise dense linear: out[b, s, h] = sum_k input[b, s, k] * weight[h, k] + bias[h]
@triton.jit
def triton_linear_row(input_ptr, weight_ptr, bias_ptr, out_ptr,
                      B, S, H, head_dim,
                      input_stride0, input_stride1, input_stride2,
                      weight_stride0, weight_stride1,
                      out_stride0, out_stride1, out_stride2,
                      BLOCK_K: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    # Base offsets
    base_input = b * input_stride0 + s * input_stride1
    base_out = b * out_stride0 + s * out_stride1 + h * out_stride2

    acc = tl.zeros((), dtype=tl.float32)
    # Loop over K dimension
    for k0 in range(0, head_dim, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < head_dim
        # Load input row slice: shape [BLOCK_K]
        x = tl.load(input_ptr + base_input + k_offsets * input_stride2, mask=mask_k, other=0.0)
        # Load weight slice for h: shape [BLOCK_K]
        w = tl.load(weight_ptr + h * weight_stride0 + k_offsets * weight_stride1, mask=mask_k, other=0.0)
        # Accumulate dot product
        acc += tl.sum(x * w, axis=0)

    # Add bias
    bval = tl.load(bias_ptr + h)
    acc += bval

    # Store result
    tl.store(out_ptr + base_out, acc)

# Kernel 2: RMSNorm per row over last dim: out_row = weight[h] * x_row / sqrt(mean(x_row^2) + eps)
@triton.jit
def triton_rmsnorm_row(x_ptr, out_ptr, weight_ptr, eps, B, H, S,
                        x_stride0, x_stride1,
                        out_stride0, out_stride1,
                        BLOCK_D: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)

    row_start = b * S
    sum_sq = 0.0
    # First pass: compute sum of squares over S
    for d in range(0, S, BLOCK_D):
        offs = d + tl.arange(0, BLOCK_D)
        mask = offs < S
        x = tl.load(x_ptr + row_start + offs * x_stride1, mask=mask, other=0.0)
        sum_sq += tl.sum(x * x, axis=0)

    mean = sum_sq / S
    inv_rms = 1.0 / tl.sqrt(mean + eps)
    scale = tl.load(weight_ptr + h) * inv_rms

    # Second pass: write normalized row
    for d in range(0, S, BLOCK_D):
        offs = d + tl.arange(0, BLOCK_D)
        mask = offs < S
        x = tl.load(x_ptr + row_start + offs * x_stride1, mask=mask, other=0.0)
        y = x * scale
        tl.store(out_ptr + b * out_stride0 + h * out_stride1 + offs * out_stride1, y, mask=mask)

# Kernel 3: Apply RoPE to Q or K: q1, q2 = q[:64], q[64:], q_rot = [-q2, q1], q_out = q*cos + q_rot*sin
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

    # First half
    for d0 in range(0, head_dim // 2, BLOCK_D):
        d = d0
        offs = d + tl.arange(0, BLOCK_D)
        mask = offs < (head_dim // 2)
        x1 = tl.load(x_ptr + base_x + offs, mask=mask, other=0.0)
        # Load cos/sin for these offsets
        cos_vals = tl.load(cos_ptr + s * cos_stride0 + offs * cos_stride1, mask=mask, other=0.0)
        sin_vals = tl.load(sin_ptr + s * sin_stride0 + offs * sin_stride1, mask=mask, other=0.0)
        # Second half rotated
        x2 = tl.load(x_ptr + base_x + (head_dim // 2) + offs, mask=mask, other=0.0)
        q1 = -x2
        q2 = x1
        rotated = q1 * cos_vals + q2 * sin_vals
        out = x1 * cos_vals + rotated
        tl.store(out_ptr + base_out + offs, out, mask=mask)

# Kernel 4: Expand KV from 8 heads to 96 heads by repeating groups (groups = 96 // 8 = 12)
@triton.jit
def triton_expand_kv_groups(kv_ptr, out_ptr, B, Hkv, S, Hgrp, head_dim,
                            kv_stride0, kv_stride1, kv_stride2, kv_stride3,
                            out_stride0, out_stride1, out_stride2, out_stride3,
                            BLOCK_D: tl.constexpr):
    b = tl.program_id(0)
    hkv = tl.program_id(1)  # 0..7
    s = tl.program_id(2)    # 0..S-1
    group = tl.program_id(3)  # 0..Hgrp-1
    hout = hkv * Hgrp + group  # 0..95

    base_kv = b * kv_stride0 + hkv * kv_stride1 + s * kv_stride2
    base_out = b * out_stride0 + hout * out_stride3 + s * kv_stride2  # last dim stride2

    # Copy the vector of length head_dim
    for d0 in range(0, head_dim, BLOCK_D):
        d = d0 + tl.arange(0, BLOCK_D)
        mask = d < head_dim
        vals = tl.load(kv_ptr + base_kv + d * kv_stride3, mask=mask, other=0.0)
        tl.store(out_ptr + base_out + d * out_stride3, vals, mask=mask)

# Kernel 5: Row-wise attention: compute output for each (b,h,i)
@triton.jit
def triton_attention_row(Q_ptr, K_ptr, V_ptr, mask_ptr, Out_ptr,
                         B, H, S, head_dim,
                         Q_stride0, Q_stride1, Q_stride2,
                         K_stride0, K_stride1, K_stride2,
                         V_stride0, V_stride1, V_stride2,
                         Out_stride0, Out_stride1,
                         scaling,  # 1/sqrt(head_dim)
                         BLOCK_J: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    i = tl.program_id(2)  # i is the query position within the sequence

    # First pass: compute max for numerical stability
    m = -float('inf')
    for j0 in range(0, S, BLOCK_J):
        j = j0 + tl.arange(0, BLOCK_J)
        mask_j = j < S
        q = tl.load(Q_ptr + b * Q_stride0 + i * Q_stride1 + h * Q_stride2, mask=True, other=0.0)  # scalar load
        # Load K rows j and compute dot with q across head_dim
        scores = tl.zeros((BLOCK_J,), dtype=tl.float32)
        for kd in range(0, head_dim, BLOCK_J):
            d = kd + tl.arange(0, BLOCK_J)
            mask_d = d < head_dim
            k = tl.load(K_ptr + b * K_stride0 + j * K_stride1 + h * K_stride2 + d * K_stride3, mask=mask_j & mask_d, other=0.0)
            scores += q * k  # q is scalar, k is vector
        scores = scores * scaling
        # Apply causal mask: load mask[i, j]
        causal = tl.load(mask_ptr + i * mask_ptr.stride(0) + j * mask_ptr.stride(1), mask=mask_j, other=-float('inf'))
        scores = scores + causal
        # Mask out-of-range j
        scores = tl.where(mask_j, scores, -float('inf'))
        # Update max
        block_max = tl.max(scores, axis=0)
        m = tl.maximum(m, block_max)

    # Second pass: compute output
    acc = tl.zeros((), dtype=tl.float32)
    for j0 in range(0, S, BLOCK_J):
        j = j0 + tl.arange(0, BLOCK_J)
        mask_j = j < S
        q = tl.load(Q_ptr + b * Q_stride0 + i * Q_stride1 + h * Q_stride2)  # scalar
        scores = tl.zeros((BLOCK_J,), dtype=tl.float32)
        for kd in range(0, head_dim, BLOCK_J):
            d = kd + tl.arange(0, BLOCK_J)
            mask_d = d < head_dim
            k = tl.load(K_ptr + b * K_stride0 + j * K_stride1 + h * K_stride2 + d * K_stride3, mask=mask_j & mask_d, other=0.0)
            scores += q * k
        scores = scores * scaling
        causal = tl.load(mask_ptr + i * mask_ptr.stride(0) + j * mask_ptr.stride(1), mask=mask_j, other=-float('inf'))
        scores = scores + causal
        scores = tl.where(mask_j, scores, -float('inf'))
        # Subtract max
        scores = scores - m
        exp_scores = tl.exp(scores)
        # Load V[j] and accumulate
        v = tl.load(V_ptr + b * V_stride0 + j * V_stride1 + h * V_stride2, mask=mask_j, other=0.0)
        acc += tl.sum(exp_scores * v, axis=0)

    # Store output
    tl.store(Out_ptr + b * Out_stride0 + h * Out_stride1 + i, acc)

# Host code: ModelNew
class ModelNew(nn.Module):
    def __init__(self, batch_size, seq_length, num_attention_heads=96, num_key_value_heads=8, head_dim=128, rms_norm_eps=1e-6):
        super().__init__()
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.scaling = 1.0 / (head_dim ** 0.5)
        self.rms_norm_eps = rms_norm_eps

    def forward(self, hidden_states, q_proj_weight, q_proj_bias, k_proj_weight, k_proj_bias, v_proj_weight, v_proj_bias,
                o_proj_weight, q_norm_weight, k_norm_weight, cos, sin):
        # Ensure CUDA
        device = hidden_states.device
        B = hidden_states.shape[0]
        S = hidden_states.shape[1]

        # Prepare weights: reshape to match QKV projection shapes
        # Q, K, V: [B, S, H] where H=num_attention_heads=96
        H = self.num_attention_heads
        Hkv = self.num_key_value_heads

        # Allocate outputs for Q, K, V
        Q = torch.empty((B, S, H), dtype=hidden_states.dtype, device=device)
        K = torch.empty((B, S, Hkv), dtype=hidden_states.dtype, device=device)
        V = torch.empty((B, S, Hkv), dtype=hidden_states.dtype, device=device)

        # Launch linear row kernels for Q, K, V
        # Grid: (B, H, S)
        grid_linear = (B, H, S)
        triton_linear_row[grid_linear](
            hidden_states, q_proj_weight, q_proj_bias, Q,
            B, S, H, self.head_dim,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            q_proj_weight.stride(0), q_proj_weight.stride(1),
            Q.stride(0), Q.stride(1), Q.stride(2),
            BLOCK_K=128,
            num_warps=4
        )
        triton_linear_row[grid_linear](
            hidden_states, k_proj_weight, k_proj_bias, K,
            B, S, Hkv, self.head_dim,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            k_proj_weight.stride(0), k_proj_weight.stride(1),
            K.stride(0), K.stride(1), K.stride(2),
            BLOCK_K=128,
            num_warps=4
        )
        triton_linear_row[grid_linear](
            hidden_states, v_proj_weight, v_proj_bias, V,
            B, S, Hkv, self.head_dim,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            v_proj_weight.stride(0), v_proj_weight.stride(1),
            V.stride(0), V.stride(1), V.stride(2),
            BLOCK_K=128,
            num_warps=4
        )

        # RMSNorm on Q and K
        # Treat Q and K as [B*H, S]
        Q_2d = Q.view(B * H, S)
        K_2d = K.view(B * Hkv, S)
        Q_norm = torch.empty_like(Q_2d)
        K_norm = torch.empty_like(K_2d)

        grid_rms = (B, H)
        triton_rmsnorm_row[grid_rms](
            Q_2d, Q_norm, q_norm_weight, self.rms_norm_eps,
            B, H, S,
            Q_2d.stride(0), Q_2d.stride(1),
            Q_norm.stride(0), Q_norm.stride(1),
            BLOCK_D=128,
            num_warps=4
        )
        triton_rmsnorm_row[grid_rms](
            K_2d, K_norm, k_norm_weight, self.rms_norm_eps,
            B, Hkv, S,
            K_2d.stride(0), K_2d.stride(1),
            K_norm.stride(0), K_norm.stride(1),
            BLOCK_D=128,
            num_warps=4
        )
        Q = Q_norm.view(B, H, S)
        K = K_norm.view(B, Hkv, S)

        # Apply RoPE to Q and K
        Q_rot = torch.empty_like(Q)
        K_rot = torch.empty_like(K)

        grid_rope = (B, H, S)
        triton_rope_row[grid_rope](
            Q, cos, sin, Q_rot,
            B, H, S, self.head_dim,
            Q.stride(0), Q.stride(1), Q.stride(2),
            cos.stride(0), cos.stride(1),
            sin.stride(0), sin.stride(1),
            Q_rot.stride(0), Q_rot.stride(1), Q_rot.stride(2),
            BLOCK_D=64,
            num_warps=4
        )
        triton_rope_row[grid_rope](
            K, cos, sin, K_rot,
            B, Hkv, S, self.head_dim,
            K.stride(0), K.stride(1), K.stride(2),
            cos.stride(0), cos.stride(1),
            sin.stride(0), sin.stride(1),
            K_rot.stride(0), K_rot.stride(1), K_rot.stride(2),
            BLOCK_D=64,
            num_warps=4
        )

        # Expand KV heads to 96 groups
        groups = H // Hkv  # 96 // 8 = 12
        K_expanded = torch.empty((B, H, S, self.head_dim), dtype=K.dtype, device=device)
        V_expanded = torch.empty((B, H, S, self.head_dim), dtype=V.dtype, device=device)

        grid_expand = (B, Hkv, S, groups)
        triton_expand_kv_groups[grid_expand](
            K, K_expanded, B, Hkv, S, groups, self.head_dim,
            K_rot.stride(0), K_rot.stride(1), K_rot.stride(2), K_rot.stride(3),
            K_expanded.stride(0), K_expanded.stride(1), K_expanded.stride(2), K_expanded.stride(3),
            BLOCK_D=128,
            num_warps=4
        )
        triton_expand_kv_groups[grid_expand](
            V, V_expanded, B, Hkv, S, groups, self.head_dim,
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            V_expanded.stride(0), V_expanded.stride(1), V_expanded.stride(2), V_expanded.stride(3),
            BLOCK_D=128,
            num_warps=4
        )

        # Compute attention output with Triton row-wise kernel: Out [B, H, S]
        Out = torch.empty((B, H, S), dtype=torch.float32, device=device)

        # Causal mask: [S, S], -inf for i >= j
        causal_mask = torch.triu(
            torch.full((S, S), float('-inf'), device=device, dtype=torch.float32),
            diagonal=1
        )

        grid_attn = (B, H, S)
        triton_attention_row[grid_attn](
            Q_rot, K_expanded, V_expanded, causal_mask, Out,
            B, H, S, self.head_dim,
            Q_rot.stride(0), Q_rot.stride(1), Q_rot.stride(2),
            K_expanded.stride(0), K_expanded.stride(1), K_expanded.stride(2),
            V_expanded.stride(0), V_expanded.stride(1), V_expanded.stride(2),
            Out.stride(0), Out.stride(1),
            self.scaling,
            BLOCK_J=128,
            num_warps=4
        )

        # Reshape and final output projection
        out = Out.transpose(1, 2).contiguous()  # [B, S, H]
        out = out.reshape(B, S, H * self.head_dim)  # [B, S, 12288]
        # o_proj_weight: [H, H*head_dim] = [96, 12288]
        o = nn.functional.linear(out, o_proj_weight, None)

        return o


def run(*args):
    return ModelNew()(*args)
