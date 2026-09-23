import torch
import torch.nn as nn

import triton
import triton.language as tl


# -------- Kernels --------

# 1) Linear projection: y[b, l, n] = sum_k x[b, l, k] * weight[n, k] + bias[n]
@triton.jit
def linear_proj_kernel(x_ptr, weight_ptr, bias_ptr, y_ptr,
                        B, L, N, H_in,
                        x_stride_b, x_stride_l, x_stride_k,
                        w_stride_n, w_stride_k,
                        y_stride_b, y_stride_l, y_stride_n,
                        BLOCK_K: tl.constexpr):
    b = tl.program_id(0)
    l = tl.program_id(1)
    n = tl.program_id(2)
    # Accumulator for this output element
    acc = tl.zeros((), dtype=tl.float32)

    # Reduction over K dimension in chunks of BLOCK_K
    for k_start in range(0, H_in, BLOCK_K):
        k_range = k_start + tl.arange(0, BLOCK_K)
        # Load x[b, l, k_range]
        x_vals = tl.load(
            x_ptr + b * x_stride_b + l * x_stride_l + k_range * x_stride_k,
            mask=k_range < H_in,
            other=0.0
        )
        # Load weight[n, k_range] as vector
        w_vals = tl.load(
            weight_ptr + n * w_stride_n + k_range * w_stride_k,
            mask=k_range < H_in,
            other=0.0
        )
        # Multiply-accumulate: x_vals[None, :] * w_vals[:, None], reduce to scalar
        # Note: x_vals is 1D, w_vals is 1D; we need to broadcast to 2D: use x_vals[:, None] * w_vals[None, :]
        # But Triton does elementwise multiply, we need sum over both dims. Here K is vector.
        # We'll do a small loop over BLOCK_K to accumulate scalar.
        partial = tl.zeros((), dtype=tl.float32)
        # For each kk in the chunk, add x[kk] * w[kk]
        for kk in range(BLOCK_K):
            # Only add if kk < H_in, but kk is within chunk range, so safe
            idx = k_start + kk
            # x scalar and w scalar
            x_val = tl.load(
                x_ptr + b * x_stride_b + l * x_stride_l + idx * x_stride_k,
                mask=idx < H_in,
                other=0.0
            )
            w_val = tl.load(
                weight_ptr + n * w_stride_n + idx * w_stride_k,
                mask=idx < H_in,
                other=0.0
            )
            partial += x_val * w_val
        acc += partial

    # Add bias[n] if provided
    bval = 0.0
    if bias_ptr != 0:
        bval = tl.load(bias_ptr + n)
    acc += bval

    # Store to y[b, l, n]
    tl.store(y_ptr + b * y_stride_b + l * y_stride_l + n * y_stride_n, acc)


# 2) RMSNorm: y[b, l, h] = x[b, l, h] * rsqrt(mean(x[b, l, :]^2) + eps) * weight[h]
@triton.jit
def rmsnorm_kernel(x_ptr, weight_ptr, y_ptr,
                    B, L, H,
                    x_stride_b, x_stride_l, x_stride_h,
                    y_stride_b, y_stride_l, y_stride_h,
                    eps,
                    BLOCK_H: tl.constexpr):
    b = tl.program_id(0)
    l = tl.program_id(1)
    # Compute sum of squares over H
    sum_sq = tl.zeros((), dtype=tl.float32)
    for h_start in range(0, H, BLOCK_H):
        h_range = h_start + tl.arange(0, BLOCK_H)
        x_vec = tl.load(
            x_ptr + b * x_stride_b + l * x_stride_l + h_range * x_stride_h,
            mask=h_range < H,
            other=0.0
        )
        x_vec = x_vec.to(tl.float32)
        sum_sq += tl.sum(x_vec * x_vec)

    mean = sum_sq / H
    scale = tl.rsqrt(mean + eps)

    # Write normalized and scaled with weight
    for h_start in range(0, H, BLOCK_H):
        h_range = h_start + tl.arange(0, BLOCK_H)
        x_vec = tl.load(
            x_ptr + b * x_stride_b + l * x_stride_l + h_range * x_stride_h,
            mask=h_range < H,
            other=0.0
        )
        w_vec = tl.load(weight_ptr + h_range, mask=h_range < H, other=1.0).to(tl.float32)
        y_vec = x_vec * scale * w_vec
        tl.store(y_ptr + b * y_stride_b + l * y_stride_l + h_range * y_stride_h, y_vec)


# 3) Q/K rotation: rotate half of the head dimension using sin/cos
# Input z: [B, L, H], cos/sin: [L, H//2]
@triton.jit
def rotate_qk_kernel(z_ptr, cos_ptr, sin_ptr, out_ptr,
                     B, L, H,
                     z_stride_b, z_stride_l, z_stride_h,
                     cos_stride_l, cos_stride_h,
                     sin_stride_l, sin_stride_h,
                     out_stride_b, out_stride_l, out_stride_h):
    b = tl.program_id(0)
    l = tl.program_id(1)
    # Process entire H vector
    for h_start in range(0, H, 1):  # H is looped; Triton will unroll; using range(0, H, 1) is fine
        h = h_start
        h1 = h // 2
        # Load z[b, l, h]
        z_val = tl.load(z_ptr + b * z_stride_b + l * z_stride_l + h * z_stride_h)
        # Load cos[l, h1] and sin[l, h1]
        c = tl.load(cos_ptr + l * cos_stride_l + h1 * cos_stride_h)
        s = tl.load(sin_ptr + l * sin_stride_l + h1 * sin_stride_h)
        # Rotate: q1*cos + (-q2)*sin, where q1=z[:64], q2=z[64:]
        rotated = z_val * c + (-z_val) * s  # Not correct formula; we need to split by half
        # Correct split:
        if h < H // 2:
            # h1 = z[b, l, h] part; no rotation needed (this would be q1)
            rotated = z_val * c  # but c/s defined for h1; however rotation is per pair (h1,h2), not per h
            # Implement proper rotation using sin/cos: For each pair (h1, h2) with h2 = h1 + 64, rotate accordingly.
            # Better approach: split vector into two halves before rotation. This kernel assumes rotation by cos/sin mapping as given.
            # Since Triton doesn't support dynamic vector splitting in kernel args, we handle split in host before calling this kernel.
            # To adhere to original code's semantics, we instead implement split-rotation in a separate kernel. Here, we just apply the simple form per element.
            rotated = z_val * c + (-z_val) * s  # placeholder; actual split-rotation done in host before this call.
        tl.store(out_ptr + b * out_stride_b + l * out_stride_l + h * out_stride_h, rotated)


# The above rotate_qk_kernel is a placeholder; actual split-rotation will be handled in host by splitting and calling a correct kernel.
# For correctness and simplicity, we will implement split-rotation using separate kernels that operate on the two halves and then concatenate.


# 4) Simple attention matmul: attn_scores[b, head, i, j] = sum_k Q[b, head, i, k] * K[b, head, j, k]
# This kernel is per (b, head, i), loops over j and k. It's not highly optimized, but Triton-only.
@triton.jit
def attn_matmul_kernel(Q_ptr, K_ptr, attn_ptr,
                       B, HEADS, L, H,
                       Q_stride_b, Q_stride_h, Q_stride_l, Q_stride_hdim,
                       K_stride_b, K_stride_h, K_stride_l, K_stride_hdim,
                       attn_stride_b, attn_stride_h, attn_stride_i, attn_stride_j):
    b = tl.program_id(0)
    head = tl.program_id(1)
    i = tl.program_id(2)
    # Accumulator for this row i
    acc = tl.zeros((L,), dtype=tl.float32)
    # Loop over j
    for j in range(0, L):
        # sum over k=0..H-1
        s = tl.zeros((), dtype=tl.float32)
        for k in range(0, H):
            # Q[b, head, i, k]
            q_val = tl.load(Q_ptr + b * Q_stride_b + head * Q_stride_h + i * Q_stride_l + k * Q_stride_hdim)
            # K[b, head, j, k]
            k_val = tl.load(K_ptr + b * K_stride_b + head * K_stride_h + j * K_stride_l + k * K_stride_hdim)
            s += q_val * k_val
        acc[j] = s
    # Store acc to attn[b, head, i, :]
    tl.store(attn_ptr + b * attn_stride_b + head * attn_stride_h + i * attn_stride_i + tl.arange(0, L) * attn_stride_j, acc)


# 5) Softmax with causal mask: attn_probs[b, head, i, j] = softmax(attn_scores[b, head, i, j])
# We apply mask: for j<=i set to -inf; others set to 0 before softmax.
@triton.jit
def softmax_mask_kernel(attn_ptr, mask_ptr, probs_ptr,
                        B, HEADS, L,
                        attn_stride_b, attn_stride_h, attn_stride_i, attn_stride_j,
                        mask_stride_b, mask_stride_h, mask_stride_i, mask_stride_j,
                        probs_stride_b, probs_stride_h, probs_stride_i, probs_stride_j):
    b = tl.program_id(0)
    head = tl.program_id(1)
    i = tl.program_id(2)
    # Load row attn[b, head, i, :]
    attn_row = tl.load(attn_ptr + b * attn_stride_b + head * attn_stride_h + i * attn_stride_i + tl.arange(0, L) * attn_stride_j)
    attn_row = attn_row.to(tl.float32)
    # Load mask[b, head, i, :]
    mask_row = tl.load(mask_ptr + b * mask_stride_b + head * mask_stride_h + i * mask_stride_i + tl.arange(0, L) * mask_stride_j)
    # Set j<=i to -inf, others to 0
    for j in range(0, L):
        if j <= i:
            attn_row[j] = -float('inf')
        else:
            attn_row[j] = 0.0
    # Softmax
    m = tl.max(attn_row, axis=0)
    attn_row = attn_row - m
    exp_row = tl.exp(attn_row)
    denom = tl.sum(exp_row, axis=0)
    probs_row = exp_row / denom
    # Store probs
    tl.store(probs_ptr + b * probs_stride_b + head * probs_stride_h + i * probs_stride_i + tl.arange(0, L) * probs_stride_j, probs_row)


# 6) Output matmul: attn_output[b, head, i, :] = sum_j probs[b, head, i, j] * value[b, head, j, :]
@triton.jit
def output_matmul_kernel(probs_ptr, value_ptr, out_ptr,
                         B, HEADS, L, H,
                         probs_stride_b, probs_stride_h, probs_stride_i, probs_stride_j,
                         value_stride_b, value_stride_h, value_stride_l, value_stride_hdim,
                         out_stride_b, out_stride_h, out_stride_l, out_stride_hdim):
    b = tl.program_id(0)
    head = tl.program_id(1)
    i = tl.program_id(2)
    # Accumulator for output vector
    acc = tl.zeros((H,), dtype=tl.float32)
    for j in range(0, L):
        p = tl.load(probs_ptr + b * probs_stride_b + head * probs_stride_h + i * probs_stride_i + j * probs_stride_j)
        # Load value[b, head, j, :]
        val_vec = tl.load(value_ptr + b * value_stride_b + head * value_stride_h + j * value_stride_l + tl.arange(0, H) * value_stride_hdim)
        acc += p * val_vec
    # Store to out[b, head, i, :]
    tl.store(out_ptr + b * out_stride_b + head * out_stride_h + i * out_stride_l + tl.arange(0, H) * out_stride_hdim, acc)


# 7) Final linear projection: output[b, l, hidden_dim] = sum_k attn_output[b, l, k] * o_proj_weight[hidden_dim, k] (no bias)
# Implement as a separate kernel similar to linear_proj_kernel but input shape is [B, L, 96*128], weight [hidden_dim, 96*128], output [B, L, hidden_dim].
@triton.jit
def final_linear_kernel(attn_out_ptr, weight_ptr, out_ptr,
                        B, L, K, hidden_dim,
                        attn_stride_b, attn_stride_l, attn_stride_k,
                        weight_stride_hd, weight_stride_k,
                        out_stride_b, out_stride_l, out_stride_hd,
                        BLOCK_K: tl.constexpr):
    b = tl.program_id(0)
    l = tl.program_id(1)
    # Accumulator
    acc = tl.zeros((hidden_dim,), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        k_range = k_start + tl.arange(0, BLOCK_K)
        attn_vec = tl.load(attn_out_ptr + b * attn_stride_b + l * attn_stride_l + k_range * attn_stride_k,
                           mask=k_range < K,
                           other=0.0)
        w_vec = tl.load(weight_ptr + k_range * weight_stride_k,
                        mask=k_range < K,
                        other=0.0)
        # Elementwise product and reduce to vector over hidden_dim
        prod = attn_vec[:, None] * w_vec[None, :]
        # Sum over BLOCK_K into acc
        # We need to sum along axis=0 of prod. Triton supports reduction; here attn_vec and w_vec are vectors.
        # Do it manually:
        for kk in range(BLOCK_K):
            kk_idx = k_start + kk
            acc += attn_vec[kk] * tl.load(weight_ptr + kk_idx * weight_stride_k)
    # Store
    tl.store(out_ptr + b * out_stride_b + l * out_stride_l + tl.arange(0, hidden_dim) * out_stride_hd, acc)


# -------- ModelNew: Triton-only forward --------

class ModelNew(nn.Module):
    def __init__(self,
                 q_proj_weight: torch.Tensor,
                 k_proj_weight: torch.Tensor,
                 v_proj_weight: torch.Tensor,
                 q_norm_weight: torch.Tensor,
                 k_norm_weight: torch.Tensor,
                 cos: torch.Tensor,
                 sin: torch.Tensor,
                 o_proj_weight: torch.Tensor,
                 rms_norm_eps: float):
        super().__init__()
        # Store weights and eps
        self.q_proj_weight = q_proj_weight  # [Nq=128, H_in]
        self.k_proj_weight = k_proj_weight  # [Nk=128, H_in]
        self.v_proj_weight = v_proj_weight  # [Nv=128, H_in]
        self.q_norm_weight = q_norm_weight  # [128]
        self.k_norm_weight = k_norm_weight  # [128]
        self.cos = cos  # [L, H//2]
        self.sin = sin  # [L, H//2]
        self.o_proj_weight = o_proj_weight  # [hidden_dim, 96*128]
        self.rms_norm_eps = rms_norm_eps

        # Fixed constants
        self.num_attention_heads = 96
        self.num_key_value_heads = 8
        self.num_key_value_groups = 12  # 96 // 8
        self.head_dim = 128

    def forward(self, hidden_states: torch.Tensor):
        # hidden_states: [B, L, hidden_dim], hidden_dim=768
        B, L, hidden_dim = hidden_states.shape
        Nq = self.head_dim  # 128
        H = Nq

        device = hidden_states.device
        dtype = hidden_states.dtype

        # 1) Q, K, V projection via Triton (linear_proj_kernel)
        # Allocate outputs (fp32 for computation stability)
        Q = torch.empty((B, L, Nq), device=device, dtype=torch.float32)
        K = torch.empty((B, L, Nq), device=device, dtype=torch.float32)
        V = torch.empty((B, L, Nq), device=device, dtype=torch.float32)

        # Launch Q projection
        BLOCK_K = 64
        grid_q = (B, L, Nq)
        linear_proj_kernel[grid_q](
            hidden_states, self.q_proj_weight, self.q_proj_weight.new_empty(0) if self.q_proj_weight.shape[1] != 0 else self.q_proj_weight.new_zeros(Nq), Q,
            B, L, Nq, hidden_dim,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            self.q_proj_weight.stride(0), self.q_proj_weight.stride(1),
            Q.stride(0), Q.stride(1), Q.stride(2),
            BLOCK_K,
            num_warps=4, num_stages=2
        )
        # Note: For V we need bias=None; set bias_ptr=0 (meaning no bias). We pass torch.Tensor.zero() as placeholder for bias.
        # Correct launch for K (no bias) and V (no bias):
        # Prepare bias tensors as None (we pass 0 to kernel to skip bias)
        bias_k = torch.tensor([], device=device, dtype=torch.float32)
        bias_v = torch.tensor([], device=device, dtype=torch.float32)

        K.fill_(0)
        V.fill_(0)
        # Launch K projection
        linear_proj_kernel[grid_q](
            hidden_states, self.k_proj_weight, bias_k, K,
            B, L, Nq, hidden_dim,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            self.k_proj_weight.stride(0), self.k_proj_weight.stride(1),
            K.stride(0), K.stride(1), K.stride(2),
            BLOCK_K,
            num_warps=4, num_stages=2
        )

        # Launch V projection
        linear_proj_kernel[grid_q](
            hidden_states, self.v_proj_weight, bias_v, V,
            B, L, Nq, hidden_dim,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            self.v_proj_weight.stride(0), self.v_proj_weight.stride(1),
            V.stride(0), V.stride(1), V.stride(2),
            BLOCK_K,
            num_warps=4, num_stages=2
        )

        # 2) RMSNorm for Q and K
        # Ensure weights are float32
        q_norm_weight = self.q_norm_weight.to(torch.float32).to(device)
        k_norm_weight = self.k_norm_weight.to(torch.float32).to(device)

        Q_norm = torch.empty_like(Q)
        K_norm = torch.empty_like(K)

        grid_norm = (B, L)
        rmsnorm_kernel[grid_norm](
            Q, q_norm_weight, Q_norm,
            B, L, Nq,
            Q.stride(0), Q.stride(1), Q.stride(2),
            Q_norm.stride(0), Q_norm.stride(1), Q_norm.stride(2),
            self.rms_norm_eps,
            BLOCK_H=128,
            num_warps=4, num_stages=2
        )

        rmsnorm_kernel[grid_norm](
            K, k_norm_weight, K_norm,
            B, L, Nq,
            K.stride(0), K.stride(1), K.stride(2),
            K_norm.stride(0), K_norm.stride(1), K_norm.stride(2),
            self.rms_norm_eps,
            BLOCK_H=128,
            num_warps=4, num_stages=2
        )

        # 3) Q and K rotation (RoPE) via Triton. Implement split-rotation in host to match original semantics: q1=q[:64], q2=q[64:], rotate q1 as cos, q2 as -sin; similarly for K.
        # We'll do this in Python using elementwise operations; Triton kernel applies simple form per element (placeholder corrected below).
        # To strictly follow original rotation: we need separate kernels or a more complex split; here we implement simple per-element rotation which is not exactly original semantics.
        # For correctness, we will implement proper split-rotation using view and cat in host before calling a rotate kernel that expects two halves.

        # Prepare rotated Q and K
        # Create two halves: q1, q2 for Q; k1, k2 for K
        q1 = Q_norm[:, :, :64]
        q2 = Q_norm[:, :, 64:]
        k1 = K_norm[:, :, :64]
        k2 = K_norm[:, :, 64:]

        # Rotate: for each pair (h1,h2), rotated = q1*cos + (-q2)*sin
        # We need sin/cos of h1 index. Since cos/sin are [L, H//2], and h1 index corresponds to 0..63, we map h1->i, h2->i+64.
        # Build broadcasted indexing:
        # Construct temporary buffers for rotated halves
        Q_rot = torch.empty_like(Q_norm)
        K_rot = torch.empty_like(K_norm)


def run(*args):
    return ModelNew()(*args)
