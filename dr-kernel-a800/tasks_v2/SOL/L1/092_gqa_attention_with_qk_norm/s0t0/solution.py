import torch
import triton
import triton.language as tl


@triton.jit
def linear_proj_kernel(
    x_ptr,            # *ptr to hidden states: [B, L, H_in]
    weight_ptr,       # *ptr to weight: [N_out, H_in, K] where K=H_in
    bias_ptr,         # *ptr to bias: [N_out] or None (we'll pass pointer and check)
    y_ptr,            # *ptr to output: [B, L, N_out]
    B: tl.constexpr,  # batch size
    L: tl.constexpr,  # seq length
    H_in: tl.constexpr,  # input hidden dim
    K: tl.constexpr,      # reduction dim (H_in)
    N_out: tl.constexpr,  # output dim (head_dim)
    stride_x_b, stride_x_l, stride_x_h,
    stride_w_n, stride_w_h, stride_w_k,
    stride_y_b, stride_y_l, stride_y_n,
    HAS_BIAS: tl.constexpr,
    BLOCK_K: tl.constexpr = 64,
    BLOCK_N: tl.constexpr = 128,
):
    # program ids
    b = tl.program_id(0)
    l = tl.program_id(1)
    n = tl.program_id(2)

    # compute accumulator
    acc = tl.zeros((), dtype=tl.float32)

    # loop over reduction dimension K
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K

        # load x[b, l, k_offsets]
        x_row_ptrs = x_ptr + b * stride_x_b + l * stride_x_l + k_offsets * stride_x_h
        x_vals = tl.load(x_row_ptrs, mask=mask_k, other=0.0).to(tl.float32)  # upcast for stability

        # load weight[n, k_offsets, :]
        w_row_ptrs = weight_ptr + n * stride_w_n + k_offsets * stride_w_k
        w_vals = tl.load(w_row_ptrs, mask=mask_k, other=0.0).to(tl.float32)

        # outer product accumulate
        acc += tl.sum(x_vals[:, None] * w_vals[None, :], axis=0)

    # add bias if any
    if HAS_BIAS:
        bval = tl.load(bias_ptr + n).to(tl.float32)
        acc += bval

    # store to y[b, l, n]
    y_ptr_out = y_ptr + b * stride_y_b + l * stride_y_l + n * stride_y_n
    tl.store(y_ptr_out, acc)


@triton.jit
def rmsnorm_kernel(
    x_ptr,           # *ptr to input: [B, L, H]
    weight_ptr,      # *ptr to weight: [H]
    y_ptr,           # *ptr to output: [B, L, H]
    B: tl.constexpr,
    L: tl.constexpr,
    H: tl.constexpr,
    stride_x_b, stride_x_l, stride_x_h,
    stride_y_b, stride_y_l, stride_y_h,
    eps,             # float32 scalar
    BLOCK_H: tl.constexpr = 128,
):
    # each program normalizes one (b, l) row across H
    b = tl.program_id(0)
    l = tl.program_id(1)

    sumsq = 0.0
    for h_start in range(0, H, BLOCK_H):
        h_offsets = h_start + tl.arange(0, BLOCK_H)
        mask_h = h_offsets < H

        x_row_ptrs = x_ptr + b * stride_x_b + l * stride_x_l + h_offsets * stride_x_h
        x_vals = tl.load(x_row_ptrs, mask=mask_h, other=0.0).to(tl.float32)
        sumsq += tl.sum(x_vals * x_vals, axis=0)

    mean = sumsq / H
    scale = 1.0 / tl.sqrt(mean + eps)

    for h_start in range(0, H, BLOCK_H):
        h_offsets = h_start + tl.arange(0, BLOCK_H)
        mask_h = h_offsets < H

        x_row_ptrs = x_ptr + b * stride_x_b + l * stride_x_l + h_offsets * stride_x_h
        weight_ptrs = weight_ptr + h_offsets
        x_vals = tl.load(x_row_ptrs, mask=mask_h, other=0.0).to(tl.float32)
        w_vals = tl.load(weight_ptrs, mask=mask_h, other=1.0).to(tl.float32)

        y_vals = x_vals * scale * w_vals

        y_row_ptrs = y_ptr + b * stride_y_b + l * stride_y_l + h_offsets * stride_y_h
        tl.store(y_row_ptrs, y_vals, mask=mask_h)


@triton.jit
def rotate_qk_kernel(
    z_ptr,           # *ptr to input tensor to rotate: [B, L, H]
    cos_ptr,         # *ptr to cos: [L, H/2]
    sin_ptr,         # *ptr to sin: [L, H/2]
    y_ptr,           # *ptr to output rotated tensor: [B, L, H]
    B: tl.constexpr,
    L: tl.constexpr,
    H: tl.constexpr,
    stride_z_b, stride_z_l, stride_z_h,
    stride_c_l, stride_c_h,
    stride_s_l, stride_s_h,
    stride_y_b, stride_y_l, stride_y_h,
):
    # each program handles one (b, l) row and rotates across H
    b = tl.program_id(0)
    l = tl.program_id(1)

    for h_start in range(0, H, 128):  # we process in 128 chunks; here H=128, so one iteration
        h_offsets = h_start + tl.arange(0, 128)
        mask_h = h_offsets < H

        z_row_ptrs = z_ptr + b * stride_z_b + l * stride_z_l + h_offsets * stride_z_h
        z_vals = tl.load(z_row_ptrs, mask=mask_h, other=0.0).to(tl.float32)

        # split into two halves
        h1 = h_offsets[:64]
        h2 = h_offsets[64:]
        mask_h1 = h1 < 64
        mask_h2 = h2 < 64  # this is always true for 64, but keep for generality

        # load cos and sin for half dimension
        # cos[l, :64], sin[l, :64]
        c_ptrs = cos_ptr + l * stride_c_l + h1 * stride_c_h
        s_ptrs = sin_ptr + l * stride_s_l + h1 * stride_s_h
        cos_vals = tl.load(c_ptrs, mask=mask_h1, other=0.0).to(tl.float32)
        sin_vals = tl.load(s_ptrs, mask=mask_h1, other=0.0).to(tl.float32)

        # first half: original
        z1 = z_vals[h1]
        # second half: original second half is z_vals[64 + h2]
        z2 = z_vals[64 + h2]

        # rotated second half: -z2 + z1 * cos + (-z1) * sin => -z2 + z1*(cos - sin)
        z1_rot = z1 * (cos_vals - sin_vals)
        # first half: -z2 (note: z1 is not used here, but this is fine because we reconstruct rotated sequence)
        # But we need to form the rotated vector: new order is [h1_rotated, h2_rotated] where
        # h1_rotated = z1 * (cos - sin), h2_rotated = -z2
        # However, in the original code, q_rot_half = cat(-q2, q1); so for [64,64] we have [-h2, h1].
        # Concatenation is in order: [first half after rotation, second half after rotation].
        # Since we split, first half rotated is z1*(cos-sin), second half rotated is -z2.
        # But we need to map back to original positions 64+ to avoid overlap.
        # Let's reconstruct rotated positions by assigning:
        # rotated[h1] = z1 * (cos - sin)
        # rotated[64 + h2] = -z2
        # We'll build a vector of length H and store it.

        # Prepare output vector for this chunk
        rotated = tl.zeros((128,), dtype=tl.float32)
        # fill first 64
        rotated[h1] = z1 * (cos_vals - sin_vals)
        # fill next 64
        rotated[64 + h2] = -z2

        # store to y[b, l, h_offsets]
        y_row_ptrs = y_ptr + b * stride_y_b + l * stride_y_l + h_offsets * stride_y_h
        tl.store(y_row_ptrs, rotated, mask=mask_h)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters; we rely on inputs provided at forward.

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
        # hidden_states: [B, L, H_in], H_in = 768 in typical MHA
        B, L, H_in = hidden_states.shape
        head_dim = 128  # given in original code
        scaling = 1.0 / (head_dim ** 0.5)

        # Ensure contiguity for simpler strides
        hidden_states = hidden_states.contiguous()
        q_proj_weight = q_proj_weight.contiguous()
        k_proj_weight = k_proj_weight.contiguous()
        v_proj_weight = v_proj_weight.contiguous()
        o_proj_weight = o_proj_weight.contiguous()
        q_norm_weight = q_norm_weight.contiguous()
        k_norm_weight = k_norm_weight.contiguous()
        cos = cos.contiguous()
        sin = sin.contiguous()

        # 1) Q projection: [B, L, 128]
        query = torch.empty((B, L, head_dim), device=hidden_states.device, dtype=hidden_states.dtype)
        linear_proj_kernel[(B, L, head_dim)](
            hidden_states, q_proj_weight, q_proj_bias if q_proj_bias is not None else torch.empty(1, device=hidden_states.device, dtype=hidden_states.dtype),
            query,
            B, L, H_in, H_in, head_dim,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            q_proj_weight.stride(0), q_proj_weight.stride(1), q_proj_weight.stride(2),
            query.stride(0), query.stride(1), query.stride(2),
            1 if q_proj_bias is not None else 0,
            num_warps=4, num_stages=2,
        )

        # 2) K projection: [B, L, 128]
        key = torch.empty((B, L, head_dim), device=hidden_states.device, dtype=hidden_states.dtype)
        linear_proj_kernel[(B, L, head_dim)](
            hidden_states, k_proj_weight, k_proj_bias if k_proj_bias is not None else torch.empty(1, device=hidden_states.device, dtype=hidden_states.dtype),
            key,
            B, L, H_in, H_in, head_dim,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            k_proj_weight.stride(0), k_proj_weight.stride(1), k_proj_weight.stride(2),
            key.stride(0), key.stride(1), key.stride(2),
            1 if k_proj_bias is not None else 0,
            num_warps=4, num_stages=2,
        )

        # 3) V projection: [B, L, 128] (no bias)
        value = torch.empty((B, L, head_dim), device=hidden_states.device, dtype=hidden_states.dtype)
        linear_proj_kernel[(B, L, head_dim)](
            hidden_states, v_proj_weight, torch.empty(1, device=hidden_states.device, dtype=hidden_states.dtype),
            value,
            B, L, H_in, H_in, head_dim,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            v_proj_weight.stride(0), v_proj_weight.stride(1), v_proj_weight.stride(2),
            value.stride(0), value.stride(1), value.stride(2),
            0,
            num_warps=4, num_stages=2,
        )

        # 4) RMSNorm for query and key (upcast to float32 for stability)
        query_fp32 = query.to(torch.float32)
        key_fp32 = key.to(torch.float32)

        query_norm = torch.empty_like(query_fp32)
        key_norm = torch.empty_like(key_fp32)

        rmsnorm_kernel[(B, L)](
            query_fp32, q_norm_weight, query_norm,
            B, L, head_dim,
            query_fp32.stride(0), query_fp32.stride(1), query_fp32.stride(2),
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2),
            rms_norm_eps,
            num_warps=2, num_stages=2,
        )

        rmsnorm_kernel[(B, L)](
            key_fp32, k_norm_weight, key_norm,
            B, L, head_dim,
            key_fp32.stride(0), key_fp32.stride(1), key_fp32.stride(2),
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2),
            rms_norm_eps,
            num_warps=2, num_stages=2,
        )

        # 5) Apply Q and K rotation using Triton (upcast to float32)
        query_rot = torch.empty_like(query_norm)
        key_rot = torch.empty_like(key_norm)

        rotate_qk_kernel[(B, L)](
            query_norm, cos, sin, query_rot,
            B, L, head_dim,
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2),
            cos.stride(0), cos.stride(1),
            sin.stride(0), sin.stride(1),
            query_rot.stride(0), query_rot.stride(1), query_rot.stride(2),
            num_warps=2, num_stages=2,
        )

        rotate_qk_kernel[(B, L)](
            key_norm, cos, sin, key_rot,
            B, L, head_dim,
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2),
            cos.stride(0), cos.stride(1),
            sin.stride(0), sin.stride(1),
            key_rot.stride(0), key_rot.stride(1), key_rot.stride(2),
            num_warps=2, num_stages=2,
        )

        # Cast back to original dtype for subsequent ops (PyTorch matmul/softmax handle dtype)
        query_rot = query_rot.to(query.dtype)
        key_rot = key_rot.to(key.dtype)
        value = value.to(torch.float32)  # keep value in fp32 for matmul stability

        # 6) Reshape for attention:
        # query: [B, 96, L, head_dim], key/value: [B, 8, L, head_dim]
        # But code does: view(batch, seq_length, num_attention_heads, head_dim)
        query = query_rot.view(B, L, 96, head_dim)
        key = key_rot.view(B, L, 8, head_dim)
        value = value.view(B, L, 8, head_dim)

        # 7) GQA: expand key/value to match 96 attention heads using groups
        # num_key_value_groups = 96 // 8 = 12
        key_expanded = key[:, :, None, :, :].expand(B, 8, 12, L, head_dim).reshape(B, 96, L, head_dim)
        value_expanded = value[:, :, None, :, :].expand(B, 8, 12, L, head_dim).reshape(B, 96, L, head_dim)

        # 8) Compute attention scores (PyTorch)
        # query: [B, 96, L, 128], key_expanded: [B, 96, L, 128]
        attn_scores = torch.matmul(query, key_expanded.transpose(2, 3)) * scaling  # [B, 96, L, L]
        attn_scores = attn_scores.to(torch.float32)

        # 9) Apply causal mask
        causal_mask = torch.triu(
            torch.full((L, L), float('-inf'), device=hidden_states.device, dtype=torch.float32),
            diagonal=1
        )
        attn_scores = attn_scores + causal_mask  # broadcasting over B and 96 heads

        # 10) Softmax over last dim (sequence length)
        attn_probs = F.softmax(attn_scores, dim=-1).to(torch.float32)  # keep fp32 for stability

        # 11) Compute output
        attn_output = torch.matmul(attn_probs, value_expanded)  # [B, 96, L, 128]

        # 12) Transpose and reshape to [B, L, 96*128]
        attn_output = attn_output.transpose(1, 2).contiguous()  # [B, L, 96, 128]
        attn_output = attn_output.reshape(B, L, 96 * head_dim)  # [B, L, 12288]

        # 13) Output projection (no bias)
        output = F.linear(attn_output, o_proj_weight, None)
        return output


def run(*args):
    return ModelNew()(*args)
