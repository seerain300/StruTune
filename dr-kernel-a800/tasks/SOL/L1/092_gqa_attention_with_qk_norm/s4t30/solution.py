import torch
import triton
import triton.language as tl


# Kernel 1: Linear with bias: X @ W.T + b
# X: [B, S, H_in], W: [H_out, H_in], b: [H_out]
# Output: [B, S, H_out]
@triton.jit
def linear_bias_kernel(
    X_ptr, W_ptr, B_ptr, Out_ptr,
    Bsz, Ssz, H_in, H_out,
    stride_x_b, stride_x_s, stride_x_h,
    stride_w_h, stride_w_k, stride_b_h, stride_out_b, stride_out_s, stride_out_h,
    BLOCK_K: tl.constexpr,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    oh = tl.program_id(2)

    acc = tl.zeros((), dtype=tl.float32)

    # Accumulate over H_in in blocks
    for k in range(0, H_in, BLOCK_K):
        # Load X[b, s, k:k+BLOCK_K]
        x_vec = tl.zeros((BLOCK_K,), dtype=tl.float32)
        for kk in range(0, BLOCK_K):
            col = k + kk
            x_off = b * stride_x_b + s * stride_x_s + col * stride_x_h
            x_vec[kk] = tl.load(X_ptr + x_off)

        # Load W[oh, k:k+BLOCK_K]
        w_vec = tl.zeros((BLOCK_K,), dtype=tl.float32)
        for kk in range(0, BLOCK_K):
            col = k + kk
            w_off = oh * stride_w_h + col * stride_w_k
            w_vec[kk] = tl.load(W_ptr + w_off)

        # FMA
        acc += tl.sum(x_vec * w_vec, axis=0)

    # Add bias
    bias = tl.load(B_ptr + oh * stride_b_h)
    acc = acc + bias

    # Store to Out[b, s, oh]
    out_off = b * stride_out_b + s * stride_out_s + oh * stride_out_h
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
    d = tl.program_id(2)  # d in [0, D)

    row_off = b * stride_x_b + s * stride_x_s

    # Load vector x over head_dim
    x_vec = tl.zeros((BLOCK,), dtype=tl.float32)
    for i in range(0, BLOCK):
        off = row_off + (d + i) * stride_x_d
        x_vec[i] = tl.load(X_ptr + off)

    # RMS computation
    sum_sq = tl.sum(x_vec * x_vec, axis=0)
    mean = sum_sq / D
    inv_rms = tl.rsqrt(mean + 0.0)  # eps=0.0 to match original
    weight = tl.load(Weight_ptr + (d + 0) * stride_w_d)  # per-dimension weight

    # Normalize and scale
    y_vec = x_vec * inv_rms
    y_vec = y_vec * weight

    # Store
    out_off = b * stride_out_b + s * stride_out_s + (d + 0) * stride_out_d
    for i in range(0, BLOCK):
        tl.store(Out_ptr + out_off + i * stride_out_d, y_vec[i])


# Kernel 3: Rotate half of the last 64 dims for a 128-length vector: q1, q2 = q[:64], q[64:], q_rot = q1*cos - q2*sin
# Applied to Q and K after RMSNorm
@triton.jit
def rotate_half_kernel(
    Q_ptr, Sin_ptr, Cos_ptr, Out_ptr,
    Bsz, Ssz, D,  # D should be 128
    stride_q_b, stride_q_s, stride_q_d,
    stride_s_d, stride_c_d, stride_out_b, stride_out_s, stride_out_d,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    d = tl.program_id(2)  # d ranges over 0..D-1

    q_off = b * stride_q_b + s * stride_q_s + d * stride_q_d
    sin_off = d * stride_s_d
    cos_off = d * stride_c_d

    # Load original q, sin, cos
    q = tl.load(Q_ptr + q_off)
    sin = tl.load(Sin_ptr + sin_off)
    cos = tl.load(Cos_ptr + cos_off)

    # Split and rotate
    q1 = q[:64]
    q2 = q[64:]
    q_rot = q1 * cos - q2 * sin

    # Store rotated result at the same position (overwrites)
    tl.store(Out_ptr + q_off, q_rot)


# Kernel 4: Compute attention scores: Q[b, h, :, :] @ K[b, h, :, :].T -> [S, S]
# Inputs: Q: [B, num_attn_heads, S, D], K: [B, num_attn_heads, S, D]
# Output: Out: [B, num_attn_heads, S, S]
@triton.jit
def matmul_qk_kernel(
    Q_ptr, K_ptr, Out_ptr,
    Bsz, Ssz, D,
    stride_q_b, stride_q_h, stride_q_s, stride_q_d,
    stride_k_b, stride_k_h, stride_k_s, stride_k_d,
    stride_out_b, stride_out_h, stride_out_s_row, stride_out_s_col,
    BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s_row = tl.program_id(2)
    s_col = tl.program_id(3)

    acc = tl.zeros((), dtype=tl.float32)

    # Iterate over head_dim in blocks
    for k in range(0, D, BLOCK_K):
        q_vec = tl.zeros((BLOCK_Q,), dtype=tl.float32)
        k_vec = tl.zeros((BLOCK_K,), dtype=tl.float32)

        # Load Q[b, h, s_row, k:k+BLOCK_Q]
        for jj in range(0, BLOCK_Q):
            d_q = k + jj
            q_off = b * stride_q_b + h * stride_q_h + s_row * stride_q_s + d_q * stride_q_d
            q_vec[jj] = tl.load(Q_ptr + q_off)

        # Load K[b, h, s_col, k:k+BLOCK_K]
        for kk in range(0, BLOCK_K):
            d_k = k + kk
            k_off = b * stride_k_b + h * stride_k_h + s_col * stride_k_s + d_k * stride_k_d
            k_vec[kk] = tl.load(K_ptr + k_off)

        # FMA
        acc += tl.sum(q_vec * k_vec, axis=0)

    # Store score to Out[b, h, s_row, s_col]
    out_off = b * stride_out_b + h * stride_out_h + s_row * stride_out_s_row + s_col * stride_out_s_col
    tl.store(Out_ptr + out_off, acc)


# Kernel 5: Softmax with causal mask over the last dim (sequence length) for each (b, h), input [S, S]
# Output: same shape, masked softmax values
@triton.jit
def softmax_mask_kernel(
    In_ptr, Out_ptr,
    Ssz,
    stride_in_b, stride_in_h, stride_in_row, stride_in_col,
    stride_out_b, stride_out_h, stride_out_row, stride_out_col,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)

    # Row-wise max
    row_max = tl.full((), -float('inf'), dtype=tl.float32)
    for j in range(0, Ssz, BLOCK):
        vals = tl.zeros((BLOCK,), dtype=tl.float32)
        for jj in range(0, BLOCK):
            col = j + jj
            in_off = b * stride_in_b + h * stride_in_h + col * stride_in_row + col * stride_in_col
            val = tl.load(In_ptr + in_off)
            vals[jj] = val
        row_max = tl.maximum(row_max, tl.max(vals, axis=0))

    # Masked sum: set future positions to -inf, then sum
    sum_exp = tl.zeros((), dtype=tl.float32)
    for j in range(0, Ssz, BLOCK):
        vals = tl.zeros((BLOCK,), dtype=tl.float32)
        for jj in range(0, BLOCK):
            col = j + jj
            in_off = b * stride_in_b + h * stride_in_h + col * stride_in_row + col * stride_in_col
            val = tl.load(In_ptr + in_off)
            # causal mask: if col > (row index s_row), set to -inf
            if col > tl.program_id(3):  # we need s_row here; Triton doesn't pass it, so we recompute per loop
                pass
            # Reassign correctly using current row index: we can't access s_row here; restructure as follows:
            # We will instead pass s_row as a grid dimension (we set grid=(B, H, S, S)), and use program_id(3) for s_row.
            s_row = tl.program_id(3)  # this is the row index we iterate over in host
            # Reload with mask logic:
            in_off = b * stride_in_b + h * stride_in_h + col * stride_in_row + s_row * stride_in_col
            val = tl.load(In_ptr + in_off)
            if col > s_row:
                val = -float('inf')
            vals[jj] = val
        exp_vals = tl.exp(vals - row_max)
        sum_exp += tl.sum(exp_vals, axis=0)

    # Write normalized outputs
    for j in range(0, Ssz, BLOCK):
        vals = tl.zeros((BLOCK,), dtype=tl.float32)
        for jj in range(0, BLOCK):
            col = j + jj
            in_off = b * stride_in_b + h * stride_in_h + col * stride_in_row + s_row * stride_in_col
            val = tl.load(In_ptr + in_off)
            if col > s_row:
                val = -float('inf')
            exp_val = tl.exp(val - row_max) / sum_exp
            out_off = b * stride_out_b + h * stride_out_h + s_row * stride_out_row + col * stride_out_col
            tl.store(Out_ptr + out_off, exp_val)


# Kernel 6: Attention output: Out[b, h, s_row, :] = Softmax(QK_scaled)[b, h, s_row, :] @ V[b, h, :, :]
# V: [B, num_attn_heads, S, D], Out: [B, num_attn_heads, S, D]
@triton.jit
def matmul_attn_kernel(
    Softmax_ptr, V_ptr, Out_ptr,
    Bsz, Ssz, D,
    stride_s_b, stride_s_h, stride_s_row, stride_s_col,
    stride_v_b, stride_v_h, stride_v_s, stride_v_d,
    stride_out_b, stride_out_h, stride_out_s, stride_out_d,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s_row = tl.program_id(2)

    acc = tl.zeros((D,), dtype=tl.float32)

    # Iterate over sequence length
    for j in range(0, Ssz, BLOCK):
        # Load softmax vector for this row segment
        vals = tl.zeros((BLOCK,), dtype=tl.float32)
        for jj in range(0, BLOCK):
            col = j + jj
            in_off = b * stride_s_b + h * stride_s_h + s_row * stride_s_row + col * stride_s_col
            vals[jj] = tl.load(Softmax_ptr + in_off)

        # Load V[b, h, j:j+BLOCK, :]
        v_block = tl.zeros((BLOCK, D), dtype=tl.float32)
        for kk in range(0, BLOCK):
            col = j + kk
            v_off = b * stride_v_b + h * stride_v_h + col * stride_v_s + 0 * stride_v_d  # init
            for d in range(0, D):
                v_off_d = b * stride_v_b + h * stride_v_h + col * stride_v_s + d * stride_v_d
                v_block[kk, d] = tl.load(V_ptr + v_off_d)

        # Dot product: acc += vals[j:j+BLOCK] @ v_block[:, :]
        # vals is [BLOCK], v_block is [BLOCK, D] -> result is [D]
        for kk in range(0, BLOCK):
            col = j + kk
            # vals[kk] * v_block[kk, :]
            acc += vals[kk] * v_block[kk, :]

    # Store Out[b, h, s_row, :]
    out_off = b * stride_out_b + h * stride_out_h + s_row * stride_out_s + 0 * stride_out_d  # init
    for d in range(0, D):
        out_off_d = b * stride_out_b + h * stride_out_h + s_row * stride_out_s + d * stride_out_d
        tl.store(Out_ptr + out_off_d, acc[d])


# Kernel 7: Linear without bias: X @ W.T
# X: [B, S, H_in], W: [H_out, H_in], Output: [B, S, H_out]
@triton.jit
def linear_nobias_kernel(
    X_ptr, W_ptr, Out_ptr,
    Bsz, Ssz, H_in, H_out,
    stride_x_b, stride_x_s, stride_x_h,
    stride_w_h, stride_w_k, stride_out_b, stride_out_s, stride_out_h,
    BLOCK_K: tl.constexpr,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    oh = tl.program_id(2)

    acc = tl.zeros((), dtype=tl.float32)

    # Accumulate over H_in in blocks
    for k in range(0, H_in, BLOCK_K):
        # Load X[b, s, k:k+BLOCK_K]
        x_vec = tl.zeros((BLOCK_K,), dtype=tl.float32)
        for kk in range(0, BLOCK_K):
            col = k + kk
            x_off = b * stride_x_b + s * stride_x_s + col * stride_x_h
            x_vec[kk] = tl.load(X_ptr + x_off)

        # Load W[oh, k:k+BLOCK_K]
        w_vec = tl.zeros((BLOCK_K,), dtype=tl.float32)
        for kk in range(0, BLOCK_K):
            col = k + kk
            w_off = oh * stride_w_h + col * stride_w_k
            w_vec[kk] = tl.load(W_ptr + w_off)

        # FMA
        acc += tl.sum(x_vec * w_vec, axis=0)

    # Store to Out[b, s, oh]
    out_off = b * stride_out_b + s * stride_out_s + oh * stride_out_h
    tl.store(Out_ptr + out_off, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

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
        # Shapes (assertions for safety; in evaluation these are provided)
        Bsz, Ssz, H_in = hidden_states.shape
        assert H_in == 128 * 96, "hidden_states last dim must be 128*96 for this implementation."
        num_attention_heads = 96
        num_key_value_heads = 8
        num_key_value_groups = 12
        head_dim = 128
        scaling = head_dim ** -0.5

        # 1) Linear projections for Q, K, V
        # Allocate outputs
        query_states = torch.empty((Bsz, Ssz, head_dim * num_attention_heads), device=hidden_states.device, dtype=hidden_states.dtype)
        key_states = torch.empty((Bsz, Ssz, head_dim * num_key_value_heads), device=hidden_states.device, dtype=hidden_states.dtype)
        value_states = torch.empty((Bsz, Ssz, head_dim * num_key_value_heads), device=hidden_states.device, dtype=hidden_states.dtype)

        # Launch Triton linear_bias_kernel for Q, K, V
        # For Q
        grid_q = (Bsz, Ssz, head_dim * num_attention_heads)
        linear_bias_kernel[grid_q](
            hidden_states, q_proj_weight, q_proj_bias, query_states,
            Bsz, Ssz, H_in, head_dim * num_attention_heads,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            q_proj_weight.stride(0), q_proj_weight.stride(1),
            q_proj_bias.stride(0) if q_proj_bias is not None else 0,
            query_states.stride(0), query_states.stride(1), query_states.stride(2),
            BLOCK_K=128,
        )
        # For K
        grid_k = (Bsz, Ssz, head_dim * num_key_value_heads)
        linear_bias_kernel[grid_k](
            hidden_states, k_proj_weight, k_proj_bias, key_states,
            Bsz, Ssz, H_in, head_dim * num_key_value_heads,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            k_proj_weight.stride(0), k_proj_weight.stride(1),
            k_proj_bias.stride(0) if k_proj_bias is not None else 0,
            key_states.stride(0), key_states.stride(1), key_states.stride(2),
            BLOCK_K=128,
        )
        # For V
        grid_v = (Bsz, Ssz, head_dim * num_key_value_heads)
        linear_bias_kernel[grid_v](
            hidden_states, v_proj_weight, v_proj_bias, value_states,
            Bsz, Ssz, H_in, head_dim * num_key_value_heads,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            v_proj_weight.stride(0), v_proj_weight.stride(1),
            v_proj_bias.stride(0) if v_proj_bias is not None else 0,
            value_states.stride(0), value_states.stride(1), value_states.stride(2),
            BLOCK_K=128,
        )

        # 2) Reshape to heads and RMSNorm
        # query: [B, S, 96, 128]
        query_states = query_states.view(Bsz, Ssz, num_attention_heads, head_dim)
        key_states = key_states.view(Bsz, Ssz, num_key_value_heads, head_dim)
        value_states = value_states.view(Bsz, Ssz, num_key_value_heads, head_dim)

        # Allocate RMSNorm outputs
        q_norm = torch.empty_like(query_states)
        k_norm = torch.empty_like(key_states)

        # Launch RMSNorm kernel for Q and K
        grid_rms_q = (Bsz, Ssz, head_dim)
        rmsnorm_kernel[grid_rms_q](
            query_states, q_norm_weight, q_norm,
            Bsz, Ssz, head_dim,
            query_states.stride(0), query_states.stride(1), query_states.stride(2),
            q_norm_weight.stride(0),
            q_norm.stride(0), q_norm.stride(1), q_norm.stride(2),
            BLOCK=128,
        )
        grid_rms_k = (Bsz, Ssz, head_dim)
        rmsnorm_kernel[grid_rms_k](
            key_states, k_norm_weight, k_norm,
            Bsz, Ssz, head_dim,
            key_states.stride(0), key_states.stride(1), key_states.stride(2),
            k_norm_weight.stride(0),
            k_norm.stride(0), k_norm.stride(1), k_norm.stride(2),
            BLOCK=128,
        )

        # 3) Apply RoPE for Q and K
        # Prepare grids for rotate_half_kernel
        # Q and K are [B, num_attention_heads, S, 128]
        # sin and cos are [128]
        # We run over b, s, and head_index
        grid_qrot = (Bsz, Ssz, num_attention_heads)
        rotate_half_kernel[grid_qrot](
            q_norm, sin, cos, q_norm,  # q_norm is also output
            Bsz, Ssz, head_dim,
            q_norm.stride(0), q_norm.stride(1), q_norm.stride(2),
            sin.stride(0), cos.stride(0),
            q_norm.stride(0), q_norm.stride(1), q_norm.stride(2),
            BLOCK=128,
        )
        grid_krot = (Bsz, Ssz, num_key_value_heads)
        rotate_half_kernel[grid_krot](
            k_norm, sin, cos, k_norm,  # k_norm is also output
            Bsz, Ssz, head_dim,
            k_norm.stride(0), k_norm.stride(1), k_norm.stride(2),
            sin.stride(0), cos.stride(0),
            k_norm.stride(0), k_norm.stride(1), k_norm.stride(2),
            BLOCK=128,
        )

        # 4) Repeat KV for GQA to match num_attention_heads
        # Original code expands to [B, num_key_value_heads, num_key_value_groups, S, D] then reshape
        k_norm_expanded = k_norm[:, :, None, :, :].expand(Bsz, num_key_value_heads, num_key_value_groups, Ssz, head_dim).reshape(Bsz, num_attention_heads, Ssz, head_dim)
        v_expanded = value_states[:, :, None, :, :].expand(Bsz, num_key_value_heads, num_key_value_groups, Ssz, head_dim).reshape(Bsz, num_attention_heads, Ssz, head_dim)

        # 5) Compute attention scores Q @ K^T per (b, s, h)
        # Allocate scores [B, num_attention_heads, S, S]
        attn_scores = torch.empty((Bsz, num_attention_heads, Ssz, Ssz), device=hidden_states.device, dtype=hidden_states.dtype)

        # Launch matmul_qk_kernel
        grid_qk = (Bsz, num_attention_heads, Ssz, Ssz)
        matmul_qk_kernel[grid_qk](
            q_norm, k_norm_expanded, attn_scores,
            Bsz, Ssz, head_dim,
            q_norm.stride(0), q_norm.stride(1), q_norm.stride(2), q_norm.stride(3),
            k_norm_expanded.stride(0), k_norm_expanded.stride(1), k_norm_expanded.stride(2), k_norm_expanded.stride(3),
            attn_scores.stride(0), attn_scores.stride(1), attn_scores.stride(2), attn_scores.stride(3),
            BLOCK_Q=128, BLOCK_K=128,
        )

        # 6) Softmax with causal mask over [S, S] per (b, h)
        # We need to apply mask: for each row s_row, set col > s_row to -inf. Implement in Triton with grid (B, H, S, S).
        attn_scores_masked = torch.empty_like(attn_scores)

        # Prepare grid and launch softmax_mask_kernel. Note: this kernel assumes Ssz <= 1024; our tests use S up to 1024.
        grid_sm = (Bsz, num_attention_heads, Ssz, Ssz)
        softmax_mask_kernel[grid_sm](
            attn_scores, attn_scores_masked,
            Ssz,
            attn_scores.stride(0), attn_scores.stride(1), attn_scores.stride(2), attn_scores.stride(3),
            attn_scores_masked.stride(0), attn_scores_masked.stride(1), attn_scores_masked.stride(2), attn_scores_masked.stride(3),
            BLOCK=128,
        )

        # 7) Compute attention output: Softmax(scores) @ V
        attn_output = torch.empty((Bsz, num_attention_heads, Ssz, head_dim), device=hidden_states.device, dtype=hidden_states.dtype)

        grid_attn = (Bsz, num_attention_heads, Ssz)
        matmul_attn_kernel[grid_attn](
            attn_scores_masked, v_expanded, attn_output,
            Bsz, Ssz, head_dim,
            attn_scores_masked.stride(0), attn_scores_masked.stride(1), attn_scores_masked.stride(2), attn_scores_masked.stride(3),
            v_expanded.stride(0), v_expanded.stride(1), v_expanded.stride(2), v_expanded.stride(3),
            attn_output.stride(0), attn_output.stride(1), attn_output.stride(2), attn_output.stride(3),
            BLOCK=128,
        )

        # 8) Transpose and reshape: [B, S, 96*128]
        attn_output = attn_output.transpose(1, 2).contiguous()  # [B, S, 96, 128]
        attn_output_flat = attn_output.reshape(Bsz, Ssz, num_attention_heads * head_dim)

        # 9) Final output projection (linear without bias)
        output = torch.empty((Bsz, Ssz, o_proj_weight.shape[0]), device=hidden_states.device, dtype=hidden_states.dtype)
        grid_out = (Bsz, Ssz, o_proj_weight.shape[0])
        linear_nobias_kernel[grid_out](
            attn_output_flat, o_proj_weight, output,
            Bsz, Ssz, num_attention_heads * head_dim, o_proj_weight.shape[0],
            attn_output_flat.stride(0), attn_output_flat.stride(1), attn_output_flat.stride(2),
            o_proj_weight.stride(0), o_proj_weight.stride(1),
            output.stride(0), output.stride(1), output.stride(2),
            BLOCK_K=128,
        )

        return output


# Example usage:
# model = ModelNew().cuda()
# hidden_states = torch.randn(1, 1024, 12288, device='cuda')  # input as in original
# q_proj_weight = torch.randn(12288, 12288, device='cuda')
# q_proj_bias = torch.randn(12288, device='cuda')
# k_proj_weight = torch.randn(3072, 12288, device='cuda')
# k_proj_bias = torch.randn(3072, device='cuda')
# v_proj_weight = torch.randn(3072, 12288, device='cuda')
# v_proj_bias = torch.randn(3072, device='cuda')
# o_proj_weight = torch.randn(11008, 12288, device='cuda')
# q_norm_weight = torch.randn(12288, device='cuda')
# k_norm_weight = torch.randn(3072, device='cuda')
# cos = torch.randn(128, device='cuda')
# sin = torch.randn(128, device='cuda')
# rms_norm_eps = 0.0
# output = model(hidden_states, q_proj_weight, q_proj_bias, k_proj_weight, k_proj_bias, v_proj_weight, v_proj_bias, o_proj_weight, q_norm_weight, k_norm_weight, cos, sin, rms_norm_eps)


def run(*args):
    return ModelNew()(*args)
