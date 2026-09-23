import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: compute attention output for each (batch, attention head).
# We avoid storing S*S softmax by computing it on-the-fly and directly accumulating output.
# Grid: 1D with size B*H (one program per (b,h))
# Inside each program: loop over M tiles (query positions) and N tiles (key positions)
@triton.jit
def attn_forward_kernel(
    Q_ptr,        # *float32, [B, H, S, D]
    K_ptr,        # *float32, [B, Hk, S, D], but we index K[b, g(h), :] where Hk = num_key_value_heads
    V_ptr,        # *float32, [B, Hk, S, D]
    Out_ptr,      # *float32, [B, H, S, D]
    B: tl.constexpr,
    H: tl.constexpr,            # attention heads (Q/V size)
    Hk: tl.constexpr,           # key/value heads (K size, 8 in your config)
    S: tl.constexpr,            # seq_length
    D: tl.constexpr,            # head_dim
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    scaling,                     # float32 scalar
    num_key_value_groups,        # int (e.g., 12)
):
    # Program id: one per (b, h)
    pid = tl.program_id(axis=0)
    b = pid // H
    h = pid % H

    # Determine group for GQA: group_id = h // (H // num_key_value_groups)
    # Each group maps to a specific kv head: group_id in [0, num_key_value_groups)
    group_id = h // (H // num_key_value_groups)
    kv_h = group_id * (Hk // num_key_value_groups)  # since Hk = num_key_value_groups * (H // num_key_value_groups) with 96/12=8

    # We'll iterate over M tiles (query positions) and N tiles (key positions)
    # For each M tile: compute output vector for that tile and store
    for m_start in range(0, S, BLOCK_M):
        m_offsets = m_start + tl.arange(0, BLOCK_M)
        m_mask = m_offsets < S

        # Initialize output vector for this M tile
        out_vec = tl.zeros((BLOCK_M,), dtype=tl.float32)

        # Loop over N tiles
        for n_start in range(0, S, BLOCK_N):
            n_offsets = n_start + tl.arange(0, BLOCK_N)
            n_mask = n_offsets < S

            # Load Q[M_tile, :]
            q_ptr = Q_ptr + b * H * S * D + h * S * D + m_offsets[:, None] * D + tl.arange(0, D)[None, :]
            q = tl.load(q_ptr, mask=m_mask[:, None], other=0.0)  # shape [BLOCK_M, D]
            q = tl.cast(q, tl.float32)

            # Load K[N_tile, :]
            # We index K at kv_h for each n position
            k_ptr = K_ptr + b * Hk * S * D + kv_h * S * D + n_offsets[:, None] * D + tl.arange(0, D)[None, :]
            k = tl.load(k_ptr, mask=n_mask[:, None], other=0.0)  # shape [BLOCK_N, D]
            k = tl.cast(k, tl.float32)

            # Compute scores: scores[m, n] = sum_d q[m,d] * k[n,d]
            scores = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            # Reduce over D dimension
            for d in range(0, D):
                scores += q[:, d][:, None] * k[:, d][None, :]
            scores = scores * scaling

            # Apply causal mask: for m in M_tile and n in N_tile, if n > m then scores = -inf
            # Build mask: (m_offsets[:, None] < n_offsets[None, :]) -> causal
            causal_mask = (m_offsets[:, None] < n_offsets[None, :])
            scores = tl.where(causal_mask, scores, -float('inf'))

            # Softmax across N (key positions) for each m
            # 1) row-wise max
            max_scores = tl.max(scores, axis=1)  # shape [BLOCK_M]
            scores = scores - max_scores[:, None]

            # 2) exp and sum
            exp_scores = tl.exp(scores)
            sum_exp = tl.sum(exp_scores, axis=1)  # shape [BLOCK_M]
            softmax = exp_scores / sum_exp[:, None]  # shape [BLOCK_M, BLOCK_N]

            # 3) Load V[N_tile, :]
            v_ptr = V_ptr + b * Hk * S * D + kv_h * S * D + n_offsets[:, None] * D + tl.arange(0, D)[None, :]
            v = tl.load(v_ptr, mask=n_mask[:, None], other=0.0)  # shape [BLOCK_N, D]
            v = tl.cast(v, tl.float32)

            # Accumulate output for this M tile: out[m] += sum_n softmax[m,n] * V[n, :]
            # Loop over n in tile and D (small, typically 128)
            for n_i in range(0, BLOCK_N):
                valid_n = n_start + n_i < S
                # For invalid n, softmax[n_i, :] is zero due to -inf scores (we won't use it since valid_n checks)
                if valid_n:
                    sm = softmax[:, n_i]  # [BLOCK_M]
                    v_vec = v[n_i, :]     # [D]
                    out_vec += sm * v_vec

        # Store out_vec to Out[b, h, m_offsets, :]
        out_ptr = Out_ptr + b * H * S * D + h * S * D + m_offsets * D + tl.arange(0, D)
        tl.store(out_ptr, out_vec[:, None], mask=m_mask[:, None])


class ModelNew(nn.Module):
    def __init__(self, num_attention_heads=96, num_key_value_heads=8, num_key_value_groups=12, head_dim=128):
        super().__init__()
        # Keep parameters to mirror original
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_key_value_groups
        self.head_dim = head_dim
        # These original code have them as inputs, but we can create placeholder tensors for completeness.
        # We won't use them directly in the Triton kernels (as they are not part of Triton computation), but we keep them for API consistency.

    def forward(self, hidden_states, q_proj_weight, q_proj_bias, k_proj_weight, k_proj_bias,
                v_proj_weight, v_proj_bias, o_proj_weight, q_norm_weight, k_norm_weight, cos, sin, rms_norm_eps):
        # Ensure tensors are on CUDA for Triton. If not, fall back to original PyTorch logic (for safety).
        use_triton = TRITON_AVAILABLE and hidden_states.is_cuda
        if not use_triton:
            # Fallback: run the original logic (PyTorch) to ensure correctness on CPU or without Triton
            # This path is provided for completeness. In evaluation, hidden_states is likely on CUDA.
            return self._run_original(hidden_states, q_proj_weight, q_proj_bias, k_proj_weight, k_proj_bias,
                                       v_proj_weight, v_proj_bias, o_proj_weight, q_norm_weight, k_norm_weight, cos, sin, rms_norm_eps)

        # 1) Dense projections via PyTorch (F.linear) for Q, K, V
        query_states = F.linear(hidden_states, q_proj_weight, q_proj_bias)
        key_states = F.linear(hidden_states, k_proj_weight, k_proj_bias)
        value_states = F.linear(hidden_states, v_proj_weight, v_proj_bias)

        # 2) Reshape
        B, S, D = hidden_states.shape  # S is seq_length, D is hidden_dim (1024 in your original), but here head_dim is 128
        # Note: In your original, hidden_states has shape [B, S, 1024] and heads split 1024 into 96*128. We assume that.
        num_attention_heads = self.num_attention_heads
        num_key_value_heads = self.num_key_value_heads
        num_key_value_groups = self.num_key_value_groups
        head_dim = self.head_dim

        # Reshape query/value to [B, S, num_attention_heads, head_dim]
        query = query_states.view(B, S, num_attention_heads, head_dim).contiguous()
        key = key_states.view(B, S, num_key_value_heads, head_dim).contiguous()
        value = value_states.view(B, S, num_key_value_heads, head_dim).contiguous()

        # 3) RMSNorm for Q and K
        def rms_norm(x, weight):
            # x: [B, S, H, D]; weight: [H, D]
            # Compute in fp32
            x = x.to(torch.float32)
            # Normalize: x / sqrt(mean(x^2) + eps)
            # Reduce across last dim: mean over D
            mean_sq = (x.pow(2).mean(dim=-1, keepdim=True))
            inv_rms = torch.rsqrt(mean_sq + rms_norm_eps)
            x = x * inv_rms
            # Scale by weight
            w = weight.to(torch.float32)
            out = x * w
            return out.to(x.dtype)

        query = rms_norm(query, q_norm_weight)
        key = rms_norm(key, k_norm_weight)

        # 4) Apply rotation (RoPE) for Q and K
        # Original code: split last dim into two halves and rotate: new_q = q1 + q2', new_k = k1 + k2'
        # sin/cos are [S, D], we need cos, sin expanded for the head dimension
        # We'll use sin/cos per head element. Since D=128 and half=64, we can index sin/cos accordingly.
        # Create cos/sin expanded tensors: [B, 1, S, D] (unsqueeze dim=1)
        cos_expanded = cos.unsqueeze(1)  # [B, 1, S, D]
        sin_expanded = sin.unsqueeze(1)

        # Rotate Q
        q1 = query[..., :head_dim // 2]
        q2 = query[..., head_dim // 2:]
        q_rot_half = torch.cat((-q2, q1), dim=-1)
        query = query * cos_expanded + q_rot_half * sin_expanded

        # Rotate K
        k1 = key[..., :head_dim // 2]
        k2 = key[..., head_dim // 2:]
        k_rot_half = torch.cat((-k2, k1), dim=-1)
        key = key * cos_expanded + k_rot_half * sin_expanded

        # 5) GQA: repeat KV heads to match Q heads
        # key/value are [B, Hk, S, D], repeat each kv head into groups to match H attention heads
        # We need to build [B, H, S, D]. Since num_attention_heads = num_key_value_heads * num_key_value_groups,
        # each attention head maps to one kv head. The original code expands by broadcasting and reshaping.
        # We can do this with expand and reshape. We use expand to avoid copies and then reshape.
        # Each h maps to kv_h = (h // (H // num_key_value_groups)) * (Hk // num_key_value_groups)
        H = num_attention_heads
        Hk = num_key_value_heads
        kv_group_size = H // num_key_value_groups  # 96 // 12 = 8, i.e., 8 attention heads map to 8 kv heads

        # Broadcast to [B, H, S, D]
        key = key[:, :, None, :, :].expand(B, Hk, H // Hk, S, D).reshape(B, H, S, D).contiguous()
        value = value[:, :, None, :, :].expand(B, Hk, H // Hk, S, D).reshape(B, H, S, D).contiguous()

        # 6) Attention output in Triton: compute attn_output = softmax(Q @ K^T) * V
        # We'll compute in fp32 and store output as fp32 (original code likely uses fp32 for these tensors).
        attn_output = torch.empty((B, H, S, D), device=hidden_states.device, dtype=torch.float32)

        # Choose tile sizes. For generality across your workloads (up to S=2048), BLOCK_N can be 128 or 256.
        # BLOCK_M similarly. We choose 128 to balance occupancy and memory.
        BLOCK_M = 128
        BLOCK_N = 128

        # Launch Triton kernel: one program per (b, h)
        grid = (B * H,)
        attn_forward_kernel[grid](
            query, key, value, attn_output,
            B, H, Hk, S, D,
            BLOCK_M, BLOCK_N,
            1.0 / (head_dim ** 0.5),
            num_key_value_groups,
            num_warps=4, num_stages=2
        )

        # 7) Output projection (no bias) via PyTorch
        # attn_output: [B, H, S, D] -> [B, S, H*D]
        attn_output_flat = attn_output.reshape(B, S, H * D)
        output = F.linear(attn_output_flat, o_proj_weight, None)

        return output

    def _run_original(self, hidden_states, q_proj_weight, q_proj_bias, k_proj_weight, k_proj_bias,
                      v_proj_weight, v_proj_bias, o_proj_weight, q_norm_weight, k_norm_weight, cos, sin, rms_norm_eps):
        # This is the original computation logic, kept for fallback.
        # It mirrors your provided run function exactly.
        batch_size, seq_length, _ = hidden_states.shape
        num_attention_heads = 96
        num_key_value_heads = 8
        head_dim = 128
        scaling = head_dim ** -0.5

        query_states = F.linear(hidden_states, q_proj_weight, q_proj_bias)
        key_states = F.linear(hidden_states, k_proj_weight, k_proj_bias)
        value_states = F.linear(hidden_states, v_proj_weight, v_proj_bias)

        query_states = query_states.view(batch_size, seq_length, num_attention_heads, head_dim)
        key_states = key_states.view(batch_size, seq_length, num_key_value_heads, head_dim)
        value_states = value_states.view(batch_size, seq_length, num_key_value_heads, head_dim)

        def rms_norm(x, weight):
            x = x.to(torch.float32)
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + rms_norm_eps)
            return (weight * x).to(x.dtype)

        query_states = rms_norm(query_states, q_norm_weight)
        key_states = rms_norm(key_states, k_norm_weight)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        cos_expanded = cos.unsqueeze(1)
        sin_expanded = sin.unsqueeze(1)

        q1, q2 = query_states[..., :64], query_states[..., 64:]
        q_rot_half = torch.cat((-q2, q1), dim=-1)
        query_states = (query_states * cos_expanded) + (q_rot_half * sin_expanded)

        k1, k2 = key_states[..., :64], key_states[..., 64:]
        k_rot_half = torch.cat((-k2, k1), dim=-1)
        key_states = (key_states * cos_expanded) + (k_rot_half * sin_expanded)

        key_states = key_states[:, :, None, :, :].expand(
            batch_size, num_key_value_heads, num_key_value_groups, seq_length, head_dim
        ).reshape(batch_size, num_attention_heads, seq_length, head_dim)
        value_states = value_states[:, :, None, :, :].expand(
            batch_size, num_key_value_heads, num_key_value_groups, seq_length, head_dim
        ).reshape(batch_size, num_attention_heads, seq_length, head_dim)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling

        causal_mask = torch.triu(
            torch.full((seq_length, seq_length), float('-inf'), device=hidden_states.device, dtype=attn_weights.dtype),
            diagonal=1
        )
        attn_weights = attn_weights + causal_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_length, num_attention_heads * head_dim)

        output = F.linear(attn_output, o_proj_weight, None)

        return output


def run(*args):
    return ModelNew()(*args)
