import torch
import triton
import triton.language as tl

# 1) Triton linear_bsh: computes out[b, s, h] = sum_k input[b, :, k] * weight[h, k] + bias[h]
# input: [B, S, K], weight: [H, K], out: [B, S, H]
@triton.jit
def triton_linear_bsh(input_ptr, weight_ptr, bias_ptr, out_ptr,
                      B, S, H, K,
                      input_stride0, input_stride1, input_stride2,
                      weight_stride0, weight_stride1,
                      out_stride0, out_stride1, out_stride2,
                      BLOCK_K: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    base_input = b * input_stride0  # we iterate over K across hidden dimension for each batch row
    acc = tl.zeros((), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        mask_k = k < K
        # input[b, :, k] -> vector over hidden dim
        x = tl.load(input_ptr + base_input + k * input_stride2, mask=mask_k, other=0.0)
        w = tl.load(weight_ptr + h * weight_stride0 + k * weight_stride1, mask=mask_k, other=0.0)
        acc += tl.sum(x * w, axis=0)
    # add bias
    bval = tl.load(bias_ptr + h)
    acc += bval
    tl.store(out_ptr + b * out_stride0 + s * out_stride1 + h * out_stride2, acc)


# 2) Triton RMSNorm per row: out_row[b,h,:] = weight[h] * x_row / sqrt(mean(x_row^2) + eps)
@triton.jit
def triton_rmsnorm_row(x_ptr, out_ptr, weight_ptr, eps, B, H, S,
                        x_stride0, x_stride1,
                        out_stride0, out_stride1,
                        BLOCK_D: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    row_start = b * S
    sum_sq = 0.0
    for d in range(0, S, BLOCK_D):
        offs = d + tl.arange(0, BLOCK_D)
        mask = offs < S
        x = tl.load(x_ptr + row_start + offs * x_stride1, mask=mask, other=0.0)
        sum_sq += tl.sum(x * x, axis=0)
    mean = sum_sq / S
    inv_rms = 1.0 / tl.sqrt(mean + eps)
    scale = tl.load(weight_ptr + h) * inv_rms
    for d in range(0, S, BLOCK_D):
        offs = d + tl.arange(0, BLOCK_D)
        mask = offs < S
        x = tl.load(x_ptr + row_start + offs * x_stride1, mask=mask, other=0.0)
        y = x * scale
        tl.store(out_ptr + b * out_stride0 + h * out_stride1 + offs * out_stride1, y, mask=mask)


# 3) Triton RoPE: rotate query/key row: split 128 into 64+64; q_out = q*cos + [-q2, q1]*sin
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
        q1 = q[:64]
        q2 = q[64:]
        rotated = -q2 + q1  # [-q2, q1]
        q_out = q * cos_vec + rotated * sin_vec
        tl.store(out_ptr + base_out + d, q_out, mask=mask)


# 4) Triton expand KV: expand KV heads from KVH to H via groups GROUPS = H // KVH
@triton.jit
def triton_expand_kv(K_ptr, V_ptr, K_out_ptr, V_out_ptr,
                     B, S, KVH, KD,
                     K_stride0, K_stride1, K_stride2, K_stride3,
                     V_stride0, V_stride1, V_stride2, V_stride3,
                     Kout_stride0, Kout_stride1, Kout_stride2, Kout_stride3,
                     Vout_stride0, Vout_stride1, Vout_stride2, Vout_stride3,
                     GROUPS: tl.constexpr):
    for kh in range(0, KVH):
        for g in range(0, GROUPS):
            h_target = kh * GROUPS + g
            for j in range(0, S):
                k_val = tl.load(K_ptr + b * K_stride0 + j * K_stride1 + kh * K_stride2 + 0 * K_stride3)
                tl.store(K_out_ptr + b * Kout_stride0 + j * Kout_stride1 + h_target * Kout_stride2 + 0 * Kout_stride3, k_val)
                v_val = tl.load(V_ptr + b * V_stride0 + j * V_stride1 + kh * V_stride2 + 0 * V_stride3)
                tl.store(V_out_ptr + b * Vout_stride0 + j * Vout_stride1 + h_target * Vout_stride2 + 0 * Vout_stride3, v_val)


# 5) Triton attention compute: for each (b,h), compute attn scores for all i and j, softmax along j, accumulate output
@triton.jit
def triton_attention(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    B, S, H,
    Q_stride0, Q_stride1, Q_stride2, Q_stride3,
    K_stride0, K_stride1, K_stride2, K_stride3,
    V_stride0, V_stride1, V_stride2, V_stride3,
    Out_stride0, Out_stride1, Out_stride2,
    BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr,
    SCALE: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    # 2D tiling over sequence positions
    for i0 in range(0, S, BLOCK_I):
        i = i0 + tl.arange(0, BLOCK_I)
        mask_i = i < S
        acc = tl.zeros((BLOCK_I,), dtype=tl.float32)
        for j0 in range(0, S, BLOCK_J):
            j = j0 + tl.arange(0, BLOCK_J)
            mask_j = j < S
            # Load Q[i, :] and K[j, :]
            q_row = tl.load(Q_ptr + b * Q_stride0 + i * Q_stride1 + h * Q_stride2 + 0 * Q_stride3, mask=mask_i, other=0.0)  # shape [BLOCK_I]
            k_row = tl.load(K_ptr + b * K_stride0 + j * K_stride1 + h * K_stride2 + 0 * K_stride3, mask=mask_j, other=0.0)  # shape [BLOCK_J]
            # Compute scores: q_row[:, None] * k_row[None, :]
            # Note: q_row is [BLOCK_I], k_row is [BLOCK_J]
            scores = q_row[:, None] * k_row[None, :] * SCALE  # [BLOCK_I, BLOCK_J]
            # Causal mask: i >= j positions should be -inf, otherwise 0
            # Build mask
            i_mat = i[:, None]
            j_mat = j[None, :]
            causal = (i_mat >= j_mat)  # True where i >= j
            # Apply mask: where causal is False, set to -inf
            scores = tl.where(causal, scores, -float('inf'))
            # Softmax along j axis (per i)
            exp_scores = tl.exp(scores)
            row_sum = tl.sum(exp_scores, axis=1)  # [BLOCK_I]
            probs = exp_scores / row_sum[:, None]  # [BLOCK_I, BLOCK_J]
            # Load V[j, :] and accumulate
            v_row = tl.load(V_ptr + b * V_stride0 + j * V_stride1 + h * V_stride2 + 0 * V_stride3, mask=mask_j, other=0.0)  # [BLOCK_J]
            acc += tl.sum(probs * v_row[None, :], axis=1)  # [BLOCK_I]
        # Store acc to Out[b, i, h]
        tl.store(Out_ptr + b * Out_stride0 + i * Out_stride1 + h * Out_stride2, acc, mask=mask_i)


# 6) Triton linear for final output projection: Out[b, s, h] = sum_k attn_output[b, s, k] * o_proj_weight[h, k]
@triton.jit
def triton_linear_proj(input_ptr, weight_ptr, out_ptr,
                       B, S, H, K,
                       input_stride0, input_stride1, input_stride2,
                       weight_stride0, weight_stride1,
                       out_stride0, out_stride1, out_stride2,
                       BLOCK_K: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    base_input = b * input_stride0
    acc = tl.zeros((), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        mask_k = k < K
        x = tl.load(input_ptr + base_input + s * input_stride1 + k * input_stride2, mask=mask_k, other=0.0)
        w = tl.load(weight_ptr + h * weight_stride0 + k * weight_stride1, mask=mask_k, other=0.0)
        acc += tl.sum(x * w, axis=0)
    tl.store(out_ptr + b * out_stride0 + s * out_stride1 + h * out_stride2, acc)

# ModelNew: Triton-only forward
class ModelNew(torch.nn.Module):
    def __init__(self, num_attention_heads=96, num_key_value_heads=8, head_dim=128, rms_norm_eps=1e-8):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.rms_norm_eps = rms_norm_eps
        # Constants for attention scale
        self.scale = 1.0 / (head_dim ** 0.5)

    def forward(self, hidden_states: torch.Tensor,
                q_proj_weight: torch.Tensor, q_proj_bias: torch.Tensor,
                k_proj_weight: torch.Tensor, k_proj_bias: torch.Tensor,
                v_proj_weight: torch.Tensor, v_proj_bias: torch.Tensor,
                o_proj_weight: torch.Tensor,
                q_norm_weight: torch.Tensor, k_norm_weight: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor):
        # Ensure CUDA
        device = hidden_states.device
        if not hidden_states.is_cuda:
            hidden_states = hidden_states.to('cuda')
        if not q_proj_weight.is_cuda:
            q_proj_weight = q_proj_weight.to('cuda')
        if not q_proj_bias.is_cuda:
            q_proj_bias = q_proj_bias.to('cuda')
        if not k_proj_weight.is_cuda:
            k_proj_weight = k_proj_weight.to('cuda')
        if not k_proj_bias.is_cuda:
            k_proj_bias = k_proj_bias.to('cuda')
        if not v_proj_weight.is_cuda:
            v_proj_weight = v_proj_weight.to('cuda')
        if not v_proj_bias.is_cuda:
            v_proj_bias = v_proj_bias.to('cuda')
        if not o_proj_weight.is_cuda:
            o_proj_weight = o_proj_weight.to('cuda')
        if not q_norm_weight.is_cuda:
            q_norm_weight = q_norm_weight.to('cuda')
        if not k_norm_weight.is_cuda:
            k_norm_weight = k_norm_weight.to('cuda')
        if not cos.is_cuda:
            cos = cos.to('cuda')
        if not sin.is_cuda:
            sin = sin.to('cuda')

        B, S, K = hidden_states.shape
        assert K == self.head_dim, "hidden_states last dim must equal head_dim (128)"

        # 1) Compute Q, K, V using Triton linear
        Q = torch.empty((B, S, self.num_attention_heads), device=device, dtype=hidden_states.dtype)
        Kt = torch.empty((B, S, self.num_key_value_heads), device=device, dtype=hidden_states.dtype)
        Vt = torch.empty((B, S, self.num_key_value_heads), device=device, dtype=hidden_states.dtype)

        grid_linear = (B, self.num_attention_heads, S)
        triton_linear_bsh[grid_linear](
            hidden_states, q_proj_weight, q_proj_bias, Q,
            B, S, self.num_attention_heads, self.head_dim,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            q_proj_weight.stride(0), q_proj_weight.stride(1),
            Q.stride(0), Q.stride(1), Q.stride(2),
            BLOCK_K=self.head_dim, num_warps=4, num_stages=2
        )

        grid_linear_k = (B, self.num_key_value_heads, S)
        triton_linear_bsh[grid_linear_k](
            hidden_states, k_proj_weight, k_proj_bias, Kt,
            B, S, self.num_key_value_heads, self.head_dim,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            k_proj_weight.stride(0), k_proj_weight.stride(1),
            Kt.stride(0), Kt.stride(1), Kt.stride(2),
            BLOCK_K=self.head_dim, num_warps=4, num_stages=2
        )

        grid_linear_v = (B, self.num_key_value_heads, S)
        triton_linear_bsh[grid_linear_v](
            hidden_states, v_proj_weight, v_proj_bias, Vt,
            B, S, self.num_key_value_heads, self.head_dim,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            v_proj_weight.stride(0), v_proj_weight.stride(1),
            Vt.stride(0), Vt.stride(1), Vt.stride(2),
            BLOCK_K=self.head_dim, num_warps=4, num_stages=2
        )

        # 2) RMSNorm for Q and K
        Q_norm = torch.empty_like(Q)
        Kt_norm = torch.empty_like(Kt)
        grid_rms_q = (B, self.num_attention_heads)
        triton_rmsnorm_row[grid_rms_q](
            Q, Q_norm, q_norm_weight, self.rms_norm_eps, B, self.num_attention_heads, S,
            Q.stride(0), Q.stride(2),
            Q_norm.stride(0), Q_norm.stride(2),
            BLOCK_D=self.head_dim, num_warps=4, num_stages=2
        )

        grid_rms_k = (B, self.num_key_value_heads)
        triton_rmsnorm_row[grid_rms_k](
            Kt, Kt_norm, k_norm_weight, self.rms_norm_eps, B, self.num_key_value_heads, S,
            Kt.stride(0), Kt.stride(2),
            Kt_norm.stride(0), Kt_norm.stride(2),
            BLOCK_D=self.head_dim, num_warps=4, num_stages=2
        )

        # 3) Apply RoPE to Q and K
        Q_rot = torch.empty_like(Q_norm)
        Kt_rot = torch.empty_like(Kt_norm)
        grid_rope_q = (B, self.num_attention_heads, S)
        triton_rope_row[grid_rope_q](
            Q_norm, cos, sin, Q_rot,
            B, self.num_attention_heads, S, self.head_dim,
            Q_norm.stride(0), Q_norm.stride(2), Q_norm.stride(1),
            cos.stride(0), cos.stride(1),
            sin.stride(0), sin.stride(1),
            Q_rot.stride(0), Q_rot.stride(2), Q_rot.stride(1),
            BLOCK_D=self.head_dim, num_warps=4, num_stages=2
        )

        grid_rope_k = (B, self.num_key_value_heads, S)
        triton_rope_row[grid_rope_k](
            Kt_norm, cos, sin, Kt_rot,
            B, self.num_key_value_heads, S, self.head_dim,
            Kt_norm.stride(0), Kt_norm.stride(2), Kt_norm.stride(1),
            cos.stride(0), cos.stride(1),
            sin.stride(0), sin.stride(1),
            Kt_rot.stride(0), Kt_rot.stride(2), Kt_rot.stride(1),
            BLOCK_D=self.head_dim, num_warps=4, num_stages=2
        )

        # 4) GQA expansion: expand KV heads from 8 to 96 via groups=12
        K_expanded = torch.empty((B, self.num_attention_heads, S, self.head_dim), device=device, dtype=hidden_states.dtype)
        V_expanded = torch.empty((B, self.num_attention_heads, S, self.head_dim), device=device, dtype=hidden_states.dtype)

        grid_expand = (B, self.num_key_value_heads, 12, S)  # GROUPS = 12
        triton_expand_kv[grid_expand](
            Kt_rot, Vt, K_expanded, V_expanded,
            B, S, self.num_key_value_heads, self.head_dim,
            Kt_rot.stride(0), Kt_rot.stride(2), Kt_rot.stride(1), 1,  # last dim is 128
            Vt.stride(0), Vt.stride(2), Vt.stride(1), 1,
            K_expanded.stride(0), K_expanded.stride(2), K_expanded.stride(1), 1,
            V_expanded.stride(0), V_expanded.stride(2), V_expanded.stride(1), 1,
            GROUPS=12, num_warps=4, num_stages=2
        )

        # Now K_expanded, V_expanded have shape [B, 96, S, 128]

        # 5) Triton attention: compute attn_output[b, :, :] per (b,h) using K_expanded/V_expanded
        attn_output = torch.empty((B, S, self.num_attention_heads), device=device, dtype=hidden_states.dtype)
        grid_attn = (B, self.num_attention_heads)
        triton_attention[grid_attn](
            Q_rot, K_expanded, V_expanded, attn_output,
            B, S, self.num_attention_heads,
            Q_rot.stride(0), Q_rot.stride(2), Q_rot.stride(1), 1,
            K_expanded.stride(0), K_expanded.stride(2), K_expanded.stride(1), 1,
            V_expanded.stride(0), V_expanded.stride(2), V_expanded.stride(1), 1,
            attn_output.stride(0), attn_output.stride(2), attn_output.stride(1),
            BLOCK_I=64, BLOCK_J=64, SCALE=self.scale,
            num_warps=4, num_stages=2
        )

        # 6) Final output projection using o_proj_weight (no bias)
        Out = torch.empty((B, S, self.num_attention_heads), device=device, dtype=hidden_states.dtype)
        grid_proj = (B, self.num_attention_heads, S)
        triton_linear_proj[grid_proj](
            attn_output, o_proj_weight, Out,
            B, S, self.num_attention_heads, self.head_dim,
            attn_output.stride(0), attn_output.stride(2), attn_output.stride(1),
            o_proj_weight.stride(0), o_proj_weight.stride(1),
            Out.stride(0), Out.stride(2), Out.stride(1),
            BLOCK_K=self.head_dim, num_warps=4, num_stages=2
        )

        return Out


def run(*args):
    return ModelNew()(*args)
