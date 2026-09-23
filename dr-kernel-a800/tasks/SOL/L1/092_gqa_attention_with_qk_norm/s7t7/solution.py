import torch
import triton
import triton.language as tl

# 1) Triton linear_bsh: computes out[b, s, h] = sum_k input[b, :, k] * weight[h, k] + bias[h]
# This matches F.linear(input [B, S, K], weight [H, K], bias [H]) -> out [B, S, H]
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

    base_input = b * input_stride0
    acc = tl.zeros((), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        mask = k < K
        x = tl.load(input_ptr + base_input + k * input_stride2, mask=mask, other=0.0)
        w = tl.load(weight_ptr + h * weight_stride0 + k * weight_stride1, mask=mask, other=0.0)
        acc += tl.sum(x * w, axis=0)

    bval = 0.0
    if bias_ptr is not None:
        bval = tl.load(bias_ptr + h)
    acc += bval

    tl.store(out_ptr + b * out_stride0 + s * out_stride1 + h * out_stride2, acc)

# 2) Triton RMSNorm per row over last dim S: out_row = weight[h] * x_row / sqrt(mean(x_row^2) + eps)
# Assumes row x is laid out as contiguous vector of length S with stride x_stride1
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

# 3) Triton Expand KV from num_key_value_heads (KVH) to num_attention_heads (H) with groups mapping
# target_h = kh * GROUPS + g, where GROUPS = H // KVH
@triton.jit
def triton_expand_kv(K_ptr, V_ptr, K_out_ptr, V_out_ptr,
                     B, S, KVH, KD,
                     K_stride0, K_stride1, K_stride2, K_stride3,
                     V_stride0, V_stride1, V_stride2, V_stride3,
                     Kout_stride0, Kout_stride1, Kout_stride2, Kout_stride3,
                     Vout_stride0, Vout_stride1, Vout_stride2, Vout_stride3,
                     GROUPS: tl.constexpr):
    # Note: KD is the last dim of K/V; we assume it's head_dim (128).
    for kh in range(0, KVH):
        for g in range(0, GROUPS):
            h_target = kh * GROUPS + g
            # Copy K rows for each position j
            for j in range(0, S):
                # Load K[b, kh, j, :]
                k_ptr = K_ptr + b * K_stride0 + j * K_stride1 + kh * K_stride2 + 0 * K_stride3
                k_val = tl.load(k_ptr)
                # Store to K_out[b, h_target, j, :]
                tl.store(K_out_ptr + b * Kout_stride0 + j * Kout_stride1 + h_target * Kout_stride2 + 0 * Kout_stride3, k_val)
                # Load V[b, kh, j, :]
                v_ptr = V_ptr + b * V_stride0 + j * V_stride1 + kh * V_stride2 + 0 * V_stride3
                v_val = tl.load(v_ptr)
                # Store to V_out[b, h_target, j, :]
                tl.store(V_out_ptr + b * Vout_stride0 + j * Vout_stride1 + h_target * Vout_stride2 + 0 * Vout_stride3, v_val)

# 4) Triton main attention kernel: for each (b, h), compute attention output row
#    It loads Q row for position i, K rows for all j tiles, computes scores, applies scaling, causal mask, softmax, and accumulates output with V.
#    Assumes that Q, K, V have been pre-processed (RMSNorm and RoPE applied, GQA expanded to 96 heads).
@triton.jit
def triton_attention_main(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    B, S, H,
    Q_stride0, Q_stride1, Q_stride2, Q_stride3,
    K_stride0, K_stride1, K_stride2, K_stride3,
    V_stride0, V_stride1, V_stride2, V_stride3,
    Out_stride0, Out_stride1,
    BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)

    scaling = 1.0 / tl.sqrt(128.0)  # head_dim = 128

    # Initialize output accumulator for this (b, h)
    out_row = tl.zeros((S,), dtype=tl.float32)

    # Iterate over query positions in tiles
    for i0 in range(0, S, BLOCK_I):
        i = i0 + tl.arange(0, BLOCK_I)
        mask_i = i < S

        # Load Q[b, h, i, :]
        q_row = tl.zeros((BLOCK_I,), dtype=tl.float32)
        for idx in range(0, BLOCK_I):
            if i[idx] < S:
                q_val = tl.load(Q_ptr + b * Q_stride0 + i[idx] * Q_stride1 + h * Q_stride2 + 0 * Q_stride3)
                q_row[idx] = q_val

        # Compute scores[i, :] = sum_j q_row * K[b, h, j, :] for j in tiles
        for j0 in range(0, S, BLOCK_J):
            j = j0 + tl.arange(0, BLOCK_J)
            mask_j = j < S

            # Initialize scores[i, :]
            scores = tl.zeros((BLOCK_I,), dtype=tl.float32)

            # For each j position, compute dot product with q_row
            for jj in range(0, BLOCK_J):
                if j[jj] < S:
                    k_row = tl.load(K_ptr + b * K_stride0 + j[jj] * K_stride1 + h * K_stride2 + 0 * K_stride3)
                    # scores += q_row * k_row
                    scores += q_row * k_row

            # Apply scaling
            scores = scores * scaling

            # Apply causal mask: for each i, j >= i -> -inf
            for ii in range(0, BLOCK_I):
                if i[ii] < S:
                    for jj in range(0, BLOCK_J):
                        if j[jj] < S and j[jj] >= (i[ii] + i0):
                            scores[ii] = -float('inf')

            # Softmax across j for each i
            # First, compute max per i
            max_scores = -float('inf')
            for ii in range(0, BLOCK_I):
                if i[ii] < S:
                    # For each jj in this tile, update max
                    for jj in range(0, BLOCK_J):
                        if j[jj] < S:
                            if scores[ii] > max_scores:
                                max_scores = scores[ii]
            # Compute exp and sum
            sum_exp = 0.0
            for ii in range(0, BLOCK_I):
                if i[ii] < S:
                    for jj in range(0, BLOCK_J):
                        if j[jj] < S:
                            sum_exp += tl.exp(scores[ii] - max_scores)
            # Normalize
            out_scale = 0.0
            for ii in range(0, BLOCK_I):
                if i[ii] < S:
                    for jj in range(0, BLOCK_J):
                        if j[jj] < S:
                            out_scale += tl.exp(scores[ii] - max_scores) * V_ptr[b * V_stride0 + j[jj] * V_stride1 + h * V_stride2 + 0 * V_stride3]
            # Store out_row[i] = out_scale
            for ii in range(0, BLOCK_I):
                if i[ii] < S:
                    out_row[i[ii]] = out_scale

    # Write out_row to Out[b, h, :]
    for i in range(0, S):
        tl.store(Out_ptr + b * Out_stride0 + h * Out_stride1 + i * Out_stride1, out_row[i])


class ModelNew(torch.nn.Module):
    def __init__(self,
                 num_attention_heads: int = 96,
                 num_key_value_heads: int = 8,
                 head_dim: int = 128,
                 rms_norm_eps: float = 1e-6):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.rms_norm_eps = rms_norm_eps

    def forward(self,
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
                sin: torch.Tensor):
        # Ensure CUDA tensors
        device = hidden_states.device
        if hidden_states.is_cuda is False:
            hidden_states = hidden_states.to('cuda')
        if q_proj_weight.is_cuda is False:
            q_proj_weight = q_proj_weight.to('cuda')
        if q_proj_bias.is_cuda is False:
            q_proj_bias = q_proj_bias.to('cuda')
        if k_proj_weight.is_cuda is False:
            k_proj_weight = k_proj_weight.to('cuda')
        if k_proj_bias is False:
            k_proj_bias = k_proj_bias.to('cuda')
        if v_proj_weight.is_cuda is False:
            v_proj_weight = v_proj_weight.to('cuda')
        if v_proj_bias is False:
            v_proj_bias = v_proj_bias.to('cuda')
        if o_proj_weight.is_cuda is False:
            o_proj_weight = o_proj_weight.to('cuda')
        if q_norm_weight.is_cuda is False:
            q_norm_weight = q_norm_weight.to('cuda')
        if k_norm_weight.is_cuda is False:
            k_norm_weight = k_norm_weight.to('cuda')
        if cos.is_cuda is False:
            cos = cos.to('cuda')
        if sin.is_cuda is False:
            sin = sin.to('cuda')

        B, S, K = hidden_states.shape
        assert K == self.head_dim, "hidden_states last dim must equal head_dim (128)"

        # 1) Compute Q, K, V using Triton linear_bsh
        # Shapes: Q [B, S, H], K [B, S, KVH], V [B, S, KVH]
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

        # 2) Apply RMSNorm to Q and K: grid (B, H)
        Q_norm = torch.empty_like(Q)
        K_norm = torch.empty_like(Kt)

        grid_rms_q = (B, self.num_attention_heads)
        triton_rmsnorm_row[grid_rms_q](
            Q, Q_norm, q_norm_weight, self.rms_norm_eps, B, self.num_attention_heads, S,
            Q.stride(0), Q.stride(1),
            Q_norm.stride(0), Q_norm.stride(1),
            BLOCK_D=self.head_dim, num_warps=4, num_stages=2
        )

        grid_rms_k = (B, self.num_key_value_heads)
        triton_rmsnorm_row[grid_rms_k](
            Kt, K_norm, k_norm_weight, self.rms_norm_eps, B, self.num_key_value_heads, S,
            Kt.stride(0), Kt.stride(1),
            K_norm.stride(0), K_norm.stride(1),
            BLOCK_D=self.head_dim, num_warps=4, num_stages=2
        )

        # 3) Apply Rotary Position Embedding (RoPE) to Q and K: q1, q2 = q[:64], q[64:], q_out = q*cos + q_rot*sin
        # Implement elementwise kernels (grid over (B, H, S))
        # For simplicity, assume head_dim=128, split into 64+64
        Q_rope = torch.empty_like(Q_norm)
        K_rope = torch.empty_like(K_norm)

        grid_rope = (B, self.num_attention_heads, S)
        triton_rope_elements[grid_rope](
            Q_norm, cos, sin, Q_rope,
            B, self.num_attention_heads, S, self.head_dim,
            Q_norm.stride(0), Q_norm.stride(1), Q_norm.stride(2),
            cos.stride(0), cos.stride(1),
            sin.stride(0), sin.stride(1),
            Q_rope.stride(0), Q_rope.stride(1), Q_rope.stride(2),
            BLOCK_D=self.head_dim, num_warps=4, num_stages=2
        )

        grid_rope_k = (B, self.num_key_value_heads, S)
        triton_rope_elements[grid_rope_k](
            K_norm, cos, sin, K_rope,
            B, self.num_key_value_heads, S, self.head_dim,
            K_norm.stride(0), K_norm.stride(1), K_norm.stride(2),
            cos.stride(0), cos.stride(1),
            sin.stride(0), sin.stride(1),
            K_rope.stride(0), K_rope.stride(1), K_rope.stride(2),
            BLOCK_D=self.head_dim, num_warps=4, num_stages=2
        )

        # 4) Grouped-Query Attention (GQA) expansion: replicate 8 KV heads to 96 groups (groups = 96//8 = 12)
        K_expanded = torch.empty((B, self.num_attention_heads, S, self.head_dim), device=device, dtype=K_norm.dtype)
        V_expanded = torch.empty((B, self.num_attention_heads, S, self.head_dim), device=device, dtype=Vt.dtype)

        grid_expand = (B, self.num_key_value_heads, 12, S)
        triton_expand_kv[grid_expand](
            K_rope, Vt, K_expanded, V_expanded,
            B, S, self.num_key_value_heads, self.head_dim,
            K_rope.stride(0), K_rope.stride(1), K_rope.stride(2), K_rope.stride(3),
            Vt.stride(0), Vt.stride(1), Vt.stride(2), Vt.stride(3),
            K_expanded.stride(0), K_expanded.stride(1), K_expanded.stride(2), K_expanded.stride(3),
            V_expanded.stride(0), V_expanded.stride(1), V_expanded.stride(2), V_expanded.stride(3),
            GROUPS=12, num_warps=4, num_stages=2
        )

        # 5) Compute attention: main Triton attention kernel grid (B, H)
        Out = torch.empty((B, self.num_attention_heads, S, self.head_dim), device=device, dtype=hidden_states.dtype)

        grid_att = (B, self.num_attention_heads)
        triton_attention_main[grid_att](
            Q_rope, K_expanded, V_expanded, Out,
            B, S, self.num_attention_heads,
            Q_rope.stride(0), Q_rope.stride(1), Q_rope.stride(2), Q_rope.stride(3),
            K_expanded.stride(0), K_expanded.stride(1), K_expanded.stride(2), K_expanded.stride(3),
            V_expanded.stride(0), V_expanded.stride(1), V_expanded.stride(2), V_expanded.stride(3),
            Out.stride(0), Out.stride(1),
            BLOCK_I=64, BLOCK_J=64, num_warps=4, num_stages=2
        )

        # 6) Output projection (no bias): [B, S, H]
        Out_final = torch.empty((B, S, self.num_attention_heads * self.head_dim), device=device, dtype=hidden_states.dtype)

        grid_out = (B, self.num_attention_heads, S)
        triton_linear_bsh[grid_out](
            Out, o_proj_weight, None, Out_final,
            B, S, self.num_attention_heads, self.head_dim,
            Out.stride(0), Out.stride(1), Out.stride(2),
            o_proj_weight.stride(0), o_proj_weight.stride(1),
            Out_final.stride(0), Out_final.stride(1), Out_final.stride(2),
            BLOCK_K=self.head_dim, num_warps=4, num_stages=2
        )

        return Out_final


def run(*args):
    return ModelNew()(*args)
