import torch
import triton
import triton.language as tl


# 1) Triton dense linear: out[b, s, h] = sum_k hidden[b, s, k] * weight[h, k] + bias[h]
# hidden shape [B, S, K], weight shape [H, K], out shape [B, S, H]
@triton.jit
def triton_linear_bsh(x_ptr, weight_ptr, bias_ptr, out_ptr,
                       B, S, K, H,
                       x_stride0, x_stride1, x_stride2,   # hidden strides for (B,S,K)
                       wt_stride0, wt_stride1,            # weight strides for (H,K)
                       out_stride0, out_stride1, out_stride2,  # out strides (B,S,H)
                       BLOCK_K: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    base_x = b * x_stride0 + s * x_stride1
    base_out = b * out_stride0 + s * out_stride1 + h * out_stride2

    acc = 0.0
    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        mask = k < K
        x = tl.load(x_ptr + base_x + k * x_stride2, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + h * wt_stride0 + k * wt_stride1, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(x * w, axis=0)

    bval = tl.load(bias_ptr + h).to(tl.float32)
    tl.store(out_ptr + base_out, acc + bval)


# 2) Triton RMSNorm per row over last dim S: out[b, h, :] = x * (weight[h] / sqrt(mean(x^2) + eps))
# x is [B, S, H], strides: (B, S, H)
@triton.jit
def triton_rmsnorm_row(x_ptr, weight_ptr, out_ptr,
                       B, S, H,
                       x_stride0, x_stride1, x_stride2,
                       out_stride0, out_stride1, out_stride2,
                       eps: tl.constexpr,
                       BLOCK_S: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    base_x = b * x_stride0 + h * x_stride1
    base_out = b * out_stride0 + h * out_stride2

    sum_sq = 0.0
    for s0 in range(0, S, BLOCK_S):
        s = s0 + tl.arange(0, BLOCK_S)
        mask = s < S
        x = tl.load(x_ptr + base_x + s * x_stride2, mask=mask, other=0.0).to(tl.float32)
        sum_sq += tl.sum(x * x, axis=0)

    mean = sum_sq / S
    inv_rms = 1.0 / tl.sqrt(mean + eps)
    scale = inv_rms * tl.load(weight_ptr + h).to(tl.float32)

    for s0 in range(0, S, BLOCK_S):
        s = s0 + tl.arange(0, BLOCK_S)
        mask = s < S
        x = tl.load(x_ptr + base_x + s * x_stride2, mask=mask, other=0.0).to(tl.float32)
        y = x * scale
        tl.store(out_ptr + base_out + s * out_stride2, y, mask=mask)


# 3) Triton RoPE per row: rotate 128-dim vector into out
# For head_dim=128, split into 64+64: q_out = q*cos + [-q2, q1]*sin
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
        q = tl.load(x_ptr + base_x + d, mask=mask, other=0.0).to(tl.float32)
        cos_vec = tl.load(cos_ptr + d, mask=mask, other=0.0).to(tl.float32)
        sin_vec = tl.load(sin_ptr + d, mask=mask, other=0.0).to(tl.float32)
        q1 = q[:64]
        q2 = q[64:]
        rotated = -q2 + q1
        q_out = q * cos_vec + rotated * sin_vec
        tl.store(out_ptr + base_out + d, q_out, mask=mask)


# 4) Triton attention: compute scores over a small window and softmax along j, then accumulate
# Grid: (B, H), loop over i (query positions) with BLOCK_I, and j (key positions) within a small tile BLOCK_J.
@triton.jit
def attention_compute_windowed(Q_ptr, K_ptr, V_ptr, Out_ptr,
                               B, S, H,
                               Q_stride0, Q_stride1, Q_stride2,
                               K_stride0, K_stride1, K_stride2,
                               V_stride0, V_stride1, V_stride2,
                               Out_stride0, Out_stride1, Out_stride2,
                               scaling: tl.constexpr,
                               BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)

    base_out = b * Out_stride0 + h * Out_stride2

    for i0 in range(0, S, BLOCK_I):
        i = i0 + tl.arange(0, BLOCK_I)
        mask_i = i < S

        # Compute scores for j-window within [i0, i0+BLOCK_J), but we set window to S for simplicity
        for j0 in range(0, S, BLOCK_J):
            j = j0 + tl.arange(0, BLOCK_J)
            mask_j = j < S

            # Load Q row for i
            q_row = tl.zeros([BLOCK_I], dtype=tl.float32)
            for k_i in range(0, 128):  # head_dim is 128 in this implementation
                q_col = tl.load(Q_ptr + b * Q_stride0 + i * Q_stride1 + k_i * Q_stride2, mask=mask_i, other=0.0).to(tl.float32)
                q_row += q_col  # accumulate across all k_i? In fact we need per k_i, but Triton does elementwise; we'll use q_row as a vector
            # The above snippet is illustrative; in practice we should load Q[:, i, :] directly. Triton supports vectorized loads.

            # Build scores matrix: scores[i, j] = sum_k Q[i, k] * K[j, k] * scaling
            scores = tl.zeros([BLOCK_I, BLOCK_J], dtype=tl.float32)
            for k in range(0, 128):
                q_vec = tl.load(Q_ptr + b * Q_stride0 + i * Q_stride1 + k * Q_stride2, mask=mask_i, other=0.0).to(tl.float32)  # vector of length BLOCK_I
                k_vec = tl.load(K_ptr + b * K_stride0 + j * K_stride1 + k * K_stride2, mask=mask_j, other=0.0).to(tl.float32)  # vector of length BLOCK_J
                scores += q_vec[:, None] * k_vec[None, :]

            scores = scores * scaling

            # Apply causal mask: for i < j, set to -inf
            for m in range(0, BLOCK_I):
                for n in range(0, BLOCK_J):
                    if (i0 + m) < (j0 + n):
                        scores[m, n] = -float('inf')

            # Softmax along j
            row_max = tl.max(scores, axis=1)
            scores = scores - row_max[:, None]
            exp_scores = tl.exp(scores)
            row_sum = tl.sum(exp_scores, axis=1)
            attn = exp_scores / row_sum[:, None]

            # Accumulate output: out[i] += sum_j attn[j] * V[j, :]
            out_vec = tl.zeros([BLOCK_I], dtype=tl.float32)
            for n in range(0, BLOCK_J):
                v_vec = tl.load(V_ptr + b * V_stride0 + (j0 + n) * V_stride1 + h * V_stride2, mask=(j0 + n < S), other=0.0).to(tl.float32)
                out_vec += attn[:, n] * v_vec

            # Store results
            tl.store(Out_ptr + base_out + i * Out_stride2, out_vec, mask=mask_i)


class ModelNew(torch.nn.Module):
    def __init__(self, hidden_dim: int, head_dim: int, num_attention_heads: int, num_key_value_heads: int,
                 rms_norm_eps: float, q_norm_weight: torch.Tensor, k_norm_weight: torch.Tensor,
                 cos: torch.Tensor, sin: torch.Tensor, o_proj_weight: torch.Tensor):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.q_norm_weight = q_norm_weight
        self.k_norm_weight = k_norm_weight
        self.cos = cos
        self.sin = sin
        self.o_proj_weight = o_proj_weight

    def forward(self, hidden_states: torch.Tensor,
                q_proj_weight: torch.Tensor, q_proj_bias: torch.Tensor,
                k_proj_weight: torch.Tensor, k_proj_bias: torch.Tensor,
                v_proj_weight: torch.Tensor, v_proj_bias: torch.Tensor):
        # Ensure CUDA tensors
        assert hidden_states.is_cuda, "hidden_states must be on CUDA device"
        assert q_proj_weight.is_cuda and q_proj_bias.is_cuda, "Q/W/B must be on CUDA"
        assert k_proj_weight.is_cuda and k_proj_bias.is_cuda, "K/W/B must be on CUDA"
        assert v_proj_weight.is_cuda and v_proj_bias.is_cuda, "V/W/B must be on CUDA"
        assert self.q_norm_weight.is_cuda and self.k_norm_weight.is_cuda, "Norm weights must be on CUDA"
        assert self.cos.is_cuda and self.sin.is_cuda, "cos/sin must be on CUDA"
        assert self.o_proj_weight.is_cuda, "o_proj_weight must be on CUDA"

        B, S, K = hidden_states.shape
        H = self.num_attention_heads

        # 1) Dense linear for Q, K, V using Triton
        Q = torch.empty((B, S, H), device=hidden_states.device, dtype=torch.float32)
        K = torch.empty((B, S, H), device=hidden_states.device, dtype=torch.float32)
        V = torch.empty((B, S, H), device=hidden_states.device, dtype=torch.float32)

        triton_linear_bsh[ (B, H, S) ](
            hidden_states, q_proj_weight, q_proj_bias, Q,
            B, S, K, H,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            q_proj_weight.stride(0), q_proj_weight.stride(1),
            Q.stride(0), Q.stride(1), Q.stride(2),
            BLOCK_K=128
        )

        triton_linear_bsh[ (B, H, S) ](
            hidden_states, k_proj_weight, k_proj_bias, K,
            B, S, K, H,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            k_proj_weight.stride(0), k_proj_weight.stride(1),
            K.stride(0), K.stride(1), K.stride(2),
            BLOCK_K=128
        )

        triton_linear_bsh[ (B, H, S) ](
            hidden_states, v_proj_weight, v_proj_bias, V,
            B, S, K, H,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            v_proj_weight.stride(0), v_proj_weight.stride(1),
            V.stride(0), V.stride(1), V.stride(2),
            BLOCK_K=128
        )

        # 2) RMSNorm for Q and K
        Q_norm = torch.empty_like(Q)
        K_norm = torch.empty_like(K)

        triton_rmsnorm_row[ (B, H) ](
            Q, self.q_norm_weight, Q_norm,
            B, S, H,
            Q.stride(0), Q.stride(1), Q.stride(2),
            Q_norm.stride(0), Q_norm.stride(1), Q_norm.stride(2),
            eps=self.rms_norm_eps,
            BLOCK_S=128
        )

        triton_rmsnorm_row[ (B, H) ](
            K, self.k_norm_weight, K_norm,
            B, S, H,
            K.stride(0), K.stride(1), K.stride(2),
            K_norm.stride(0), K_norm.stride(1), K_norm.stride(2),
            eps=self.rms_norm_eps,
            BLOCK_S=128
        )

        # 3) Rotary Position Embedding
        Q_rope = torch.empty_like(Q_norm)
        K_rope = torch.empty_like(K_norm)

        triton_rope_row[ (B, H, S) ](
            Q_norm, self.cos, self.sin, Q_rope,
            B, H, S, self.head_dim,
            Q_norm.stride(0), Q_norm.stride(1), Q_norm.stride(2),
            self.cos.stride(0), self.cos.stride(1),
            self.sin.stride(0), self.sin.stride(1),
            Q_rope.stride(0), Q_rope.stride(1), Q_rope.stride(2),
            BLOCK_D=64
        )

        triton_rope_row[ (B, H, S) ](
            K_norm, self.cos, self.sin, K_rope,
            B, H, S, self.head_dim,
            K_norm.stride(0), K_norm.stride(1), K_norm.stride(2),
            self.cos.stride(0), self.cos.stride(1),
            self.sin.stride(0), self.sin.stride(1),
            K_rope.stride(0), K_rope.stride(1), K_rope.stride(2),
            BLOCK_D=64
        )

        # 4) Attention compute in Triton with windowed softmax
        attn_out = torch.empty((B, S, H), device=hidden_states.device, dtype=torch.float32)

        attention_compute_windowed[ (B, H) ](
            Q_rope, K_rope, V, attn_out,
            B, S, H,
            Q_rope.stride(0), Q_rope.stride(1), Q_rope.stride(2),
            K_rope.stride(0), K_rope.stride(1), K_rope.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            attn_out.stride(0), attn_out.stride(1), attn_out.stride(2),
            scaling=1.0 / (self.head_dim ** 0.5),
            BLOCK_I=64, BLOCK_J=64
        )

        # 5) Output projection: out = linear(attn_out, o_proj_weight, no bias)
        output = torch.empty((B, S, H), device=hidden_states.device, dtype=torch.float32)

        triton_linear_bsh[ (B, H, S) ](
            attn_out, self.o_proj_weight, None, output,
            B, S, H, H,
            attn_out.stride(0), attn_out.stride(1), attn_out.stride(2),
            self.o_proj_weight.stride(0), self.o_proj_weight.stride(1),
            output.stride(0), output.stride(1), output.stride(2),
            BLOCK_K=128
        )

        return output


def run(*args):
    return ModelNew()(*args)
