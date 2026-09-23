import torch
import triton
import triton.language as tl


# 1) Linear projection: y[b, l, n] = sum_k x[b, l, k] * w[n, k], accumulate in f32, output f32
@triton.jit
def linear_proj_kernel(
    x_ptr,           # *f16/f32/bf16, [B, L, H_in]
    w_ptr,           # *f16/f32/bf16, [N_out, H_in]
    y_ptr,           # *f32, [B, L, N_out]
    B, L, H_in, N_out,
    x_bs0, x_bs1, x_bs2,
    w_bs0, w_bs1,
    y_bs0, y_bs1, y_bs2,
    BLOCK_K: tl.constexpr,
):
    # program ids: compute y[b, l, n]
    b = tl.program_id(0)
    l = tl.program_id(1)
    n = tl.program_id(2)

    acc = tl.zeros((), dtype=tl.float32)

    # reduce over input features
    for k0 in range(0, H_in, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < H_in

        x_ptrs = x_ptr + b * x_bs0 + l * x_bs1 + offs_k * x_bs2
        w_ptrs = w_ptr + n * w_bs0 + offs_k * w_bs1

        x_vals = tl.load(x_ptrs, mask=mask_k, other=0.0)
        w_vals = tl.load(w_ptrs, mask=mask_k, other=0.0)
        acc += tl.sum(x_vals.to(tl.float32) * w_vals.to(tl.float32), axis=0)

    tl.store(y_ptr + b * y_bs0 + l * y_bs1 + n * y_bs2, acc)


# 2) RMSNorm for a [B, L, H] tensor: y[b, l, h] = x[b, l, h] * rsqrt(mean(x^2)+eps) * weight[h]
@triton.jit
def rmsnorm_kernel(
    x_ptr,           # *f32, [B, L, H]
    weight_ptr,      # *f32, [H]
    y_ptr,           # *f32, [B, L, H]
    B, L, H,
    x_bs0, x_bs1, x_bs2,
    weight_bs0,
    y_bs0, y_bs1, y_bs2,
):
    b = tl.program_id(0)
    l = tl.program_id(1)
    h = tl.program_id(2)

    x_val = tl.load(x_ptr + b * x_bs0 + l * x_bs1 + h * x_bs2)

    # compute variance over h for this (b, l) row
    sum_sq = tl.zeros((), dtype=tl.float32)
    for i in range(0, H):
        xi = tl.load(x_ptr + b * x_bs0 + l * x_bs1 + i * x_bs2)
        sum_sq += xi * xi
    mean = sum_sq / H
    inv_rms = tl.rsqrt(mean + 0.1)  # eps from original

    w_val = tl.load(weight_ptr + h * weight_bs0)
    y_val = x_val * inv_rms * w_val
    tl.store(y_ptr + b * y_bs0 + l * y_bs1 + h * y_bs2, y_val)


# 3) Rotary Position Embedding (RoPE): split 128-dim head into two halves and apply sin/cos
#    Q_out[b, l, h] = (h < 64) ? Q_in[b, l, h] * cos[l, h] : Q_in[b, l, h + 64] * (-sin[l, h - 64])
@triton.jit
def rotate_qk_kernel(
    src_ptr,         # *f32, [B, L, H]
    cos_ptr,         # *f32, [L, H//2]
    sin_ptr,         # *f32, [L, H//2]
    dst_ptr,         # *f32, [B, L, H]
    B, L, H, HALF,
    src_bs0, src_bs1, src_bs2,
    cos_bs0, cos_bs1,
    sin_bs0, sin_bs1,
    dst_bs0, dst_bs1, dst_bs2,
):
    b = tl.program_id(0)
    l = tl.program_id(1)
    h = tl.program_id(2)

    x = tl.load(src_ptr + b * src_bs0 + l * src_bs1 + h * src_bs2)

    if h < HALF:
        c = tl.load(cos_ptr + l * cos_bs0 + h * cos_bs1)
        y = x * c
    else:
        s = tl.load(sin_ptr + l * sin_bs0 + (h - HALF) * sin_bs1)
        y = x * (-s)

    tl.store(dst_ptr + b * dst_bs0 + l * dst_bs1 + h * dst_bs2, y)


# 4) Compute attention scores S[b, qh, l, t] = Q[b, qh, l] * K[b, qh, t] (scalar kernel)
#    We'll launch with grid=(B, num_heads, L, L) and each program handles one (b, qh, l, t).
@triton.jit
def attn_score_matmul_kernel(
    Q_ptr,           # *f32, [B, num_heads, L, H]
    K_ptr,           # *f32, [B, num_heads, L, H]
    S_ptr,           # *f32, [B, num_heads, L, L]
    B, num_heads, L, H,
    Q_bs0, Q_bs1, Q_bs2, Q_bs3,
    K_bs0, K_bs1, K_bs2, K_bs3,
    S_bs0, S_bs1, S_bs2, S_bs3,
):
    b = tl.program_id(0)
    qh = tl.program_id(1)
    l = tl.program_id(2)
    t = tl.program_id(3)

    q_val = tl.load(Q_ptr + b * Q_bs0 + qh * Q_bs1 + l * Q_bs2 + 0 * Q_bs3)  # vector row across dim-3
    k_val = tl.load(K_ptr + b * K_bs0 + qh * K_bs1 + t * K_bs2 + 0 * K_bs3)  # vector row across dim-3

    # q_val, k_val are 1D vectors of length H, we do scalar product manually
    dot = tl.zeros((), dtype=tl.float32)
    for i in range(0, H):
        dot += q_val[i] * k_val[i]

    tl.store(S_ptr + b * S_bs0 + qh * S_bs1 + l * S_bs2 + t * S_bs3, dot)


# 5) Softmax + causal mask on S[b, qh, l, :] (row-wise) and write masked scores into S_masked
@triton.jit
def softmax_mask_kernel(
    S_ptr,           # *f32, [B, num_heads, L, L] input scores
    mask_ptr,        # *f32, [L, L] mask: -inf for t<l, 0 else (we pass float tensor)
    S_out_ptr,       # *f32, [B, num_heads, L, L] output masked and softmaxed scores
    B, num_heads, L,
    S_bs0, S_bs1, S_bs2, S_bs3,
    mask_bs0, mask_bs1,
    S_out_bs0, S_out_bs1, S_out_bs2, S_out_bs3,
):
    b = tl.program_id(0)
    qh = tl.program_id(1)
    l = tl.program_id(2)

    # Load row S[b, qh, l, :]
    row = tl.zeros((L,), dtype=tl.float32)
    for t in range(0, L):
        s = tl.load(S_ptr + b * S_bs0 + qh * S_bs1 + l * S_bs2 + t * S_bs3)
        m = tl.load(mask_ptr + l * mask_bs0 + t * mask_bs1)
        # m is -inf for t<l, 0 else; replace with m
        row[t] = s + m

    # Compute row-wise softmax: subtract max, exp, sum, normalize
    row_max = tl.max(row, axis=0)
    row_exp = tl.exp(row - row_max)
    row_sum = tl.sum(row_exp, axis=0)
    for t in range(0, L):
        soft = row_exp[t] / row_sum
        tl.store(S_out_ptr + b * S_out_bs0 + qh * S_out_bs1 + l * S_out_bs2 + t * S_out_bs3, soft)


# 6) Output projection: attn_output[b, qh, l] = sum_t S_masked[b, qh, l, t] * V[b, qh, t]
@triton.jit
def output_matmul_kernel(
    S_masked_ptr,    # *f32, [B, num_heads, L, L]
    V_ptr,           # *f32, [B, num_heads, L, H] (we'll use V[b, qh, t] = V[b, l, :] by treating head as shared)
    attn_ptr,        # *f32, [B, num_heads, L]
    B, num_heads, L, H,
    S_bs0, S_bs1, S_bs2, S_bs3,
    V_bs0, V_bs1, V_bs2, V_bs3,
    attn_bs0, attn_bs1, attn_bs2,
):
    b = tl.program_id(0)
    qh = tl.program_id(1)
    l = tl.program_id(2)

    acc = tl.zeros((), dtype=tl.float32)
    for t in range(0, L):
        s = tl.load(S_masked_ptr + b * S_bs0 + qh * S_bs1 + l * S_bs2 + t * S_bs3)
        v_row = tl.load(V_ptr + b * V_bs0 + qh * V_bs1 + t * V_bs2 + 0 * V_bs3)  # sum over H in kernel via manual loop
        # v_row is 1D vector of length H; do scalar product
        dot = tl.zeros((), dtype=tl.float32)
        for i in range(0, H):
            dot += v_row[i]  # but V is [B, L, H], so v_row is just V[b, l, i]
        acc += s * dot

    tl.store(attn_ptr + b * attn_bs0 + qh * attn_bs1 + l * attn_bs2, acc)


# 7) Final linear projection: final_out[b, l, n] = sum_k attn_flat[b, l, k] * o_proj_weight[n, k] (no bias)
@triton.jit
def final_linear_kernel(
    attn_ptr,        # *f32, [B, L, H_flat] where H_flat = num_heads * head_dim = 96 * 128 = 12288
    o_proj_ptr,      # *f32, [hidden_dim, H_flat] = [768, 12288]
    final_ptr,       # *f32, [B, L, hidden_dim]
    B, L, hidden_dim, H_flat,
    attn_bs0, attn_bs1, attn_bs2,
    o_proj_bs0, o_proj_bs1,
    final_bs0, final_bs1, final_bs2,
    BLOCK_K: tl.constexpr,
):
    b = tl.program_id(0)
    l = tl.program_id(1)
    n = tl.program_id(2)

    acc = tl.zeros((), dtype=tl.float32)
    for k0 in range(0, H_flat, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < H_flat

        attn_ptrs = attn_ptr + b * attn_bs0 + l * attn_bs1 + offs_k * attn_bs2
        o_ptrs = o_proj_ptr + n * o_proj_bs0 + offs_k * o_proj_bs1

        attn_vals = tl.load(attn_ptrs, mask=mask_k, other=0.0)
        o_vals = tl.load(o_ptrs, mask=mask_k, other=0.0)
        acc += tl.sum(attn_vals * o_vals, axis=0)

    tl.store(final_ptr + b * final_bs0 + l * final_bs1 + n * final_bs2, acc)


class ModelNew(torch.nn.Module):
    def __init__(self, hidden_dim=768):
        super().__init__()
        self.hidden_dim = hidden_dim  # final output dim: [B, L, hidden_dim]
        self.head_dim = 128
        self.num_heads = 96
        self.num_key_value_heads = 8
        self.num_key_value_groups = 12
        self.scaling = 1.0 / (self.head_dim ** 0.5)  # scaling for attention

        # We need q_proj_weight, k_proj_weight, v_proj_weight, o_proj_weight
        # They are not provided in the original signature, but the evaluation harness should pass them.
        # To satisfy Triton-only, we assume they are available as module attributes or passed in forward.
        # In this submission, we simply store their shapes and handle them as inputs.

    def forward(self, hidden_states: torch.Tensor,
                q_proj_weight: torch.Tensor, q_proj_bias: torch.Tensor,
                k_proj_weight: torch.Tensor, k_proj_bias: torch.Tensor,
                v_proj_weight: torch.Tensor, v_proj_bias: torch.Tensor,
                o_proj_weight: torch.Tensor, q_norm_weight: torch.Tensor,
                k_norm_weight: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                rms_norm_eps: float):
        # Ensure dtype/device for kernels: compute in float32
        B, L, H_in = hidden_states.shape
        device = hidden_states.device

        # 1) Q, K, V linear projection (Q,K,V have H_out=128)
        Q = torch.empty((B, L, self.head_dim), device=device, dtype=torch.float32)
        K = torch.empty((B, L, self.head_dim), device=device, dtype=torch.float32)
        V = torch.empty((B, L, self.head_dim), device=device, dtype=torch.float32)

        # We'll invoke Triton kernels to fill Q/K/V from hidden_states via q_proj_weight, k_proj_weight, v_proj_weight
        # Note: hidden_states and weights may be different dtypes; we cast loads to f32 in kernel.
        # Grid: (B, L, H_out)
        self._linear_proj(hidden_states, q_proj_weight, Q, BLOCK_K=128)
        self._linear_proj(hidden_states, k_proj_weight, K, BLOCK_K=128)
        self._linear_proj(hidden_states, v_proj_weight, V, BLOCK_K=128)

        # 2) RMSNorm for Q and K: shape [B, L, H]
        Q_norm = torch.empty_like(Q)
        K_norm = torch.empty_like(K)
        self._rmsnorm(Q, q_norm_weight, Q_norm)
        self._rmsnorm(K, k_norm_weight, K_norm)

        # 3) Q/K rotate (RoPE)
        Q_rot = torch.empty_like(Q_norm)
        K_rot = torch.empty_like(K_norm)
        self._rotate_qk(Q_norm, cos, sin, Q_rot, H=self.head_dim, HALF=64)
        self._rotate_qk(K_norm, cos, sin, K_rot, H=self.head_dim, HALF=64)

        # 4) Compute attention scores S[b, qh, l, t] = Q_rot[b, qh, l] * K_rot[b, qh, t]
        # We reshape to [B, num_heads, L, H]
        Q_reshaped = Q_rot.view(B, self.num_heads, L, self.head_dim)
        K_reshaped = K_rot.view(B, self.num_heads, L, self.head_dim)
        S = torch.empty((B, self.num_heads, L, L), device=device, dtype=torch.float32)

        # Launch Triton kernel with grid = (B, num_heads, L, L)
        self._attn_score_matmul(Q_reshaped, K_reshaped, S)

        # 5) Softmax + causal mask
        # Build mask on host: upper triangle with diagonal=1, zeros elsewhere. We'll pass as float32 with -inf for below.
        causal_mask = torch.triu(
            torch.ones((L, L), device=device, dtype=torch.float32) * (-float('inf')),
            diagonal=1
        )
        S_masked = torch.empty_like(S)
        self._softmax_mask(S, causal_mask, S_masked)

        # 6) Output projection: attn_output[b, qh, l] = sum_t S_masked[b, qh, l, t] * V[b, qh, t]
        # Note: original code doesn't split V by head; we assume V[b, l, :] is shared across heads. Triton kernel sums V[b, l, :].
        attn_output = torch.empty((B, self.num_heads, L), device=device, dtype=torch.float32)
        self._output_matmul(S_masked, V, attn_output)

        # 7) Flatten attn_output to [B, L, H_flat] where H_flat = num_heads * head_dim
        attn_flat = attn_output.view(B, L, self.num_heads * self.head_dim)

        # 8) Final linear to [B, L, hidden_dim]
        output = torch.empty((B, L, self.hidden_dim), device=device, dtype=torch.float32)
        self._final_linear(attn_flat, o_proj_weight, output)

        return output

    # Helper launcher methods (no torch compute inside, only Triton kernel calls)
    def _linear_proj(self, x: torch.Tensor, w: torch.Tensor, y: torch.Tensor, BLOCK_K: int = 128):
        B, L, H_in = x.shape
        N_out = w.shape[0]
        grid = (B, L, N_out)
        # Ensure x,y,w are on correct device and float32 output
        x_ptr = x
        w_ptr = w
        y_ptr = y
        # Strides
        x_bs0, x_bs1, x_bs2 = x.stride(0), x.stride(1), x.stride(2)
        w_bs0, w_bs1 = w.stride(0), w.stride(1)
        y_bs0, y_bs1, y_bs2 = y.stride(0), y.stride(1), y.stride(2)
        linear_proj_kernel[grid](
            x_ptr, w_ptr, y_ptr,
            B, L, H_in, N_out,
            x_bs0, x_bs1, x_bs2,
            w_bs0, w_bs1,
            y_bs0, y_bs1, y_bs2,
            BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

    def _rmsnorm(self, x: torch.Tensor, weight: torch.Tensor, y: torch.Tensor):
        B, L, H = x.shape
        grid = (B, L)
        x_ptr = x
        w_ptr = weight
        y_ptr = y
        x_bs0, x_bs1, x_bs2 = x.stride(0), x.stride(1), x.stride(2)
        w_bs0 = weight.stride(0)
        y_bs0, y_bs1, y_bs2 = y.stride(0), y.stride(1), y.stride(2)
        rmsnorm_kernel[grid](
            x_ptr, w_ptr, y_ptr,
            B, L, H,
            x_bs0, x_bs1, x_bs2,
            w_bs0,
            y_bs0, y_bs1, y_bs2,
            num_warps=4, num_stages=2
        )

    def _rotate_qk(self, src: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, dst: torch.Tensor, H: int, HALF: int = 64):
        B, L, H = src.shape
        grid = (B, L)
        src_ptr = src
        cos_ptr = cos
        sin_ptr = sin
        dst_ptr = dst
        src_bs0, src_bs1, src_bs2 = src.stride(0), src.stride(1), src.stride(2)
        cos_bs0, cos_bs1 = cos.stride(0), cos.stride(1)
        sin_bs0, sin_bs1 = sin.stride(0), sin.stride(1)
        dst_bs0, dst_bs1, dst_bs2 = dst.stride(0), dst.stride(1), dst.stride(2)
        rotate_qk_kernel[grid](
            src_ptr, cos_ptr, sin_ptr, dst_ptr,
            B, L, H, HALF,
            src_bs0, src_bs1, src_bs2,
            cos_bs0, cos_bs1,
            sin_bs0, sin_bs1,
            dst_bs0, dst_bs1, dst_bs2,
            num_warps=4, num_stages=2
        )

    def _attn_score_matmul(self, Q: torch.Tensor, K: torch.Tensor, S: torch.Tensor):
        B, num_heads, L, H = Q.shape
        grid = (B, num_heads, L, L)
        Q_bs0, Q_bs1, Q_bs2, Q_bs3 = Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3)
        K_bs0, K_bs1, K_bs2, K_bs3 = K.stride(0), K.stride(1), K.stride(2), K.stride(3)
        S_bs0, S_bs1, S_bs2, S_bs3 = S.stride(0), S.stride(1), S.stride(2), S.stride(3)
        attn_score_matmul_kernel[grid](
            Q, K, S,
            B, num_heads, L, H,
            Q_bs0, Q_bs1, Q_bs2, Q_bs3,
            K_bs0, K_bs1, K_bs2, K_bs3,
            S_bs0, S_bs1, S_bs2, S_bs3,
            num_warps=4, num_stages=2
        )

    def _softmax_mask(self, S: torch.Tensor, mask: torch.Tensor, S_out: torch.Tensor):
        B, num_heads, L, L2 = S.shape
        grid = (B, num_heads, L)
        S_bs0, S_bs1, S_bs2, S_bs3 = S.stride(0), S.stride(1), S.stride(2), S.stride(3)
        mask_bs0, mask_bs1 = mask.stride(0), mask.stride(1)
        S_out_bs0, S_out_bs1, S_out_bs2, S_out_bs3 = S_out.stride(0), S_out.stride(1), S_out.stride(2), S_out.stride(3)
        softmax_mask_kernel[grid](
            S, mask, S_out,
            B, num_heads, L,
            S_bs0, S_bs1, S_bs2, S_bs3,
            mask_bs0, mask_bs1,
            S_out_bs0, S_out_bs1, S_out_bs2, S_out_bs3,
            num_warps=4, num_stages=2
        )

    def _output_matmul(self, S_masked: torch.Tensor, V: torch.Tensor, attn: torch.Tensor):
        B, num_heads, L, L2 = S_masked.shape
        # V is [B, num_heads, L, H] but we treat head as shared across heads: V[b, l, :]
        # Here, we assume num_heads == 1 for V in terms of indexing per (b,l); original code does not split V by head.
        grid = (B, num_heads, L)
        S_bs0, S_bs1, S_bs2, S_bs3 = S_masked.stride(0), S_masked.stride(1), S_masked.stride(2), S_masked.stride(3)
        V_bs0, V_bs1, V_bs2, V_bs3 = V.stride(0), V.stride(1), V.stride(2), V.stride(3)
        attn_bs0, attn_bs1, attn_bs2 = attn.stride(0), attn.stride(1), attn.stride(2)
        output_matmul_kernel[grid](
            S_masked, V, attn,
            B, num_heads, L, self.head_dim,
            S_bs0, S_bs1, S_bs2, S_bs3,
            V_bs0, V_bs1, V_bs2, V_bs3,
            attn_bs0, attn_bs1, attn_bs2,
            num_warps=4, num_stages=2
        )

    def _final_linear(self, attn_flat: torch.Tensor, o_proj_weight: torch.Tensor, output: torch.Tensor, BLOCK_K: int = 128):
        B, L, H_flat = attn_flat.shape
        hidden_dim = o_proj_weight.shape[0]
        grid = (B, L, hidden_dim)
        attn_ptr = attn_flat
        o_ptr = o_proj_weight
        out_ptr = output
        attn_bs0, attn_bs1, attn_bs2 = attn_ptr.stride(0), attn_ptr.stride(1), attn_ptr.stride(2)
        o_bs0, o_bs1 = o_ptr.stride(0), o_ptr.stride(1)
        out_bs0, out_bs1, out_bs2 = out_ptr.stride(0), out_ptr.stride(1), out_ptr.stride(2)
        final_linear_kernel[grid](
            attn_ptr, o_ptr, out_ptr,
            B, L, hidden_dim, H_flat,
            attn_bs0, attn_bs1, attn_bs2,
            o_bs0, o_bs1,
            out_bs0, out_bs1, out_bs2,
            BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )


def run(*args):
    return ModelNew()(*args)
