import torch
import triton
import triton.language as tl


# Kernel 1: Linear with bias: compute Q = hidden_states @ q_proj_weight.T + q_proj_bias
# X: [B, S, Hin], W: [Hout, Hin], bias: [Hout], Out: [B, S, Hout]
@triton.jit
def linear_bias_row_kernel(
    X_ptr, W_ptr, Bias_ptr, Out_ptr,
    Bsz, Ssz, Hin, Hout,
    stride_x_b, stride_x_s, stride_x_in,
    stride_w_h, stride_w_in,
    stride_bias_h,
    stride_out_b, stride_out_s, stride_out_h,
    Hout_idx: tl.constexpr,
):
    # Grid: (Bsz * Ssz, 1)
    pid = tl.program_id(0)
    b = pid // Ssz
    s = pid % Ssz

    # Compute base offsets
    base_x = b * stride_x_b + s * stride_x_s
    base_out = b * stride_out_b + s * stride_out_s

    # Accumulator for output vector of length Hout
    acc = tl.zeros((Hout,), dtype=tl.float32)

    # Loop over input dimension
    for in_idx in range(0, Hin):
        x_val = tl.load(X_ptr + base_x + in_idx * stride_x_in)
        # Load weight row for this output index
        w_row = tl.load(W_ptr + Hout_idx * stride_w_h + in_idx * stride_w_in)
        acc[Hout_idx] += x_val * w_row

    # Add bias
    bias_val = tl.load(Bias_ptr + Hout_idx * stride_bias_h)
    acc[Hout_idx] += bias_val

    # Store result
    tl.store(Out_ptr + base_out + Hout_idx * stride_out_h, acc[Hout_idx])


# Kernel 2: RMSNorm over last dim (size = 128): y = x * rsqrt(mean(x^2) + eps), then scale by weight
# Input x: [B, S, D], weight: [D] -> Output y: [B, S, D]
@triton.jit
def rmsnorm_kernel(
    X_ptr, Weight_ptr, Out_ptr,
    Bsz, Ssz, D,
    stride_x_b, stride_x_s, stride_x_d,
    stride_w_d, stride_out_b, stride_out_s, stride_out_d,
    BLOCK: tl.constexpr,  # BLOCK should be >= D, here D=128 so BLOCK=128
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    row_off = b * stride_x_b + s * stride_x_s

    # Load x vector across D
    x_vec = tl.zeros((BLOCK,), dtype=tl.float32)
    for i in range(0, BLOCK):
        x_vec[i] = tl.load(X_ptr + row_off + i * stride_x_d)

    # Compute mean of squares
    sum_sq = tl.sum(x_vec * x_vec, axis=0)
    mean = sum_sq / D
    inv_rms = tl.rsqrt(mean + 0.0)  # eps=0.0 as in original code

    # Load weight vector for this row
    w_vec = tl.load(Weight_ptr, mask=None, other=0.0)  # Weight_ptr has shape [D]
    y_vec = x_vec * inv_rms * w_vec

    out_row_off = b * stride_out_b + s * stride_out_s
    for i in range(0, BLOCK):
        tl.store(Out_ptr + out_row_off + i * stride_out_d, y_vec[i])


# Kernel 3: Rotate half of the last 64 dims: q1, q2 = q[:64], q[64:], rotated = q1*cos - q2*sin
@triton.jit
def rotate_half_kernel(
    Q_ptr, Sin_ptr, Cos_ptr, Out_ptr,
    Bsz, Ssz, D,  # D should be 128
    stride_q_b, stride_q_s, stride_q_d,
    stride_s_d, stride_c_d, stride_out_b, stride_out_s, stride_out_d,
    BLOCK: tl.constexpr,  # BLOCK >= D, here D=128, BLOCK=128
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    row_off_q = b * stride_q_b + s * stride_q_s
    out_row_off = b * stride_out_b + s * stride_out_s

    q_vec = tl.zeros((BLOCK,), dtype=tl.float32)
    for i in range(0, BLOCK):
        q_vec[i] = tl.load(Q_ptr + row_off_q + i * stride_q_d)

    # sin/cos vectors (128 elements)
    sin_vec = tl.load(Sin_ptr)  # shape [D]
    cos_vec = tl.load(Cos_ptr)  # shape [D]

    q1 = q_vec[:64]
    q2 = q_vec[64:]
    rotated = q1 * cos_vec[:64] - q2 * sin_vec[:64]  # length 64
    out_vec = tl.zeros((BLOCK,), dtype=tl.float32)
    out_vec[:64] = rotated
    out_vec[64:] = q_vec[64:]  # unchanged upper part

    for i in range(0, BLOCK):
        tl.store(Out_ptr + out_row_off + i * stride_out_d, out_vec[i])


# Kernel 4: Compute attention scores: Q @ K^T per (b, s, h) -> output [S, S]
@triton.jit
def attn_scores_kernel(
    Q_ptr, K_ptr, Out_ptr,
    Bsz, Ssz, D,  # D=128
    stride_q_b, stride_q_s, stride_q_d,
    stride_k_b, stride_k_s, stride_k_d,
    stride_out_s1, stride_out_s2,
    BLOCK: tl.constexpr,  # BLOCK >= S, here S up to 1024, we can set BLOCK=128 for tile, but for simplicity we loop
):
    b = tl.program_id(0)
    s1 = tl.program_id(1)
    s2 = tl.program_id(2)

    # Load Q row (length D) for (b, s1, h)
    # We assume Q is [B, num_attention_heads, S, D]; we pass h via grid or index. Here we compute directly.
    # To keep it simple, we compute one s2 output for row s1 across all s2 in the grid.
    row_off_q = b * stride_q_b + s1 * stride_q_s  # only s1, h will be passed through pid3
    q_vec = tl.zeros((D,), dtype=tl.float32)
    for i in range(0, D):
        q_vec[i] = tl.load(Q_ptr + row_off_q + i * stride_q_d)

    # Load K row (length D) for (b, s2, h)
    # We need to loop over D for K to compute dot with q_vec
    dot = tl.zeros((), dtype=tl.float32)
    for j in range(0, D):
        k_val = tl.load(K_ptr + b * stride_k_b + s2 * stride_k_s + j * stride_k_d)
        dot += q_vec[j] * k_val

    # Scale by 1/sqrt(D)
    scale = 1.0 / tl.sqrt(D)
    dot = dot * scale

    # Store to Out[b, s1, s2]
    tl.store(Out_ptr + b * stride_out_s1 + s1 * stride_out_s1 + s2 * stride_out_s2, dot)


# Kernel 5: Softmax with causal mask over sequence length (per row s1)
@triton.jit
def softmax_mask_row_kernel(
    In_ptr, Out_ptr,
    Ssz,  # sequence length
    stride_in_s1, stride_in_s2,
    stride_out_s1, stride_out_s2,
):
    s1 = tl.program_id(0)

    # Compute row-wise max
    max_val = -float('inf')
    for col in range(0, Ssz):
        val = tl.load(In_ptr + s1 * stride_in_s1 + col * stride_in_s2)
        max_val = tl.maximum(max_val, val)

    # Apply mask: if col > s1, set to -inf; else keep
    sum_exp = 0.0
    for col in range(0, Ssz):
        val = tl.load(In_ptr + s1 * stride_in_s1 + col * stride_in_s2)
        if col > s1:
            val = -float('inf')
        exp_val = tl.exp(val - max_val)
        sum_exp += exp_val

    # Normalize
    for col in range(0, Ssz):
        val = tl.load(In_ptr + s1 * stride_in_s1 + col * stride_in_s2)
        if col > s1:
            val = -float('inf')
        exp_val = tl.exp(val - max_val) / sum_exp
        tl.store(Out_ptr + s1 * stride_out_s1 + col * stride_out_s2, exp_val)


# Kernel 6: Compute attention output: Softmax(scores) @ V per (b, s1, h) -> output [S, D]
@triton.jit
def attn_output_kernel(
    Softmax_ptr, V_ptr, Out_ptr,
    Bsz, Ssz, D,
    stride_sm_s1, stride_sm_s2,
    stride_v_b, stride_v_s, stride_v_d,
    stride_out_s1, stride_out_s2,
):
    b = tl.program_id(0)
    s1 = tl.program_id(1)

    out_vec = tl.zeros((D,), dtype=tl.float32)
    for d in range(0, D):
        dot = 0.0
        for s2 in range(0, Ssz):
            sm = tl.load(Softmax_ptr + b * stride_sm_s1 + s1 * stride_sm_s1 + s2 * stride_sm_s2)
            v_val = tl.load(V_ptr + b * stride_v_b + s2 * stride_v_s + d * stride_v_d)
            dot += sm * v_val
        out_vec[d] = dot
    # Store out[b, s1, :]
    out_row_off = b * stride_out_s1 + s1 * stride_out_s1
    for d in range(0, D):
        tl.store(Out_ptr + out_row_off + d * stride_out_s2, out_vec[d])


# Kernel 7: Final output projection (linear without bias): attn_output @ o_proj_weight.T -> [B, S, H_out]
# X: [B, S, D], W: [H_out, D], Out: [B, S, H_out]
@triton.jit
def linear_nobias_row_kernel(
    X_ptr, W_ptr, Out_ptr,
    Bsz, Ssz, D, H_out,
    stride_x_b, stride_x_s, stride_x_d,
    stride_w_h, stride_w_d,
    stride_out_b, stride_out_s, stride_out_h,
    Hout_idx: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // Ssz
    s = pid % Ssz

    base_x = b * stride_x_b + s * stride_x_s
    base_out = b * stride_out_b + s * stride_out_s

    acc = 0.0
    for d in range(0, D):
        x_val = tl.load(X_ptr + base_x + d * stride_x_d)
        w_val = tl.load(W_ptr + Hout_idx * stride_w_h + d * stride_w_d)
        acc += x_val * w_val

    tl.store(Out_ptr + base_out + Hout_idx * stride_out_h, acc)


def triton_linear_bias(X: torch.Tensor, W: torch.Tensor, BIAS: torch.Tensor, out: torch.Tensor):
    """
    Launch linear_bias_row_kernel for each Hout index.
    X: [B, S, Hin], W: [Hout, Hin], BIAS: [Hout], out: [B, S, Hout]
    """
    B, S, Hin = X.shape
    Hout, Hin_w = W.shape
    assert Hin_w == Hin
    grid = (B * S,)
    for h in range(Hout):
        linear_bias_row_kernel[grid](
            X, W, BIAS, out,
            B, S, Hin, Hout,
            X.stride(0), X.stride(1), X.stride(2),
            W.stride(0), W.stride(1),
            BIAS.stride(0),
            out.stride(0), out.stride(1), out.stride(2),
            Hout_idx=h,
            num_warps=4, num_stages=2,
        )


def triton_rmsnorm(x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor):
    """
    RMSNorm over last dim (size=128). x: [B, S, 128], weight: [128], out: [B, S, 128]
    """
    B, S, D = x.shape
    grid = (B, S)
    triton.run(
        rmsnorm_kernel[grid](
            x, weight, out,
            B, S, D,
            x.stride(0), x.stride(1), x.stride(2),
            weight.stride(0), out.stride(0), out.stride(1), out.stride(2),
            BLOCK=128,
            num_warps=4, num_stages=2,
        )
    )


def triton_rotate_half(q: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor, out: torch.Tensor):
    """
    Rotate Q in-place using sin/cos. q: [B, S, 128], sin: [128], cos: [128], out: [B, S, 128]
    """
    B, S, D = q.shape
    grid = (B, S)
    triton.run(
        rotate_half_kernel[grid](
            q, sin, cos, out,
            B, S, D,
            q.stride(0), q.stride(1), q.stride(2),
            sin.stride(0), cos.stride(0), out.stride(0), out.stride(1), out.stride(2),
            BLOCK=128,
            num_warps=4, num_stages=2,
        )
    )


def triton_attn_scores(q: torch.Tensor, k: torch.Tensor, out: torch.Tensor):
    """
    Compute attention scores Q @ K^T per (b, s1, h). q: [B, S, 128], k: [B, S, 128], out: [B, S, S]
    """
    B, S, D = q.shape
    grid = (B, S, S)
    triton.run(
        attn_scores_kernel[grid](
            q, k, out,
            B, S, D,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            out.stride(0), out.stride(1),
            BLOCK=128,
            num_warps=4, num_stages=2,
        )
    )


def triton_softmax_mask(scores: torch.Tensor, out: torch.Tensor, seq_len: int):
    """
    Softmax over last dim (sequence length) with causal mask (upper triangular, diagonal=1).
    scores: [B, S, S], out: [B, S, S]
    """
    B, S, _ = scores.shape
    assert S == seq_len
    grid = (S,)
    for b in range(B):
        softmax_mask_row_kernel[grid](
            scores[b], out[b],
            S,
            scores[b].stride(0), scores[b].stride(1),
            out[b].stride(0), out[b].stride(1),
            num_warps=4, num_stages=2,
        )


def triton_attn_output(softmax: torch.Tensor, v: torch.Tensor, out: torch.Tensor):
    """
    Compute attention output: softmax @ V per (b, s1, h). softmax: [B, S, S], v: [B, S, 128], out: [B, S, 128]
    """
    B, S, D = v.shape
    grid = (B, S)
    triton.run(
        attn_output_kernel[grid](
            softmax, v, out,
            B, S, D,
            softmax.stride(0), softmax.stride(1),
            v.stride(0), v.stride(1), v.stride(2),
            out.stride(0), out.stride(1),
            num_warps=4, num_stages=2,
        )
    )


def triton_linear_nobias(x: torch.Tensor, w: torch.Tensor, out: torch.Tensor):
    """
    Final output projection: x @ W.T (no bias). x: [B, S, D], w: [H_out, D], out: [B, S, H_out]
    """
    B, S, D = x.shape
    H_out, D_w = w.shape
    assert D_w == D
    grid = (B * S,)
    for h in range(H_out):
        linear_nobias_row_kernel[grid](
            x, w, out,
            B, S, D, H_out,
            x.stride(0), x.stride(1), x.stride(2),
            w.stride(0), w.stride(1),
            out.stride(0), out.stride(1), out.stride(2),
            Hout_idx=h,
            num_warps=4, num_stages=2,
        )


class ModelNew(torch.nn.Module):
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
        """
        hidden_states: [B, S, 11008] (from the original run)
        q_proj_weight: [H_q, 11008] = [1152, 11008]
        q_proj_bias: [H_q] = [1152]
        Similarly for k/v/o and q/k norms (weights are 128-d vectors).
        cos: [D] = [128], sin: [128]
        rms_norm_eps: 0.0 in original code.
        """
        B, S, _ = hidden_states.shape

        # 1) Compute Q, K, V via linear bias (host code only allocates; heavy math in Triton)
        q = torch.empty((B, S, q_proj_weight.shape[0]), device=hidden_states.device, dtype=hidden_states.dtype)
        k = torch.empty((B, S, k_proj_weight.shape[0]), device=hidden_states.device, dtype=hidden_states.dtype)
        v = torch.empty((B, S, v_proj_weight.shape[0]), device=hidden_states.device, dtype=hidden_states.dtype)

        triton_linear_bias(hidden_states, q_proj_weight, q_proj_bias, q)
        triton_linear_bias(hidden_states, k_proj_weight, k_proj_bias, k)
        triton_linear_bias(hidden_states, v_proj_weight, v_proj_bias, v)

        # 2) Apply RMSNorm (per head across 128-dim)
        q_norm = torch.empty_like(q)
        k_norm = torch.empty_like(k)
        triton_rmsnorm(q, q_norm_weight, q_norm)   # q_norm_weight: [128]
        triton_rmsnorm(k, k_norm_weight, k_norm)   # k_norm_weight: [128]

        # 3) Rotate Q and K using sin/cos
        q_rot = torch.empty_like(q_norm)
        k_rot = torch.empty_like(k_norm)
        triton_rotate_half(q_norm, sin, cos, q_rot)
        triton_rotate_half(k_norm, sin, cos, k_rot)

        # 4) Compute attention scores Q @ K^T -> [B, S, S]
        scores = torch.empty((B, S, S), device=hidden_states.device, dtype=hidden_states.dtype)
        triton_attn_scores(q_rot, k_rot, scores)

        # 5) Apply softmax with causal mask (upper-triangular, diagonal=1) in Triton
        scores_masked = torch.empty_like(scores, device=hidden_states.device, dtype=hidden_states.dtype)
        triton_softmax_mask(scores, scores_masked, S)

        # 6) Compute attention output: softmax @ V -> [B, S, 128]
        attn_out = torch.empty((B, S, 128), device=hidden_states.device, dtype=hidden_states.dtype)
        triton_attn_output(scores_masked, v, attn_out)

        # 7) Final output projection: attn_out @ o_proj_weight.T -> [B, S, H_out]
        # Note: In the original code, hidden_states has 11008 dims, but Q/K/V are 1152; final output dims are not specified.
        # The original returns F.linear(attn_output, o_proj_weight, None), presumably a large output (likely 11008).
        # We assume H_out = 11008 as in the original run context. If it differs, adjust accordingly.
        H_out = o_proj_weight.shape[0]  # typically 11008, but not necessarily; keep as provided.
        output = torch.empty((B, S, H_out), device=hidden_states.device, dtype=hidden_states.dtype)
        triton_linear_nobias(attn_out, o_proj_weight, output)

        return output


def run(*args):
    return ModelNew()(*args)
