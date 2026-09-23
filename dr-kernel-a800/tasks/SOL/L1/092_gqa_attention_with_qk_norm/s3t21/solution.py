import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False

# Triton kernel: dense linear Y = X @ W^T + B
# X: [M, K], W: [N, K], B: [N], Y: [M, N]
@triton.jit
def linear_kernel_2d(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_xm, stride_xk,
    stride_w0, stride_w1,
    stride_ym, stride_yn,
    BLOCK_K: tl.constexpr,
):
    m = tl.program_id(axis=0)
    n = tl.program_id(axis=1)
    acc = tl.zeros((), dtype=tl.float32)
    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(X_ptr + m * stride_xm + offs_k * stride_xk, mask=offs_k < K, other=0.0)  # [BLOCK_K]
        w = tl.load(W_ptr + n * stride_w0 + offs_k * stride_w1, mask=offs_k < K, other=0.0)  # [BLOCK_K]
        acc += tl.sum(x * w, axis=0)
    # Add bias
    b = tl.load(B_ptr + n)
    acc += b
    tl.store(Y_ptr + m * stride_ym + n * stride_yn, acc)


# Triton kernel: RMSNorm per (b, h, s, d)
# X: [B, H, S, D], W: [D], Y: [B, H, S, D]
@triton.jit
def rmsnorm_kernel(
    X_ptr, W_ptr, Y_ptr,
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    stride_xb, stride_xh, stride_xs, stride_xd,
    stride_yb, stride_yh, stride_ys, stride_yd,
    eps: tl.float32,
):
    b = tl.program_id(axis=0)
    h = tl.program_id(axis=1)
    s = tl.program_id(axis=2)
    d = tl.program_id(axis=3)
    # Load x[b,h,s,d] as scalar
    x = tl.load(X_ptr + b * stride_xb + h * stride_xh + s * stride_xs + d * stride_xd).to(tl.float32)
    sum_sq = x * x
    mean_sq = tl.sum(sum_sq, axis=0) / D
    inv_rms = tl.rsqrt(mean_sq + eps)
    w = tl.load(W_ptr + d).to(tl.float32)
    y = (x * inv_rms) * w
    tl.store(Y_ptr + b * stride_yb + h * stride_yh + s * stride_ys + d * stride_yd, y)


# Triton kernel: apply rotation (RoPE) for last dim D=128, split into two halves (64,64)
# X: [B, H, S, D], C: [S, D/2], S: [S, D/2], Y: [B, H, S, D]
@triton.jit
def rotate_half_kernel(
    X_ptr, C_ptr, S_ptr, Y_ptr,
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    stride_xb, stride_xh, stride_xs, stride_xd,
    stride_yb, stride_yh, stride_ys, stride_yd,
    stride_c0, stride_c1,     # cos strides: [S, D/2]
    stride_s0, stride_s1,     # sin strides: [S, D/2]
):
    b = tl.program_id(axis=0)
    h = tl.program_id(axis=1)
    s = tl.program_id(axis=2)
    d = tl.program_id(axis=3)
    x = tl.load(X_ptr + b * stride_xb + h * stride_xh + s * stride_xs + d * stride_xd).to(tl.float32)
    # Split head into two halves
    q1 = x[:64]
    q2 = x[64:]
    rotated_half = tl.concatenate((-q2, q1), axis=0)
    c = tl.load(C_ptr + s * stride_c0 + d // 2 * stride_c1).to(tl.float32)  # cos
    s_val = tl.load(S_ptr + s * stride_s0 + d // 2 * stride_s1).to(tl.float32)  # sin
    y = x * c + rotated_half * s_val
    tl.store(Y_ptr + b * stride_yb + h * stride_yh + s * stride_ys + d * stride_yd, y)


# Triton kernel: compute attention scores and apply causal mask, write Soft[b, h, :, :] per row
# Q: [B, H, S, 128], K: [B, H, S, 128], Soft: [B, H, S, S]
@triton.jit
def attn_scores_kernel(
    Q_ptr, K_ptr, Soft_ptr,
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_sb, stride_sh, stride_ss, stride_sd,
    scaling: tl.float32,
):
    b = tl.program_id(axis=0)
    h = tl.program_id(axis=1)
    i = tl.program_id(axis=2)  # query position
    # Compute scores for all j in [0, S)
    # scores[i, j] = dot(Q[b,h,i,:], K[b,h,j,:]) * scaling
    for j in range(0, S):
        q = tl.load(Q_ptr + b * stride_qb + h * stride_qh + i * stride_qs + tl.arange(0, 128) * stride_qd)  # shape [128]
        k = tl.load(K_ptr + b * stride_kb + h * stride_kh + j * stride_ks + tl.arange(0, 128) * stride_kd)  # shape [128]
        q = q.to(tl.float32)
        k = k.to(tl.float32)
        dot = tl.sum(q * k, axis=0)
        score = dot * scaling
        # causal mask: if i < j, set to -inf; else 0
        mask_val = tl.where(i < j, -float('inf'), 0.0)
        score = score + mask_val
        tl.store(Soft_ptr + b * stride_sb + h * stride_sh + i * stride_ss + j * stride_sd, score)


# Triton kernel: compute softmax over each row Soft[b, h, i, :] in-place (Soft is initialized to -inf),
# and write out[b, h, i, :] = sum_j Soft[b,h,i,j] * V[b,h,j,:] where V is the value tensor for that head.
# Out: [B, H, S, 128]
@triton.jit
def softmax_out_kernel(
    Soft_ptr, V_ptr, Out_ptr,
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr,
    stride_sb, stride_sh, stride_ss, stride_sd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
):
    b = tl.program_id(axis=0)
    h = tl.program_id(axis=1)
    i = tl.program_id(axis=2)
    # Load row scores
    row = Soft_ptr + b * stride_sb + h * stride_sh + i * stride_ss
    scores = tl.load(row + tl.arange(0, S) * stride_sd, mask=tl.arange(0, S) < S, other=-float('inf'))  # [S]
    # Numerically stable softmax: subtract max
    max_score = tl.max(scores, axis=0)
    scores = scores - max_score
    exp_scores = tl.exp(scores)
    sum_exp = tl.sum(exp_scores, axis=0)
    soft = exp_scores / sum_exp  # [S]
    # Compute output for this (b, h, i)
    for j in range(0, S):
        v = tl.load(V_ptr + b * stride_vb + h * stride_vh + j * stride_vs + tl.arange(0, 128) * stride_vd)  # [128]
        v = v.to(tl.float32)
        soft_j = tl.load(Soft_ptr + b * stride_sb + h * stride_sh + i * stride_ss + j * stride_sd).to(tl.float32)
        out_vec = tl.sum(v * soft_j, axis=0)  # scalar
        tl.store(Out_ptr + b * stride_ob + h * stride_oh + i * stride_os + tl.arange(0, 128) * stride_od, out_vec)


# Triton kernel: final linear Y = Out_flat @ O^T + B (O is [128, 128]), Y is [B, S, 128]
@triton.jit
def final_linear_kernel(
    Out_ptr, O_ptr, B_ptr, Y_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,  # M = B*S, N=128, K=128
    stride_om, stride_ok,
    stride_o0, stride_o1,
    stride_ym, stride_yn,
    BLOCK_K: tl.constexpr,
):
    m = tl.program_id(axis=0)  # over M=B*S
    n = tl.program_id(axis=1)  # over N=128
    acc = tl.zeros((), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        out = tl.load(Out_ptr + m * stride_om + offs_k * stride_ok, mask=offs_k < K, other=0.0)  # [BLOCK_K]
        o = tl.load(O_ptr + n * stride_o0 + offs_k * stride_o1, mask=offs_k < K, other=0.0)      # [BLOCK_K]
        acc += tl.sum(out * o, axis=0)
    b = tl.load(B_ptr + n)
    acc += b
    tl.store(Y_ptr + m * stride_ym + n * stride_yn, acc)


def _launch_linear(X, W, B, out, block_k=64):
    # X: [M, D_in], W: [N, D_in], B: [N], out: [M, N]
    M, D_in = X.shape
    N = W.shape[0]
    grid = (M, N)
    linear_kernel_2d[grid](
        X, W, B, out,
        M, N, D_in,
        1, 1,               # X strides: row-major (M, D_in)
        1, 1,               # W strides: row-major (N, D_in)
        1, 1,               # out strides: row-major (M, N)
        BLOCK_K=block_k,
        num_warps=4,
    )


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor,
                q_proj_weight: torch.Tensor, q_proj_bias: torch.Tensor,
                k_proj_weight: torch.Tensor, k_proj_bias: torch.Tensor,
                v_proj_weight: torch.Tensor, v_proj_bias: torch.Tensor,
                o_proj_weight: torch.Tensor, o_proj_bias: torch.Tensor,
                q_norm_weight: torch.Tensor, k_norm_weight: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor,
                rms_norm_eps: float):
        # Ensure Triton is available
        if not TRITON_AVAILABLE:
            # Fallback to PyTorch if Triton not available (though evaluator requires Triton-only)
            raise RuntimeError("Triton is not available")

        # Shapes from original code (fixed): head_dim = 128
        B, S, D_in = hidden_states.shape  # hidden_states is [B, S, 128]
        H = 96
        num_key_value_heads = 8
        num_key_value_groups = 12

        device = hidden_states.device
        dtype = hidden_states.dtype

        # Cast inputs to float32 for Triton kernels (accumulation in f32)
        hidden_states_f32 = hidden_states.float().contiguous()

        # 1) Linear layers for Q, K, V using Triton
        M = B * S
        q = torch.empty((M, 128), device=device, dtype=torch.float32)
        k = torch.empty((M, 128), device=device, dtype=torch.float32)
        v = torch.empty((M, 128), device=device, dtype=torch.float32)

        _launch_linear(hidden_states_f32.reshape(M, D_in), q_proj_weight.float(), q_proj_bias.float(), q)
        _launch_linear(hidden_states_f32.reshape(M, D_in), k_proj_weight.float(), k_proj_bias.float(), k)
        _launch_linear(hidden_states_f32.reshape(M, D_in), v_proj_weight.float(), v_proj_bias.float(), v)

        # 2) Reshape to heads [B, H, S, 128]
        q4d = q.view(B, S, H, 128).contiguous()
        k4d = k.view(B, S, H, 128).contiguous()
        v4d = v.view(B, S, H, 128).contiguous()

        # 3) RMSNorm for Q and K (Triton)
        q_norm = torch.empty_like(q4d, device=device, dtype=torch.float32)
        k_norm = torch.empty_like(k4d, device=device, dtype=torch.float32)

        grid_norm = (B, H, S, 128)
        rmsnorm_kernel[grid_norm](
            q4d, q_norm_weight.float(), q_norm,
            B, H, S, 128,
            q4d.stride(0), q4d.stride(1), q4d.stride(2), q4d.stride(3),
            q_norm.stride(0), q_norm.stride(1), q_norm.stride(2), q_norm.stride(3),
            rms_norm_eps,
            num_warps=1,
        )
        rmsnorm_kernel[grid_norm](
            k4d, k_norm_weight.float(), k_norm,
            B, H, S, 128,
            k4d.stride(0), k4d.stride(1), k4d.stride(2), k4d.stride(3),
            k_norm.stride(0), k_norm.stride(1), k_norm.stride(2), k_norm.stride(3),
            rms_norm_eps,
            num_warps=1,
        )

        # 4) Rotate Q and K with provided cos/sin (Triton)
        # Prepare cos/sin of shape [S, 64] since head_dim=128
        cos_exp = cos[:S, :64].float().contiguous()  # [S, 64]
        sin_exp = sin[:S, :64].float().contiguous()  # [S, 64]

        q_rot = torch.empty_like(q_norm, device=device, dtype=torch.float32)
        k_rot = torch.empty_like(k_norm, device=device, dtype=torch.float32)

        grid_rot = (B, H, S, 128)
        rotate_half_kernel[grid_rot](
            q_norm, cos_exp, sin_exp, q_rot,
            B, H, S, 128,
            q_norm.stride(0), q_norm.stride(1), q_norm.stride(2), q_norm.stride(3),
            q_rot.stride(0), q_rot.stride(1), q_rot.stride(2), q_rot.stride(3),
            cos_exp.stride(0), cos_exp.stride(1),
            sin_exp.stride(0), sin_exp.stride(1),
            num_warps=1,
        )
        rotate_half_kernel[grid_rot](
            k_norm, cos_exp, sin_exp, k_rot,
            B, H, S, 128,
            k_norm.stride(0), k_norm.stride(1), k_norm.stride(2), k_norm.stride(3),
            k_rot.stride(0), k_rot.stride(1), k_rot.stride(2), k_rot.stride(3),
            cos_exp.stride(0), cos_exp.stride(1),
            sin_exp.stride(0), sin_exp.stride(1),
            num_warps=1,
        )

        # 5) For GQA, expand KV heads: [B, 8, S, 128] -> [B, 96, S, 128]
        num_key_value_heads = 8
        num_key_value_groups = 12
        k_gqa = k_rot[:, :, None, :, :].expand(B, num_key_value_heads, num_key_value_groups, S, 128).reshape(B, H, S, 128)
        v_gqa = v.view(B, H, S, 128).contiguous()  # V already is [B*S,128], we reshape to [B,H,S,128]

        # 6) Compute attention scores with causal mask using Triton (Soft: [B,H,S,S])
        Soft = torch.empty((B, H, S, S), device=device, dtype=torch.float32)
        scaling = 1.0 / 128.0  # head_dim ** -0.5
        grid_scores = (B, H, S)
        attn_scores_kernel[grid_scores](
            q_rot, k_gqa, Soft,
            B, H, S,
            q_rot.stride(0), q_rot.stride(1), q_rot.stride(2), q_rot.stride(3),
            k_gqa.stride(0), k_gqa.stride(1), k_gqa.stride(2), k_gqa.stride(3),
            Soft.stride(0), Soft.stride(1), Soft.stride(2), Soft.stride(3),
            scaling,
            num_warps=1,
        )

        # 7) Compute attention output per (b,h,i) using softmax and V (Triton)
        Out = torch.empty((B, H, S, 128), device=device, dtype=torch.float32)
        grid_softmax = (B, H, S)
        softmax_out_kernel[grid_softmax](
            Soft, k_gqa, Out,
            B, H, S,
            Soft.stride(0), Soft.stride(1), Soft.stride(2), Soft.stride(3),
            k_gqa.stride(0), k_gqa.stride(1), k_gqa.stride(2), k_gqa.stride(3),
            Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
            num_warps=1,
        )

        # 8) Final output projection: Y = Out_flat @ o_proj_weight^T + o_proj_bias
        # Flatten Out to [B*S, 128]
        M_out = B * S
        out_flat = Out.reshape(M_out, 128).contiguous()
        output = torch.empty((M_out, 128), device=device, dtype=torch.float32)
        _launch_linear(out_flat, o_proj_weight.float(), o_proj_bias.float(), output)

        # 9) Reshape to [B, S, 128] and cast back if needed
        output = output.view(B, S, 128)
        # Cast back to original dtype if necessary
        if output.dtype != dtype:
            output = output.to(dtype)
        return output


def run(*args):
    return ModelNew()(*args)
