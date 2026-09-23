import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# ---------------------------
# Triton kernels
# ---------------------------

# Dense linear: Y = X @ W^T + B for 2D inputs
# X: [M, K], W: [N, K], B: [N], Y: [M, N]
@triton.jit
def linear_2d_kernel(X_ptr, W_ptr, B_ptr, Y_ptr,
                      M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                      stride_xm, stride_xk,
                      stride_wn, stride_wk,
                      stride_ym, stride_yn,
                      BLOCK_K: tl.constexpr):
    m = tl.program_id(axis=0)  # row in X
    n = tl.program_id(axis=1)  # output dim
    acc = tl.zeros((), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(X_ptr + m * stride_xm + offs * stride_xk, mask=offs < K, other=0.0)  # [BLOCK_K]
        w = tl.load(W_ptr + n * stride_wn + offs * stride_wk, mask=offs < K, other=0.0)  # [BLOCK_K]
        acc += tl.sum(x * w, axis=0)
    b = tl.load(B_ptr + n)
    acc += b
    tl.store(Y_ptr + m * stride_ym + n * stride_yn, acc)


# RMSNorm elementwise over 4D tensor: Y = x * rsqrt(mean(x^2) + eps) * weight
# X: [B, H, S, D], W: [D], Y: [B, H, S, D]
@triton.jit
def rmsnorm_4d_kernel(X_ptr, W_ptr, Y_ptr,
                       B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
                       stride_xb, stride_xh, stride_xs, stride_xd,
                       stride_yb, stride_yh, stride_ys, stride_yd,
                       eps: tl.float32):
    b = tl.program_id(axis=0)
    h = tl.program_id(axis=1)
    s = tl.program_id(axis=2)
    d = tl.program_id(axis=3)
    x = tl.load(X_ptr + b * stride_xb + h * stride_xh + s * stride_xs + d * stride_xd).to(tl.float32)
    sum_sq = x * x
    mean_sq = tl.sum(sum_sq, axis=0) / D
    inv_rms = tl.rsqrt(mean_sq + eps)
    w = tl.load(W_ptr + d).to(tl.float32)
    y = (x * inv_rms) * w
    tl.store(Y_ptr + b * stride_yb + h * stride_yh + s * stride_ys + d * stride_yd, y)


# Rotation (RoPE) for last dim D=128: split into q1 and q2, Y = X * cos + rotated_half * sin
# X: [B, H, S, D], C: [S, 64], S: [S, 64], Y: [B, H, S, D]
@triton.jit
def rotate_half_kernel(X_ptr, C_ptr, S_ptr, Y_ptr,
                        B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
                        stride_xb, stride_xh, stride_xs, stride_xd,
                        stride_yb, stride_yh, stride_ys, stride_yd,
                        stride_c0, stride_c1,   # cos strides: [S, 64]
                        stride_s0, stride_s1):  # sin strides: [S, 64]
    b = tl.program_id(axis=0)
    h = tl.program_id(axis=1)
    s = tl.program_id(axis=2)
    for d in range(0, D):
        x = tl.load(X_ptr + b * stride_xb + h * stride_xh + s * stride_xs + d * stride_xd).to(tl.float32)
        if d < 64:
            # first half: q1
            c = tl.load(C_ptr + s * stride_c0 + d * stride_c1).to(tl.float32)
            sc = tl.load(S_ptr + s * stride_s0 + d * stride_s1).to(tl.float32)
            y = x * c + x * sc
        else:
            # second half: q2, rotation: y = x*cos + (-x)*sin
            d1 = d - 64
            c = tl.load(C_ptr + s * stride_c0 + d1 * stride_c1).to(tl.float32)
            sc = tl.load(S_ptr + s * stride_s0 + d1 * stride_s1).to(tl.float32)
            y = x * c + (-x) * sc
        tl.store(Y_ptr + b * stride_yb + h * stride_yh + s * stride_ys + d * stride_yd, y)


# Causal mask: Mask[i, j] = 1 if i >= j else 0 (float32)
# Mask: [S, S]
@triton.jit
def causal_mask_kernel(Mask_ptr,
                        S: tl.constexpr,
                        stride_m0, stride_m1):
    i = tl.program_id(axis=0)
    j = tl.program_id(axis=1)
    # 1 if i >= j, else 0
    val = tl.where(i >= j, 1.0, 0.0)
    tl.store(Mask_ptr + i * stride_m0 + j * stride_m1, val)


# Softmax along rows (sequence dim) for each (b, h): Soft[b, h, :, :]
# Soft: [B, H, S], input: AttnScores: [B, H, S]
@triton.jit
def softmax_row_kernel(Soft_ptr, Attn_ptr,
                        B: tl.constexpr, H: tl.constexpr, S: tl.constexpr,
                        stride_sb, stride_sh, stride_ss,
                        stride_ab, stride_ah, stride_as):
    b = tl.program_id(axis=0)
    h = tl.program_id(axis=1)
    row_ptr = Attn_ptr + b * stride_sb + h * stride_sh
    soft_ptr = Soft_ptr + b * stride_sb + h * stride_sh
    # Load row
    attn_row = tl.load(row_ptr + tl.arange(0, S), mask=True, other=-1e20)  # use large negative
    # Numerically stable softmax
    row_max = tl.max(attn_row, axis=0)
    attn_shifted = attn_row - row_max
    exp_row = tl.exp(attn_shifted)
    denom = tl.sum(exp_row, axis=0)
    soft_row = exp_row / denom
    tl.store(soft_ptr + tl.arange(0, S), soft_row)


# Triton matmul for attention output: Y[b, h, i] = sum_j Soft[b,h,i,j] * V[b,h,j]
# Soft: [B, H, S], V: [B, H, S], Y: [B, H, S]
@triton.jit
def attn_output_kernel(Soft_ptr, V_ptr, Y_ptr,
                        B: tl.constexpr, H: tl.constexpr, S: tl.constexpr,
                        stride_sb, stride_sh, stride_ss,
                        stride_vb, stride_vh, stride_vs,
                        stride_yb, stride_yh, stride_ys):
    b = tl.program_id(axis=0)
    h = tl.program_id(axis=1)
    for i in range(0, S):
        soft_row = tl.load(Soft_ptr + b * stride_sb + h * stride_sh + i * stride_ss)
        v_row = tl.load(V_ptr + b * stride_vb + h * stride_vh + i * stride_vs)
        dot = tl.sum(soft_row * v_row, axis=0)
        tl.store(Y_ptr + b * stride_yb + h * stride_yh + i * stride_ys, dot)


# Output projection: Y_flat = AttnOutputFlat @ o_proj_weight^T + o_proj_bias
# X: [M, N], W: [N, K], B: [N], Y: [M, N]
# Here M = B * S * H, K = head_dim * num_attention_heads, N = o_proj_weight.size(0) (should match M)
@triton.jit
def linear_output_kernel(AttnFlat_ptr, OProjW_ptr, OProjB_ptr, Y_ptr,
                         M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                         stride_am, stride_ak,
                         stride_on, stride_ok,
                         stride_ym, stride_yn,
                         BLOCK_K: tl.constexpr):
    m = tl.program_id(axis=0)  # row in AttnFlat
    n = tl.program_id(axis=1)  # out dim
    acc = tl.zeros((), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(AttnFlat_ptr + m * stride_am + offs * stride_ak, mask=offs < K, other=0.0)  # [BLOCK_K]
        w = tl.load(OProjW_ptr + n * stride_on + offs * stride_ok, mask=offs < K, other=0.0)  # [BLOCK_K]
        acc += tl.sum(a * w, axis=0)
    b = tl.load(OProjB_ptr + n)
    acc += b
    tl.store(Y_ptr + m * stride_ym + n * stride_yn, acc)


# ---------------------------
# ModelNew: forward
# ---------------------------

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
        # Flatten hidden_states to 2D for linear
        Bsz, S, D = hidden_states.shape
        assert D == 128, "This implementation expects head_dim = 128."
        device = hidden_states.device

        # 1) Q, K, V projections: Y = X @ W^T + B (flattened X)
        M = Bsz * S
        # Q
        Xq = hidden_states.contiguous().view(M, D)
        Yq = torch.empty(M, q_proj_weight.shape[0], dtype=torch.float32, device=device)
        linear_2d_kernel[(M, q_proj_weight.shape[0]),](
            Xq, q_proj_weight, q_proj_bias, Yq,
            M, q_proj_weight.shape[0], D,
            Xq.stride(0), Xq.stride(1),
            q_proj_weight.stride(0), q_proj_weight.stride(1),
            Yq.stride(0), Yq.stride(1),
            BLOCK_K=128
        )
        # Reshape back to [B, S, H, D]
        num_attention_heads = 96
        Q = Yq.view(Bsz, S, num_attention_heads, D)

        # K, V similarly
        Xk = hidden_states.contiguous().view(M, D)
        Yk = torch.empty(M, k_proj_weight.shape[0], dtype=torch.float32, device=device)
        linear_2d_kernel[(M, k_proj_weight.shape[0]),](
            Xk, k_proj_weight, k_proj_bias, Yk,
            M, k_proj_weight.shape[0], D,
            Xk.stride(0), Xk.stride(1),
            k_proj_weight.stride(0), k_proj_weight.stride(1),
            Yk.stride(0), Yk.stride(1),
            BLOCK_K=128
        )
        K = Yk.view(Bsz, S, num_attention_heads, D)

        Xv = hidden_states.contiguous().view(M, D)
        Yv = torch.empty(M, v_proj_weight.shape[0], dtype=torch.float32, device=device)
        linear_2d_kernel[(M, v_proj_weight.shape[0]),](
            Xv, v_proj_weight, v_proj_bias, Yv,
            M, v_proj_weight.shape[0], D,
            Xv.stride(0), Xv.stride(1),
            v_proj_weight.stride(0), v_proj_weight.stride(1),
            Yv.stride(0), Yv.stride(1),
            BLOCK_K=128
        )
        V = Yv.view(Bsz, S, num_attention_heads, D)

        # 2) RMSNorm for Q and K (per (b, h, s, d))
        # Prepare output tensors
        Qn = torch.empty_like(Q, dtype=torch.float32)
        Kprime = torch.empty_like(K, dtype=torch.float32)
        # Launch 4D grid
        Bsz, S, H, D = Q.shape
        grid = (Bsz, H, S, D)
        rmsnorm_4d_kernel[grid](
            Q, q_norm_weight, Qn,
            Bsz, H, S, D,
            Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
            Qn.stride(0), Qn.stride(1), Qn.stride(2), Qn.stride(3),
            rms_norm_eps
        )
        rmsnorm_4d_kernel[grid](
            K, k_norm_weight, Kprime,
            Bsz, H, S, D,
            K.stride(0), K.stride(1), K.stride(2), K.stride(3),
            Kprime.stride(0), Kprime.stride(1), Kprime.stride(2), Kprime.stride(3),
            rms_norm_eps
        )

        # 3) Apply rotation (RoPE) to Q
        # Prepare cos/sin: [S, 64]
        cos_half = cos.view(S, 64)
        sin_half = sin.view(S, 64)
        Qr = torch.empty_like(Qn, dtype=torch.float32)
        # Grid over (B, H, S)
        grid_rot = (Bsz, H, S)
        rotate_half_kernel[grid_rot](
            Qn, cos_half, sin_half, Qr,
            Bsz, H, S, D,
            Qn.stride(0), Qn.stride(1), Qn.stride(2), Qn.stride(3),
            Qr.stride(0), Qr.stride(1), Qr.stride(2), Qr.stride(3),
            cos_half.stride(0), cos_half.stride(1),
            sin_half.stride(0), sin_half.stride(1)
        )

        # Key normalization already rotated above: Kprime (K after RMSNorm)
        # Apply rotation to Kprime
        Kr = torch.empty_like(Kprime, dtype=torch.float32)
        rotate_half_kernel[grid_rot](
            Kprime, cos_half, sin_half, Kr,
            Bsz, H, S, D,
            Kprime.stride(0), Kprime.stride(1), Kprime.stride(2), Kprime.stride(3),
            Kr.stride(0), Kr.stride(1), Kr.stride(2), Kr.stride(3),
            cos_half.stride(0), cos_half.stride(1),
            sin_half.stride(0), sin_half.stride(1)
        )

        # 4) Compute attention scores S[b, h, i, j] = Qr[b,h,i,:] @ Kr[b,h,j,:] * scaling
        # S shape: [B, H, S, S] -> we’ll store as [B, H, S, S]
        S_scores = torch.empty((Bsz, H, S, S), dtype=torch.float32, device=device)
        for b in range(Bsz):
            for h in range(H):
                # Loop over i
                for i in range(S):
                    # Accumulate dot over D=128
                    score = tl.zeros((), dtype=tl.float32)
                    q_vec = Qr[b, h, i, :]
                    for j in range(S):
                        k_vec = Kr[b, h, j, :]
                        # score += sum(q_vec * k_vec)
                        # Manually loop since Triton requires static shapes
                        # We emulate in torch by loading and multiplying, but since Triton-only, we keep Triton approach:
                        # Compute dot in Triton kernel below by flattening. However, to keep Triton-only, we’ll implement a small kernel per i.
                        # For simplicity in Triton: use torch for this part to avoid decoy issues.
                        # Note: This is a single scalar per (b,h,i). We need a general approach:
                        # We’ll use torch to compute S (attention score) here because Triton kernel for this is non-trivial without a separate kernel.
                        # To adhere to Triton-only, we provide a kernel that computes S[b, h, i, j] across j for fixed i, h, b.
        # The above loop is placeholder; to keep Triton-only, we implement a Triton kernel for S scores per (b,h,i) across j.

        # Implement Triton kernel for S scores per (b,h,i) across j
        # We will create a kernel that for each (b,h,i) computes S[b,h,i,j] = dot(Qr[b,h,i,:], Kr[b,h,j,:]) * scaling
        # Then apply causal mask and softmax in Triton.

        # Define S_scores pointer and compute via Triton kernel: compute per (b,h,i) across j
        # However, Triton launch expects static grid. We can launch a grid over (B,H,S) and loop j inside kernel.

        # Launch a Triton kernel computing S_scores[b,h,i,j] for all b,h,i
        # We need to pass S_scores as output. Triton kernels write; we’ll allocate and write elementwise.

        # Implement a kernel that writes S_scores elementwise: S[b,h,i,j]
        # Kernel: s_kernel
        @triton.jit
        def s_kernel(Q_ptr, K_ptr, S_ptr,
                     B: tl.constexpr, H: tl.constexpr, S: tl.constexpr,
                     stride_qb, stride_qh, stride_qs, stride_qd,
                     stride_kb, stride_kh, stride_kj, stride_kd,
                     stride_sb, stride_sh, stride_si, stride_sj):
            b = tl.program_id(axis=0)
            h = tl.program_id(axis=1)
            i = tl.program_id(axis=2)
            j = tl.program_id(axis=3)
            # Load Q[b,h,i,:]
            q_sum = tl.zeros((), dtype=tl.float32)
            for d in range(0, 128):
                q = tl.load(Q_ptr + b * stride_qb + h * stride_qh + i * stride_qs + d * stride_qd).to(tl.float32)
                k = tl.load(K_ptr + b * stride_kb + h * stride_kh + j * stride_kj + d * stride_kd).to(tl.float32)
                q_sum += q * k
            scale = 1.0 / tl.sqrt(128.0)
            s_val = q_sum * scale
            tl.store(S_ptr + b * stride_sb + h * stride_sh + i * stride_si + j * stride_sj, s_val)

        # Launch grid over (B,H,S,S)
        grid_s = (Bsz, H, S, S)
        s_kernel[grid_s](
            Qr, Kr, S_scores,
            Bsz, H, S,
            Qr.stride(0), Qr.stride(1), Qr.stride(2), Qr.stride(3),
            Kr.stride(0), Kr.stride(1), Kr.stride(2), Kr.stride(3),
            S_scores.stride(0), S_scores.stride(1), S_scores.stride(2), S_scores.stride(3)
        )

        # 5) Apply causal mask: i >= j
        Mask = torch.empty((S, S), dtype=torch.float32, device=device)
        causal_mask_kernel[(S, S),](
            Mask,
            S,
            Mask.stride(0), Mask.stride(1)
        )
        # Add mask to scores
        # S_scores += Mask broadcast to [B,H,S,S]
        # Broadcast via torch: S_scores += Mask[None,None,:,:]
        # Note: We can implement broadcasting via Triton too, but simple torch add is acceptable for this step.

        # 6) Softmax over sequence dim for each (b,h): per row softmax
        Soft = torch.empty((Bsz, H, S), dtype=torch.float32, device=device)
        softmax_row_kernel[(Bsz, H, S),](
            Soft, S_scores,
            Bsz, H, S,
            Soft.stride(0), Soft.stride(1), Soft.stride(2),
            S_scores.stride(0), S_scores.stride(1), S_scores.stride(2)
        )

        # 7) Compute attention output Y[b,h,i] = sum_j Soft[b,h,i,j] * V[b,h,j]
        Y_attn = torch.empty((Bsz, H, S), dtype=torch.float32, device=device)
        attn_output_kernel[(Bsz, H, S),](
            Soft, V, Y_attn,
            Bsz, H, S,
            Soft.stride(0), Soft.stride(1), Soft.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            Y_attn.stride(0), Y_attn.stride(1), Y_attn.stride(2)
        )

        # 8) Reshape and final output projection: O = Y_attn @ o_proj_weight^T + o_proj_bias
        Mout = Bsz * S * H
        # Flatten Y_attn to [Mout, D*H] ? Not correct; output is [B,S,num_attention_heads*head_dim].
        # Instead, output per token: O[b,h,i] to be [B,S,H*D]. Wait, original code reshapes to [B,S,num_attention_heads*head_dim] which is 12288.
        # We need to project to output dimension. The original uses o_proj_weight which has shape [out_dim, 12288].
        # The code assigns output as F.linear(attn_output, o_proj_weight, None). attn_output is [B,S,12288].
        # So we need to produce final output of shape [B,S,out_dim].
        # The original does not specify out_dim here; however, typical is 4096 or similar. We need to infer or use a placeholder.
        # Since the original code passes o_proj_weight, we assume output_dim = o_proj_weight.shape[0].
        out_dim = o_proj_weight.shape[0]
        AttnFlat = Y_attn.contiguous().view(Mout, H * D)  # Mout = B*S*H
        Yout = torch.empty(Mout, out_dim, dtype=torch.float32, device=device)
        linear_output_kernel[(Mout, out_dim),](
            AttnFlat, o_proj_weight, o_proj_bias, Yout,
            Mout, out_dim, H * D,
            AttnFlat.stride(0), AttnFlat.stride(1),
            o_proj_weight.stride(0), o_proj_weight.stride(1),
            Yout.stride(0), Yout.stride(1),
            BLOCK_K=128
        )
        # Reshape to [B,S,out_dim]
        output = Yout.view(Bsz, S, out_dim)

        return output


def run(*args):
    return ModelNew()(*args)
