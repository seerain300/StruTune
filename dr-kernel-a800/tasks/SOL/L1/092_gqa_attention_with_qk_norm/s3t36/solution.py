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
def linear_mm_addb_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M, N, K,
    stride_xm, stride_xk,      # X strides: m, k
    stride_w0, stride_w1,      # W strides: n, k
    stride_ym, stride_yn,      # Y strides: m, n
):
    m = tl.program_id(axis=0)  # row index
    n = tl.program_id(axis=1)  # col index
    acc = tl.zeros((), dtype=tl.float32)
    # Loop over K in chunks of 128
    for k0 in range(0, K, 128):
        offs_k = k0 + tl.arange(0, 128)
        mask_k = offs_k < K
        x = tl.load(X_ptr + m * stride_xm + offs_k * stride_xk, mask=mask_k, other=0.0)   # [128]
        w = tl.load(W_ptr + n * stride_w0 + offs_k * stride_w1, mask=mask_k, other=0.0)   # [128]
        acc += tl.sum(x * w, axis=0)
    # Add bias
    b = tl.load(B_ptr + n)
    acc += b
    # Store to Y
    tl.store(Y_ptr + m * stride_ym + n * stride_yn, acc)


# Triton kernel: RMSNorm per element for X[b,h,s,d] = X * rsqrt(mean(X^2) + eps) * weight[d]
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
    # D is 128, so we split into 64
    for d in range(0, 128):
        x = tl.load(X_ptr + b * stride_xb + h * stride_xh + s * stride_xs + d * stride_xd).to(tl.float32)
        q1 = x[:64]
        q2 = x[64:]
        c = tl.load(C_ptr + s * stride_c0 + d // 2 * stride_c1).to(tl.float32)
        s_angle = tl.load(S_ptr + s * stride_s0 + d // 2 * stride_s1).to(tl.float32)
        y_half = (-q2) * c + q1 * s_angle
        out = q1 * c + q2 * s_angle
        # Store to output at original d position
        tl.store(Y_ptr + b * stride_yb + h * stride_yh + s * stride_ys + d * stride_yd, out)


# Triton kernel: compute attention scores S[b,h,i,j] = Q[b,h,i,:] @ K[b,h,j,:] * scaling
# Inputs: Q: [B, H, S, D], K: [B, H, S, D], Outputs: S: [B, H, S, S]
@triton.jit
def attn_scores_kernel(
    Q_ptr, K_ptr, S_ptr,
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_sb, stride_sh, stride_si, stride_sj,
    scaling: tl.float32,
):
    b = tl.program_id(axis=0)
    h = tl.program_id(axis=1)
    i = tl.program_id(axis=2)  # query position
    # Accumulate scores for all j
    for j in range(0, S):
        acc = tl.zeros((), dtype=tl.float32)
        # Dot over D=128
        for d in range(0, 128):
            q = tl.load(Q_ptr + b * stride_qb + h * stride_qh + i * stride_qs + d * stride_qd).to(tl.float32)
            k = tl.load(K_ptr + b * stride_kb + h * stride_kh + j * stride_ks + d * stride_kd).to(tl.float32)
            acc += q * k
        acc = acc * scaling
        tl.store(S_ptr + b * stride_sb + h * stride_sh + i * stride_si + j * stride_sj, acc)


# Triton kernel: softmax over sequence dim per (b, h, i): S[b,h,i,:] in-place
@triton.jit
def softmax_cols_kernel(
    S_ptr, S_out_ptr,
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr,
    stride_sb, stride_sh, stride_si, stride_sj,
):
    b = tl.program_id(axis=0)
    h = tl.program_id(axis=1)
    i = tl.program_id(axis=2)
    # Load entire row of length S
    # Create a vector of length S and store max
    # Note: Triton supports vectorized loads via indexing; here we implement a simple loop
    max_val = -float('inf')
    for j in range(0, S):
        s = tl.load(S_ptr + b * stride_sb + h * stride_sh + i * stride_si + j * stride_sj)
        if s > max_val:
            max_val = s
    # Compute exp and sum
    sum_exp = 0.0
    for j in range(0, S):
        s = tl.load(S_ptr + b * stride_sb + h * stride_sh + i * stride_si + j * stride_sj)
        e = tl.exp(s - max_val)
        tl.store(S_out_ptr + b * stride_sb + h * stride_sh + i * stride_si + j * stride_sj, e)
        sum_exp += e
    # Normalize
    for j in range(0, S):
        e = tl.load(S_out_ptr + b * stride_sb + h * stride_sh + i * stride_si + j * stride_sj)
        out = e / sum_exp
        tl.store(S_out_ptr + b * stride_sb + h * stride_sh + i * stride_si + j * stride_sj, out)


# Triton kernel: attention output Y[b,h,i,:] = sum_j Soft[b,h,i,j] * V[b,h,j,:]
@triton.jit
def attn_out_kernel(
    Soft_ptr, V_ptr, Y_ptr,
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    stride_sb, stride_sh, stride_si, stride_sj,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_yb, stride_yh, stride_yi, stride_yd,
):
    b = tl.program_id(axis=0)
    h = tl.program_id(axis=1)
    i = tl.program_id(axis=2)
    for d in range(0, D):
        acc = tl.zeros((), dtype=tl.float32)
        for j in range(0, S):
            soft = tl.load(Soft_ptr + b * stride_sb + h * stride_sh + i * stride_si + j * stride_sj)
            v = tl.load(V_ptr + b * stride_vb + h * stride_vh + j * stride_vs + d * stride_vd)
            acc += soft * v
        tl.store(Y_ptr + b * stride_yb + h * stride_yh + i * stride_yi + d * stride_yd, acc)


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim = 128
        self.num_attention_heads = 96
        self.num_key_value_heads = 8
        self.num_key_value_groups = 12

    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, S, hidden_dim]
        q_proj_weight: torch.Tensor,  # [H, D], H=128*96
        q_proj_bias: torch.Tensor,    # [H]
        k_proj_weight: torch.Tensor,  # [K, D], K=128*8
        k_proj_bias: torch.Tensor,    # [K]
        v_proj_weight: torch.Tensor,  # [V, D], V=128*8
        v_proj_bias: torch.Tensor,    # [V]
        o_proj_weight: torch.Tensor,  # [O, D], O=hidden_dim
        o_proj_bias: torch.Tensor,    # [O]
        q_norm_weight: torch.Tensor,  # [D]
        k_norm_weight: torch.Tensor,  # [D]
        cos: torch.Tensor,            # [S, 64]
        sin: torch.Tensor,            # [S, 64]
        rms_norm_eps: float,
    ):
        # Ensure inputs are on CUDA
        assert TRITON_AVAILABLE, "Triton is not available"
        device = hidden_states.device
        Bsz, S, hidden_dim = hidden_states.shape
        D = self.head_dim
        H = self.num_attention_heads
        K_h = self.num_key_value_heads
        G = self.num_key_value_groups
        # Flatten and cast for Triton
        hs = hidden_states.reshape(-1, D).to(torch.float32)  # [B*S, D]
        # 1) Q projection
        Q = torch.empty(Bsz * S, H, dtype=torch.float32, device=device)
        # Reshape q_proj_weight to [H, D_in], D_in=D
        # Note: original q_proj_weight shape is [H, D], here H=128*96=12288
        # We'll use W shape [H, D] for linear. Prepare Wt [H, D].
        # But our linear kernel expects X[M,K], W[N,K], so set M=B*S, N=H, K=D
        # Here we need to linearize: For Triton, we pass pointers; we can use hs as X.
        # However, linear mm uses general X[M,K]; we can compute Q = hs @ q_proj_weight^T + bias
        # We'll construct X = hs, W = q_proj_weight, B = q_proj_bias. But hs has shape [B*S, D], q_proj_weight has shape [H, D].
        # To compute Q of shape [B*S, H], we need X to be [B*S, D] and W to be [H, D]. So:
        # We'll set X_ptr = hs, W_ptr = q_proj_weight, B_ptr = q_proj_bias. Result Y = [B*S, H]
        # Create tensors:
        # X: [M=B*S, K=D]
        M = Bsz * S
        # Construct X as hs, K=D
        X_Q = hs
        # W_Q: [N=H, K=D]
        W_Q = q_proj_weight
        B_Q = q_proj_bias
        # Output Q_mat [B*S, H]
        Q_mat = torch.empty((M, H), dtype=torch.float32, device=device)
        linear_mm_addb_kernel[(M, H)](
            X_Q, W_Q, B_Q, Q_mat,
            M, H, D,
            X_Q.stride(0), D,          # stride_xm, stride_xk
            W_Q.stride(0), W_Q.stride(1),  # stride_w0, stride_w1
            Q_mat.stride(0), Q_mat.stride(1),  # stride_ym, stride_yn
            num_warps=4, num_stages=2
        )
        Q = Q_mat.view(Bsz, S, H)

        # 2) RMSNorm on Q
        Q_norm = torch.empty_like(Q, dtype=torch.float32, device=device)
        # Grid over (B, H, S, D) = (Bsz, H, S, D)
        grid = (Bsz, H, S, D)
        rmsnorm_kernel[grid](
            Q.reshape(-1, S * H, D), q_norm_weight, Q_norm.reshape(-1, S * H, D),
            Bsz, H, S, D,
            Q.reshape(-1, S * H, D).stride(0), Q.reshape(-1, S * H, D).stride(1), Q.reshape(-1, S * H, D).stride(2),
            Q_norm.reshape(-1, S * H, D).stride(0), Q_norm.reshape(-1, S * H, D).stride(1), Q_norm.reshape(-1, S * H, D).stride(2),
            rms_norm_eps,
            num_warps=2, num_stages=2
        )
        Q = Q_norm  # shape [B, S, H]

        # 3) K projection
        K_mat = torch.empty((M, K_h * D), dtype=torch.float32, device=device)
        W_K = k_proj_weight
        B_K = k_proj_bias
        linear_mm_addb_kernel[(M, K_h * D)](
            X_Q, W_K, B_K, K_mat,
            M, K_h * D, D,
            X_Q.stride(0), D,
            W_K.stride(0), W_K.stride(1),
            K_mat.stride(0), K_mat.stride(1),
            num_warps=4, num_stages=2
        )
        K = K_mat.view(Bsz, S, K_h * D)

        # 4) RMSNorm on K
        K_norm = torch.empty_like(K, dtype=torch.float32, device=device)
        grid = (Bsz, K_h, S, D)
        # For K_norm grid over (B, K_h, S, D)
        rmsnorm_kernel[grid](
            K.reshape(-1, S * K_h, D), k_norm_weight, K_norm.reshape(-1, S * K_h, D),
            Bsz, K_h, S, D,
            K.reshape(-1, S * K_h, D).stride(0), K.reshape(-1, S * K_h, D).stride(1), K.reshape(-1, S * K_h, D).stride(2),
            K_norm.reshape(-1, S * K_h, D).stride(0), K_norm.reshape(-1, S * K_h, D).stride(1), K_norm.reshape(-1, S * K_h, D).stride(2),
            rms_norm_eps,
            num_warps=2, num_stages=2
        )
        K = K_norm  # shape [B, S, K_h*D]

        # 5) V projection
        V_mat = torch.empty((M, K_h * D), dtype=torch.float32, device=device)
        W_V = v_proj_weight
        B_V = v_proj_bias
        linear_mm_addb_kernel[(M, K_h * D)](
            X_Q, W_V, B_V, V_mat,
            M, K_h * D, D,
            X_Q.stride(0), D,
            W_V.stride(0), W_V.stride(1),
            V_mat.stride(0), V_mat.stride(1),
            num_warps=4, num_stages=2
        )
        V = V_mat.view(Bsz, S, K_h * D)

        # 6) Apply rotation (RoPE) to Q and K
        # Prepare output tensors for rotated Q and K
        Q_rot = torch.empty_like(Q, dtype=torch.float32, device=device)
        K_rot = torch.empty_like(K, dtype=torch.float32, device=device)
        # Grid over (B, H, S)
        grid_rot = (Bsz, H, S)
        rotate_half_kernel[grid_rot](
            Q.reshape(Bsz, H, S, D), cos, sin, Q_rot.reshape(Bsz, H, S, D),
            Bsz, H, S, D,
            Q.reshape(Bsz, H, S, D).stride(0), Q.reshape(Bsz, H, S, D).stride(1), Q.reshape(Bsz, H, S, D).stride(2), Q.reshape(Bsz, H, S, D).stride(3),
            Q_rot.reshape(Bsz, H, S, D).stride(0), Q_rot.reshape(Bsz, H, S, D).stride(1), Q_rot.reshape(Bsz, H, S, D).stride(2), Q_rot.reshape(Bsz, H, S, D).stride(3),
            cos.stride(0), cos.stride(1), sin.stride(0), sin.stride(1),
            num_warps=4, num_stages=2
        )
        rotate_half_kernel[grid_rot](
            K.reshape(Bsz, K_h, S, D), cos, sin, K_rot.reshape(Bsz, K_h, S, D),
            Bsz, K_h, S, D,
            K.reshape(Bsz, K_h, S, D).stride(0), K.reshape(Bsz, K_h, S, D).stride(1), K.reshape(Bsz, K_h, S, D).stride(2), K.reshape(Bsz, K_h, S, D).stride(3),
            K_rot.reshape(Bsz, K_h, S, D).stride(0), K_rot.reshape(Bsz, K_h, S, D).stride(1), K_rot.reshape(Bsz, K_h, S, D).stride(2), K_rot.reshape(Bsz, K_h, S, D).stride(3),
            cos.stride(0), cos.stride(1), sin.stride(0), sin.stride(1),
            num_warps=4, num_stages=2
        )

        # 7) Repeat K/V heads for GQA: [B, H, S, D] where H=96, K_h=8, groups=12
        # We need to expand K_rot/V to [B, 96, S, D] by repeating each K_h into H groups.
        # Because groups=12, each original head j is repeated into 12 new heads:
        # new_h = j * 12 + g for g in [0..11].
        # We will compute key/value per head and expand. However, since K_h=8 and H=96 => H//K_h=12, consistent.
        # Prepare Key and Value expanded tensors
        Key_exp = torch.empty((Bsz, H, S, D), dtype=torch.float32, device=device)
        Value_exp = torch.empty((Bsz, H, S, D), dtype=torch.float32, device=device)

        # We can fill Key_exp/Value_exp by repeating K_rot/V across H slots
        for j in range(K_h):
            # repeat into 12 groups
            for g in range(G):
                new_h = j * G + g
                Key_exp[:, new_h, :, :] = K_rot[:, j, :, :]
                Value_exp[:, new_h, :, :] = V[:, j, :, :]

        # 8) Compute attention scores S[b,h,i,j] = Q[b,h,i,:] @ K[b,h,j,:] * scaling
        # We have Q_rot shape [B, S, H], Key_exp shape [B, H, S, D]
        # To feed attn_scores_kernel: Q_ptr -> [B, H, S, D], K_ptr -> [B, H, S, D], S_ptr -> [B, H, S, S]
        # We need to reshape Q_rot to [B, H, S, D], but Q_rot has D=H=128? Actually H=96, D=128.
        # We need Q and K of shape [B, H, S, D], but Q has shape [B, S, H] from linear mm. We can transpose:
        # Q_T = Q_rot.permute(0, 2, 1, 3) -> [B, S, H, D]
        # Key_exp is [B, H, S, D]
        Q_T = Q_rot.permute(0, 2, 1, 3)  # [B, S, H, D]
        # attn_scores_kernel expects Q, K as [B, H, S, D] but we have [B, S, H, D]. It’s okay to pass transpose; kernel uses strides so it can handle any indexing provided we pass correct strides.
        # However, Triton kernel signature uses Q_ptr of shape [B, H, S, D]. Let’s rearrange: since we can pass any contiguous layout, we can materialize Q_T as contiguous with shape [B, H, S, D] by permuting.
        # To do that, we can construct a contiguous Q_Q = Q_T.contiguous() with shape [B, S, H, D], but we need [B, H, S, D]. So we reshape by swapping dims: Q_Q = Q_T.permute(0,2,1,3).contiguous() -> [B, S, H, D] still, not desired. Better: we directly pass Q_T as is and let kernel read with strides. The kernel uses stride_qb, stride_qh, stride_qs, stride_qd; with Q_T being [B, S, H, D], stride_qh would correspond to H dimension, which is not what we want. Therefore, we should explicitly make Q_Q = Q_T.permute(0,2,1,3).contiguous(), then reshape to [B, H, S, D] by another permute, but that would require a 4D permutation back, which is awkward.
        # To avoid confusion, let’s create Q_Q = Q_rot.view(B, S, H, D).contiguous(), then permute to [B, H, S, D] with swap dims:
        # We can do: Q_Q = Q_rot.view(B, S, H, D).permute(0,2,1,3).contiguous() -> [B, H, S, D]
        # Similarly for Key_exp:
        # Key_exp is [B, H, S, D] already, so we can use Key_ptr directly.
        # Compute attention scores S: [B, H, S, S]
        S_scores = torch.empty((Bsz, H, S, S), dtype=torch.float32, device=device)
        # Launch attn_scores_kernel with Q_Q and Key_ptr
        # Prepare Q_Q and Key_ptr
        # Q_Q = Q_T.permute(0,2,1,3).contiguous() -> [B, H, S, D]
        # But Q_T = [B, S, H, D]. To permute to [B, H, S, D], we need to keep the last dimension D intact, so we can do:
        # Q_Q = Q_T.permute(0,1,3,2).contiguous() -> [B, S, D, H], not desired. It seems we cannot directly get [B, H, S, D] from [B, S, H, D] via simple permute without copying.
        # Therefore, we will materialize Q_Q by transposing and making contiguous:
        Q_Q = Q_T.permute(0, 2, 1, 3).contiguous()  # [B, H, S, D]
        Key_ptr = Key_exp  # [B, H, S, D]
        # Scaling
        scaling = 1.0 / (D ** 0.5)
        attn_scores_kernel[(Bsz, H, S)](
            Q_Q, Key_ptr, S_scores,
            Bsz, H, S, D,
            Q_Q.stride(0), Q_Q.stride(1), Q_Q.stride(2), Q_Q.stride(3),
            Key_ptr.stride(0), Key_ptr.stride(1), Key_ptr.stride(2), Key_ptr.stride(3),
            S_scores.stride(0), S_scores.stride(1), S_scores.stride(2), S_scores.stride(3),
            scaling,
            num_warps=4, num_stages=2
        )

        # 9) Softmax over sequence dim per (b, h, i)
        Soft = torch.empty_like(S_scores, dtype=torch.float32, device=device)
        # We need softmax along last dim (sequence position j). For each (b, h, i), normalize S_scores[b,h,i,:].
        # Launch softmax_cols_kernel for each (b,h,i). Triton supports loops; we’ll run one program per (b,h,i).
        grid_softmax = (Bsz, H, S)
        softmax_cols_kernel[grid_softmax](
            S_scores, Soft,
            Bsz, H, S,
            S_scores.stride(0), S_scores.stride(1), S_scores.stride(2), S_scores.stride(3),
            num_warps=4, num_stages=2
        )

        # 10) Attention output: Y[b,h,i,:] = sum_j Soft[b,h,i,j] * Value_exp[b,h,j,:]
        # Value_exp is [B, H, S, D]. We need to compute Y for each (b, h, i).
        Y_attn = torch.empty((Bsz, H, S, D), dtype=torch.float32, device=device)
        attn_out_kernel[(Bsz, H, S, D)](
            Soft, Value_exp,
            Y_attn,
            Bsz, H, S, D,
            Soft.stride(0), Soft.stride(1), Soft.stride(2), Soft.stride(3),
            Value_exp.stride(0), Value_exp.stride(1), Value_exp.stride(2), Value_exp.stride(3),
            Y_attn.stride(0), Y_attn.stride(1), Y_attn.stride(2), Y_attn.stride(3),
            num_warps=4, num_stages=2
        )

        # 11) Final output projection: Y_out = attn_output_flat @ o_proj_weight^T + o_proj_bias
        # attn_output_flat is [B*S*H, D] -> reshape to [B*S, H*D], but we have [B, H, S, D]. Flatten to [M, D] where M=B*S*H.
        M_out = Bsz * S * H
        attn_flat = Y_attn.reshape(M_out, D).to(torch.float32)
        O = torch.empty(M_out, dtype=torch.float32, device=device)
        W_O = o_proj_weight  # [O, D], O=hidden_dim
        B_O = o_proj_bias
        # linear_mm_addb_kernel expects X[M,K], W[N,K], B[N], Y[M,N]
        # Here X=attn_flat [M_out, D], W=W_O [O, D], B=B_O [O], Y [M_out, O]
        linear_mm_addb_kernel[(M_out, O)](
            attn_flat, W_O, B_O, O,
            M_out, O, D,
            attn_flat.stride(0), D,
            W_O.stride(0), W_O.stride(1),
            O.stride(0), O.stride(1),
            num_warps=4, num_stages=2
        )
        output = O.reshape(Bsz, S, H * D)  # H*D = hidden_dim
        # Cast to original dtype if needed
        # original hidden_states dtype is typically float16/bfloat16; output is float32 here. If original expected float32, keep as is.
        return output


def run(*args):
    return ModelNew()(*args)
