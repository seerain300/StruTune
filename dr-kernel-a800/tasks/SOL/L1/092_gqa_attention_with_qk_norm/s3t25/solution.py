import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: Y = X @ W^T + B
# X: [M, K], W: [N, K], B: [N], Y: [M, N]
# We will call it with M = B*S, K = input_dim, N = output_dim
@triton.jit
def linear_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    BLOCK_K: tl.constexpr,
):
    m = tl.program_id(axis=0)  # row in X/Y, corresponds to b*s
    n = tl.program_id(axis=1)  # output feature
    acc = tl.zeros((), dtype=tl.float32)
    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        x = tl.load(X_ptr + m * stride_xm + offs_k * stride_xk, mask=mask_k, other=0.0)  # [BLOCK_K]
        w = tl.load(W_ptr + n * stride_wn + offs_k * stride_wk, mask=mask_k, other=0.0)  # [BLOCK_K]
        acc += tl.sum(x * w, axis=0)
    # Add bias
    b = tl.load(B_ptr + n)
    acc += b
    # Store Y[m, n]
    tl.store(Y_ptr + m * stride_ym + n * stride_yn, acc)


# Triton kernel: RMSNorm per element (b, h, s, d) using weight of length D
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
    # Load x (single scalar)
    x = tl.load(X_ptr + b * stride_xb + h * stride_xh + s * stride_xs + d * stride_xd).to(tl.float32)
    # Compute mean of x^2 over the entire feature vector (here it's a single scalar, but we keep signature generic)
    # For a single element, mean is just x^2 / D, but since d varies, we assume X is [B, H, S, D] and we compute per (b,h,s,d)
    # So mean_sq = x^2 / D
    mean_sq = x * x / D
    inv_rms = tl.rsqrt(mean_sq + eps)
    w = tl.load(W_ptr + d).to(tl.float32)
    y = (x * inv_rms) * w
    tl.store(Y_ptr + b * stride_yb + h * stride_yh + s * stride_ys + d * stride_yd, y)


# Triton kernel: apply rotation (RoPE) for half dims: 64 elements
# X: [B, H, S, 128], C: [S, 64], S: [S, 64], Y: [B, H, S, 128]
# Rotate only the first half (64 dims) and apply to both halves:
# q1 = X[..., :64], q2 = X[..., 64:], rotated_half = cat((-q2, q1), -1)
# Y = X * cos + rotated_half * sin
@triton.jit
def rotate_half_kernel(
    X_ptr, C_ptr, S_ptr, Y_ptr,
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr,
    stride_xb, stride_xh, stride_xs, stride_xd,
    stride_yb, stride_yh, stride_ys, stride_yd,
    stride_c0, stride_c1,     # cos strides: [S, 64]
    stride_s0, stride_s1,     # sin strides: [S, 64]
    HALF: tl.constexpr = 64,  # half dimension
):
    b = tl.program_id(axis=0)
    h = tl.program_id(axis=1)
    s = tl.program_id(axis=2)
    # Load full 128-d vector
    vec = tl.load(X_ptr + b * stride_xb + h * stride_xh + s * stride_xs + tl.arange(0, 128) * stride_xd)
    # Split into q1 and q2
    q1 = vec[:64]
    q2 = vec[64:]
    # Load cos/sin for this s
    cos_vals = tl.load(C_ptr + s * stride_c0 + tl.arange(0, HALF) * stride_c1)  # [64]
    sin_vals = tl.load(S_ptr + s * stride_s0 + tl.arange(0, HALF) * stride_s1)  # [64]
    rotated_half = -q2 * cos_vals + q1 * sin_vals
    y = vec * cos_vals + rotated_half * sin_vals
    tl.store(Y_ptr + b * stride_yb + h * stride_yh + s * stride_ys + tl.arange(0, 128) * stride_yd, y)


class ModelNew(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int, num_key_value_heads: int,
                 num_key_value_groups: int, head_dim: int, q_norm_eps: float, k_norm_eps: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_key_value_groups
        self.head_dim = head_dim
        self.q_norm_eps = q_norm_eps
        self.k_norm_eps = k_norm_eps
        # Dummy weights/bias placeholders to pass to kernels; actual weights are expected to be provided at runtime.
        # We won't use nn.Parameters here; instead we will pass actual torch tensors to forward.
        self.scaling = 1.0  # not used in this simplified version

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
        o_proj_bias: torch.Tensor,
        q_norm_weight: torch.Tensor,
        k_norm_weight: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ):
        # Ensure device and dtype; compute in float32
        device = hidden_states.device
        # Prepare shapes
        B = batch_size
        S = seq_len
        D_in = hidden_states.shape[-1]  # 1280 in original
        # Flatten [B, S, D_in] -> [M, D_in] for linear, where M = B*S
        X = hidden_states.reshape(B * S, D_in).contiguous().to(torch.float32)
        M = B * S

        # Kernels require contiguous strides; we pass row-major [M, K] -> strides (M, K)
        # 1) Q projection
        Q = torch.empty((M, q_proj_weight.shape[0]), dtype=torch.float32, device=device)
        if TRITON_AVAILABLE:
            grid_q = (M, q_proj_weight.shape[0])
            linear_kernel[grid_q](
                X, q_proj_weight, q_proj_bias,
                Q,
                M, D_in, q_proj_weight.shape[0],
                X.stride(0), X.stride(1),
                q_proj_weight.stride(0), q_proj_weight.stride(1),
                Q.stride(0), Q.stride(1),
                BLOCK_K=64,
            )
        else:
            Q = F.linear(hidden_states, q_proj_weight, q_proj_bias)

        # 2) K projection
        K_raw = torch.empty((M, k_proj_weight.shape[0]), dtype=torch.float32, device=device)
        if TRITON_AVAILABLE:
            grid_k = (M, k_proj_weight.shape[0])
            linear_kernel[grid_k](
                X, k_proj_weight, k_proj_bias,
                K_raw,
                M, D_in, k_proj_weight.shape[0],
                X.stride(0), X.stride(1),
                k_proj_weight.stride(0), k_proj_weight.stride(1),
                K_raw.stride(0), K_raw.stride(1),
                BLOCK_K=64,
            )
        else:
            K_raw = F.linear(hidden_states, k_proj_weight, k_proj_bias)

        # 3) V projection
        V_raw = torch.empty((M, v_proj_weight.shape[0]), dtype=torch.float32, device=device)
        if TRITON_AVAILABLE:
            grid_v = (M, v_proj_weight.shape[0])
            linear_kernel[grid_v](
                X, v_proj_weight, v_proj_bias,
                V_raw,
                M, D_in, v_proj_weight.shape[0],
                X.stride(0), X.stride(1),
                v_proj_weight.stride(0), v_proj_weight.stride(1),
                V_raw.stride(0), V_raw.stride(1),
                BLOCK_K=64,
            )
        else:
            V_raw = F.linear(hidden_states, v_proj_weight, v_proj_bias)

        # 4) RMSNorm for Q and K
        # Reshape Q/K_raw to [B, H, S, D]
        D_out = q_proj_weight.shape[0]  # same for Q and K projection outputs (head_dim)
        assert D_out == self.head_dim, "Output dim mismatch"
        # We need to infer H and S after projection, but original code has hidden_states [B, S, D_in], and we already flattened to [B*S, D_in].
        # After projection, Q/K/V are [B*S, D_out], then we reshape to [B, num_attention_heads, S, D_out].
        # Here num_attention_heads = 96, D_out = head_dim = 128
        H_q = self.num_attention_heads
        H_k = self.num_key_value_heads
        # We'll compute Q/K/V as [B, H, S, D_out]
        Q4 = Q.view(B, H_q, S, D_out)
        K4 = K_raw.view(B, H_k, S, D_out)
        V4 = V_raw.view(B, H_k, S, D_out)

        # Launch RMSNorm for Q and K
        # Ensure contiguity
        Q4 = Q4.contiguous()
        K4 = K4.contiguous()
        V4 = V4.contiguous()

        # Allocate outputs for RMSNorm
        Q_norm = torch.empty_like(Q4, dtype=torch.float32, device=device)
        K_norm = torch.empty_like(K4, dtype=torch.float32, device=device)

        if TRITON_AVAILABLE:
            grid_rms = (B, H_q, S, D_out)
            rmsnorm_kernel[grid_rms](
                Q4, q_norm_weight, Q_norm,
                B, H_q, S, D_out,
                Q4.stride(0), Q4.stride(1), Q4.stride(2), Q4.stride(3),
                Q_norm.stride(0), Q_norm.stride(1), Q_norm.stride(2), Q_norm.stride(3),
                self.q_norm_eps,
            )
            grid_rms_k = (B, H_k, S, D_out)
            rmsnorm_kernel[grid_rms_k](
                K4, k_norm_weight, K_norm,
                B, H_k, S, D_out,
                K4.stride(0), K4.stride(1), K4.stride(2), K4.stride(3),
                K_norm.stride(0), K_norm.stride(1), K_norm.stride(2), K_norm.stride(3),
                self.k_norm_eps,
            )
        else:
            # Fallback RMSNorm using PyTorch
            # For Q
            # Compute per (b,h,s): mean = q.mean(-1, keepdim=True), inv = 1/sqrt(mean + eps), y = q * inv * weight
            for b in range(B):
                for h in range(H_q):
                    q = Q4[b, h]  # [S, D_out]
                    mean = q.pow(2).mean(-1, keepdim=True)
                    inv = torch.rsqrt(mean + self.q_norm_eps)
                    Q_norm[b, h] = q * inv * q_norm_weight
            # For K
            for b in range(B):
                for h in range(H_k):
                    k = K4[b, h]  # [S, D_out]
                    mean = k.pow(2).mean(-1, keepdim=True)
                    inv = torch.rsqrt(mean + self.k_norm_eps)
                    K_norm[b, h] = k * inv * k_norm_weight

        # 5) Rotate (RoPE) for Q and K
        # Rotate half: first 64 dims
        # Ensure cos/sin are contiguous and on device
        cos_ = cos.to(torch.float32).contiguous()
        sin_ = sin.to(torch.float32).contiguous()
        # We need to pass C and S as [S, HALF]
        C = cos_[:, :64].contiguous()  # [S, 64]
        S_ = sin_[:, :64].contiguous() # [S, 64]

        # Allocate rotated tensors
        Q_rot = torch.empty_like(Q_norm, dtype=torch.float32, device=device)
        K_rot = torch.empty_like(K_norm, dtype=torch.float32, device=device)

        if TRITON_AVAILABLE:
            grid_rot_q = (B, H_q, S)
            rotate_half_kernel[grid_rot_q](
                Q_norm, C, S_, Q_rot,
                B, H_q, S,
                Q_norm.stride(0), Q_norm.stride(1), Q_norm.stride(2), Q_norm.stride(3),
                Q_rot.stride(0), Q_rot.stride(1), Q_rot.stride(2), Q_rot.stride(3),
                C.stride(0), C.stride(1),
                S_.stride(0), S_.stride(1),
                HALF=64,
            )
            grid_rot_k = (B, H_k, S)
            rotate_half_kernel[grid_rot_k](
                K_norm, C, S_, K_rot,
                B, H_k, S,
                K_norm.stride(0), K_norm.stride(1), K_norm.stride(2), K_norm.stride(3),
                K_rot.stride(0), K_rot.stride(1), K_rot.stride(2), K_rot.stride(3),
                C.stride(0), C.stride(1),
                S_.stride(0), S_.stride(1),
                HALF=64,
            )
        else:
            # Fallback rotation using PyTorch: split and apply rotation formula
            for b in range(B):
                for h in range(H_q):
                    q = Q_norm[b, h]  # [S, D_out]
                    q1 = q[:, :64]
                    q2 = q[:, 64:]
                    cos_s = C[s]  # [64]
                    sin_s = S_[s]  # [64]
                    rotated_half = -q2 * cos_s + q1 * sin_s
                    Q_rot[b, h] = q * cos_s + rotated_half * sin_s
                for h in range(H_k):
                    k = K_norm[b, h]  # [S, D_out]
                    k1 = k[:, :64]
                    k2 = k[:, 64:]
                    rotated_half = -k2 * cos_s + k1 * sin_s
                    K_rot[b, h] = k * cos_s + rotated_half * sin_s

        # Now we have rotated Q and K per attention head. Next, attention and output.
        # Repeat KV heads for GQA: [B, H_k, S, D_out] -> [B, H_q, S, D_out]
        K_rot_expanded = K_rot[:, :, None, :, :].expand(B, H_k, H_q // H_k, S, D_out).reshape(B, H_q, S, D_out)
        V4_expanded = V4[:, :, None, :, :].expand(B, H_k, H_q // H_k, S, D_out).reshape(B, H_q, S, D_out)

        # Compute attention: scores = Q_rot @ K_rot_expanded^T
        # We'll use PyTorch for this part to ensure correctness (softmax + causal mask + matmul)
        # Reshape to [B, H_q, S, D_out]
        attn_weights = torch.matmul(Q_rot, K_rot_expanded.transpose(2, 3)) * (self.head_dim ** -0.5)
        # Causal mask: upper triangular, keep lower triangle; for (i, j) if i < j, set -inf
        # attn_weights shape: [B, H_q, S, S]
        # Build causal mask
        causal_mask = torch.triu(torch.full((S, S), float('-inf'), device=device, dtype=torch.float32), diagonal=1)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)  # [1, 1, S, S]; broadcasting over B and H_q
        attn_weights = attn_weights + causal_mask

        # Softmax over last dim (sequence length S)
        attn_weights = F.softmax(attn_weights, dim=-1)

        # Compute attention output: [B, H_q, S, D_out] = attn_weights @ V4_expanded
        attn_output = torch.matmul(attn_weights, V4_expanded)

        # Transpose and reshape to [B, S, H_q * D_out]
        attn_output = attn_output.transpose(1, 2).contiguous()  # [B, S, H_q * D_out]
        attn_output = attn_output.reshape(B, S, H_q * D_out)

        # 6) Final output projection
        output = torch.empty((B * S, o_proj_weight.shape[0]), dtype=torch.float32, device=device)
        if TRITON_AVAILABLE:
            grid_o = (B * S, o_proj_weight.shape[0])
            linear_kernel[grid_o](
                attn_output.reshape(B * S, H_q * D_out),
                o_proj_weight, o_proj_bias,
                output,
                B * S, H_q * D_out, o_proj_weight.shape[0],
                attn_output.reshape(B * S, H_q * D_out).stride(0), attn_output.reshape(B * S, H_q * D_out).stride(1),
                o_proj_weight.stride(0), o_proj_weight.stride(1),
                output.stride(0), output.stride(1),
                BLOCK_K=64,
            )
        else:
            output = F.linear(attn_output, o_proj_weight, o_proj_bias)

        return output


# Original Model uses these axes in evaluation. We keep them for compatibility.
@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    q_proj_weight: torch.Tensor,
    q_proj_bias: torch.Tensor,
    k_proj_weight: torch.Tensor,
    k_proj_bias: torch.Tensor,
    v_proj_weight: torch.Tensor,
    v_proj_bias: torch.Tensor,
    o_proj_weight: torch.Tensor,
    o_proj_bias: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rms_norm_eps: float,
    batch_size: int,
    seq_len: int,
):
    # Pass eps for Q and K separately if needed; original code uses rms_norm_eps for both. We simplify here.
    # Create a ModelNew instance; but since the evaluator calls ModelNew.forward directly, we don't need a module.
    # Just call the forward function below with provided tensors.
    # We will construct the axes as needed; the evaluator supplies them.
    # Forward function expects axes in signature; we pass None for batch_size/seq_len and compute internally.
    # However, to match original signature, we keep the call compatible.
    # The evaluator provides batch_size and seq_len via axes dict. We pass them.
    return ModelNew(
        hidden_size=hidden_states.shape[-1],
        num_attention_heads=96,
        num_key_value_heads=8,
        num_key_value_groups=12,
        head_dim=128,
        q_norm_eps=rms_norm_eps,
        k_norm_eps=rms_norm_eps,
    ).forward(
        hidden_states,
        q_proj_weight, q_proj_bias,
        k_proj_weight, k_proj_bias,
        v_proj_weight, v_proj_bias,
        o_proj_weight, o_proj_bias,
        q_norm_weight, k_norm_weight,
        cos, sin,
        batch_size=batch_size,
        seq_len=seq_len,
    )


class Model(nn.Module):
    def forward(self, *args):
        # We rely on run function which calls ModelNew.forward; args match original signature.
        return run(*args)


def run(*args):
    return ModelNew()(*args)
