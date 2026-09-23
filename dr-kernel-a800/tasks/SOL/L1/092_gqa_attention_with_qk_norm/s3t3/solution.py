import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False

# Triton kernel for dense linear: Y = X @ W^T + B
# X: [B, S, D_in], W: [D_out, D_in], B: [D_out] or None, Y: [B, S, D_out]
@triton.jit
def linear_kernel(
    X_ptr,           # *fp32, input [B, S, D_in]
    W_ptr,           # *fp32, weight [D_out, D_in]
    B_ptr,           # *fp32, bias [D_out] or dummy
    Y_ptr,           # *fp32, output [B, S, D_out]
    B: tl.constexpr, S: tl.constexpr, D_in: tl.constexpr, D_out: tl.constexpr,
    stride_xb, stride_xs, stride_xd,   # strides for X: (B,S,D_in)
    stride_w0, stride_w1,              # strides for W: (D_out, D_in)
    stride_yb, stride_ys, stride_yd,   # strides for Y: (B,S,D_out)
    BLOCK_K: tl.constexpr,             # reduction tile over D_in, e.g., 64
):
    pid = tl.program_id(axis=0)  # one program per (b, s)
    b = pid // S
    s = pid % S

    # Accumulator for output vector of length D_out
    acc = tl.zeros((D_out,), dtype=tl.float32)

    # Loop over K dimension in tiles of BLOCK_K
    for k0 in range(0, D_in, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        # Load X[b, s, k_offsets]
        x = tl.load(
            X_ptr + b * stride_xb + s * stride_xs + k_offsets * stride_xd,
            mask=k_offsets < D_in,
            other=0.0
        )  # [BLOCK_K]
        # Load W[:, k_offsets] -> shape [D_out, BLOCK_K]
        w = tl.load(
            W_ptr + k_offsets[None, :] * stride_w1 + tl.arange(0, D_out)[:, None] * stride_w0,
            mask=(tl.arange(0, D_out)[:, None] < D_out) & (k_offsets[None, :] < D_in),
            other=0.0
        )  # [D_out, BLOCK_K]
        # acc += sum over k of x[k] * W[:, k]
        acc += tl.sum(w * x[None, :], axis=1)  # reduce over BLOCK_K -> [D_out]

    # Add bias if provided
    if B_ptr is not None:
        b = tl.load(B_ptr + tl.arange(0, D_out), mask=tl.arange(0, D_out) < D_out, other=0.0)
        acc += b

    # Store result Y[b, s, :]
    tl.store(Y_ptr + b * stride_yb + s * stride_ys + tl.arange(0, D_out) * stride_yd, acc, mask=tl.arange(0, D_out) < D_out)

# Triton kernel: RMSNorm over last dim, elementwise: y = weight * x * rsqrt(mean(x^2)+eps)
# Inputs are [B, H, S, D] with arbitrary strides; weight is [D].
@triton.jit
def rmsnorm_kernel(
    X_ptr,        # *fp32, input [B, H, S, D]
    W_ptr,        # *fp32, weight [D]
    Y_ptr,        # *fp32, output [B, H, S, D]
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    stride_xb, stride_xh, stride_xs, stride_xd,
    stride_yb, stride_yh, stride_ys, stride_yd,
    eps,          # float32
):
    b = tl.program_id(axis=0)
    h = tl.program_id(axis=1)
    s = tl.program_id(axis=2)
    d_offsets = tl.arange(0, D)
    x = tl.load(
        X_ptr + b * stride_xb + h * stride_xh + s * stride_xs + d_offsets * stride_xd,
        mask=d_offsets < D,
        other=0.0
    )
    x32 = x.to(tl.float32)
    mean_sq = tl.sum(x32 * x32, axis=0) / D
    inv_rms = tl.rsqrt(mean_sq + eps)
    w = tl.load(W_ptr + d_offsets, mask=d_offsets < D, other=1.0)
    y = (x32 * inv_rms) * w
    tl.store(
        Y_ptr + b * stride_yb + h * stride_yh + s * stride_ys + d_offsets * stride_yd,
        y,
        mask=d_offsets < D
    )

# Triton kernel: apply rotation (RoPE) for last dim D=128, split into two halves (64,64)
# X: [B, H, S, D], C: [S, 64], S: [S, 64], Y: [B, H, S, D]
@triton.jit
def rotate_half_kernel(
    X_ptr,        # *fp32, input [B, H, S, D]
    C_ptr,        # *fp32, cos [S, D/2] flattened
    S_ptr,        # *fp32, sin [S, D/2] flattened
    Y_ptr,        # *fp32, output [B, H, S, D]
    B: tl.constexpr, H: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    stride_xb, stride_xh, stride_xs, stride_xd,
    stride_yb, stride_yh, stride_ys, stride_yd,
    stride_c0, stride_c1,     # strides for cos: (S, D/2)
    stride_s0, stride_s1,     # strides for sin: (S, D/2)
):
    b = tl.program_id(axis=0)
    h = tl.program_id(axis=1)
    s = tl.program_id(axis=2)
    d_offsets = tl.arange(0, D)

    # Load original X[b, h, s, :]
    x = tl.load(
        X_ptr + b * stride_xb + h * stride_xh + s * stride_xs + d_offsets * stride_xd,
        mask=d_offsets < D,
        other=0.0
    )  # [D]

    # Split into two halves
    d_half = D // 2
    d1 = d_offsets[:d_half]        # first half [0:64]
    d2 = d_offsets[d_half:]        # second half [64:128]
    q1 = x[:d_half]
    q2 = x[d_half:]

    # Load cos and sin for this s
    c = tl.load(C_ptr + s * stride_c0 + tl.arange(0, d_half) * stride_c1, mask=tl.arange(0, d_half) < d_half, other=0.0)
    s_rot = tl.load(S_ptr + s * stride_s0 + tl.arange(0, d_half) * stride_s1, mask=tl.arange(0, d_half) < d_half, other=0.0)

    # Rotate second half: new_q2 = -q2 * s + q1 * c ; new_q1 = q1 * c + q2 * s
    new_q2 = -q2 * s_rot + q1 * c
    new_q1 = q1 * c + q2 * s_rot

    # Concatenate
    y = tl.zeros((D,), dtype=tl.float32)
    y[:d_half] = new_q1
    y[d_half:] = new_q2

    tl.store(
        Y_ptr + b * stride_yb + h * stride_yh + s * stride_ys + d_offsets * stride_yd,
        y,
        mask=d_offsets < D
    )

# The original attention and output projection (F.linear + softmax + F.linear) remain in PyTorch
# to ensure correctness and performance. We will only use Triton for Q,K,V,O, RMSNorm, and rotation.


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()
        # constants from the original code (fixed for this demo)
        self.num_attention_heads = 96
        self.num_key_value_heads = 8
        self.head_dim = 128
        self.num_key_value_groups = 12
        self.scaling = 1.0 / (self.head_dim ** 0.5)

    def forward(self, hidden_states, q_proj_weight, q_proj_bias, k_proj_weight, k_proj_bias, v_proj_weight, v_proj_bias, o_proj_weight, q_norm_weight, k_norm_weight, cos, sin, rms_norm_eps):
        B, S, _ = hidden_states.shape
        D_in = self.num_attention_heads * self.head_dim
        device = hidden_states.device

        # Ensure fp32 for computation
        hidden_states_fp32 = hidden_states.to(torch.float32)

        # 1) Dense projections via Triton: Q, K, V
        Q = torch.empty((B, S, self.head_dim), device=device, dtype=torch.float32)
        K = torch.empty((B, S, self.head_dim), device=device, dtype=torch.float32)
        V = torch.empty((B, S, self.head_dim), device=device, dtype=torch.float32)

        # Launch one program per (b, s)
        grid_q = (B * S,)
        linear_kernel[grid_q](
            hidden_states_fp32, q_proj_weight.to(torch.float32), q_proj_bias if q_proj_bias is not None else torch.empty(0, device=device, dtype=torch.float32),
            Q,
            B=B, S=S, D_in=D_in, D_out=self.head_dim,
            stride_xb=hidden_states_fp32.stride(0), stride_xs=hidden_states_fp32.stride(1), stride_xd=hidden_states_fp32.stride(2),
            stride_w0=q_proj_weight.stride(0), stride_w1=q_proj_weight.stride(1),
            stride_yb=Q.stride(0), stride_ys=Q.stride(1), stride_yd=Q.stride(2),
            BLOCK_K=64,
        )

        grid_k = (B * S,)
        linear_kernel[grid_k](
            hidden_states_fp32, k_proj_weight.to(torch.float32), k_proj_bias if k_proj_bias is not None else torch.empty(0, device=device, dtype=torch.float32),
            K,
            B=B, S=S, D_in=D_in, D_out=self.head_dim,
            stride_xb=hidden_states_fp32.stride(0), stride_xs=hidden_states_fp32.stride(1), stride_xd=hidden_states_fp32.stride(2),
            stride_w0=k_proj_weight.stride(0), stride_w1=k_proj_weight.stride(1),
            stride_yb=K.stride(0), stride_ys=K.stride(1), stride_yd=K.stride(2),
            BLOCK_K=64,
        )

        grid_v = (B * S,)
        linear_kernel[grid_v](
            hidden_states_fp32, v_proj_weight.to(torch.float32), v_proj_bias if v_proj_bias is not None else torch.empty(0, device=device, dtype=torch.float32),
            V,
            B=B, S=S, D_in=D_in, D_out=self.head_dim,
            stride_xb=hidden_states_fp32.stride(0), stride_xs=hidden_states_fp32.stride(1), stride_xd=hidden_states_fp32.stride(2),
            stride_w0=v_proj_weight.stride(0), stride_w1=v_proj_weight.stride(1),
            stride_yb=V.stride(0), stride_ys=V.stride(1), stride_yd=V.stride(2),
            BLOCK_K=64,
        )

        # 2) RMSNorm for Q and K using Triton: elementwise per (b,h,s, :)
        # Reshape to [B, H, S, D] where H = num_attention_heads
        Q4 = Q.view(B, self.num_attention_heads, S, self.head_dim).contiguous()
        K4 = K.view(B, self.num_key_value_heads, S, self.head_dim).contiguous()

        # Allocate normalized tensors
        Q_norm = torch.empty_like(Q4)
        K_norm = torch.empty_like(K4)

        # Launch 3D grid over (B, H, S)
        grid_norm_q = (B, self.num_attention_heads, S)
        rmsnorm_kernel[grid_norm_q](
            Q4, q_norm_weight.to(torch.float32), Q_norm,
            B=B, H=self.num_attention_heads, S=S, D=self.head_dim,
            stride_xb=Q4.stride(0), stride_xh=Q4.stride(1), stride_xs=Q4.stride(2), stride_xd=Q4.stride(3),
            stride_yb=Q_norm.stride(0), stride_yh=Q_norm.stride(1), stride_ys=Q_norm.stride(2), stride_yd=Q_norm.stride(3),
            eps=rms_norm_eps,
        )

        grid_norm_k = (B, self.num_key_value_heads, S)
        rmsnorm_kernel[grid_norm_k](
            K4, k_norm_weight.to(torch.float32), K_norm,
            B=B, H=self.num_key_value_heads, S=S, D=self.head_dim,
            stride_xb=K4.stride(0), stride_xh=K4.stride(1), stride_xs=K4.stride(2), stride_xd=K4.stride(3),
            stride_yb=K_norm.stride(0), stride_yh=K_norm.stride(1), stride_ys=K_norm.stride(2), stride_yd=K_norm.stride(3),
            eps=rms_norm_eps,
        )

        # 3) Rotate (RoPE) Q and K via Triton
        # cos and sin are [S, D/2] and are provided. We flatten them for pointer arithmetic in kernel.
        Q_rot = torch.empty_like(Q_norm)
        K_rot = torch.empty_like(K_norm)

        grid_rot_q = (B, self.num_attention_heads, S)
        rotate_half_kernel[grid_rot_q](
            Q_norm, cos.to(torch.float32), sin.to(torch.float32),
            Q_rot,
            B=B, H=self.num_attention_heads, S=S, D=self.head_dim,
            stride_xb=Q_norm.stride(0), stride_xh=Q_norm.stride(1), stride_xs=Q_norm.stride(2), stride_xd=Q_norm.stride(3),
            stride_yb=Q_rot.stride(0), stride_yh=Q_rot.stride(1), stride_ys=Q_rot.stride(2), stride_yd=Q_rot.stride(3),
            stride_c0=cos.stride(0), stride_c1=cos.stride(1),
            stride_s0=sin.stride(0), stride_s1=sin.stride(1),
        )

        grid_rot_k = (B, self.num_key_value_heads, S)
        rotate_half_kernel[grid_rot_k](
            K_norm, cos.to(torch.float32), sin.to(torch.float32),
            K_rot,
            B=B, H=self.num_key_value_heads, S=S, D=self.head_dim,
            stride_xb=K_norm.stride(0), stride_xh=K_norm.stride(1), stride_xs=K_norm.stride(2), stride_xd=K_norm.stride(3),
            stride_yb=K_rot.stride(0), stride_yh=K_rot.stride(1), stride_ys=K_rot.stride(2), stride_yd=K_rot.stride(3),
            stride_c0=cos.stride(0), stride_c1=cos.stride(1),
            stride_s0=sin.stride(0), stride_s1=sin.stride(1),
        )

        # 4) Grouped Query Attention (GQA): repeat KV heads to match Q heads
        # Original code: key_states = key_states[:, :, None, :, :].expand(...) then reshape.
        # We mimic that by keeping K_rot as [B, Hk, S, D] with Hk=8 and H=96, then expand; but since we want values to line up with attention, we also need V.
        # V has shape [B, Hk, S, D]. We need [B, H, S, D] where H=96. Mapping: for each attention head h, take the corresponding kv_head = (h // num_groups) * (Hk // num_groups).
        # We’ll construct V_rep accordingly (expand is cheap as a view), then compute attention in PyTorch.
        # Compute which KV head each attention head uses
        Hk = self.num_key_value_heads
        num_groups = self.num_key_value_groups
        H = self.num_attention_heads
        D = self.head_dim
        kv_per_group = Hk // num_groups  # 8 // 12 -> 8 (matching), so each group maps to one KV head.

        # For each h in [0..H), its kv_head = (h // num_groups) * kv_per_group
        # We’ll build a list of KV tensors by selecting from original Q_rot, K_rot, V; but to keep it simple, we can construct a dict mapping h -> kv head and expand views.
        # However, since Triton kernels are not allowed for this step, we do it in PyTorch as views (expand).
        # First, expand Q_rot, K_rot to [B, H, S, D]
        Q_rot_rep = Q_rot[:, :, None, :].expand(B, H, S, D).contiguous()
        K_rot_rep = K_rot[:, :, None, :].expand(B, H, S, D).contiguous()  # expand doesn’t copy; it’s a view for S and D. We need contiguous for matmul in PyTorch later.
        # We need V as [B, H, S, D]; V currently is [B, Hk, S, D]. We can select the appropriate kv head for each h.
        # Since Hk == H * (Hk / H), mapping is straightforward: h uses kv_head = (h // num_groups) * (Hk // num_groups).
        # But we also don’t have num_groups in V; we can compute it consistently and use original K’s kv head mapping.
        # In this code, original H=96, Hk=8, num_groups=12 => each h uses kv head index h // num_groups.
        # Note: V was computed from v_proj_weight, but original mapping uses the same KV heads as K. We can directly expand V as we do for K since we don’t have per-h selection.
        # To be exact, we should gather V per h. Since Triton is not used here for gather, we’ll do a correct gather in PyTorch:
        # Build index per (b,h): kv_idx[h] = (h // num_groups) * (Hk // num_groups)
        kv_idx = (torch.arange(H, device=device) // num_groups) * (Hk // num_groups)  # [H], e.g., [0,1,2,...,7,0,1,...]
        V_sel = V.index_select(1, kv_idx)  # [B, H, S, D]
        V_rep = V_sel  # already [B, H, S, D]

        # 5) Compute attention in PyTorch: scores = Q_rot_rep @ K_rot_rep^T * scaling, apply causal mask, softmax, then attn_output = scores @ V_rep
        # Shapes: Q_rot_rep: [B, H, S, D], K_rot_rep: [B, H, S, D] (but we want actual KV values per h). Actually, K_rot_rep was expanded from 8 KV heads, which is incorrect for softmax.
        # We must recompute K per attention head by selecting from the original K_norm[K_rot] before rotation. We cannot do Triton gather here, so we gather K per h:
        # K_selected = torch.stack([K_norm[:, i, :, :], i in kv_idx], dim=1) -> [B, H, S, D]
        K_sel_list = []
        for h in range(H):
            kh = kv_idx[h].item()
            K_sel_list.append(K_norm[:, kh, :, :])  # [B, S, D]
        K_sel = torch.stack(K_sel_list, dim=1)  # [B, H, S, D]
        K_sel_rot = torch.empty_like(K_sel)
        grid_rot_k_sel = (B, H, S)
        rotate_half_kernel[grid_rot_k_sel](
            K_sel, cos.to(torch.float32), sin.to(torch.float32),
            K_sel_rot,
            B=B, H=H, S=S, D=D,
            stride_xb=K_sel.stride(0), stride_xh=K_sel.stride(1), stride_xs=K_sel.stride(2), stride_xd=K_sel.stride(3),
            stride_yb=K_sel_rot.stride(0), stride_yh=K_sel_rot.stride(1), stride_ys=K_sel_rot.stride(2), stride_yd=K_sel_rot.stride(3),
            stride_c0=cos.stride(0), stride_c1=cos.stride(1),
            stride_s0=sin.stride(0), stride_s1=sin.stride(1),
        )

        # V_sel already constructed by gathering V from kv_idx
        # Compute scores = Q @ K^T per (b,h) over S,S
        attn_scores = torch.matmul(Q_rot_rep, K_sel_rot.transpose(2, 3))  # [B, H, S, S] * scaling
        attn_scores = attn_scores * self.scaling

        # Causal mask: upper-triangular with diagonal=1
        causal_mask = torch.triu(
            torch.full((S, S), float('-inf'), device=device, dtype=torch.float32),
            diagonal=1
        ).unsqueeze(0).unsqueeze(1)  # [1, 1, S, S]
        attn_scores = attn_scores + causal_mask

        # Softmax over last dim (sequence length)
        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32)  # [B, H, S, S]

        # Output = attn_weights @ V_sel
        attn_output = torch.matmul(attn_weights, V_sel)  # [B, H, S, D]

        # 6) Output projection via PyTorch F.linear (no bias)
        output = F.linear(attn_output.reshape(B, S, H * D), o_proj_weight.to(torch.float32), None)
        output = output.reshape(B, S, H * D)

        return output


def run(*args):
    return ModelNew()(*args)
