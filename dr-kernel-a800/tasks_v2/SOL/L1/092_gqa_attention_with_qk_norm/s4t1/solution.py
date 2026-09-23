import torch
import triton
import triton.language as tl

# Kernel 1: Linear with bias: X @ W.T + b
# X: [B, S, H_in], W: [H_out, H_in], b: [H_out] -> Out: [B, S, H_out]
@triton.jit
def linear_bias_kernel(
    X_ptr, W_ptr, B_ptr, Out_ptr,
    Bsz: tl.constexpr, Ssz: tl.constexpr, H_in: tl.constexpr, H_out: tl.constexpr,
    stride_x_b, stride_x_s, stride_x_h,
    stride_w_o, stride_w_i,
    stride_out_b, stride_out_s, stride_out_h,
    BLOCK_IN: tl.constexpr, BLOCK_OUT: tl.constexpr,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    o = tl.program_id(2)

    acc = tl.zeros((), dtype=tl.float32)

    for i in range(0, H_out, BLOCK_OUT):
        o_offsets = i + tl.arange(0, BLOCK_OUT)
        mask_o = o_offsets < H_out

        # load b vector for output channel o_offsets
        b_vals = tl.load(B_ptr + o_offsets, mask=mask_o, other=0.0).to(tl.float32)

        acc = tl.zeros([BLOCK_OUT], dtype=tl.float32)

        for j in range(0, H_in, BLOCK_IN):
            i_offsets = j + tl.arange(0, BLOCK_IN)
            mask_i = i_offsets < H_in

            # X[b, s, i_offsets]
            x_ptrs = X_ptr + b * stride_x_b + s * stride_x_s + i_offsets * stride_x_h
            x_vals = tl.load(x_ptrs, mask=mask_i, other=0.0).to(tl.float32)

            # W[o_offsets, i_offsets] -> shape [BLOCK_OUT, BLOCK_IN]
            w_ptrs = W_ptr + o_offsets[:, None] * stride_w_o + i_offsets[None, :] * stride_w_i
            w_vals = tl.load(w_ptrs, mask=mask_o[:, None] & mask_i[None, :], other=0.0).to(tl.float32)

            acc += tl.sum(w_vals * x_vals[None, :], axis=1)

        out_ptrs = Out_ptr + b * stride_out_b + s * stride_out_s + o_offsets * stride_out_h
        tl.store(out_ptrs, acc + b_vals, mask=mask_o)


# Kernel 2: RMSNorm: Y = X * rsqrt(mean(X^2) + eps), then scale by weight
# X: [B, S, H], weight: [H] -> Y: [B, S, H]
@triton.jit
def rmsnorm_kernel(
    X_ptr, W_ptr, Y_ptr,
    Bsz: tl.constexpr, Ssz: tl.constexpr, H: tl.constexpr,
    stride_x_b, stride_x_s, stride_x_h,
    stride_w_h,
    stride_y_b, stride_y_s, stride_y_h,
    eps: tl.constexpr,  # float
    BLOCK_H: tl.constexpr,
):
    b = tl.program_id(0)
    s = tl.program_id(1)

    sum_sq = tl.zeros((), dtype=tl.float32)
    for h in range(0, H, BLOCK_H):
        h_offsets = h + tl.arange(0, BLOCK_H)
        mask_h = h_offsets < H
        x = tl.load(X_ptr + b * stride_x_b + s * stride_x_s + h_offsets * stride_x_h, mask=mask_h, other=0.0).to(tl.float32)
        sum_sq += tl.sum(x * x, axis=0)

    mean = sum_sq / H
    inv_rms = 1.0 / tl.sqrt(mean + eps)

    for h in range(0, H, BLOCK_H):
        h_offsets = h + tl.arange(0, BLOCK_H)
        mask_h = h_offsets < H
        x = tl.load(X_ptr + b * stride_x_b + s * stride_x_s + h_offsets * stride_x_h, mask=mask_h, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + h_offsets * stride_w_h, mask=mask_h, other=1.0).to(tl.float32)
        y = x * inv_rms
        y = y * w
        tl.store(Y_ptr + b * stride_y_b + s * stride_y_s + h_offsets * stride_y_h, y, mask=mask_h)


# Kernel 3: Softmax with mask over last dimension (S dimension), output [B, H, S, S]
# AttnScores: [B, H, S, S], Mask: same shape, Out: same shape
@triton.jit
def softmax_mask_kernel(
    Scores_ptr, Mask_ptr, Out_ptr,
    Bsz: tl.constexpr, H: tl.constexpr, Ssz: tl.constexpr,
    stride_scores_b, stride_scores_h, stride_scores_s1, stride_scores_s2,
    stride_mask_b, stride_mask_h, stride_mask_s1, stride_mask_s2,
    stride_out_b, stride_out_h, stride_out_s1, stride_out_s2,
    BLOCK_S: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s_row = tl.program_id(2)

    # Load row across S dimension
    col = tl.arange(0, BLOCK_S)
    row_start = b * stride_scores_b + h * stride_scores_h + s_row * stride_scores_s1
    scores = tl.load(Scores_ptr + row_start + col * stride_scores_s2, mask=col < Ssz, other=-float('inf')).to(tl.float32)

    # Load mask
    mask_row_start = b * stride_mask_b + h * stride_mask_h + s_row * stride_mask_s1
    mask = tl.load(Mask_ptr + mask_row_start + col * stride_mask_s2, mask=col < Ssz, other=0.0).to(tl.float32)

    scores = scores + mask

    m = tl.max(scores, axis=0)
    scores = scores - m
    exp_scores = tl.exp(scores)
    sum_exp = tl.sum(exp_scores, axis=0)
    softmax = exp_scores / sum_exp

    out_row_start = b * stride_out_b + h * stride_out_h + s_row * stride_out_s1
    tl.store(Out_ptr + out_row_start + col * stride_out_s2, softmax, mask=col < Ssz)


# Kernel 4: Rotate half for Q and K: q1, q2 = q[..., :64], q[..., 64:], q_rot = [q1, -q2]
# X: [B, S, H], cos: [1, H], sin: [1, H] -> Y: [B, S, H]
@triton.jit
def rotate_half_kernel(
    X_ptr, Cos_ptr, Sin_ptr, Y_ptr,
    Bsz: tl.constexpr, Ssz: tl.constexpr, H: tl.constexpr,
    stride_x_b, stride_x_s, stride_x_h,
    stride_cos_h, stride_sin_h,
    stride_y_b, stride_y_s, stride_y_h,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h = tl.program_id(2)

    h_offsets = h + tl.arange(0, 128)
    mask_h = h_offsets < H

    x = tl.load(X_ptr + b * stride_x_b + s * stride_x_s + h_offsets * stride_x_h, mask=mask_h, other=0.0).to(tl.float32)

    half = head_dim // 2  # 64
    q1 = x[:half]
    q2 = x[half:]

    cos_vals = tl.load(Cos_ptr + h_offsets * stride_cos_h, mask=mask_h, other=1.0).to(tl.float32)
    sin_vals = tl.load(Sin_ptr + h_offsets * stride_sin_h, mask=mask_h, other=1.0).to(tl.float32)

    q_rot_half = tl.concatenate([-q2, q1], axis=0)
    y = x * cos_vals + q_rot_half * sin_vals

    tl.store(Y_ptr + b * stride_y_b + s * stride_y_s + h_offsets * stride_y_h, y, mask=mask_h)


# Kernel 5: AttnScores = (Q @ K^T) * scaling
# Q: [B, S, H], K^T: [S, H] -> AttnScores: [B, S, H]
@triton.jit
def matmul_qk_kernel(
    Q_ptr, KT_ptr, Out_ptr,
    Bsz: tl.constexpr, Ssz: tl.constexpr, H: tl.constexpr,
    stride_q_b, stride_q_s, stride_q_h,
    stride_kt_s, stride_kt_h,
    stride_out_b, stride_out_s, stride_out_h,
    scaling: tl.constexpr,  # float
    BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)  # over B*S
    pid_n = tl.program_id(1)  # over output channels H
    pid_k_blk = tl.program_id(2)  # over K blocks

    b = pid_m // Ssz
    s = pid_m % Ssz

    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = n_offsets < H

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    for k_start in range(0, BLOCK_K):
        k = k_start + pid_k_blk * BLOCK_K

        # load Q[b, s, k]
        q_ptr = Q_ptr + b * stride_q_b + s * stride_q_s + k * stride_q_h
        q_val = tl.load(q_ptr).to(tl.float32)

        # load KT[k, n_offsets]
        kt_ptrs = KT_ptr + k * stride_kt_s + n_offsets * stride_kt_h
        kt_vals = tl.load(kt_ptrs, mask=mask_n, other=0.0).to(tl.float32)

        acc += q_val * kt_vals

    acc *= scaling
    out_ptrs = Out_ptr + b * stride_out_b + s * stride_out_s + n_offsets * stride_out_h
    tl.store(out_ptrs, acc, mask=mask_n)


# Kernel 6: Attn output: Out = AttnWeights @ V
# AttnWeights: [B, S, H], V: [S, H] -> Output: [B, S, H]
@triton.jit
def matmul_attn_kernel(
    W_ptr, V_ptr, Out_ptr,
    Bsz: tl.constexpr, Ssz: tl.constexpr, H: tl.constexpr,
    stride_w_b, stride_w_s, stride_w_h,
    stride_v_s, stride_v_h,
    stride_out_b, stride_out_s, stride_out_h,
    BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)  # over B*S
    pid_n = tl.program_id(1)  # over output channels H
    pid_k_blk = tl.program_id(2)  # over K blocks

    b = pid_m // Ssz
    s = pid_m % Ssz

    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = n_offsets < H

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    for k_start in range(0, BLOCK_K):
        k = k_start + pid_k_blk * BLOCK_K

        # load W[b, s, k]
        w_ptr = W_ptr + b * stride_w_b + s * stride_w_s + k * stride_w_h
        w_val = tl.load(w_ptr).to(tl.float32)

        # load V[k, n_offsets]
        v_ptrs = V_ptr + k * stride_v_s + n_offsets * stride_v_h
        v_vals = tl.load(v_ptrs, mask=mask_n, other=0.0).to(tl.float32)

        acc += w_val * v_vals

    out_ptrs = Out_ptr + b * stride_out_b + s * stride_out_s + n_offsets * stride_out_h
    tl.store(out_ptrs, acc, mask=mask_n)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

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
        device = hidden_states.device
        assert device.type == "cuda", "This Triton version requires CUDA device."

        Bsz, Ssz, H_in = hidden_states.shape
        head_dim = 128
        num_attention_heads = 96
        num_key_value_heads = 8
        num_key_value_groups = 12
        scaling = 1.0 / (head_dim ** 0.5)

        # 1) Linear projections for Q, K, V
        Q = torch.empty((Bsz, Ssz, head_dim), device=device, dtype=torch.float32)
        linear_bias_kernel[(Bsz, Ssz, head_dim)](
            hidden_states, q_proj_weight, q_proj_bias, Q,
            Bsz, Ssz, H_in, head_dim,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            q_proj_weight.stride(0), q_proj_weight.stride(1),
            Q.stride(0), Q.stride(1), Q.stride(2),
            BLOCK_IN=128, BLOCK_OUT=64,
            num_warps=4,
        )

        K = torch.empty((Bsz, Ssz, head_dim), device=device, dtype=torch.float32)
        linear_bias_kernel[(Bsz, Ssz, head_dim)](
            hidden_states, k_proj_weight, k_proj_bias, K,
            Bsz, Ssz, H_in, head_dim,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            k_proj_weight.stride(0), k_proj_weight.stride(1),
            K.stride(0), K.stride(1), K.stride(2),
            BLOCK_IN=128, BLOCK_OUT=64,
            num_warps=4,
        )

        V = torch.empty((Bsz, Ssz, head_dim), device=device, dtype=torch.float32)
        linear_bias_kernel[(Bsz, Ssz, head_dim)](
            hidden_states, v_proj_weight, v_proj_bias, V,
            Bsz, Ssz, H_in, head_dim,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            v_proj_weight.stride(0), v_proj_weight.stride(1),
            V.stride(0), V.stride(1), V.stride(2),
            BLOCK_IN=128, BLOCK_OUT=64,
            num_warps=4,
        )

        # 2) Reshape to [B, S, num_heads, head_dim] and transpose to [B, num_heads, S, head_dim]
        Q_heads = Q.view(Bsz, Ssz, num_attention_heads, head_dim).transpose(1, 2)  # [B, 96, S, 128]
        K_heads = K.view(Bsz, Ssz, num_key_value_heads, head_dim).transpose(1, 2)  # [B, 8, S, 128]
        V_heads = V.view(Bsz, Ssz, num_key_value_heads, head_dim).transpose(1, 2)  # [B, 8, S, 128]

        # 3) RMSNorm on Q and K
        Q_norm = torch.empty_like(Q_heads, device=device, dtype=torch.float32)
        rmsnorm_kernel[(Bsz, Ssz)](
            Q_heads, q_norm_weight, Q_norm,
            Bsz, Ssz, head_dim,
            Q_heads.stride(0), Q_heads.stride(1), Q_heads.stride(2),
            q_norm_weight.stride(0),
            Q_norm.stride(0), Q_norm.stride(1), Q_norm.stride(2),
            eps=rms_norm_eps,
            BLOCK_H=128,
            num_warps=4,
        )

        K_norm = torch.empty_like(K_heads, device=device, dtype=torch.float32)
        rmsnorm_kernel[(Bsz, Ssz)](
            K_heads, k_norm_weight, K_norm,
            Bsz, Ssz, head_dim,
            K_heads.stride(0), K_heads.stride(1), K_heads.stride(2),
            k_norm_weight.stride(0),
            K_norm.stride(0), K_norm.stride(1), K_norm.stride(2),
            eps=rms_norm_eps,
            BLOCK_H=128,
            num_warps=4,
        )

        # 4) Rotate half for Q and K
        Q_rot = torch.empty_like(Q_norm, device=device, dtype=torch.float32)
        rotate_half_kernel[(Bsz, Ssz, head_dim)](
            Q_norm, cos, sin, Q_rot,
            Bsz, Ssz, head_dim,
            Q_norm.stride(0), Q_norm.stride(1), Q_norm.stride(2),
            cos.stride(0), sin.stride(0),
            Q_rot.stride(0), Q_rot.stride(1), Q_rot.stride(2),
            num_warps=4,
        )

        K_rot = torch.empty_like(K_norm, device=device, dtype=torch.float32)
        rotate_half_kernel[(Bsz, Ssz, head_dim)](
            K_norm, cos, sin, K_rot,
            Bsz, Ssz, head_dim,
            K_norm.stride(0), K_norm.stride(1), K_norm.stride(2),
            cos.stride(0), sin.stride(0),
            K_rot.stride(0), K_rot.stride(1), K_rot.stride(2),
            num_warps=4,
        )

        # 5) Grouped-Query Attention: expand KV to 96 heads and reshape
        # Note: K/V are [B, 8, S, 128], expand num_key_value_groups=12 along that dimension
        # Grouped mapping: group_id = head % num_key_value_groups, map to key_idx = group_id * (num_attention_heads // num_key_value_heads)
        # Here num_attention_heads // num_key_value_heads == 12, which matches num_key_value_groups.
        K_rot_exp = K_rot[:, :, None, :, :].expand(Bsz, num_key_value_heads, num_key_value_groups, Ssz, head_dim).reshape(Bsz, num_attention_heads, Ssz, head_dim)
        V_heads_exp = V_heads[:, :, None, :, :].expand(Bsz, num_key_value_heads, num_key_value_groups, Ssz, head_dim).reshape(Bsz, num_attention_heads, Ssz, head_dim)

        # 6) Compute attention scores: AttnScores = Q_rot @ K_rot^T * scaling
        # We need KT of shape [S, H] for each (b, s) row
        AttnScores = torch.empty((Bsz, Ssz, head_dim), device=device, dtype=torch.float32)
        # For matmul, we pass K_rot as [B, S, H], and treat KT as transposed view [S, H] by using strides accordingly.
        # We'll flatten B and S into M=B*S and use kernel over (M, H, K_blocks).
        M = Bsz * Ssz
        grid = (M, head_dim, 1)  # K blocks = 1 since H=Sz=seq_len, but we set BLOCK_K to 64 or 128 and loop
        matmul_qk_kernel[grid](
            Q_rot, K_rot, AttnScores,
            Bsz, Ssz, head_dim,
            Q_rot.stride(0), Q_rot.stride(1), Q_rot.stride(2),
            K_rot.stride(1), K_rot.stride(2),
            AttnScores.stride(0), AttnScores.stride(1), AttnScores.stride(2),
            scaling,
            BLOCK_K=64, BLOCK_M=1, BLOCK_N=128,
            num_warps=4,
        )

        # 7) Apply causal mask: upper-triangular with diagonal=1
        # Build mask [Ssz, Ssz] on device and broadcast to [B, num_attention_heads, Ssz, Ssz]
        causal_mask = torch.triu(
            torch.full((Ssz, Ssz), -float('inf'), device=device, dtype=torch.float32),
            diagonal=1
        )
        Mask = causal_mask.unsqueeze(0).unsqueeze(1).expand(Bsz, num_attention_heads, Ssz, Ssz).contiguous()

        # 8) Softmax over last dim (S) for each (b, h) row
        SoftmaxOut = torch.empty_like(AttnScores, device=device, dtype=torch.float32)
        softmax_mask_kernel[(Bsz, num_attention_heads, Ssz)](
            AttnScores, Mask, SoftmaxOut,
            Bsz, num_attention_heads, Ssz,
            AttnScores.stride(0), AttnScores.stride(1), AttnScores.stride(2), AttnScores.stride(3),
            Mask.stride(0), Mask.stride(1), Mask.stride(2), Mask.stride(3),
            SoftmaxOut.stride(0), SoftmaxOut.stride(1), SoftmaxOut.stride(2), SoftmaxOut.stride(3),
            BLOCK_S=128,
            num_warps=4,
        )

        # 9) Compute attention output: Out = SoftmaxOut @ V
        Out = torch.empty((Bsz, Ssz, head_dim), device=device, dtype=torch.float32)
        # Here, SoftmaxOut [B, S, H], V [B, S, H] (since we reshaped expanded V to match num_attention_heads). However, original uses [S, H] per head, we need to align properly.
        # Note: V_heads_exp is [B, 96, S, 128] after group expansion. For matmul_attn, we need V as [S, H] per (b). We can flatten B*S for each head.
        # Use V_heads_exp as [B*S, H] by flattening appropriate strides. More cleanly, we can take V per (b, s) for each head.
        # We need V per (b, s, head) which is already available as expanded. To feed matmul_attn kernel, we need V in [S, H] per (b). We can get V for each (b, s) by iterating heads, but simpler: use V directly.
        # Reuse V as [B, S, H]: we have V_heads_exp, but for output projection we only need actual V from original projection (not expanded). We need to compute V for each head's sequence independently. The simplest is to use V (original [B, S, H]) for output projection, but here we need per-head V corresponding to query heads. Since GQA uses same V for all query heads, we can use expanded V for each head's softmax row. In code, we have V_heads_exp already.
        # However, to keep correctness, we'll use original V (not expanded), as output projection does not depend on group mapping.
        V_for_output = V  # [B, S, H]
        matmul_attn_kernel[(M, head_dim, 1)](
            SoftmaxOut, V_for_output,
            Out,
            Bsz, Ssz, head_dim,
            SoftmaxOut.stride(0), SoftmaxOut.stride(1), SoftmaxOut.stride(2),
            V_for_output.stride(0), V_for_output.stride(1),
            Out.stride(0), Out.stride(1), Out.stride(2),
            BLOCK_K=64, BLOCK_M=1, BLOCK_N=128,
            num_warps=4,
        )

        # 10) Transpose and reshape: [B, S, H] -> [B, S, num_attention_heads*head_dim]
        Out = Out.transpose(1, 2).contiguous()  # [B, 128, S]
        Out = Out.reshape(Bsz, Ssz, num_attention_heads * head_dim)  # [B, S, 12288]

        # 11) Output projection (no bias)
        Output = torch.empty((Bsz, Ssz, o_proj_weight.shape[0]), device=device, dtype=torch.float32)
        # o_proj_weight shape is [H_out, H_in] where H_out may not be head_dim. We must compute it correctly.
        # From original: output = linear(attn_output, o_proj_weight, None). attn_output shape is [B, S, num_attention_heads * head_dim] i.e., [B, S, 12288].
        # So H_in = 12288, H_out = o_proj_weight.shape[0].
        H_in_out = num_attention_heads * head_dim
        linear_bias_kernel[(Bsz, Ssz, o_proj_weight.shape[0])](
            Out, o_proj_weight, torch.zeros((o_proj_weight.shape[0],), device=device, dtype=torch.float32),
            Output,
            Bsz, Ssz, H_in_out, o_proj_weight.shape[0],
            Out.stride(0), Out.stride(1), Out.stride(2),
            o_proj_weight.stride(0), o_proj_weight.stride(1),
            Output.stride(0), Output.stride(1), Output.stride(2),
            BLOCK_IN=128, BLOCK_OUT=64,
            num_warps=4,
        )

        return Output


def run(*args):
    return ModelNew()(*args)
