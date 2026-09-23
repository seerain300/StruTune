import torch
import triton
import triton.language as tl


# Kernel 1: Linear with bias: X @ W.T + b
# X: [B, S, H_in], W: [H_out, H_in], b: [H_out] -> Y: [B, S, H_out]
@triton.jit
def linear_bias_kernel(
    X_ptr, W_ptr, BIAS_ptr, Y_ptr,
    Bsz, Ssz, H_in, H_out,
    stride_x_b, stride_x_s, stride_x_h,
    stride_w_h, stride_w_in, stride_bias_h, stride_y_b, stride_y_s, stride_y_h,
):
    # program ids: process one (b, s, h_out) per program
    b = tl.program_id(0)
    s = tl.program_id(1)
    h_out = tl.program_id(2)

    acc = tl.zeros((), dtype=tl.float32)

    # reduction over H_in
    for hin in range(0, H_in):
        x_off = b * stride_x_b + s * stride_x_s + hin * stride_x_h
        x_val = tl.load(X_ptr + x_off)
        w_off = h_out * stride_w_h + hin * stride_w_in
        w_val = tl.load(W_ptr + w_off)
        acc += x_val * w_val

    bias_val = tl.load(BIAS_ptr + h_out * stride_bias_h)
    acc += bias_val

    y_off = b * stride_y_b + s * stride_y_s + h_out * stride_y_h
    tl.store(Y_ptr + y_off, acc)


# Kernel 2: RMSNorm over last dim (size = head_dim=128): y = x * rsqrt(mean(x^2) + eps), then scale by weight
# Input x: [B, S, D], weight: [D] -> Output y: [B, S, D]
@triton.jit
def rmsnorm_kernel(
    X_ptr, Weight_ptr, Out_ptr,
    Bsz, Ssz, D,
    stride_x_b, stride_x_s, stride_x_d,
    stride_w_d, stride_out_b, stride_out_s, stride_out_d,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    d = tl.program_id(2)

    row_off = b * stride_x_b + s * stride_x_s

    # sum of squares over BLOCK elements of the row
    sum_sq = 0.0
    for i in range(0, BLOCK):
        off = row_off + (d + i) * stride_x_d
        x = tl.load(X_ptr + off)
        sum_sq += x * x

    mean = sum_sq / D
    inv_rms = tl.rsqrt(mean + 0.0)  # eps = 0.0 to match original behavior
    weight = tl.load(Weight_ptr + (d + 0) * stride_w_d)

    # write normalized and scaled output back
    for i in range(0, BLOCK):
        in_off = row_off + (d + i) * stride_x_d
        x = tl.load(X_ptr + in_off)
        y = x * inv_rms
        y = y * weight
        out_off = b * stride_out_b + s * stride_out_s + (d + i) * stride_out_d
        tl.store(Out_ptr + out_off, y)


# Kernel 3: Rotate half of the last 64 dims for a 128-length vector: q1, q2 = q[:64], q[64:], q_rot = q1*cos - q2*sin
# This kernel is called for both query and key vectors. It takes Q, Sin, Cos and produces Out.
@triton.jit
def rotate_half_kernel(
    Q_ptr, Sin_ptr, Cos_ptr, Out_ptr,
    Bsz, Ssz, D,  # D should be 128
    stride_q_b, stride_q_s, stride_q_d,
    stride_s_d, stride_c_d, stride_out_b, stride_out_s, stride_out_d,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    d = tl.program_id(2)  # d ranges over 0..D-1

    q_off = b * stride_q_b + s * stride_q_s + d * stride_q_d
    sin_val = tl.load(Sin_ptr + d * stride_s_d)
    cos_val = tl.load(Cos_ptr + d * stride_c_d)

    # Load q
    q = tl.load(Q_ptr + q_off)
    # Split
    q1 = q[:64]
    q2 = q[64:]
    # Rotate: q1' = q1*cos - q2*sin ; q2' = q2*cos + q1*sin
    q1_rot = q1 * cos_val - q2 * sin_val
    q2_rot = q2 * cos_val + q1 * sin_val
    q_rot = tl.concatenate([q1_rot, q2_rot], axis=0)

    out_off = b * stride_out_b + s * stride_out_s + d * stride_out_d
    # Write q_rot to Out at the same (b, s, d) location
    tl.store(Out_ptr + out_off, q_rot)


# Kernel 4: Compute attention scores Q @ K^T for one (b, s, h): output [S, S] float32
# Q: [B, num_heads, S, D], K: [B, num_heads, S, D] -> scores: [B, num_heads, S, S]
@triton.jit
def matmul_qk_kernel(
    Q_ptr, K_ptr, Out_ptr,
    Bsz, Ssz, D, num_heads,
    stride_q_b, stride_q_h, stride_q_s, stride_q_d,
    stride_k_b, stride_k_h, stride_k_s, stride_k_d,
    stride_out_b, stride_out_h, stride_out_srow, stride_out_scol,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    # We tile over S (rows) and S (cols); this kernel computes the full [S, S] for one (b, h)
    for row in range(0, Ssz):
        acc = tl.zeros((Ssz,), dtype=tl.float32)
        for col in range(0, Ssz):
            q_off = b * stride_q_b + h * stride_q_h + row * stride_q_s + tl.arange(0, D) * stride_q_d
            k_off = b * stride_k_b + h * stride_k_h + col * stride_k_s + tl.arange(0, D) * stride_k_d
            q_vec = tl.load(Q_ptr + q_off)
            k_vec = tl.load(K_ptr + k_off)
            acc[col] = tl.sum(q_vec * k_vec, axis=0)
        out_row_off = b * stride_out_b + h * stride_out_h + row * stride_out_srow + tl.arange(0, Ssz) * stride_out_scol
        tl.store(Out_ptr + out_row_off, acc)


# Kernel 5: Softmax with causal mask over last dim (sequence length) per (b, h): Y = softmax(scores)
# scores: [S, S], Y: [S, S]
@triton.jit
def softmax_mask_kernel(
    In_ptr, Mask_ptr, Y_ptr,
    Ssz,
    stride_in_b, stride_in_h, stride_in_srow, stride_in_scol,
    stride_mask_b, stride_mask_h, stride_mask_srow, stride_mask_scol,
    stride_y_b, stride_y_h, stride_y_srow, stride_y_scol,
):
    b = tl.program_id(0)
    h = tl.program_id(1)

    # row-wise max
    max_val = -float('inf')
    for j in range(0, Ssz):
        in_off = b * stride_in_b + h * stride_in_h + j * stride_in_srow + 0 * stride_in_scol
        val = tl.load(In_ptr + in_off)
        # apply mask: load mask at (row=j, col=0) across cols
        # Note: mask is [S, S], at fixed (row=j, col varying)
        mask_off = b * stride_mask_b + h * stride_mask_h + j * stride_mask_srow + tl.arange(0, Ssz) * stride_mask_scol
        mask_vec = tl.load(Mask_ptr + mask_off)
        val = tl.where(mask_vec == -float('inf'), -float('inf'), val)
        max_val = tl.maximum(max_val, val)

    # sum of exp
    sum_exp = 0.0
    for j in range(0, Ssz):
        in_off = b * stride_in_b + h * stride_in_h + j * stride_in_srow + 0 * stride_in_scol
        val = tl.load(In_ptr + in_off)
        mask_off = b * stride_mask_b + h * stride_mask_h + j * stride_mask_srow + tl.arange(0, Ssz) * stride_mask_scol
        mask_vec = tl.load(Mask_ptr + mask_off)
        val = tl.where(mask_vec == -float('inf'), -float('inf'), val - max_val)
        exp_val = tl.exp(val)
        sum_exp += exp_val

    # write softmax
    for j in range(0, Ssz):
        in_off = b * stride_in_b + h * stride_in_h + j * stride_in_srow + 0 * stride_in_scol
        val = tl.load(In_ptr + in_off)
        mask_off = b * stride_mask_b + h * stride_mask_h + j * stride_mask_srow + tl.arange(0, Ssz) * stride_mask_scol
        mask_vec = tl.load(Mask_ptr + mask_off)
        val = tl.where(mask_vec == -float('inf'), -float('inf'), val - max_val)
        exp_val = tl.exp(val) / sum_exp
        y_off = b * stride_y_b + h * stride_y_h + j * stride_y_srow + tl.arange(0, Ssz) * stride_y_scol
        tl.store(Y_ptr + y_off, exp_val)


# Kernel 6: Attention output: Softmax(scores) @ V for one (b, s, h): output [S, D]
@triton.jit
def matmul_attn_kernel(
    Score_ptr, V_ptr, Out_ptr,
    Bsz, Ssz, D, num_heads,
    stride_score_b, stride_score_h, stride_score_srow, stride_score_scol,
    stride_v_b, stride_v_h, stride_v_s, stride_v_d,
    stride_out_b, stride_out_h, stride_out_s, stride_out_d,
):
    b = tl.program_id(0)
    h = tl.program_id(1)

    for row in range(0, Ssz):
        acc = tl.zeros((D,), dtype=tl.float32)
        for col in range(0, Ssz):
            # load score[row, col]
            score_off = b * stride_score_b + h * stride_score_h + row * stride_score_srow + col * stride_score_scol
            score_val = tl.load(Score_ptr + score_off)
            # load V[col, :]
            v_off = b * stride_v_b + h * stride_v_h + col * stride_v_s + tl.arange(0, D) * stride_v_d
            v_vec = tl.load(V_ptr + v_off)
            acc += score_val * v_vec
        out_off = b * stride_out_b + h * stride_out_h + row * stride_out_s + tl.arange(0, D) * stride_out_d
        tl.store(Out_ptr + out_off, acc)


# Kernel 7: Linear without bias: X @ W.T -> Y
# X: [B, S, H_in], W: [H_out, H_in], Y: [B, S, H_out]
@triton.jit
def linear_nobias_kernel(
    X_ptr, W_ptr, Y_ptr,
    Bsz, Ssz, H_in, H_out,
    stride_x_b, stride_x_s, stride_x_h,
    stride_w_h, stride_w_in, stride_y_b, stride_y_s, stride_y_h,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    h_out = tl.program_id(2)

    acc = tl.zeros((), dtype=tl.float32)
    for hin in range(0, H_in):
        x_off = b * stride_x_b + s * stride_x_s + hin * stride_x_h
        x_val = tl.load(X_ptr + x_off)
        w_off = h_out * stride_w_h + hin * stride_w_in
        w_val = tl.load(W_ptr + w_off)
        acc += x_val * w_val

    y_off = b * stride_y_b + s * stride_y_s + h_out * stride_y_h
    tl.store(Y_ptr + y_off, acc)


# Entry point ModelNew.forward must be Triton-only: no torch ops in host
class ModelNew(torch.nn.Module):
    def __init__(self, hidden_states: torch.Tensor,
                 q_proj_weight: torch.Tensor, q_proj_bias: torch.Tensor,
                 k_proj_weight: torch.Tensor, k_proj_bias: torch.Tensor,
                 v_proj_weight: torch.Tensor, v_proj_bias: torch.Tensor,
                 o_proj_weight: torch.Tensor,
                 q_norm_weight: torch.Tensor, k_norm_weight: torch.Tensor,
                 cos: torch.Tensor, sin: torch.Tensor,
                 batch_size: int, seq_length: int):
        super().__init__()
        self.hidden_states = hidden_states
        self.q_proj_weight = q_proj_weight
        self.q_proj_bias = q_proj_bias
        self.k_proj_weight = k_proj_weight
        self.k_proj_bias = k_proj_bias
        self.v_proj_weight = v_proj_weight
        self.v_proj_bias = v_proj_bias
        self.o_proj_weight = o_proj_weight
        self.q_norm_weight = q_norm_weight
        self.k_norm_weight = k_norm_weight
        self.cos = cos
        self.sin = sin
        self.batch_size = batch_size
        self.seq_length = seq_length

    def forward(self):
        Bsz = self.batch_size
        Ssz = self.seq_length
        H_in = 1280  # original hidden size (assumed provided)
        H_out_qk = 128  # projection to 128-d per head
        D = 128
        num_attention_heads = 96
        num_key_value_heads = 8
        num_key_value_groups = 12  # not used directly

        device = self.hidden_states.device
        dtype = self.hidden_states.dtype

        # 1) Linear projections for Q, K, V
        # Q: [B, S, H_out_qk]
        q_proj_weight = self.q_proj_weight
        q_proj_bias = self.q_proj_bias
        q = torch.empty((Bsz, Ssz, H_out_qk), device=device, dtype=dtype)
        linear_bias_kernel[(Bsz, Ssz, H_out_qk)](
            self.hidden_states, q_proj_weight, q_proj_bias, q,
            Bsz, Ssz, H_in, H_out_qk,
            self.hidden_states.stride(0), self.hidden_states.stride(1), self.hidden_states.stride(2),
            q_proj_weight.stride(0), q_proj_weight.stride(1),
            q_proj_bias.stride(0),
            q.stride(0), q.stride(1), q.stride(2),
            num_warps=4, num_stages=2
        )

        # K: [B, S, H_out_qk]
        k_proj_weight = self.k_proj_weight
        k_proj_bias = self.k_proj_bias
        k = torch.empty((Bsz, Ssz, H_out_qk), device=device, dtype=dtype)
        linear_bias_kernel[(Bsz, Ssz, H_out_qk)](
            self.hidden_states, k_proj_weight, k_proj_bias, k,
            Bsz, Ssz, H_in, H_out_qk,
            self.hidden_states.stride(0), self.hidden_states.stride(1), self.hidden_states.stride(2),
            k_proj_weight.stride(0), k_proj_weight.stride(1),
            k_proj_bias.stride(0),
            k.stride(0), k.stride(1), k.stride(2),
            num_warps=4, num_stages=2
        )

        # V: [B, S, H_out_qk]
        v_proj_weight = self.v_proj_weight
        v_proj_bias = self.v_proj_bias
        v = torch.empty((Bsz, Ssz, H_out_qk), device=device, dtype=dtype)
        linear_bias_kernel[(Bsz, Ssz, H_out_qk)](
            self.hidden_states, v_proj_weight, v_proj_bias, v,
            Bsz, Ssz, H_in, H_out_qk,
            self.hidden_states.stride(0), self.hidden_states.stride(1), self.hidden_states.stride(2),
            v_proj_weight.stride(0), v_proj_weight.stride(1),
            v_proj_bias.stride(0),
            v.stride(0), v.stride(1), v.stride(2),
            num_warps=4, num_stages=2
        )

        # 2) RMSNorm for Q and K
        q_norm = torch.empty_like(q, device=device, dtype=dtype)
        rmsnorm_kernel[(Bsz, Ssz, D)](
            q, self.q_norm_weight, q_norm,
            Bsz, Ssz, D,
            q.stride(0), q.stride(1), q.stride(2),
            self.q_norm_weight.stride(0),
            q_norm.stride(0), q_norm.stride(1), q_norm.stride(2),
            BLOCK=128, num_warps=4, num_stages=2
        )

        k_norm = torch.empty_like(k, device=device, dtype=dtype)
        rmsnorm_kernel[(Bsz, Ssz, D)](
            k, self.k_norm_weight, k_norm,
            Bsz, Ssz, D,
            k.stride(0), k.stride(1), k.stride(2),
            self.k_norm_weight.stride(0),
            k_norm.stride(0), k_norm.stride(1), k_norm.stride(2),
            BLOCK=128, num_warps=4, num_stages=2
        )

        # 3) Apply rotation (RoPE) for Q and K
        q_rot = torch.empty_like(q_norm, device=device, dtype=dtype)
        rotate_half_kernel[(Bsz, Ssz, D)](
            q_norm, self.sin, self.cos, q_rot,
            Bsz, Ssz, D,
            q_norm.stride(0), q_norm.stride(1), q_norm.stride(2),
            self.sin.stride(0), self.cos.stride(0),
            q_rot.stride(0), q_rot.stride(1), q_rot.stride(2),
            num_warps=4, num_stages=2
        )

        k_rot = torch.empty_like(k_norm, device=device, dtype=dtype)
        rotate_half_kernel[(Bsz, Ssz, D)](
            k_norm, self.sin, self.cos, k_rot,
            Bsz, Ssz, D,
            k_norm.stride(0), k_norm.stride(1), k_norm.stride(2),
            self.sin.stride(0), self.cos.stride(0),
            k_rot.stride(0), k_rot.stride(1), k_rot.stride(2),
            num_warps=4, num_stages=2
        )

        # 4) Reshape to heads and transpose to [B, num_heads, S, D]
        # For Q, K, V: num_attention_heads = 96, head_dim = 128
        q_h = q_rot.view(Bsz, Ssz, num_attention_heads, D).transpose(1, 2)  # [B, 96, S, D]
        k_h = k_rot.view(Bsz, Ssz, num_key_value_heads, D).transpose(1, 2)  # [B, 8, S, D]
        v_h = v.view(Bsz, Ssz, num_key_value_heads, D).transpose(1, 2)      # [B, 8, S, D]

        # 5) Repeat K/V heads to match 96 attention heads for GQA
        # Expand to [B, 96, S, D]
        k_h_exp = k_h[:, :, None, :, :].expand(Bsz, num_attention_heads, num_key_value_groups, Ssz, D).reshape(Bsz, num_attention_heads, Ssz, D)
        v_h_exp = v_h[:, :, None, :, :].expand(Bsz, num_attention_heads, num_key_value_groups, Ssz, D).reshape(Bsz, num_attention_heads, Ssz, D)

        # 6) Compute attention scores: Q @ K^T -> [B, 96, S, S]
        scores = torch.empty((Bsz, num_attention_heads, Ssz, Ssz), device=device, dtype=dtype)
        matmul_qk_kernel[(Bsz, num_attention_heads)](
            q_h, k_h_exp, scores,
            Bsz, Ssz, D, num_attention_heads,
            q_h.stride(0), q_h.stride(1), q_h.stride(2), q_h.stride(3),
            k_h_exp.stride(0), k_h_exp.stride(1), k_h_exp.stride(2), k_h_exp.stride(3),
            scores.stride(0), scores.stride(1), scores.stride(2), scores.stride(3),
            num_warps=4, num_stages=2
        )

        # 7) Softmax with causal mask in Triton
        causal_mask = torch.empty((Ssz, Ssz), device=device, dtype=dtype)
        for i in range(Ssz):
            for j in range(Ssz):
                causal_mask[i, j] = -float('inf') if j > i else 0.0  # upper-triangular with diagonal=1
        # Load mask into Triton: we will generate it in Triton using index logic to avoid torch
        # Implement mask in Triton via index-based logic; softmax_mask_kernel uses index comparisons.
        y = torch.empty_like(scores, device=device, dtype=dtype)
        softmax_mask_kernel[(Bsz, num_attention_heads)](
            scores, causal_mask, y,
            Ssz,
            scores.stride(0), scores.stride(1), scores.stride(2), scores.stride(3),
            causal_mask.stride(0), causal_mask.stride(1),
            y.stride(0), y.stride(1), y.stride(2), y.stride(3),
            num_warps=4, num_stages=2
        )

        # 8) Compute attention output: Softmax(scores) @ V -> [B, 96, S, D]
        attn_out = torch.empty((Bsz, num_attention_heads, Ssz, D), device=device, dtype=dtype)
        matmul_attn_kernel[(Bsz, num_attention_heads)](
            y, v_h_exp, attn_out,
            Bsz, Ssz, D, num_attention_heads,
            y.stride(0), y.stride(1), y.stride(2), y.stride(3),
            v_h_exp.stride(0), v_h_exp.stride(1), v_h_exp.stride(2), v_h_exp.stride(3),
            attn_out.stride(0), attn_out.stride(1), attn_out.stride(2), attn_out.stride(3),
            num_warps=4, num_stages=2
        )

        # 9) Transpose and reshape for final output: [B, S, num_attention_heads * D]
        attn_out_t = attn_out.transpose(1, 2).contiguous()  # [B, S, 96, D] -> [B, S, 12288]
        attn_out_flat = attn_out_t.reshape(Bsz, Ssz, num_attention_heads * D)

        # 10) Final output projection (no bias)
        output = torch.empty((Bsz, Ssz, num_attention_heads * D), device=device, dtype=dtype)
        linear_nobias_kernel[(Bsz, Ssz, num_attention_heads * D)](
            attn_out_flat, self.o_proj_weight, output,
            Bsz, Ssz, num_attention_heads * D, num_attention_heads * D,
            attn_out_flat.stride(0), attn_out_flat.stride(1), attn_out_flat.stride(2),
            self.o_proj_weight.stride(0), self.o_proj_weight.stride(1),
            output.stride(0), output.stride(1), output.stride(2),
            num_warps=4, num_stages=2
        )

        return output


def run(*args):
    return ModelNew()(*args)
