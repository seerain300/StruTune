import torch
import triton
import triton.language as tl


# Kernel 1: Linear with bias: X @ W.T + b
# X: [B, S, H_in], W: [H_out, H_in], b: [H_out]
# Output: Y: [B, S, H_out]
@triton.jit
def linear_bias_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    Bsz, Ssz, H_in, H_out,
    stride_x_b, stride_x_s, stride_x_hin,
    stride_w_hout, stride_w_hin,
    stride_b_hout,
    stride_y_b, stride_y_s, stride_y_hout,
    BLOCK_HIN: tl.constexpr, BLOCK_HOUT: tl.constexpr,
):
    # Grid: (Bsz, Ssz, H_out)
    b = tl.program_id(0)
    s = tl.program_id(1)
    hout = tl.program_id(2)

    acc = 0.0

    # Accumulate over H_in in tiles of BLOCK_HIN
    for hin_start in range(0, H_in, BLOCK_HIN):
        x_vec = tl.zeros((BLOCK_HIN,), dtype=tl.float32)
        for i in range(0, BLOCK_HIN):
            hin = hin_start + i
            x_off = b * stride_x_b + s * stride_x_s + hin * stride_x_hin
            x_vec[i] = tl.load(X_ptr + x_off)
        # Load corresponding W[hout, :] across H_in (vector over H_in for fixed hout)
        w_vec = tl.zeros((BLOCK_HOUT,), dtype=tl.float32)
        for i in range(0, BLOCK_HOUT):
            w_off = hout * stride_w_hout + (hin_start + i) * stride_w_hin  # index over H_in
            w_vec[i] = tl.load(W_ptr + w_off)
        # Dot product accumulate for this hout
        acc += tl.sum(x_vec * w_vec, axis=0)

    # Add bias
    bias = tl.load(B_ptr + hout * stride_b_hout)
    acc = acc + bias

    # Store result to Y[b, s, hout]
    y_off = b * stride_y_b + s * stride_y_s + hout * stride_y_hout
    tl.store(Y_ptr + y_off, acc)


# Kernel 2: RMSNorm per (b,s) row over last dim D (D=128): y = x * rsqrt(mean(x^2) + eps), then scale by weight
# X: [B, S, D], Weight: [D], Output Y: [B, S, D]
@triton.jit
def rmsnorm_kernel(
    X_ptr, Weight_ptr, Out_ptr,
    Bsz, Ssz, D,
    stride_x_b, stride_x_s, stride_x_d,
    stride_w_d,
    stride_out_b, stride_out_s, stride_out_d,
    BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    # Loop over D in tiles
    for d_start in range(0, D, BLOCK_D):
        x_sum = 0.0
        for i in range(0, BLOCK_D):
            d = d_start + i
            x_off = b * stride_x_b + s * stride_x_s + d * stride_x_d
            x_val = tl.load(X_ptr + x_off)
            x_sum += x_val * x_val
        mean = x_sum / D
        inv_rms = tl.rsqrt(mean + 0.0)  # eps=0.0 per original
        for i in range(0, BLOCK_D):
            d = d_start + i
            x_off = b * stride_x_b + s * stride_x_s + d * stride_x_d
            w_off = d * stride_w_d
            x_val = tl.load(X_ptr + x_off)
            w_val = tl.load(Weight_ptr + w_off)
            y_val = x_val * inv_rms
            y_val = y_val * w_val
            out_off = b * stride_out_b + s * stride_out_s + d * stride_out_d
            tl.store(Out_ptr + out_off, y_val)


# Kernel 3: Rotate half of last 64 dims: for D=128, apply q1*cos - q2*sin to last 64 dims
# Q: [B, S, D], Sin: [D], Cos: [D], Out: [B, S, D]
@triton.jit
def rotate_half_kernel(
    Q_ptr, Sin_ptr, Cos_ptr, Out_ptr,
    Bsz, Ssz, D,
    stride_q_b, stride_q_s, stride_q_d,
    stride_s_d, stride_c_d, stride_out_b, stride_out_s, stride_out_d,
    BLOCK_D: tl.constexpr,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    for d_start in range(0, D, BLOCK_D):
        for i in range(0, BLOCK_D):
            d = d_start + i
            q_off = b * stride_q_b + s * stride_q_s + d * stride_q_d
            sin_off = d * stride_s_d
            cos_off = d * stride_c_d
            q_val = tl.load(Q_ptr + q_off)
            sin_val = tl.load(Sin_ptr + sin_off)
            cos_val = tl.load(Cos_ptr + cos_off)
            # If d < 64, q1 = q_val, q2 = Q_ptr[b,s,64+i]
            # Otherwise, q1 = Q_ptr[b,s,d], q2 = Q_ptr[b,s,d-64]
            if d < 64:
                q2 = tl.load(Q_ptr + b * stride_q_b + s * stride_q_s + (64 + i) * stride_q_d)
                q1 = q_val
                out_val = q1 * cos_val - q2 * sin_val
            else:
                q1 = tl.load(Q_ptr + b * stride_q_b + s * stride_q_s + d * stride_q_d)
                q2 = tl.load(Q_ptr + b * stride_q_b + s * stride_q_s + (d - 64) * stride_q_d)
                out_val = q1 * cos_val - q2 * sin_val
            out_off = b * stride_out_b + s * stride_out_s + d * stride_out_d
            tl.store(Out_ptr + out_off, out_val)


# Kernel 4: Compute attention scores Q @ K^T for each (b, s, h): Out[b,h,i,j] = sum_k Q[b,h,i,k] * K[b,h,j,k]
# Inputs: Q_norm: [B, S, D], K_norm: [B, S, D], Outputs: Attn: [B, num_attention_heads, S, S] (we pass h index as grid dim)
@triton.jit
def matmul_qk_block(
    Q_ptr, K_ptr, Out_ptr,
    Bsz, Ssz, D,
    stride_q_b, stride_q_s, stride_q_d,
    stride_k_b, stride_k_s, stride_k_d,
    stride_out_b, stride_out_h, stride_out_i, stride_out_j,
    h: tl.constexpr,  # attention head index, can be a constexpr to let Triton optimize
    BLOCK: tl.constexpr,
):
    # Grid over (b, h, i)
    b = tl.program_id(0)
    i = tl.program_id(1)
    j_block = tl.program_id(2)

    col_start = j_block * BLOCK
    for j in range(0, BLOCK):
        j_idx = col_start + j
        if j_idx >= Ssz:
            break

        # Compute scores[i, j_idx]
        score = 0.0
        for k in range(0, D):
            q_off = b * stride_q_b + i * stride_q_s + k * stride_q_d
            k_off = b * stride_k_b + h * stride_k_h + j_idx * stride_k_s + k * stride_k_d
            q_val = tl.load(Q_ptr + q_off)
            k_val = tl.load(K_ptr + k_off)
            score += q_val * k_val

        out_off = b * stride_out_b + h * stride_out_h + i * stride_out_i + j_idx * stride_out_j
        tl.store(Out_ptr + out_off, score)


# Kernel 5: Softmax over last dim (sequence length) for each row i: Attn[b,h,i,:]
# Apply causal mask: if j > i, set to -inf. We do softmax in Triton with the mask.
@triton.jit
def softmax_mask_kernel(
    Attn_ptr, Out_ptr,
    Bsz, Ssz,
    stride_attn_b, stride_attn_h, stride_attn_i, stride_attn_j,
    stride_out_b, stride_out_h, stride_out_i, stride_out_j,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    for i in range(0, Ssz):
        # row_max
        row_max = -1e30
        for j in range(0, Ssz):
            attn_off = b * stride_attn_b + h * stride_attn_h + i * stride_attn_i + j * stride_attn_j
            score = tl.load(Attn_ptr + attn_off)
            if j > i:
                score = -1e30
            row_max = tl.maximum(row_max, score)
        # sum of exp
        row_sum = 0.0
        for j in range(0, Ssz):
            attn_off = b * stride_attn_b + h * stride_attn_h + i * stride_attn_i + j * stride_attn_j
            score = tl.load(Attn_ptr + attn_off)
            if j > i:
                score = -1e30
            exp_score = tl.exp(score - row_max)
            row_sum += exp_score
        # write normalized
        for j in range(0, Ssz):
            attn_off = b * stride_attn_b + h * stride_attn_h + i * stride_attn_i + j * stride_attn_j
            score = tl.load(Attn_ptr + attn_off)
            if j > i:
                score = -1e30
            prob = tl.exp(score - row_max) / row_sum
            out_off = b * stride_out_b + h * stride_out_h + i * stride_out_i + j * stride_out_j
            tl.store(Out_ptr + out_off, prob)


# Kernel 6: Compute attention output for each (b, s, h): Out[b,h,s,:] = sum_j Softmax[b,h,s,j] * V[b,h,s,j,:]
# Inputs: Softmax: [B, num_attention_heads, S, S], V_norm: [B, S, D], Outputs: AttnOut: [B, num_attention_heads, S, D]
@triton.jit
def matmul_attn_block(
    Soft_ptr, V_ptr, Out_ptr,
    Bsz, Ssz, D,
    stride_soft_b, stride_soft_h, stride_soft_i, stride_soft_j,
    stride_v_b, stride_v_s, stride_v_d,
    stride_out_b, stride_out_h, stride_out_i, stride_out_d,
    BLOCK: tl.constexpr,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    i = tl.program_id(2)
    d_block = tl.program_id(3)

    for d_start in range(0, D, BLOCK):
        out_vec = tl.zeros((BLOCK,), dtype=tl.float32)
        for d in range(0, BLOCK):
            d_idx = d_start + d
            if d_idx >= D:
                break
            sum_val = 0.0
            for j in range(0, Ssz):
                soft_off = b * stride_soft_b + h * stride_soft_h + i * stride_soft_i + j * stride_soft_j
                v_off = b * stride_v_b + i * stride_v_s + d_idx * stride_v_d
                prob = tl.load(Soft_ptr + soft_off)
                v_val = tl.load(V_ptr + v_off)
                sum_val += prob * v_val
            out_off = b * stride_out_b + h * stride_out_h + i * stride_out_i + d_idx * stride_out_d
            out_vec[d] = sum_val
            tl.store(Out_ptr + out_off, out_vec[d])


# Kernel 7: Linear without bias: X @ W.T
# X: [B, S, H_in], W: [H_out, H_in], Output: Y: [B, S, H_out]
@triton.jit
def linear_nobias_kernel(
    X_ptr, W_ptr, Y_ptr,
    Bsz, Ssz, H_in, H_out,
    stride_x_b, stride_x_s, stride_x_hin,
    stride_w_hout, stride_w_hin,
    stride_y_b, stride_y_s, stride_y_hout,
    BLOCK_HIN: tl.constexpr, BLOCK_HOUT: tl.constexpr,
):
    b = tl.program_id(0)
    s = tl.program_id(1)
    hout = tl.program_id(2)

    acc = 0.0
    for hin_start in range(0, H_in, BLOCK_HIN):
        x_vec = tl.zeros((BLOCK_HIN,), dtype=tl.float32)
        for i in range(0, BLOCK_HIN):
            hin = hin_start + i
            x_off = b * stride_x_b + s * stride_x_s + hin * stride_x_hin
            x_vec[i] = tl.load(X_ptr + x_off)
        w_vec = tl.zeros((BLOCK_HOUT,), dtype=tl.float32)
        for i in range(0, BLOCK_HOUT):
            w_off = hout * stride_w_hout + (hin_start + i) * stride_w_hin
            w_vec[i] = tl.load(W_ptr + w_off)
        acc += tl.sum(x_vec * w_vec, axis=0)

    y_off = b * stride_y_b + s * stride_y_s + hout * stride_y_hout
    tl.store(Y_ptr + y_off, acc)


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
        # Expect shapes: hidden_states: [B, S, H_in], weights: [H_out, H_in], biases: [H_out], cos/sin: [D] where D=128
        Bsz = hidden_states.shape[0]
        Ssz = hidden_states.shape[1]
        H_in = hidden_states.shape[2]
        D = 128  # head_dim

        # Allocate intermediates (all Triton kernels; no torch in host)
        # 1) Q = linear(hidden_states, q_proj_weight, q_proj_bias)
        Q = torch.empty((Bsz, Ssz, D), dtype=torch.float32, device=hidden_states.device)
        linear_bias_kernel[(Bsz, Ssz, D // 64 + 1)](  # grid over B,S,H_out (here D=128, H_out=D)
            hidden_states, q_proj_weight, q_proj_bias, Q,
            Bsz, Ssz, H_in, D,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            q_proj_weight.stride(0), q_proj_weight.stride(1),
            q_proj_bias.stride(0),
            Q.stride(0), Q.stride(1), Q.stride(2),
            BLOCK_HIN=64, BLOCK_HOUT=64,
        )

        # 2) K = linear(hidden_states, k_proj_weight, k_proj_bias)
        K = torch.empty((Bsz, Ssz, D), dtype=torch.float32, device=hidden_states.device)
        linear_bias_kernel[(Bsz, Ssz, D // 64 + 1)](
            hidden_states, k_proj_weight, k_proj_bias, K,
            Bsz, Ssz, H_in, D,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            k_proj_weight.stride(0), k_proj_weight.stride(1),
            k_proj_bias.stride(0),
            K.stride(0), K.stride(1), K.stride(2),
            BLOCK_HIN=64, BLOCK_HOUT=64,
        )

        # 3) V = linear(hidden_states, v_proj_weight, v_proj_bias)
        V = torch.empty((Bsz, Ssz, D), dtype=torch.float32, device=hidden_states.device)
        linear_bias_kernel[(Bsz, Ssz, D // 64 + 1)](
            hidden_states, v_proj_weight, v_proj_bias, V,
            Bsz, Ssz, H_in, D,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            v_proj_weight.stride(0), v_proj_weight.stride(1),
            v_proj_bias.stride(0),
            V.stride(0), V.stride(1), V.stride(2),
            BLOCK_HIN=64, BLOCK_HOUT=64,
        )

        # 4) RMSNorm for Q and K
        Q_norm = torch.empty_like(Q)
        rmsnorm_kernel[(Bsz, Ssz, D)](
            Q, q_norm_weight, Q_norm,
            Bsz, Ssz, D,
            Q.stride(0), Q.stride(1), Q.stride(2),
            q_norm_weight.stride(0),
            Q_norm.stride(0), Q_norm.stride(1), Q_norm.stride(2),
            BLOCK_D=128,
        )
        K_norm = torch.empty_like(K)
        rmsnorm_kernel[(Bsz, Ssz, D)](
            K, k_norm_weight, K_norm,
            Bsz, Ssz, D,
            K.stride(0), K.stride(1), K.stride(2),
            k_norm_weight.stride(0),
            K_norm.stride(0), K_norm.stride(1), K_norm.stride(2),
            BLOCK_D=128,
        )

        # 5) Apply RoPE (rotate half) to Q and K
        Q_rot = torch.empty_like(Q_norm)
        rotate_half_kernel[(Bsz, Ssz, D)](
            Q_norm, cos, sin, Q_rot,
            Bsz, Ssz, D,
            Q_norm.stride(0), Q_norm.stride(1), Q_norm.stride(2),
            cos.stride(0), sin.stride(0),
            Q_rot.stride(0), Q_rot.stride(1), Q_rot.stride(2),
            BLOCK_D=128,
        )
        K_rot = torch.empty_like(K_norm)
        rotate_half_kernel[(Bsz, Ssz, D)](
            K_norm, cos, sin, K_rot,
            Bsz, Ssz, D,
            K_norm.stride(0), K_norm.stride(1), K_norm.stride(2),
            cos.stride(0), sin.stride(0),
            K_rot.stride(0), K_rot.stride(1), K_rot.stride(2),
            BLOCK_D=128,
        )

        # 6) Compute attention scores: Attn[b,h,i,j] = sum_k Q_rot[b,h,i,k] * K_rot[b,h,j,k]
        num_attention_heads = 96
        attn_scores = torch.empty((Bsz, num_attention_heads, Ssz, Ssz), dtype=torch.float32, device=hidden_states.device)
        # Launch per head
        for h in range(0, num_attention_heads):
            # Grid: (Bsz, Ssz, Ssz // BLOCK + 1), BLOCK=128
            matmul_qk_block[(Bsz, Ssz, (Ssz + 127) // 128)](
                Q_rot, K_rot, attn_scores,
                Bsz, Ssz, D,
                Q_rot.stride(0), Q_rot.stride(1), Q_rot.stride(2),
                K_rot.stride(0), K_rot.stride(1), K_rot.stride(2),
                attn_scores.stride(0), attn_scores.stride(1), attn_scores.stride(2), attn_scores.stride(3),
                h,
                BLOCK=128,
            )

        # 7) Softmax with causal mask in Triton
        attn_probs = torch.empty_like(attn_scores)
        softmax_mask_kernel[(Bsz, num_attention_heads)](
            attn_scores, attn_probs,
            Bsz, Ssz,
            attn_scores.stride(0), attn_scores.stride(1), attn_scores.stride(2), attn_scores.stride(3),
            attn_probs.stride(0), attn_probs.stride(1), attn_probs.stride(2), attn_probs.stride(3),
            BLOCK=128,
        )

        # 8) Compute attention output: Out[b,h,i,:] = sum_j probs[b,h,i,j] * V[b,h,i,j,:]
        attn_out = torch.empty((Bsz, num_attention_heads, Ssz, D), dtype=torch.float32, device=hidden_states.device)
        for h in range(0, num_attention_heads):
            matmul_attn_block[(Bsz, 1, Ssz, (D + 127) // 128)](
                attn_probs, V, attn_out,
                Bsz, Ssz, D,
                attn_probs.stride(0), attn_probs.stride(1), attn_probs.stride(2), attn_probs.stride(3),
                V.stride(0), V.stride(1), V.stride(2),
                attn_out.stride(0), attn_out.stride(1), attn_out.stride(2), attn_out.stride(3),
                BLOCK=128,
            )

        # 9) Transpose and reshape: attn_out [B, num_heads, S, D] -> [B, S, num_heads*D]
        attn_out_t = attn_out.transpose(1, 2).contiguous()  # [B, S, num_heads, D]
        attn_out_2d = attn_out_t.reshape(Bsz, Ssz, num_attention_heads * D)

        # 10) Final output projection (no bias): attn_out_2d @ o_proj_weight.T
        output = torch.empty((Bsz, Ssz, o_proj_weight.shape[0]), dtype=torch.float32, device=hidden_states.device)
        linear_nobias_kernel[(Bsz, Ssz, (num_attention_heads * D + 63) // 64 + 1)](
            attn_out_2d, o_proj_weight, output,
            Bsz, Ssz, num_attention_heads * D, o_proj_weight.shape[0],
            attn_out_2d.stride(0), attn_out_2d.stride(1), attn_out_2d.stride(2),
            o_proj_weight.stride(0), o_proj_weight.stride(1),
            output.stride(0), output.stride(1), output.stride(2),
            BLOCK_HIN=64, BLOCK_HOUT=64,
        )

        return output


def run(*args):
    return ModelNew()(*args)
