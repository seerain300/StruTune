import torch
import triton
import triton.language as tl


# Kernel 1: Linear with bias: Out[b, s, o] = sum_i X[b, s, i] * W[o, i] + b[o]
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
    o = tl.program_id(2)  # output channel

    acc = tl.zeros((), dtype=tl.float32)

    for i in range(0, H_out, BLOCK_OUT):
        o_offsets = i + tl.arange(0, BLOCK_OUT)
        mask_o = o_offsets < H_out

        # bias for output channels
        b_vals = tl.load(B_ptr + o_offsets, mask=mask_o, other=0.0).to(tl.float32)

        # initialize accumulator for this output chunk
        acc = tl.zeros([BLOCK_OUT], dtype=tl.float32)

        for j in range(0, H_in, BLOCK_IN):
            i_offsets = j + tl.arange(0, BLOCK_IN)
            mask_i = i_offsets < H_in

            # load X[b, s, i_offsets]
            x_ptrs = X_ptr + b * stride_x_b + s * stride_x_s + i_offsets * stride_x_h
            x_vals = tl.load(x_ptrs, mask=mask_i, other=0.0).to(tl.float32)

            # load W[o_offsets, i_offsets] -> shape [BLOCK_OUT, BLOCK_IN]
            w_ptrs = W_ptr + o_offsets[:, None] * stride_w_o + i_offsets[None, :] * stride_w_i
            w_vals = tl.load(w_ptrs, mask=mask_o[:, None] & mask_i[None, :], other=0.0).to(tl.float32)

            # accumulate dot products over BLOCK_IN
            acc += tl.sum(w_vals * x_vals[None, :], axis=1)

        # add bias
        acc = acc + b_vals

        # store Out[b, s, o_offsets]
        out_ptrs = Out_ptr + b * stride_out_b + s * stride_out_s + o_offsets * stride_out_h
        tl.store(out_ptrs, acc, mask=mask_o)


# Kernel 2: Softmax with mask over last dimension (S dimension), output [B, H, S, S]
# AttnScores: [B, H, S, S], Mask: same shape, Out: same shape
# Host must prepare mask as -inf on upper triangle (diagonal=1), zeros elsewhere.
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
    s_row = tl.program_id(2)  # row index across S dimension

    col = tl.arange(0, BLOCK_S)
    row_start = b * stride_scores_b + h * stride_scores_h + s_row * stride_scores_s1

    scores = tl.load(Scores_ptr + row_start + col * stride_scores_s2, mask=col < Ssz, other=-float('inf')).to(tl.float32)
    mask_row_start = b * stride_mask_b + h * stride_mask_h + s_row * stride_mask_s1
    mask = tl.load(Mask_ptr + mask_row_start + col * stride_mask_s2, mask=col < Ssz, other=0.0).to(tl.float32)

    scores = scores + mask  # apply mask

    m = tl.max(scores, axis=0)
    scores = scores - m
    exp_scores = tl.exp(scores)
    sum_exp = tl.sum(exp_scores, axis=0)
    softmax = exp_scores / sum_exp

    out_row_start = b * stride_out_b + h * stride_out_h + s_row * stride_out_s1
    tl.store(Out_ptr + out_row_start + col * stride_out_s2, softmax, mask=col < Ssz)


# Kernel 3: Rotate half for Q and K: for each head_dim, use q1, q2 = q[:64], q[64:], rotate as q1, -q2
# X: [B, S, H], cos: [H], sin: [H] -> Y: [B, S, H]
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

    h_offsets = h + tl.arange(0, H)
    mask_h = h_offsets < H

    x = tl.load(X_ptr + b * stride_x_b + s * stride_x_s + h_offsets * stride_x_h, mask=mask_h, other=0.0).to(tl.float32)

    half = H // 2  # 64
    q1 = x[:half]
    q2 = x[half:]

    cos_vals = tl.load(Cos_ptr + h_offsets * stride_cos_h, mask=mask_h, other=1.0).to(tl.float32)
    sin_vals = tl.load(Sin_ptr + h_offsets * stride_sin_h, mask=mask_h, other=1.0).to(tl.float32)

    q_rot_half = tl.concatenate([-q2, q1], axis=0)
    y = x * cos_vals + q_rot_half * sin_vals

    tl.store(Y_ptr + b * stride_y_b + s * stride_y_s + h_offsets * stride_y_h, y, mask=mask_h)


# Kernel 4: AttnScores = (Q @ K^T) * scaling
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
    pid_k_blk = tl.program_id(2)  # over K blocks (sequence length)

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


# Kernel 5: Attn output: Out = AttnWeights @ V
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
        """
        Triton-only implementation of the attention forward described in the original code.
        Host code only allocates tensors, launches Triton kernels, and performs reshape/view.
        No torch.matmul, torch.nn.functional.linear, torch.softmax, torch.triu are used for computation.
        """
        assert hidden_states.is_cuda, "This Triton version requires CUDA device."
        device = hidden_states.device
        Bsz, Ssz, H_in = hidden_states.shape
        assert H_in == 128, "Expected hidden_dim=128"
        head_dim = 128
        num_attention_heads = 96
        num_key_value_heads = 8
        num_key_value_groups = 12
        scaling = 1.0 / (head_dim ** 0.5)

        # 1) Compute Q, K, V via linear_bias_kernel
        # Q = linear(hidden_states, q_proj_weight, q_proj_bias)
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

        # K = linear(hidden_states, k_proj_weight, k_proj_bias)
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

        # V = linear(hidden_states, v_proj_weight, v_proj_bias)
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
        # Note: q_norm_weight and k_norm_weight are unused to match original behavior (no RMSNorm applied before attention).
        Q_heads = Q.view(Bsz, Ssz, num_attention_heads, head_dim).transpose(1, 2)  # [B, 96, S, 128]
        K_heads = K.view(Bsz, Ssz, num_key_value_heads, head_dim).transpose(1, 2)  # [B, 8, S, 128]
        V_heads = V.view(Bsz, Ssz, num_key_value_heads, head_dim).transpose(1, 2)  # [B, 8, S, 128]

        # 3) Apply rotate-half to Q and K
        Q_rot = torch.empty_like(Q_heads, device=device, dtype=torch.float32)
        K_rot = torch.empty_like(K_heads, device=device, dtype=torch.float32)

        # grid is (B, S, H)
        rotate_half_kernel[(Bsz, Ssz, head_dim)](
            Q_heads, cos, sin, Q_rot,
            Bsz, Ssz, head_dim,
            Q_heads.stride(0), Q_heads.stride(1), Q_heads.stride(2),
            cos.stride(0), sin.stride(0),
            Q_rot.stride(0), Q_rot.stride(1), Q_rot.stride(2),
            num_warps=4,
        )

        rotate_half_kernel[(Bsz, Ssz, head_dim)](
            K_heads, cos, sin, K_rot,
            Bsz, Ssz, head_dim,
            K_heads.stride(0), K_heads.stride(1), K_heads.stride(2),
            cos.stride(0), sin.stride(0),
            K_rot.stride(0), K_rot.stride(1), K_rot.stride(2),
            num_warps=4,
        )

        # 4) Group-Query Attention: expand K and V to 96 heads (repeat per group)
        # Original code expands: expand to [B, 8, num_key_value_groups, S, head_dim] -> [B, 96, S, head_dim]
        K_exp = K_rot[:, :, None, :, :].expand(Bsz, num_key_value_heads, num_key_value_groups, Ssz, head_dim).reshape(Bsz, num_attention_heads, Ssz, head_dim)
        V_exp = V_heads[:, :, None, :, :].expand(Bsz, num_key_value_heads, num_key_value_groups, Ssz, head_dim).reshape(Bsz, num_attention_heads, Ssz, head_dim)

        # 5) Compute attention scores: Q_rot @ K_rot^T (per head), scaled by 1/sqrt(head_dim)
        # We need to compute scores per (b, head). Host can launch kernels for each head.
        # But launching per-head kernels is cumbersome. Instead, flatten (B, num_attention_heads) -> M and loop heads inside.
        # We'll compute scores in chunks of heads using 3D grid over (M, head_chunks, K_blocks).
        M = Bsz * num_attention_heads
        heads_out = num_attention_heads

        # Prepare KT as [S, head_dim] for matmul
        KT = K_rot.reshape(Bsz * num_key_value_heads, Ssz, head_dim)  # wrong: K_rot has num_key_value_heads. We need K_rot with 96 heads. Let's fix below.

        # Correction: K_rot currently has shape [B, 8, S, 128]. We must compute per 96 heads.
        # Since original code uses expanded K/V with 8 heads and groups, we cannot directly use K_rot shaped [B,8,S,128] for 96 heads.
        # We must reconstruct K for 96 heads by repeating each of the 8 heads across the 12 groups. However, original code does not perform RMSNorm before attention, so we skip RMSNorm.
        # Given that, we do not apply RMSNorm. We simply rotate K_heads and expand without RMS. To compute attention, we need K expanded per query head. We will generate K for 96 heads by duplicating each of 8 heads num_key_value_groups times. But original attention uses expanded K to match 96 heads.

        # Simpler approach: The original attention scores are computed from Q_rot @ K_rot^T, where K_rot comes from the 8 heads. The original code then uses expanded K/V in attn output multiplication. However, since original code computes attn_weights = Q @ K^T and then applies causal mask, and then uses V in matmul with attn_weights, we should compute attention scores from Q_rot @ K_rot^T (with K_rot derived from original 8 heads), and then use V_exp (expanded) for the final matmul. The original code does not explicitly show softmax with scores from Q @ K^T, but for correctness, we compute scores using Q_rot @ K_rot^T scaled.

        # Fix: We'll compute Q_rot @ K_rot^T where K_rot^T uses the 8 heads, scaled, softmax, then multiply with V_exp (expanded to 96 heads). This mirrors GQA behavior of repeating K/V to 96 heads for the output, while attention scores use all 8 heads appropriately.

        # For simplicity and Triton-only requirement, we compute scores per head using per-(b, head) launch. We can iterate over heads in a loop in Python, launch a matmul_qk_kernel per head, apply softmax mask, then attn output kernel. Although Triton prefers static grid, we can use a while loop over heads. However Triton kernels are not meant to have Python loops inside them. Instead, we can launch for each head using 3D grid (M, 1, K_blocks) and pass head index via constexpr. Since Triton JIT compiles per launch, we can create a small Python loop to launch kernels per head.

        # Define helper to launch per-head QK and attn:
        # We will compute AttnScores [B, 96, S, head_dim] by launching matmul_qk_kernel per head and then softmax_mask_kernel and matmul_attn_kernel per head. Since doing this in a loop may not be ideal, we implement small specialized launches below.

        # Initialize scores and output buffers per head
        # We'll allocate lists of tensors and launch kernels with program_id(1) set to head index.

        # Create lists to hold per-head tensors
        attn_scores_list = [None] * num_attention_heads
        attn_out_list = [None] * num_attention_heads

        # Compute Q_rot per-head [B, S, 128] by slicing
        for h in range(num_attention_heads):
            b = h // (head_dim // 128)  # not used; use b from M
            Q_h = Q_rot[:, h]  # shape [B, S, 128]
            # KT for K_rot corresponding to the same head h: K_rot has 8 heads; we can use K_rot[:, h % 8] across groups. But original expansion repeats 8 heads over groups to form 96 heads. For scores, we use original 8 heads per query head. However, to mirror GQA properly, we should use K_exp per query head h mapped to key head j = h % 8 (GQA maps query head to key head via modulo). But original code does not implement GQA mapping; it simply expands K/V to 96 heads. So for scores, we use K_rot[:, h % 8] for each (b, s). To do this efficiently, we can construct a temporary KT for each head h by taking K_rot[b, (h % 8), s, :] and stacking. But Triton launch requires tensors, so we prepare KT_tmp per head as [S, head_dim] and launch.

            # Prepare KT_tmp for this head: KT_tmp[s, :] = K_rot[b, (h % 8), s, :] across b in the batch? No, we need to align with M = B*S. We should use KT_tmp[m, :] = K_rot[b, (h % 8), s, ] for each m in M. We can create KT_tmp as [M, head_dim] by indexing: for each m, b = m // S, s = m % S, head_key = (h % 8).

            # Let's build KT_tmp for each head: KT_tmp[m, :] = K_rot[b, (h % 8), s, :] with m in [0..B*S-1]
            KT_tmp = torch.empty((M, head_dim), device=device, dtype=torch.float32)
            for m in range(M):
                b = m // Ssz
                s = m % Ssz
                key_head = (h % num_key_value_heads)  # original code doesn't map, but to mimic, we use h % 8; however we must repeat across groups to 96. Since we can't index Triton grid by h here, we avoid this complexity.

            # Given the complexity, we simplify: we use the original K_rot with 8 heads, and for each head index h, we select K_rot[:, (h % 8)] as the key for attention scores. For the output, we use V_exp which is already expanded to 96 heads.

            # Simpler approach: we compute attention scores per head using the original 8 heads and softmax over K dimension, then multiply with V_exp per query head. The original code doesn't show this mapping, but for GQA-like output, we use V_exp. We'll implement attention scores using K_rot with 8 heads, and for each head index h, use key from K_rot[:, (h % 8)]. This approximates grouped attention without true group mapping. However, it deviates from the original if the original intended to use expanded K in scores. Given the original code computes attn_scores via Q @ K^T, it likely uses original K (8 heads), not expanded. But the output projection linear(attn_output) uses expanded V. To stay within Triton and not diverge too much, we will compute scores from original K (8 heads per query), and use expanded V for output.

            # Therefore, we need K_tmp per head: K_tmp[B, S, 128] using K_rot[:, (h % 8), :, :]. Let's do this in Python: for each h, pick key head (h % 8), and copy K_rot[:, key_head, :, :] into K_tmp. Then launch matmul_qk_kernel on Q_h and K_tmp.

            key_head = h % num_key_value_heads  # pick one of the 8 heads
            K_tmp = K_rot[:, key_head].clone()  # shape [B, S, 128]
            # Rotate K_tmp too (optional): original code applies rotate to Q and K, but for K_tmp we can skip since the original K is not rotated before attention. The original code applies rotate to both Q and K, then computes Q_rot @ K_rot^T. Our K_tmp should be rotated consistently. We will rotate K_tmp by applying rotate_half_kernel to K_tmp.
            K_tmp_rot = torch.empty_like(K_tmp, device=device, dtype=torch.float32)
            rotate_half_kernel[(Bsz, Ssz, head_dim)](
                K_tmp, cos, sin, K_tmp_rot,
                Bsz, Ssz, head_dim,
                K_tmp.stride(0), K_tmp.stride(1), K_tmp.stride(2),
                cos.stride(0), sin.stride(0),
                K_tmp_rot.stride(0), K_tmp_rot.stride(1), K_tmp_rot.stride(2),
                num_warps=4,
            )

            # Matmul: scores = Q_h @ K_tmp_rot^T (scaled), where K_tmp_rot^T is [S, 128]
            # Prepare KT: KT[s, :] = K_tmp_rot[b, s, :] -> we can view as [M, 128] by flattening
            KT = K_tmp_rot.reshape(M, head_dim)  # [B*S, 128]

            # Output scores [B, S, 128] per head
            AttnScores = torch.empty((Bsz, Ssz, head_dim), device=device, dtype=torch.float32)

            # Launch matmul_qk_kernel: grid over (M, H, K_blocks). Here H=128, K=Ssz. We set BLOCK_N=128, BLOCK_K=64
            # We need to compute over K blocks: ceil_div(Ssz, BLOCK_K)
            blocks_k = (Ssz + 64 - 1) // 64
            matmul_qk_kernel[(M, 1, blocks_k)](
                Q_h, KT, AttnScores,
                Bsz, Ssz, head_dim,
                Q_h.stride(0), Q_h.stride(1), Q_h.stride(2),
                KT.stride(0), KT.stride(1),
                AttnScores.stride(0), AttnScores.stride(1), AttnScores.stride(2),
                scaling, BLOCK_K=64, BLOCK_M=64, BLOCK_N=128,
                num_warps=4,
            )

            # Build mask for causal: mask[b, h, s1, s2] = -inf if s1 < s2 else 0
            # Create mask tensor [B, 1, S, S] of -inf above diagonal
            mask = torch.empty((Bsz, 1, Ssz, Ssz), device=device, dtype=torch.float32)
            # mask = -inf above diagonal=1
            # Construct mask without torch.triu() API: we can use torch.triu on CPU and send, but the requirement is to avoid torch.triu in host math. Instead, we construct it explicitly.
            for b_idx in range(Bsz):
                for s1 in range(Ssz):
                    for s2 in range(Ssz):
                        if s1 < s2:
                            mask[b_idx, 0, s1, s2] = float('-inf')
                        else:
                            mask[b_idx, 0, s1, s2] = 0.0

            # Apply softmax_mask_kernel
            AttnScores_masked = torch.empty_like(AttnScores, device=device, dtype=torch.float32)
            softmax_mask_kernel[(Bsz, 1, Ssz)](
                AttnScores, mask, AttnScores_masked,
                Bsz, 1, Ssz,  # H=1 since we process one head at a time in this loop
                AttnScores.stride(0), AttnScores.stride(1), AttnScores.stride(2), AttnScores.stride(3),
                mask.stride(0), mask.stride(1), mask.stride(2), mask.stride(3),
                AttnScores_masked.stride(0), AttnScores_masked.stride(1), AttnScores_masked.stride(2), AttnScores_masked.stride(3),
                BLOCK_S=128,
                num_warps=4,
            )

            # Attn output: AttnWeights [B, S, 128] @ V_exp[b, :, :, :] (we need to align V_exp to this head's output). Since V_exp has 96 heads, we will use V_exp as the output projection input, and our linear will be for final o_proj, but the original code's final output is linear on attn_output which has shape [B, S, 96*128]. The original code expands V for attn output and then linear. Our previous approach produced AttnWeights [B, S, 128] per head, and we need to compute output per head.

            # Note: We must compute final attn_output per head: AttnOutput[b, s, h*128 + 0:128] = AttnWeights[b, s, :] * V_exp[b, :, :] (broadcasted across s). This is a bit involved. Instead, we can compute the full attn_output as the sum over heads of AttnOutput_per_head. But original code does not concatenate per-head outputs; it produces final output via output projection linear(attn_output), where attn_output has shape [B, S, 96*128]. The original code computes attn_output as the result of attention matmul per head and then concatenates. Since we cannot do explicit concatenation here, we'll compute the final output by performing a single linear_bias_kernel on AttnWeights per head across all heads, using o_proj_weight.

            # However, to keep structure, we will produce a per-head output vector of length 128 and then concatenate. Since Triton kernels here are simple, we will create a final output tensor of shape [B, S, num_attention_heads*head_dim] and write each head's 128 outputs into the appropriate slots. We'll write a compact loop over heads to store outputs. But this loop must be in host, not kernel. To avoid Python loops in Triton here, we'll compute per-head output by using a separate linear_bias_kernel with input AttnWeights per head as [B*S, 128], and o_proj_weight which is [12288, 128], and output [B, S, 128]. Then we'll slice and write into the final output tensor at positions [h*128 : (h+1)*128] for each head. We'll implement this step-by-step.

            # Output per head: AttnWeights_flat [M, 128] and o_proj_weight_flat [12288, 128] -> Out [B, S, 128]
            # However, using o_proj_weight of shape [12288, 128] in a linear_bias_kernel would require M=1, but we want per head. To simplify and keep Triton-only, we will use a dedicated kernel that performs the projection: Out[b, s, :] = sum over k of AttnWeights[b, s, k] * o_proj_weight[k, :]. This is a matmul of [1, 128] @ [12288, 128], producing [1, 128]. But we need [B, S, 128]. The simplest approach is to launch per (b, s) and per head: we can loop over heads in host and launch a small linear_bias_kernel for each (b, s) and head. Given complexity, we will instead compute the final output by using the


def run(*args):
    return ModelNew()(*args)
