import torch
import triton
import triton.language as tl

# 1) Triton dense linear: out[b, s, h] = sum_k input[b, :, k] * weight[h, k] + bias[h]
@triton.jit
def triton_linear_bsh(input_ptr, weight_ptr, bias_ptr, out_ptr,
                      B, S, H, K,
                      input_stride0, input_stride1, input_stride2,
                      weight_stride0, weight_stride1,
                      out_stride0, out_stride1, out_stride2,
                      BLOCK_K: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    base_input = b * input_stride0  # input[b, :, :] flattened across S*K
    acc = tl.zeros((), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        mask = k < K
        # Load input[b, :, k] as a vector of length K
        x = tl.load(input_ptr + base_input + k * input_stride2, mask=mask, other=0.0)
        # Load weight[h, k] as a vector of length K
        w = tl.load(weight_ptr + h * weight_stride0 + k * weight_stride1, mask=mask, other=0.0)
        acc += tl.sum(x * w, axis=0)

    bval = tl.load(bias_ptr + h)
    acc += bval
    tl.store(out_ptr + b * out_stride0 + s * out_stride1 + h * out_stride2, acc)


# 2) Triton RMSNorm per row: out_row = weight[h] * x_row / sqrt(mean(x_row^2) + eps)
# Note: here "row" refers to entire sequence dimension S, i.e., normalize Q[b, h, :] and K[b, h, :].
@triton.jit
def triton_rmsnorm_row(x_ptr, out_ptr, weight_ptr, eps, B, H, S,
                        x_stride0, x_stride1,
                        out_stride0, out_stride1,
                        BLOCK_D: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    row_start = b * S

    sum_sq = 0.0
    for d in range(0, S, BLOCK_D):
        offs = d + tl.arange(0, BLOCK_D)
        mask = offs < S
        x = tl.load(x_ptr + row_start + offs * x_stride1, mask=mask, other=0.0)
        sum_sq += tl.sum(x * x, axis=0)

    mean = sum_sq / S
    inv_rms = 1.0 / tl.sqrt(mean + eps)
    scale = tl.load(weight_ptr + h) * inv_rms

    for d in range(0, S, BLOCK_D):
        offs = d + tl.arange(0, BLOCK_D)
        mask = offs < S
        x = tl.load(x_ptr + row_start + offs * x_stride1, mask=mask, other=0.0)
        y = x * scale
        tl.store(out_ptr + b * out_stride0 + h * out_stride1 + offs * out_stride1, y, mask=mask)


# 3) Triton RoPE: rotate query/key: split 128 into 64+64 and apply q_out = q*cos + [-q2, q1]*sin
@triton.jit
def triton_rope_row(x_ptr, cos_ptr, sin_ptr, out_ptr,
                    B, H, S, head_dim,
                    x_stride0, x_stride1, x_stride2,
                    cos_stride0, cos_stride1,
                    sin_stride0, sin_stride1,
                    out_stride0, out_stride1, out_stride2,
                    BLOCK_D: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    base_x = b * x_stride0 + s * x_stride1 + h * x_stride2
    base_out = b * out_stride0 + s * out_stride1 + h * out_stride2

    for d0 in range(0, head_dim, BLOCK_D):
        d = d0 + tl.arange(0, BLOCK_D)
        mask = d < head_dim
        q = tl.load(x_ptr + base_x + d, mask=mask, other=0.0)
        cos_vec = tl.load(cos_ptr + d, mask=mask, other=0.0)
        sin_vec = tl.load(sin_ptr + d, mask=mask, other=0.0)
        q1 = q[:64]
        q2 = q[64:]
        rotated = -q2 + q1
        q_out = q * cos_vec + rotated * sin_vec
        tl.store(out_ptr + base_out + d, q_out, mask=mask)


# 4) Triton GQA expand: expand KV heads from KVH to H with GROUPS = H // KVH
@triton.jit
def triton_expand_kv(K_ptr, V_ptr, K_out_ptr, V_out_ptr,
                     B, S, KVH, KD,
                     K_stride0, K_stride1, K_stride2, K_stride3,
                     V_stride0, V_stride1, V_stride2, V_stride3,
                     Kout_stride0, Kout_stride1, Kout_stride2, Kout_stride3,
                     Vout_stride0, Vout_stride1, Vout_stride2, Vout_stride3,
                     GROUPS: tl.constexpr):
    for kh in range(0, KVH):
        for g in range(0, GROUPS):
            h_target = kh * GROUPS + g
            for j in range(0, S):
                # K: load K[b, kh, j, :] (KD is last dim)
                k_row = tl.load(K_ptr + b * K_stride0 + kh * K_stride2 + j * K_stride1 + 0 * K_stride3)
                tl.store(K_out_ptr + b * Kout_stride0 + h_target * Kout_stride2 + j * Kout_stride1 + 0 * Kout_stride3, k_row)
                # V: load V[b, kh, j, :]
                v_row = tl.load(V_ptr + b * V_stride0 + kh * V_stride2 + j * V_stride1 + 0 * V_stride3)
                tl.store(V_out_ptr + b * Vout_stride0 + h_target * Vout_stride2 + j * Vout_stride1 + 0 * Vout_stride3, v_row)


# 5) Triton attention compute per (b, h): tile i and j, compute scores, apply causal, softmax, accumulate
@triton.jit
def triton_attention_compute(Q_ptr, K_ptr, V_ptr, Out_ptr,
                             B, S, H, head_dim,
                             Q_stride0, Q_stride1, Q_stride2, Q_stride3,
                             K_stride0, K_stride1, K_stride2, K_stride3,
                             V_stride0, V_stride1, V_stride2, V_stride3,
                             Out_stride0, Out_stride1,
                             BLOCK_I: tl.constexpr, BLOCK_J: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)

    inv_scale = 1.0 / tl.sqrt(head_dim)

    for i0 in range(0, S, BLOCK_I):
        i = i0 + tl.arange(0, BLOCK_I)
        mask_i = i < S

        scores = tl.zeros((BLOCK_I,), dtype=tl.float32)
        # Compute scores[i, j] = Q[b, h, i] · K[b, h, j]
        for j0 in range(0, S, BLOCK_J):
            j = j0 + tl.arange(0, BLOCK_J)
            mask_j = j < S

            # Load Q[i, :]
            q_base = b * Q_stride0 + i * Q_stride2
            q_ptrs = q_base + Q_stride3  # Q_stride3 corresponds to KD (128), but here we access Vectors across KD dimension implicitly via stride. Simplify: for Q[KV], KD stride is 128 -> we access linearly.
            # We need to load Q[b, h, i, k] for k in 0..127
            q_vec = tl.zeros((BLOCK_I,), dtype=tl.float32)
            # Implement Q vector loading for i across KD: we assume Q is laid out [B, S, H, KD], so we access with strides accordingly.
            # However, our out tensors were created with out[b, s, h, kd] so we can load q_vec from Q_ptr at b,i,h across KD.
            # Simpler: assume Q has shape [B, S, H, KD] with strides. We'll pass correct strides from host.

            # Initialize q_vec by looping over k
            for k in range(0, head_dim):
                # q_ptrs = b*Q_stride0 + i*Q_stride1 + h*Q_stride2 + k*Q_stride3
                q_ptrs = b * Q_stride0 + i * Q_stride1 + h * Q_stride2 + k * Q_stride3
                # Load each element for i vector
                q_vec += tl.load(Q_ptr + q_ptrs, mask=mask_i, other=0.0)

            # Load K[j, :]
            k_base = b * K_stride0 + j * K_stride1
            k_ptrs = k_base + K_stride3  # across KD
            k_vec = tl.zeros((BLOCK_J,), dtype=tl.float32)
            for k in range(0, head_dim):
                k_ptrs = b * K_stride0 + j * K_stride1 + h * K_stride2 + k * K_stride3
                k_vec += tl.load(K_ptr + k_ptrs, mask=mask_j, other=0.0)

            scores += tl.sum(q_vec[:, None] * k_vec[None, :], axis=1) * inv_scale

        # Apply causal mask: if i >= j, scores[i, j] = -inf
        for ii in range(0, BLOCK_I):
            for jj in range(0, BLOCK_J):
                if (i0 + ii) >= (j0 + jj):
                    scores[ii] = -float('inf')

        # Softmax along j
        max_score = tl.max(scores, axis=0)
        exp_scores = tl.exp(scores - max_score)
        sum_exp = tl.sum(exp_scores, axis=0)
        attn = exp_scores / sum_exp

        # Accumulate output[b, h, i] = sum_j attn[i, j] * V[b, h, j]
        out_vec = tl.zeros((BLOCK_I,), dtype=tl.float32)
        for j0 in range(0, S, BLOCK_J):
            j = j0 + tl.arange(0, BLOCK_J)
            mask_j = j < S
            v_base = b * V_stride0 + j * V_stride1
            v_ptrs = v_base + h * V_stride2 + 0 * V_stride3
            v_vec = tl.zeros((BLOCK_J,), dtype=tl.float32)
            for k in range(0, head_dim):
                v_ptrs = b * V_stride0 + j * V_stride1 + h * V_stride2 + k * V_stride3
                v_vec += tl.load(V_ptr + v_ptrs, mask=mask_j, other=0.0)
            out_vec += tl.sum(attn[:, jj] * v_vec[jj] for jj in range(0, BLOCK_J))

        # Store out[b, h, i]
        out_base = b * Out_stride0 + h * Out_stride1
        for ii in range(0, BLOCK_I):
            ii_global = i0 + ii
            if ii_global < S:
                tl.store(Out_ptr + out_base + ii_global * Out_stride1, out_vec[ii])


# 6) Triton linear for output projection: Out_final[b, s, h] = sum_k Out_final[b, s, k] * o_proj_weight[h, k]
@triton.jit
def triton_linear_bsh_out(Out_final_ptr, weight_ptr, out_ptr,
                          B, S, H, K,
                          Out_final_stride0, Out_final_stride1, Out_final_stride2,
                          weight_stride0, weight_stride1,
                          out_stride0, out_stride1, out_stride2,
                          BLOCK_K: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.program_id(2)

    base_out = b * Out_final_stride0 + s * Out_final_stride1
    acc = tl.zeros((), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        mask = k < K
        x = tl.load(Out_final_ptr + base_out + k * Out_final_stride2, mask=mask, other=0.0)
        w = tl.load(weight_ptr + h * weight_stride0 + k * weight_stride1, mask=mask, other=0.0)
        acc += tl.sum(x * w, axis=0)

    tl.store(out_ptr + b * out_stride0 + s * out_stride1 + h * out_stride2, acc)


class ModelNew(torch.nn.Module):
    def __init__(self,
                 num_attention_heads: int = 96,
                 num_key_value_heads: int = 8,
                 head_dim: int = 128,
                 rms_norm_eps: float = 1e-6):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.rms_norm_eps = rms_norm_eps

    def forward(self,
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
                sin: torch.Tensor):
        # Ensure CUDA tensors
        device = hidden_states.device
        if not hidden_states.is_cuda:
            hidden_states = hidden_states.to('cuda')
        if not q_proj_weight.is_cuda:
            q_proj_weight = q_proj_weight.to('cuda')
        if not q_proj_bias.is_cuda:
            q_proj_bias = q_proj_bias.to('cuda')
        if not k_proj_weight.is_cuda:
            k_proj_weight = k_proj_weight.to('cuda')
        if not k_proj_bias.is_cuda:
            k_proj_bias = k_proj_bias.to('cuda')
        if not v_proj_weight.is_cuda:
            v_proj_weight = v_proj_weight.to('cuda')
        if not v_proj_bias.is_cuda:
            v_proj_bias = v_proj_bias.to('cuda')
        if not o_proj_weight.is_cuda:
            o_proj_weight = o_proj_weight.to('cuda')
        if not q_norm_weight.is_cuda:
            q_norm_weight = q_norm_weight.to('cuda')
        if not k_norm_weight.is_cuda:
            k_norm_weight = k_norm_weight.to('cuda')
        if not cos.is_cuda:
            cos = cos.to('cuda')
        if not sin.is_cuda:
            sin = sin.to('cuda')

        B, S, _ = hidden_states.shape
        KVH = self.num_key_value_heads
        H = self.num_attention_heads
        KD = self.head_dim
        groups = H // KVH  # 12

        # 1) Dense linear for Q, K, V: out [B, S, H]
        Q = torch.empty((B, S, H), device=device, dtype=hidden_states.dtype)
        K = torch.empty((B, S, KVH), device=device, dtype=hidden_states.dtype)
        V = torch.empty((B, S, KVH), device=device, dtype=hidden_states.dtype)

        grid_linear = (B, H, S)
        triton_linear_bsh[grid_linear](
            hidden_states, q_proj_weight, q_proj_bias, Q,
            B, S, H, KD,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            q_proj_weight.stride(0), q_proj_weight.stride(1),
            Q.stride(0), Q.stride(1), Q.stride(2),
            BLOCK_K=KD, num_warps=4, num_stages=2
        )

        triton_linear_bsh[grid_linear](
            hidden_states, k_proj_weight, k_proj_bias, K,
            B, S, KVH, KD,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            k_proj_weight.stride(0), k_proj_weight.stride(1),
            K.stride(0), K.stride(1), K.stride(2),
            BLOCK_K=KD, num_warps=4, num_stages=2
        )

        triton_linear_bsh[grid_linear](
            hidden_states, v_proj_weight, v_proj_bias, V,
            B, S, KVH, KD,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            v_proj_weight.stride(0), v_proj_weight.stride(1),
            V.stride(0), V.stride(1), V.stride(2),
            BLOCK_K=KD, num_warps=4, num_stages=2
        )

        # 2) RMSNorm on Q and K
        Q_n = torch.empty_like(Q)
        K_n = torch.empty_like(K)

        grid_rms_q = (B, H)
        triton_rmsnorm_row[grid_rms_q](
            Q, Q_n, q_norm_weight, self.rms_norm_eps, B, H, S,
            Q.stride(0), Q.stride(2),
            Q_n.stride(0), Q_n.stride(1),
            BLOCK_D=KD, num_warps=4, num_stages=2
        )

        grid_rms_k = (B, KVH)
        triton_rmsnorm_row[grid_rms_k](
            K, K_n, k_norm_weight, self.rms_norm_eps, B, KVH, S,
            K.stride(0), K.stride(2),
            K_n.stride(0), K_n.stride(1),
            BLOCK_D=KD, num_warps=4, num_stages=2
        )

        # 3) Apply RoPE to Q and K
        Q_rot = torch.empty_like(Q_n)
        K_rot = torch.empty_like(K_n)
        grid_rope = (B, H, S)
        triton_rope_row[grid_rope](
            Q_n, cos, sin, Q_rot,
            B, H, S, KD,
            Q_n.stride(0), Q_n.stride(1), Q_n.stride(2),
            cos.stride(0), cos.stride(1),
            sin.stride(0), sin.stride(1),
            Q_rot.stride(0), Q_rot.stride(1), Q_rot.stride(2),
            BLOCK_D=KD, num_warps=4, num_stages=2
        )

        triton_rope_row[grid_rope](
            K_n, cos, sin, K_rot,
            B, KVH, S, KD,
            K_n.stride(0), K_n.stride(1), K_n.stride(2),
            cos.stride(0), cos.stride(1),
            sin.stride(0), sin.stride(1),
            K_rot.stride(0), K_rot.stride(1), K_rot.stride(2),
            BLOCK_D=KD, num_warps=4, num_stages=2
        )

        # 4) Expand KV from KVH to H via groups
        K_exp = torch.empty((B, H, S, KD), device=device, dtype=hidden_states.dtype)
        V_exp = torch.empty((B, H, S, KD), device=device, dtype=hidden_states.dtype)

        # We need strides for [B, KVH, S, KD] and [B, H, S, KD]
        grid_expand = (B, KVH, groups, S)
        triton_expand_kv[grid_expand](
            K_rot, V, K_exp, V_exp,
            B, S, KVH, KD,
            K_rot.stride(0), K_rot.stride(1), K_rot.stride(2), K_rot.stride(3),
            V.stride(0), V.stride(1), V.stride(2), V.stride(3),
            K_exp.stride(0), K_exp.stride(1), K_exp.stride(2), K_exp.stride(3),
            V_exp.stride(0), V_exp.stride(1), V_exp.stride(2), V_exp.stride(3),
            GROUPS=groups, num_warps=4, num_stages=2
        )

        # 5) Attention compute: out [B, H, S]
        Out = torch.empty((B, H, S), device=device, dtype=hidden_states.dtype)

        grid_attn = (B, H)
        triton_attention_compute[grid_attn](
            Q_rot, K_exp, V_exp, Out,
            B, S, H, KD,
            Q_rot.stride(0), Q_rot.stride(1), Q_rot.stride(2), Q_rot.stride(3),
            K_exp.stride(0), K_exp.stride(1), K_exp.stride(2), K_exp.stride(3),
            V_exp.stride(0), V_exp.stride(1), V_exp.stride(2), V_exp.stride(3),
            Out.stride(0), Out.stride(1),
            BLOCK_I=128, BLOCK_J=128, num_warps=4, num_stages=2
        )

        # 6) Output projection: Out -> [B, S, H]
        Out_proj = torch.empty((B, S, H), device=device, dtype=hidden_states.dtype)
        grid_linear_out = (B, H, S)
        triton_linear_bsh_out[grid_linear_out](
            Out, o_proj_weight, Out_proj,
            B, S, H, KD,
            Out.stride(0), Out.stride(1), Out.stride(2),
            o_proj_weight.stride(0), o_proj_weight.stride(1),
            Out_proj.stride(0), Out_proj.stride(1), Out_proj.stride(2),
            BLOCK_K=KD, num_warps=4, num_stages=2
        )

        # Final shape: [B, S, H * KD] -> here H=96, KD=128 => 12288, but original uses num_attention_heads * head_dim => return [B, S, 12288]
        # However, original function returns [B, S, num_attention_heads * head_dim] which is [B, S, 12288].
        return Out_proj

# End of ModelNew


def run(*args):
    return ModelNew()(*args)
