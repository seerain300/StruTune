import torch
import triton
import triton.language as tl


# =========================
# Triton kernels
# =========================

# 1) Batched GEMM (no bias): C[M, N] = A[M, K] @ B[K, N], where B is weight^T
@triton.jit
def triton_batched_gemm_no_bias(
    A_ptr,        # *fp32, [M, K]
    B_ptr,        # *fp32, [K, N]
    C_ptr,        # *fp32, [M, N]
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am, stride_ak,   # A strides (row, col)
    stride_bk, stride_bn,   # B strides (row=K, col=N)
    stride_cm, stride_cn,   # C strides (row, col)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a = tl.load(
            A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak),
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0
        )  # [BLOCK_M, BLOCK_K]
        b = tl.load(
            B_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn),
            mask=(offs_k[:, None] < K) & (offs_n[None, :] < N),
            other=0.0
        )  # [BLOCK_K, BLOCK_N]
        acc += tl.dot(a, b)
    tl.store(
        C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn),
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N)
    )


# 2) RMSNorm per head: out[b, s, h, :] = weight[h, :] * x / sqrt(mean(x^2) + eps)
# This kernel normalizes over head_dim for each (b, s, h).
@triton.jit
def triton_rmsnorm(
    X_ptr,        # *fp32, [B*S, H] (flattened batch*seq positions, head_dim dim)
    W_ptr,        # *fp32, [H] (weight vector per head)
    Out_ptr,      # *fp32, [B*S, H]
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
    stride_xb, stride_xs, stride_xh,   # X strides
    stride_oh,                       # Out stride for head dim
    eps: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_bs = tl.program_id(0)  # each program handles one (b, s) pair
    # Compute base offsets
    # Note: We assume X and Out are contiguous in last dim; we pass strides for generality.
    # Each program processes the entire head_dim vector for one position.
    # Compute sum of squares
    sum_sq = tl.zeros((), dtype=tl.float32)
    for h0 in range(0, H, BLOCK_H):
        offs_h = h0 + tl.arange(0, BLOCK_H)
        x_vec = tl.load(
            X_ptr + pid_bs * stride_xb + offs_h * stride_xh,
            mask=offs_h < H,
            other=0.0
        )
        sum_sq += tl.sum(x_vec * x_vec, axis=0)
    mean = sum_sq / H
    inv_rms = 1.0 / tl.sqrt(mean + eps)

    # Normalize and scale by weight
    for h0 in range(0, H, BLOCK_H):
        offs_h = h0 + tl.arange(0, BLOCK_H)
        x_vec = tl.load(
            X_ptr + pid_bs * stride_xb + offs_h * stride_xh,
            mask=offs_h < H,
            other=0.0
        )
        w_vec = tl.load(W_ptr + offs_h, mask=offs_h < H, other=0.0)
        y_vec = x_vec * inv_rms * w_vec
        tl.store(Out_ptr + pid_bs * stride_xb + offs_h * stride_oh, y_vec, mask=offs_h < H)


# 3) RotPE rotation: in place on X [B, S, H], where H is even (128), rotate half:
#   after rotation: X[..., :H/2] = cos * X + sin * (-X[..., H/2:]) and
#   X[..., H/2:] = cos * X[..., H/2:] + sin * X[..., :H/2]
@triton.jit
def triton_rotate_half(
    X_ptr,        # *fp32, [B*S, H]
    Cos_ptr,      # *fp32, [H] (cos values per dim)
    Sin_ptr,      # *fp32, [H] (sin values per dim)
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
    stride_xb, stride_xs, stride_xh,
):
    pid_bs = tl.program_id(0)
    for h0 in range(0, H, 32):
        offs_h = h0 + tl.arange(0, 32)
        x = tl.load(X_ptr + pid_bs * stride_xb + offs_h * stride_xh, mask=offs_h < H, other=0.0)
        cosv = tl.load(Cos_ptr + offs_h, mask=offs_h < H, other=0.0)
        sinv = tl.load(Sin_ptr + offs_h, mask=offs_h < H, other=0.0)

        half = H // 2
        x_first = x[:half]
        x_second = x[half:]

        # After rotation:
        # y_first = cos * x_first + sin * (-x_second)
        # y_second = cos * x_second + sin * x_first
        y_first = cosv[:half] * x_first - sinv[:half] * x_second
        y_second = cosv[half:] * x_second + sinv[half:] * x_first

        y = tl.zeros((H,), dtype=tl.float32)
        y[:half] = y_first
        y[half:] = y_second

        tl.store(X_ptr + pid_bs * stride_xb + offs_h * stride_xh, y, mask=offs_h < H)


# 4) Grouped Query Attention repeat: repeat each of num_key_value_heads=8 across
#    num_key_value_groups=12 to form num_attention_heads=96 heads. Output shape [B, S, 128].
#    This kernel writes repeated slices of input Q/K into output Q_exp/K_exp at positions
#    corresponding to groups. We assume out_K has already been allocated and zeroed.
@triton.jit
# out shape [B, S, H], in shape [B, S, H]
def triton_grouped_q_repeat(
    In_ptr,       # *fp32, [B*S, H] (no batch, contiguous)
    Out_ptr,      # *fp32, [B*num_attention_heads*S, H]
    B: tl.constexpr, S: tl.constexpr, H: tl.constexpr,
    num_key_value_heads: tl.constexpr, num_key_value_groups: tl.constexpr, num_attention_heads: tl.constexpr,
    stride_ib, stride_is, stride_ih,   # In strides
    stride_ob, stride_os, stride_oh,   # Out strides
):
    # Flatten (b, s) into one dimension
    total_bs = B * S
    for m in range(0, total_bs * num_attention_heads):
        b = m // (num_attention_heads * S)
        s = (m % (num_attention_heads * S)) // num_attention_heads
        head = m % num_attention_heads
        # Determine source head (KV head) and group index
        # head = orig_h * num_key_value_groups + group
        group = head % num_key_value_groups
        orig_h = head // num_key_value_groups  # in [0, num_key_value_heads-1]
        # Copy In[b, s, orig_h, :] into Out[b, s, head, :]
        # Compute base offsets
        in_off = b * stride_ib + s * stride_is + 0 * stride_ih  # since we load vector along h, but we pass stride_ih as H stride
        # Note: In is [B*S, H]; Out is [B*num_attention_heads*S, H]
        out_off = b * stride_ob + s * stride_os + head * stride_oh
        # Copy entire H vector
        for h0 in range(0, H, 32):
            offs_h = h0 + tl.arange(0, 32)
            x = tl.load(In_ptr + in_off + offs_h * stride_ih, mask=offs_h < H, other=0.0)
            tl.store(Out_ptr + out_off + offs_h * stride_oh, x, mask=offs_h < H)


# 5) Compute attention scores: S[M, N] = Q[M, D] @ K^T[N, D], where M = B * num_attention_heads, N = S.
#    Each program computes one row (m) and one tile of columns (BLOCK_N), accumulating across D in BLOCK_K chunks.
@triton.jit
def triton_score_matmul(
    Q_ptr,        # *fp32, [M, D]
    K_ptr,        # *fp32, [N, D]
    S_ptr,        # *fp32, [M, N]
    M: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    stride_qm, stride_qd,    # Q strides
    stride_kn, stride_kd,    # K strides (rows=N, cols=D)
    stride_sm, stride_sn,    # S strides
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for d0 in range(0, D, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        q = tl.load(
            Q_ptr + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd),
            mask=(offs_m[:, None] < M) & (offs_d[None, :] < D),
            other=0.0
        )  # [BLOCK_M, BLOCK_D]
        k = tl.load(
            K_ptr + (offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd),
            mask=(offs_n[:, None] < N) & (offs_d[None, :] < D),
            other=0.0
        )  # [BLOCK_D, BLOCK_N]
        acc += tl.dot(q, k)
    tl.store(
        S_ptr + (offs_m[:, None] * stride_sm + offs_n[None, :] * stride_sn),
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N)
    )


# 6) Row-wise softmax over N (sequence dim): S[M, N], output normalized S_out[M, N]
@triton.jit
def triton_softmax_rowwise(
    In_ptr,       # *fp32, [M, N]
    Out_ptr,      # *fp32, [M, N]
    M: tl.constexpr, N: tl.constexpr,
    stride_im, stride_in,   # In strides
    stride_om, stride_on,   # Out strides
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    row_start = pid_m * stride_im
    # Compute row max
    max_val = tl.full((), -1e20, dtype=tl.float32)
    for n0 in range(0, N, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        x = tl.load(In_ptr + row_start + offs_n * stride_in, mask=offs_n < N, other=-1e20)
        current_max = tl.max(x, axis=0)
        max_val = tl.maximum(max_val, current_max)
    # Compute sum of exp(x - max)
    sum_exp = tl.zeros((), dtype=tl.float32)
    for n0 in range(0, N, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        x = tl.load(In_ptr + row_start + offs_n * stride_in, mask=offs_n < N, other=-1e20)
        ex = tl.exp(x - max_val)
        sum_exp += tl.sum(ex, axis=0)
    # Normalize and store
    for n0 in range(0, N, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        x = tl.load(In_ptr + row_start + offs_n * stride_in, mask=offs_n < N, other=-1e20)
        y = tl.exp(x - max_val) / sum_exp
        tl.store(Out_ptr + pid_m * stride_om + offs_n * stride_on, y, mask=offs_n < N)


# 7) Final output: S_out[M, N] @ V_exp[M, D] where V_exp is repeated KV for each attention head,
#    D = head_dim (128). Each program computes one row output across D.
@triton.jit
def triton_final_output_rowwise(
    S_ptr,        # *fp32, [M, N] = attention scores
    V_ptr,        # *fp32, [M, D] (expanded V per head)
    Out_ptr,      # *fp32, [M, D]
    M: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    stride_sm, stride_sn,    # S strides
    stride_vm, stride_vd,    # V strides
    stride_om, stride_od,    # Out strides
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    row_s = S_ptr + pid_m * stride_sm
    row_v = V_ptr + pid_m * stride_vm
    row_o = Out_ptr + pid_m * stride_om

    for d0 in range(0, D, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        # Load vector s[pid_m, :] over N dimension
        sum_v = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for n0 in range(0, N, 32):
            offs_n = n0 + tl.arange(0, 32)
            s_vec = tl.load(row_s + offs_n * stride_sn, mask=offs_n < N, other=0.0)  # [32]
            v_vec = tl.load(row_v + (offs_d[:, None] * stride_vd + offs_n[None, :] * stride_vd),
                            mask=(offs_d[:, None] < D) & (offs_n[None, :] < N),
                            other=0.0)  # [BLOCK_D, 32]
            # sum over N: s_vec[:, None] * v_vec[None, :]
            # We need to multiply each s in the block with corresponding v across N; since N is small, we loop:
            # For each n in offs_n:
            #   sum_v += s_vec[n] * v_vec[:, n]
            # Implement outer product-like accumulation:
            # We'll do this by iterating n index:
            for nn in range(32):
                n_idx = n0 + nn
                # s_val for this chunk
                if n_idx < N:
                    s_val = s_vec[nn]
                    v_chunk = tl.load(row_v + (offs_d * stride_vd + n_idx * stride_vd),
                                      mask=offs_d < D,
                                      other=0.0)  # [BLOCK_D]
                    sum_v += s_val * v_chunk
        tl.store(row_o + offs_d * stride_od, sum_v, mask=offs_d < D)


# =========================
# ModelNew: Triton-only forward
# =========================

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # We assume the following constants from the original code:
        self.num_attention_heads = 96
        self.num_key_value_heads = 8
        self.head_dim = 128
        self.num_key_value_groups = 12
        self.rms_norm_eps = 1e-8

    def forward(self, hidden_states: torch.Tensor,
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
                sin: torch.Tensor):
        # hidden_states: [B, S, H]
        B, S, H = hidden_states.shape
        assert H == self.head_dim, "Hidden dim must equal head_dim (128)."

        # 1) Dense projections (no bias)
        # A: [B*S, H], W: [H, H] (transposed weight)
        # We create Wt for each projection (q, k, v, output). For simplicity, we use provided weights directly.

        # Use Triton GEMM for Q, K, V, and final output. We need B^T weights.
        def batched_gemm_no_bias(A, B):  # B is weight, returns C [M, N] where M=B*S, N=H
            M = A.shape[0]
            N = B.shape[1]
            K = B.shape[0]
            # Allocate output
            C = torch.empty((M, N), device=A.device, dtype=torch.float32)
            # Strides for A [M, K], B [K, N], C [M, N]
            stride_am, stride_ak = A.stride(0), A.stride(1)
            stride_bk, stride_bn = B.stride(0), B.stride(1)
            stride_cm, stride_cn = C.stride(0), C.stride(1)
            # Launch grid
            grid = (triton.cdiv(M, 64), triton.cdiv(N, 64))
            triton_batched_gemm_no_bias[grid](
                A, B, C,
                M, N, K,
                stride_am, stride_ak,
                stride_bk, stride_bn,
                stride_cm, stride_cn,
                BLOCK_M=64, BLOCK_N=64, BLOCK_K=32
            )
            return C

        # Convert hidden_states and weights to fp32 and contiguous
        hs = hidden_states.to(torch.float32).contiguous()
        # Flatten M = B * num_attention_heads for initial projections (but here we need [B*S, H]).
        # Build A for Q, K, V:
        # A_q = hs, B_q = q_proj_weight^T
        Wt_q = q_proj_weight.t().contiguous()  # [H, H]
        Wt_k = k_proj_weight.t().contiguous()
        Wt_v = v_proj_weight.t().contiguous()

        # Compute Q, K, V
        A_q = hs
        A_k = hs
        A_v = hs
        Q = batched_gemm_no_bias(A_q, Wt_q)        # [B*S, H]
        K = batched_gemm_no_bias(A_k, Wt_k)        # [B*S, H]
        V = batched_gemm_no_bias(A_v, Wt_v)        # [B*S, H]

        # 2) RMSNorm on Q and K
        # Prepare outputs
        Q_norm = torch.empty_like(Q)
        K_norm = torch.empty_like(K)
        # For RMSNorm, each row corresponds to (b, s, h). But Q,K are [B*S, H], so one program per (b*s) row.
        grid_rms = (B * S,)
        # q_norm_weight and k_norm_weight are [H]
        triton_rmsnorm[grid_rms](
            Q, q_norm_weight, Q_norm,
            B, S, H,
            Q.stride(0), Q.stride(1), Q.stride(2),
            Q_norm.stride(2),  # stride along head_dim (last dim)
            self.rms_norm_eps,
            BLOCK_H=64
        )
        triton_rmsnorm[grid_rms](
            K, k_norm_weight, K_norm,
            B, S, H,
            K.stride(0), K.stride(1), K.stride(2),
            K_norm.stride(2),
            self.rms_norm_eps,
            BLOCK_H=64
        )

        # 3) RotPE on Q and K: rotate half
        # cos, sin are [H], we pass as fp32
        cos_f = cos.to(torch.float32)
        sin_f = sin.to(torch.float32)
        triton_rotate_half[(B * S,)](
            Q_norm, cos_f, sin_f,
            B, S, H,
            Q_norm.stride(0), Q_norm.stride(1), Q_norm.stride(2),
        )
        triton_rotate_half[(B * S,)](
            K_norm, cos_f, sin_f,
            B, S, H,
            K_norm.stride(0), K_norm.stride(1), K_norm.stride(2),
        )

        # 4) Grouped Query Attention repeat for K and V to 96 heads
        # Allocate expanded K and V: [B, S, 96*H]
        K_exp = torch.empty((B, S, self.num_attention_heads * self.head_dim), device=hidden_states.device, dtype=torch.float32)
        V_exp = torch.empty((B, S, self.num_attention_heads * self.head_dim), device=hidden_states.device, dtype=torch.float32)

        # Flatten input pointers: K_flat [B*S, H], V_flat [B*S, H]
        K_flat = K_norm.view(B * S, H)
        V_flat = V_norm.view(B * S, H)
        # Out tensors flattened: [B*S*num_attention_heads, H]
        OutK = K_exp.view(B * S * self.num_attention_heads, self.head_dim)
        OutV = V_exp.view(B * S * self.num_attention_heads, self.head_dim)

        triton_grouped_q_repeat[(B * S * self.num_attention_heads,)](
            K_flat, OutK,
            B, S, H,
            self.num_key_value_heads, self.num_key_value_groups, self.num_attention_heads,
            K_flat.stride(0), K_flat.stride(1), K_flat.stride(2),
            OutK.stride(0), OutK.stride(1), OutK.stride(2),
        )
        triton_grouped_q_repeat[(B * S * self.num_attention_heads,)](
            V_flat, OutV,
            B, S, H,
            self.num_key_value_heads, self.num_key_value_groups, self.num_attention_heads,
            V_flat.stride(0), V_flat.stride(1), V_flat.stride(2),
            OutV.stride(0), OutV.stride(1), OutV.stride(2),
        )
        K_exp = K_exp.contiguous()
        V_exp = V_exp.contiguous()

        # 5) Compute attention scores S[M, N] = Q[M, H] @ K^T[N, H], where M = B*num_attention_heads, N = S
        # Flatten Q to [M, H], K_exp to [N, H]
        M_attn = B * self.num_attention_heads
        Q_flat = Q_norm.view(M_attn, H)
        S_scores = torch.empty((M_attn, S), device=hidden_states.device, dtype=torch.float32)
        grid_score = (triton.cdiv(M_attn, 64), triton.cdiv(S, 64))
        triton_score_matmul[grid_score](
            Q_flat, K_exp, S_scores,
            M_attn, S, H,
            Q_flat.stride(0), Q_flat.stride(1),
            K_exp.stride(0), K_exp.stride(1),
            S_scores.stride(0), S_scores.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_D=32
        )

        # 6) Softmax over sequence dimension per row
        S_scores_soft = torch.empty_like(S_scores)
        grid_softmax = (M_attn,)
        triton_softmax_rowwise[grid_softmax](
            S_scores, S_scores_soft,
            M_attn, S,
            S_scores.stride(0), S_scores.stride(1),
            S_scores_soft.stride(0), S_scores_soft.stride(1),
            BLOCK_N=64
        )

        # 7) Final output: softmax @ V_exp
        # Out shape [M_attn, H], V_exp shape [M_attn, H]
        attn_out = torch.empty((M_attn, H), device=hidden_states.device, dtype=torch.float32)
        grid_final = (M_attn,)
        triton_final_output_rowwise[grid_final](
            S_scores_soft, V_exp, attn_out,
            M_attn, S, H,
            S_scores_soft.stride(0), S_scores_soft.stride(1),
            V_exp.stride(0), V_exp.stride(1),
            attn_out.stride(0), attn_out.stride(1),
            BLOCK_D=64
        )

        # 8) Output projection (no bias): attn_out @ o_proj_weight^T
        # A_proj: [M_attn*H, H] flattened into [M_attn, H]
        # Bt: [H, num_attention_heads*head_dim] (not used here since attn_out is [M_attn, H], and we multiply by o_proj_weight)
        # We need to compute attn_out @ o_proj_weight^T -> [M_attn, H]
        # But o_proj_weight is [H, hidden_dim]. We can do it with Triton GEMM:
        M_proj = M_attn
        K_proj = H
        N_proj = H  # hidden_dim is H=128
        A_proj = attn_out  # already [M_attn, H]
        Bt_proj = o_proj_weight.t().contiguous()  # [H, hidden_dim]
        output = torch.empty((M_proj, N_proj), device=hidden_states.device, dtype=torch.float32)
        grid_proj = (triton.cdiv(M_proj, 64), triton.cdiv(N_proj, 64))
        triton_batched_gemm_no_bias[grid_proj](
            A_proj, Bt_proj, output,
            M_proj, N_proj, K_proj,
            A_proj.stride(0), A_proj.stride(1),
            Bt_proj.stride(0), Bt_proj.stride(1),
            output.stride(0), output.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=32
        )

        # Reshape to [B, S, hidden_dim] = [B, S, H]
        output = output.view(B, S, H)
        return output


def run(*args):
    return ModelNew()(*args)
