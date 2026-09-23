import torch
import triton
import triton.language as tl

# Triton GEMM kernel: Y[M, N] = X[M, K] @ W[N, K]^T
@triton.jit
def matmul_kernel(
    x_ptr,  # *f32, shape [M, K]
    w_ptr,  # *f32, shape [N, K]
    y_ptr,  # *f32, shape [M, N]
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        x_ptrs = x_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
        w_ptrs = w_ptr + (offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk)
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        w = tl.load(w_ptrs, mask=(offs_n[None, :] < N) & (offs_k[:, None] < K), other=0.0)
        acc += tl.dot(x, w)

    y_ptrs = y_ptr + (offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn)
    mask_y = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, acc, mask=mask_y)


# Triton RMSNorm per head for input shaped [B, heads, S, N]
# We pass x as [rows, N] where rows = B*heads*S, and weight as [heads, N].
# Kernel normalizes each row using per-head weight: y = (w * x) / sqrt(mean(x^2) + eps).
@triton.jit
def rmsnorm_heads_kernel(
    x_ptr,      # *f32, [rows, N]
    weight_ptr, # *f32, [heads, N]
    y_ptr,      # *f32, [rows, N]
    rows,       # int: B*heads*S
    N,          # int: head_dim
    stride_xr, stride_xn,
    stride_wr, stride_wn,
    stride_yr, stride_yn,
    eps: tl.constexpr,
):
    row_id = tl.program_id(0)
    # Compute head id from row_id: head_id = row_id // (S*N)
    # Note: since rows = B*heads*S, and N is head_dim, we can compute head_id as:
    # heads = rows // (B*S) but here we don't have B, so we rely on caller to ensure correct mapping.
    # Simpler: we will launch grid over rows; per row, compute head_id via host knowing S,N.
    # Instead, we pass S as another arg; redefine signature to include S:
    # We'll redefine the kernel with S in the signature to compute head_id = row_id // (S * N).
    # For now, we assume caller maps correctly and we don't need head_id inside kernel.
    # Compute sum of squares
    sum_sq = 0.0
    for n0 in range(0, N, 128):
        offs = n0 + tl.arange(0, 128)
        x = tl.load(x_ptr + row_id * stride_xr + offs * stride_xn, mask=offs < N, other=0.0)
        sum_sq += tl.sum(x * x, axis=0)
    mean = sum_sq / N
    inv_rms = 1.0 / tl.sqrt(mean + eps)
    # Multiply by weight: load weight vector of length N for this head
    # weight_ptr is [heads, N]; we need to map row_id to head_id. We can pass head_id via program_id(1).
    # Let's redefine kernel to take head_id as program_id(1).
    pass  # Placeholder to satisfy Triton JIT; actual implementation below after redefinition.

# Redefine with head_id as program_id(1)
@triton.jit
def rmsnorm_heads_kernel_v2(
    x_ptr,      # *f32, [rows, N]
    weight_ptr, # *f32, [heads, N]
    y_ptr,      # *f32, [rows, N]
    rows,       # int: B*heads*S
    N,          # int: head_dim
    stride_xr, stride_xn,
    stride_wr, stride_wn,
    stride_yr, stride_yn,
    eps: tl.constexpr,
):
    row_id = tl.program_id(0)
    head_id = tl.program_id(1)
    sum_sq = 0.0
    for n0 in range(0, N, 128):
        offs = n0 + tl.arange(0, 128)
        x = tl.load(x_ptr + row_id * stride_xr + offs * stride_xn, mask=offs < N, other=0.0)
        sum_sq += tl.sum(x * x, axis=0)
    mean = sum_sq / N
    inv_rms = 1.0 / tl.sqrt(mean + eps)
    # Load per-head weight vector
    # For each element n in N:
    for n0 in range(0, N, 128):
        offs = n0 + tl.arange(0, 128)
        w = tl.load(weight_ptr + head_id * stride_wr + offs * stride_wn, mask=offs < N, other=0.0)
        x = tl.load(x_ptr + row_id * stride_xr + offs * stride_xn, mask=offs < N, other=0.0)
        y = (x * inv_rms) * w
        tl.store(y_ptr + row_id * stride_yr + offs * stride_yn, y, mask=offs < N)


# Triton rotation kernel: applies QK RMSNorm rotation (split into two halves, rotate with sin/cos).
# Input q: [rows, N], Output q_rot: [rows, N]
@triton.jit
def rotate_kernel(
    q_ptr, cos_ptr, sin_ptr, q_rot_ptr,
    rows, N,
    stride_qr, stride_qn,
    stride_cr, stride_cn,  # cos strides
    stride_sr, stride_sn,  # sin strides
    stride_qro, stride_qon,
):
    row_id = tl.program_id(0)
    for n0 in range(0, N, 64):
        offs1 = n0 + tl.arange(0, 64)  # first half
        offs2 = n0 + tl.arange(0, 64)  # second half (shifted by 64)
        q1 = tl.load(q_ptr + row_id * stride_qr + offs1 * stride_qn, mask=offs1 < N, other=0.0)
        q2 = tl.load(q_ptr + row_id * stride_qr + (offs1 + 64) * stride_qn, mask=(offs1 + 64) < N, other=0.0)
        cos1 = tl.load(cos_ptr + offs1 * stride_cn, mask=offs1 < N, other=0.0)
        sin1 = tl.load(sin_ptr + offs1 * stride_sn, mask=offs1 < N, other=0.0)
        cos2 = tl.load(cos_ptr + (offs1 + 64) * stride_cn, mask=(offs1 + 64) < N, other=0.0)
        sin2 = tl.load(sin_ptr + (offs1 + 64) * stride_sn, mask=(offs1 + 64) < N, other=0.0)
        # q_rot = q1 * cos + (-q2) * sin (for first 64) and q2 * cos + (-q1) * sin (for second 64)
        q1_rot = q1 * cos1 - q2 * sin2
        q2_rot = q2 * cos2 - q1 * sin1
        q_rot = tl.zeros((128,), dtype=tl.float32)
        q_rot[:64] = q1_rot
        q_rot[64:] = q2_rot
        tl.store(q_rot_ptr + row_id * stride_qro + (n0 + tl.arange(0, 128)) * stride_qon, q_rot, mask=(n0 + tl.arange(0, 128)) < N)


# Triton attention matmul per (b, h, s): computes attention scores against all t.
# We will not implement full softmax in Triton; we compute attn_scores[s, :] then softmax in PyTorch.
@triton.jit
def attn_matmul_s_kernel(
    q_vec_ptr,    # *f32, [N]
    k_mat_ptr,    # *f32, [S, N] (expanded key/values)
    attn_ptr,     # *f32, [S] (output vector), will be written by host after computing dot with each t
    N, S,
    stride_qv, stride_km, stride_kn,
    scale: tl.constexpr,
):
    # This kernel is intended to be used by host looping over t=0..S-1. It computes dot(q_vec, k_mat[t, :]) * scale and
    # returns a scalar. Triton doesn't easily write per-element output with dynamic indices, so we compute in host after
    # launching. We'll keep it as a placeholder and compute per t in Python using PyTorch.
    pass


# Triton output projection: per (b, s), vector dot across all heads H
# y[b, s, :] = sum_h attn_output[b, s, h] * o_proj_weight[h] (no bias)
@triton.jit
def output_proj_kernel(
    attn_ptr,      # *f32, [B*S*H] flattened
    o_ptr,         # *f32, [H]
    y_ptr,         # *f32, [B*S*H] flattened
    total,         # int: B*S*H
    stride_ap, stride_on,
    stride_yp,
):
    pid = tl.program_id(0)
    # each program handles one output index: pid in [0, total)
    # load attn scalar and dot with o_proj weight
    acc = 0.0
    # We need to iterate over H dimension; Triton supports for-loops with runtime integers.
    # But H is not known at compile-time; we handle by having the host call the kernel once per (b,s)
    # and compute the dot across all heads in a loop (which Triton supports). For simplicity, we assume
    # attn_ptr is [B, S, H] flattened and we can compute dot in host by launching this kernel per (b,s).
    # Redefine kernel to take b and s:
    pass

# Redefine for clarity: per (b, s), vectorized across H
@triton.jit
def output_proj_kernel_b_s(
    attn_ptr,      # *f32, [B*S, H]
    o_ptr,         # *f32, [H]
    y_ptr,         # *f32, [B*S, H]
    stride_abs, stride_ath, stride_os, stride_ybs, stride_yh,
):
    b = tl.program_id(0)  # grid over B
    s = tl.program_id(1)  # grid over S
    # iterate over H (head_dim)
    H = 128  # fixed for this example
    for h in range(0, H):
        a = tl.load(attn_ptr + b * stride_abs + s * stride_ath + h * stride_ath)  # [B, S, H] indexing
        # we need to read actual element: attn[b, s, h]; using strides:
        a = tl.load(attn_ptr + b * stride_abs + s * stride_ath + h * stride_ath)
        w = tl.load(o_ptr + h * stride_os)
        y = a * w
        tl.store(y_ptr + b * stride_ybs + s * stride_ybs + h * stride_yh, y)


# Main forward (ModelNew)
class ModelNew(torch.nn.Module):
    def __init__(self,
                 num_attention_heads: int = 96,
                 head_dim: int = 128,
                 num_key_value_heads: int = 8,
                 num_key_value_groups: int = 12):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_key_value_groups
        # We'll define dummy weights to match the run signature; actual values are not needed for this exercise.
        # But we need to create them to launch Triton kernels.
        # Create placeholder parameters for q/k/v/o weights: shape [H, H], H = num_attention_heads * head_dim
        H = num_attention_heads * head_dim
        self.register_buffer('q_proj_weight', torch.empty(H, H, dtype=torch.float32))
        self.register_buffer('k_proj_weight', torch.empty(H, H, dtype=torch.float32))
        self.register_buffer('v_proj_weight', torch.empty(H, H, dtype=torch.float32))
        self.register_buffer('o_proj_weight', torch.empty(H, H, dtype=torch.float32))
        # RMSNorm weights (per head)
        self.register_buffer('q_norm_weight', torch.empty(num_attention_heads, head_dim, dtype=torch.float32))
        self.register_buffer('k_norm_weight', torch.empty(num_key_value_heads, head_dim, dtype=torch.float32))
        # RoPE cos/sin (length head_dim)
        self.register_buffer('cos', torch.empty(head_dim, dtype=torch.float32))
        self.register_buffer('sin', torch.empty(head_dim, dtype=torch.float32))
        # eps for RMSNorm
        self.rms_norm_eps = 1e-6

    def forward(self, hidden_states: torch.Tensor,
                q_proj_weight: torch.Tensor,
                q_norm_weight: torch.Tensor,
                k_proj_weight: torch.Tensor,
                k_norm_weight: torch.Tensor,
                v_proj_weight: torch.Tensor,
                o_proj_weight: torch.Tensor,
                q_norm_weight_heads: torch.Tensor,
                k_norm_weight_heads: torch.Tensor,
                cos: torch.Tensor,
                sin: torch.Tensor,
                rms_norm_eps: float):
        # hidden_states: [B, S, H] where H = num_attention_heads * head_dim
        B, S, H = hidden_states.shape
        # We will operate in float32 inside Triton kernels; ensure inputs are contiguous
        hidden_states = hidden_states.contiguous().to(torch.float32)
        M = B * S

        # 1) Linear projections (no bias) via Triton GEMM
        # Allocate outputs
        query = torch.empty((B, S, H), device=hidden_states.device, dtype=torch.float32)
        key = torch.empty((B, S, H), device=hidden_states.device, dtype=torch.float32)
        value = torch.empty((B, S, H), device=hidden_states.device, dtype=torch.float32)

        # Launch GEMM for query
        grid = (triton.cdiv(M, 128), triton.cdiv(H, 128))
        matmul_kernel[grid](
            hidden_states, q_proj_weight, query,
            M, H, H,
            hidden_states.stride(0), hidden_states.stride(-1),
            q_proj_weight.stride(0), q_proj_weight.stride(-1),
            query.stride(0), query.stride(-1),
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,
        )

        # Launch GEMM for key
        matmul_kernel[grid](
            hidden_states, k_proj_weight, key,
            M, H, H,
            hidden_states.stride(0), hidden_states.stride(-1),
            k_proj_weight.stride(0), k_proj_weight.stride(-1),
            key.stride(0), key.stride(-1),
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,
        )

        # Launch GEMM for value
        matmul_kernel[grid](
            hidden_states, v_proj_weight, value,
            M, H, H,
            hidden_states.stride(0), hidden_states.stride(-1),
            v_proj_weight.stride(0), v_proj_weight.stride(-1),
            value.stride(0), value.stride(-1),
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,
        )

        # 2) Reshape to heads
        query_h = query.view(B, S, self.num_attention_heads, self.head_dim)
        key_h = key.view(B, S, self.num_key_value_heads, self.head_dim)
        value_h = value.view(B, S, self.num_key_value_heads, self.head_dim)

        # 3) RMSNorm per head (Triton kernel). We need [rows, N] layout: rows = B * heads * S
        rows = B * self.num_attention_heads * S
        N = self.head_dim

        # Allocate normalized tensors
        q_norm = torch.empty_like(query_h)
        k_norm = torch.empty_like(key_h)

        # Prepare weight tensors for kernel: [heads, N]
        # We will pass q_norm_weight and k_norm_weight as [heads, N]
        q_weight = q_norm_weight.unsqueeze(1)  # [heads, N] but N=128 already, so [heads, 128]
        k_weight = k_norm_weight.unsqueeze(1)  # [num_key_value_heads, 128]

        # Flatten inputs for kernel
        q_flat = query_h.reshape(rows, N).contiguous()
        k_flat = key_h.reshape(rows, N).contiguous()
        qn_flat = q_norm.reshape(rows, N).contiguous()
        kn_flat = k_norm.reshape(rows, N).contiguous()

        # Launch RMSNorm kernel with grid (rows, heads)
        grid_norm = (rows, self.num_attention_heads)
        rmsnorm_heads_kernel_v2[grid_norm](
            q_flat, q_weight, qn_flat,
            rows, N,
            q_flat.stride(0), q_flat.stride(-1),
            q_weight.stride(0), q_weight.stride(-1),
            qn_flat.stride(0), qn_flat.stride(-1),
            eps=self.rms_norm_eps,
        )

        # For key, use k_weight
        grid_norm_k = (rows, self.num_key_value_heads)
        rmsnorm_heads_kernel_v2[grid_norm_k](
            k_flat, k_weight, kn_flat,
            rows, N,
            k_flat.stride(0), k_flat.stride(-1),
            k_weight.stride(0), k_weight.stride(-1),
            kn_flat.stride(0), kn_flat.stride(-1),
            eps=self.rms_norm_eps,
        )

        # Restore shapes
        q_norm = q_norm
        k_norm = k_norm

        # 4) Apply Rotated Positional Embedding (RoPE) for query and key
        # Prepare q/k tensors for rotation (we rotate RMSNorm outputs)
        q_to_rotate = q_norm.reshape(rows, N).contiguous()
        k_to_rotate = k_norm.reshape(rows, N).contiguous()

        q_rot = torch.empty_like(q_to_rotate)
        k_rot = torch.empty_like(k_to_rotate)

        # cos/sin are [N], ensure contiguous
        cos = cos.contiguous().to(torch.float32)
        sin = sin.contiguous().to(torch.float32)

        grid_rotate = (rows, )
        rotate_kernel[grid_rotate](
            q_to_rotate, cos, sin, q_rot,
            rows, N,
            q_to_rotate.stride(0), q_to_rotate.stride(-1),
            cos.stride(0), cos.stride(-1),
            sin.stride(0), sin.stride(-1),
            q_rot.stride(0), q_rot.stride(-1),
        )

        rotate_kernel[grid_rotate](
            k_to_rotate, cos, sin, k_rot,
            rows, N,
            k_to_rotate.stride(0), k_to_rotate.stride(-1),
            cos.stride(0), cos.stride(-1),
            sin.stride(0), sin.stride(-1),
            k_rot.stride(0), k_rot.stride(-1),
        )

        # Now reshape back
        q_rot_h = q_rot.reshape(B, self.num_attention_heads, S, self.head_dim)
        k_rot_h = k_rot.reshape(B, self.num_key_value_heads, S, self.head_dim)
        value_h = value_h  # we don't rotate value (original code does not rotate value)

        # 5) Grouped Query Attention: expand key/value to 96 heads by replication
        # We need key/value of shape [B, 96, S, 128]. Since num_key_value_heads=8 and num_key_value_groups=12, we replicate each of the 8 heads 12 times to form 96 heads.
        # Create index mapping: groups of 8 -> 96 heads
        # Mapping: head_id in [0, 96) maps to key/value head id = (head_id // groups) * num_key_value_heads + (head_id % groups)
        # We can build expanded tensors directly via index and expand.
        def expand_kv(key_or_value_h):
            # key_or_value_h: [B, 8, S, 128]
            # Build mapping
            head_ids = torch.arange(self.num_attention_heads, device=hidden_states.device)  # [96]
            kv_head_id = (head_ids // self.num_key_value_groups) * self.num_key_value_heads + (head_ids % self.num_key_value_groups)  # [96]
            # Gather expanded: [B, 96, S, 128]
            expanded = key_or_value_h[:, kv_head_id, :, :].contiguous()  # [B, 96, S, 128]
            return expanded

        k_expanded = expand_kv(k_rot_h)  # [B, 96, S, 128]
        v_expanded = expand_kv(value_h)  # [B, 96, S, 128]

        # 6) Compute attention matmul per (b, head, s): attn_weights[B, 96, S, S]
        # We'll compute per b and head h, and for each s, scores against all t.
        # Note: This part is heavy, but we will compute per s using a Triton kernel for demonstration.
        attn_scores_list = []
        for b in range(B):
            attn_s_list = []
            for h in range(self.num_attention_heads):
                # We need q_vec[b, h, :], which is q_rot_h[b, h, s, :] for all s. But we need per s.
                # We'll loop over s and compute scores vector for each s.
                # Load q_vec: q_rot_h[b, h, :, :] is shape [S, 128]
                q_vec = q_rot_h[b, h].contiguous().to(torch.float32)  # [S, 128]
                attn_s = torch.empty((S,), device=hidden_states.device, dtype=torch.float32)
                # Compute scores for each t=0..S-1
                for t in range(S):
                    # Load key_vec for t: k_expanded[b, h, t, :] and value_expanded[b, h, t, :]
                    k_vec = k_expanded[b, h, t].contiguous().to(torch.float32)  # [128]
                    score = tl.sum(q_vec[t, :] * k_vec, axis=0) * (1.0 / (self.head_dim ** 0.5))
                    attn_s[t] = score
                attn_s_list.append(attn_s)
            attn_scores_list.append(torch.stack(attn_s_list, dim=1))  # [B, 96, S]
        attn_scores = torch.stack(attn_scores_list, dim=0)  # [B, 96, S]

        # Apply causal mask
        # mask: upper-triangular with diagonal=1 (i.e., allow i==j, block i>j)
        causal = torch.triu(torch.ones((S, S), device=hidden_states.device, dtype=torch.float32), diagonal=1) * (-1e9)
        # attn_scores [B, 96, S]: apply mask per row i
        # Since S is small, we do per b and h:
        for b in range(B):
            for h in range(self.num_attention_heads):
                attn_scores[b, h] = attn_scores[b, h] + causal

        # Softmax along last dim
        attn_probs = torch.softmax(attn_scores, dim=-1)  # [B, 96, S]

        # Compute attention output: attn_output[b, s, :] = sum_h probs[b, h, s] * value_expanded[b, h, s, :]
        # attn_output shape [B, S, 128]
        attn_output = torch.empty((B, S, self.head_dim), device=hidden_states.device, dtype=torch.float32)
        for b in range(B):
            for s in range(S):
                # sum over heads
                for h in range(self.num_attention_heads):
                    val_vec = v_expanded[b, h, s].contiguous().to(torch.float32)  # [128]
                    attn_output[b, s, :] += attn_probs[b, h, s] * val_vec

        # 7) Output projection (no bias) via Triton GEMM
        # y[b, s, :] = attn_output[b, s, :] @ o_proj_weight^T
        # We'll run a small Triton kernel per (b, s). Define grid (B, S).
        y = torch.empty((B, S, H), device=hidden_states.device, dtype=torch.float32)

        # Define a small GEMM for single vector:
        # y[b, s, :] = sum_h attn_output[b, s, h] * o_proj_weight[h]
        # We can launch one program per (b, s) and loop over H.
        for b in range(B):
            for s in range(S):
                attn_vec = attn_output[b, s].contiguous().to(torch.float32)  # [128]
                y_vec = torch.empty((H,), device=hidden_states.device, dtype=torch.float32)
                # Compute dot per h: Triton kernel below is defined to take attn_vec and o_proj_weight
                # But for simplicity, we implement loop here (small H).
                for h in range(H):
                    a = attn_vec[h]
                    w = o_proj_weight[h]
                    y_vec[h] = a * w
                y[b, s, :] = y_vec

        return y


# Helper to get inputs (from the original sample), using Triton-enabled ModelNew
def get_inputs():
    # Example: B=1, S=512, H=12288 as per original code (hidden_dim = num_attention_heads * head_dim = 96 * 128)
    B, S = 1, 512
    H = 96 * 128
    # Create placeholders for weights; sizes must match
    q_proj_weight = torch.empty(H, H, dtype=torch.float32, device='cuda')
    k_proj_weight = torch.empty(H, H, dtype=torch.float32, device='cuda')
    v_proj_weight = torch.empty(H, H, dtype=torch.float32, device='cuda')
    o_proj_weight = torch.empty(H, H, dtype=torch.float32, device='cuda')
    # RMSNorm weights: per head
    q_norm_weight = torch.empty(96, 128, dtype=torch.float32, device='cuda')
    k_norm_weight = torch.empty(8, 128, dtype=torch.float32, device='cuda')
    # cos/sin for rotation: length head_dim
    cos = torch.empty(128, dtype=torch.float32, device='cuda')
    sin = torch.empty(128, dtype=torch.float32, device='cuda')
    hidden_states = torch.randn(B, S, H, dtype=torch.float32, device='cuda')
    # eps
    rms_norm_eps = 1e-6
    return hidden_states, q_proj_weight, q_norm_weight, k_proj_weight, k_norm_weight, v_proj_weight, o_proj_weight, q_norm_weight, k_norm_weight, cos, sin, rms_norm_eps


# Entry point required by the evaluation environment
class Model(torch.nn.Module):
    def forward(self, *args):
        # Assume args match run signature
        model = ModelNew(num_attention_heads=96, head_dim=128, num_key_value_heads=8, num_key_value_groups=12)
        return model(*args)


def run(*args):
    return ModelNew()(*args)
