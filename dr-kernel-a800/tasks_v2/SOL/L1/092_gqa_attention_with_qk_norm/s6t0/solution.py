import torch
import torch.nn as nn
import triton
import triton.language as tl

# Triton kernels: dense matmul without bias, RMSNorm, and softmax.

# Dense (batched) matmul: C[M, K] = A[M, N] @ B[K, N]
# A: hidden_states viewed as [M, N], with strides a0, a1
# B: weight transposed as [K, N], with strides b0, b1
# C: output [M, K], with strides c0, c1
@triton.jit
def triton_dense_mm_no_bias(A_ptr, B_ptr, C_ptr,
                             M, N, K,
                             a0, a1, b0, b1, c0, c1,
                             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)

    off_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    off_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    # We need to compute a partial sum for off_k across N
    # We'll do the accumulation in FP32
    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)

    for n_start in range(0, N, BLOCK_N):
        off_n = n_start + tl.arange(0, BLOCK_N)

        # Load A tile: [BLOCK_M, BLOCK_N]
        a_ptrs = A_ptr + off_m[:, None] * a0 + off_n[None, :] * a1
        a_mask = (off_m[:, None] < M) & (off_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        a = a.to(tl.float32)

        # Load B tile: [BLOCK_K, BLOCK_N] (B is [K, N])
        b_ptrs = B_ptr + off_k[:, None] * b0 + off_n[None, :] * b1
        b_mask = (off_k[:, None] < K) & (off_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        b = b.to(tl.float32)

        # acc += a @ b^T -> [BLOCK_M, BLOCK_K]
        acc += tl.dot(a, tl.trans(b))

    # Write results to C
    c_ptrs = C_ptr + off_m[:, None] * c0 + off_k[None, :] * c1
    c_mask = (off_m[:, None] < M) & (off_k[None, :] < K)
    tl.store(c_ptrs, acc, mask=c_mask)


# Triton kernel for RMSNorm applied in-place to X, with per-row weight and eps.
# Input X: [M, D], where M = batch*seq*heads (flattened), D = head_dim.
# weight: [D], float32
# eps: float32
@triton.jit
def triton_rmsnorm_inplace(X_ptr, weight_ptr, M, D, eps,
                            x0, x1, w0,
                            BLOCK_D: tl.constexpr):
    row_id = tl.program_id(0)  # each program handles one row
    # compute sum of squares over D
    sum_sq = 0.0
    for d_start in range(0, D, BLOCK_D):
        offs = d_start + tl.arange(0, BLOCK_D)
        mask = offs < D
        x = tl.load(X_ptr + row_id * x0 + offs * x1, mask=mask, other=0.0).to(tl.float32)
        sum_sq += tl.sum(x * x)
    mean = sum_sq / D
    inv_rms = 1.0 / tl.sqrt(mean + eps)

    # write normalized and scaled output
    for d_start in range(0, D, BLOCK_D):
        offs = d_start + tl.arange(0, BLOCK_D)
        mask = offs < D
        x = tl.load(X_ptr + row_id * x0 + offs * x1, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + offs * w0, mask=mask, other=1.0).to(tl.float32)
        y = x * inv_rms * w
        # store as float32; original code uses float32 for RMSNorm accumulation
        tl.store(X_ptr + row_id * x0 + offs * x1, y, mask=mask)


# Triton kernel: row-wise softmax over the last dimension (sequence length S)
# Input X: [B, H, S, S] (we will pass the 2D slice [S, S] for each (B, H) row).
# We treat X as a flat [num_rows, S] where num_rows = B * H * S.
# Output Out: same shape as X, in float32 for numerical stability.
@triton.jit
def triton_softmax_rows(X_ptr, Out_ptr,
                         num_rows, S,
                         x_row_stride, x_col_stride,
                         out_row_stride, out_col_stride,
                         BLOCK_S: tl.constexpr):
    row_id = tl.program_id(0)  # 0 .. num_rows-1
    # compute row pointer offsets
    row_x = X_ptr + row_id * x_row_stride
    row_out = Out_ptr + row_id * out_row_stride

    # 1) compute max across columns
    max_val = -1e30  # large negative
    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        x = tl.load(row_x + offs * x_col_stride, mask=mask, other=-1e30).to(tl.float32)
        # reduce max across vector
        block_max = tl.max(x, axis=0)
        max_val = tl.maximum(max_val, block_max)

    # 2) compute sum of exp(x - max) across columns
    sum_exp = 0.0
    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        x = tl.load(row_x + offs * x_col_stride, mask=mask, other=-1e30).to(tl.float32)
        e = tl.exp(x - max_val)
        sum_exp += tl.sum(e, axis=0)

    # 3) write normalized outputs
    for s_start in range(0, S, BLOCK_S):
        offs = s_start + tl.arange(0, BLOCK_S)
        mask = offs < S
        x = tl.load(row_x + offs * x_col_stride, mask=mask, other=-1e30).to(tl.float32)
        e = tl.exp(x - max_val) / sum_exp
        tl.store(row_out + offs * out_col_stride, e, mask=mask)


# Helper function to launch dense GEMM no bias
def triton_linear_no_bias(A, W, out):
    # A: [M, N], W: [K, N] (i.e., W is q/v/k weight transposed [out_features, in_features])
    # out: [M, K]
    M, N = A.shape
    K = W.shape[0]
    # Ensure contiguous or correct strides
    a = A
    b = W
    c = out
    # Strides
    a0, a1 = a.stride(0), a.stride(1)
    b0, b1 = b.stride(0), b.stride(1)
    c0, c1 = c.stride(0), c.stride(1)
    # Choose blocks
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_K))
    triton_dense_mm_no_bias[grid](a, b, c, M, N, K, a0, a1, b0, b1, c0, c1,
                                  BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                                  num_warps=4, num_stages=2)
    return c


# RMSNorm wrapper
def triton_rmsnorm_inplace_rowwise(X, weight, eps):
    # X: [B, S, H, D], we flatten M = B*S*H, D = head_dim
    B, S, H, D = X.shape
    M = B * S * H
    # We need to make X viewable as [M, D] in terms of pointer arithmetic; we'll use .view if possible
    # But Triton kernels expect simple strides. We'll pass strides from a viewable contiguous tensor.
    # To be safe, we can create a contiguous [M, D] view:
    X_flat = X.contiguous().view(M, D)
    # weight is [D], ensure float32
    w = weight.to(torch.float32).contiguous()
    x0, x1 = X_flat.stride(0), X_flat.stride(1)
    w0 = w.stride(0)
    BLOCK_D = 128
    grid = (M,)
    triton_rmsnorm_inplace[grid](X_flat, w, M, D, eps, x0, x1, w0, BLOCK_D=BLOCK_D, num_warps=4, num_stages=2)
    # Reshape back
    X.copy_(X_flat.view(B, S, H, D))
    return X


# Softmax over sequence dimension for each (batch, head) row
def triton_softmax_attention(X, num_rows, S):
    # X: [B, H, S, S], we'll flatten to [num_rows, S]
    # We'll create a 2D contiguous view
    B, H, S1, S2 = X.shape
    assert S1 == S2 == S
    X2D = X.reshape(B * H * S, S).contiguous()  # [num_rows, S]
    Out2D = torch.empty_like(X2D, dtype=torch.float32)  # compute in fp32
    row_stride = X2D.stride(0)
    col_stride = X2D.stride(1)
    out_row_stride = Out2D.stride(0)
    out_col_stride = Out2D.stride(1)
    BLOCK_S = 128
    grid = (num_rows,)
    triton_softmax_rows[grid](X2D, Out2D, num_rows, S,
                              row_stride, col_stride,
                              out_row_stride, out_col_stride,
                              BLOCK_S=BLOCK_S, num_warps=4, num_stages=2)
    # Reshape back to [B, H, S, S] with fp32 softmax result
    out = Out2D.view(B, H, S, S)
    return out


# Main ModelNew that uses Triton kernels
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # We don't store weights here; they are passed into forward.

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
                sin: torch.Tensor,
                rms_norm_eps: float):
        # Ensure device is CUDA for Triton
        assert hidden_states.is_cuda, "Triton kernels require CUDA tensors"
        # Shapes
        B, S, hidden_dim = hidden_states.shape
        num_attention_heads = 96
        num_key_value_heads = 8
        head_dim = 128
        num_key_value_groups = 12
        scaling = head_dim ** -0.5

        # 1) Dense Q, K, V (no bias) using Triton
        # Prepare A views: [M, N] where M = B*S, N = hidden_dim
        M = B * S
        A_Q = hidden_states.view(M, hidden_dim)
        A_K = hidden_states.view(M, hidden_dim)
        A_V = hidden_states.view(M, hidden_dim)

        # Weights: need [K, N] where K = hidden_dim
        W_Q = q_proj_weight.t().contiguous()  # [hidden_dim, hidden_dim] -> actual output dim is num_attention_heads*head_dim
        W_K = k_proj_weight.t().contiguous()
        W_V = v_proj_weight.t().contiguous()

        # Output tensors
        Q = torch.empty((M, num_attention_heads * head_dim), device=hidden_states.device, dtype=torch.float32)
        K = torch.empty((M, num_attention_heads * head_dim), device=hidden_states.device, dtype=torch.float32)
        V = torch.empty((M, num_attention_heads * head_dim), device=hidden_states.device, dtype=torch.float32)

        # Launch Triton GEMMs
        triton_linear_no_bias(A_Q, W_Q, Q)   # Q: [B*S, 96*128]
        triton_linear_no_bias(A_K, W_K, K)   # K: [B*S, 96*128]
        triton_linear_no_bias(A_V, W_V, V)   # V: [B*S, 8*128]

        # Reshape to heads
        Q = Q.view(B, S, num_attention_heads, head_dim)  # [B, S, 96, 128]
        K = K.view(B, S, num_attention_heads, head_dim)  # [B, S, 96, 128]
        V = V.view(B, S, num_key_value_heads, head_dim)  # [B, S, 8, 128]

        # 2) Apply RMSNorm for Q and K using Triton
        # q_norm_weight and k_norm_weight are [head_dim] per head; here we have 96 query heads and 8 key heads.
        # We'll normalize per head across dim=3 (head_dim) for each (B, S, head) row.
        # First, make contiguous views for each head
        # For query heads:
        for h in range(num_attention_heads):
            Xh = Q[:, :, h, :]  # [B, S, 128]
            # Flatten rows: M = B*S
            Xh_flat = Xh.contiguous().view(B * S, head_dim)
            q_norm_w = q_norm_weight.to(torch.float32).contiguous()
            triton_rmsnorm_inplace_rowwise(Xh_flat, q_norm_w, rms_norm_eps)
            # Reshape back
            Q[:, :, h, :] = Xh_flat.view(B, S, head_dim)

        # For key heads:
        for h in range(num_key_value_heads):
            Xh = K[:, :, h, :]  # [B, S, 128]
            Xh_flat = Xh.contiguous().view(B * S, head_dim)
            k_norm_w = k_norm_weight.to(torch.float32).contiguous()
            triton_rmsnorm_inplace_rowwise(Xh_flat, k_norm_w, rms_norm_eps)
            K[:, :, h, :] = Xh_flat.view(B, S, head_dim)

        # 3) Transpose to [B, num_heads, S, head_dim]
        Q = Q.transpose(1, 2)  # [B, 96, S, 128]
        K = K.transpose(1, 2)  # [B, 96, S, 128]
        V = V.transpose(1, 2)  # [B, 8, S, 128]

        # 4) Apply RoPE rotation (elementwise)
        # Split half: first 64 and second 64
        # Prepare cos/sin expanded: [B, 1, S, 128]
        cos_exp = cos.unsqueeze(1)  # [B, 1, S, 128]
        sin_exp = sin.unsqueeze(1)
        for b in range(B):
            for h in range(num_attention_heads):
                q = Q[b, h, :, :]  # [S, 128]
                q1 = q[:, :64]
                q2 = q[:, 64:]
                q_rot = torch.cat((-q2, q1), dim=1)  # [S, 64] for each half -> [S, 128] with second half
                q = (q * cos_exp[b]) + (q_rot * sin_exp[b])
                Q[b, h, :, :] = q

            for h in range(num_key_value_heads):
                k = K[b, h, :, :]
                k1 = k[:, :64]
                k2 = k[:, 64:]
                k_rot = torch.cat((-k2, k1), dim=1)
                k = (k * cos_exp[b]) + (k_rot * sin_exp[b])
                K[b, h, :, :] = k

        # 5) Grouped Query Attention: repeat KV heads across groups to match 96 attention heads
        # Effective reshape: expand [B, 8, S, 128] -> [B, 8, 12, S, 128] then reshape to [B, 96, S, 128]
        K_exp = K[:, :, None, :, :].expand(B, num_key_value_heads, num_key_value_groups, S, head_dim).reshape(B, num_attention_heads, S, head_dim)
        V_exp = V[:, :, None, :, :].expand(B, num_key_value_heads, num_key_value_groups, S, head_dim).reshape(B, num_attention_heads, S, head_dim)
        K = K_exp
        V = V_exp

        # 6) Compute attention scores using PyTorch matmul (due to complexity of doing full attention in Triton here)
        # attn_weights [B, 96, S, S]
        attn_weights = torch.matmul(Q, K.transpose(3, 2)) * scaling  # [B, 96, S, S]

        # 7) Causal mask: apply in PyTorch (upper-triangular with diagonal=1)
        causal_mask = torch.triu(torch.full((S, S), float('-inf'), device=hidden_states.device, dtype=attn_weights.dtype), diagonal=1)
        # Broadcast to [B, 96, S, S]
        attn_weights = attn_weights + causal_mask

        # 8) Softmax using Triton (row-wise over last dim S)
        num_rows = B * num_attention_heads * S
        attn_weights_softmax = triton_softmax_attention(attn_weights, num_rows, S)  # fp32 result

        # 9) Compute attention output
        # attn_output = softmax @ V  -> [B, 96, S, 128]
        # We need V expanded form, which we already have as [B, 96, S, 128] from grouping above.

        attn_output = torch.matmul(attn_weights_softmax, V)

        # 10) Final output projection (no bias) using Triton GEMM
        B_hat, H, S1, D1 = attn_output.shape  # B_hat should equal B
        assert B_hat == B
        # Prepare A and W for final projection: o_proj_weight is [out_features, in_features] = [hidden_dim, num_heads*head_dim]
        A = attn_output.reshape(B * H * S1, D1).contiguous()
        W_o = o_proj_weight.t().contiguous()  # [D1, hidden_dim]
        Out = torch.empty((B * H * S1, hidden_dim), device=hidden_states.device, dtype=torch.float32)
        triton_linear_no_bias(A, W_o, Out)

        # 11) Reshape to [B, S, hidden_dim]
        Out = Out.view(B, H, S1, hidden_dim)
        Out = Out.transpose(1, 2).contiguous().reshape(B, S1, H * hidden_dim)
        return Out


def run(*args):
    return ModelNew()(*args)
