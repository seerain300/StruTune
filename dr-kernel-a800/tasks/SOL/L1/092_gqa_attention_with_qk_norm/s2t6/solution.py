import math
import torch
import triton
import triton.language as tl


# Triton GEMM kernel: C[M, N] = A[M, K] @ B[N, K]^T
@triton.jit
def matmul_no_bias_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = B_ptr + (offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk)
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_n[None, :] < N) & (offs_k[:, None] < K), other=0.0)
        acc += tl.dot(a, b)

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# Triton RMSNorm per row (last dim): X[M, N] -> X_norm[M, N]
# We'll implement per (B,S,head): normalize over last dim (head_dim), scale by per-head weight.
@triton.jit
def rmsnorm_row_kernel(
    X_ptr,  # [M, N]
    W_ptr,  # [N], per-head scale
    Y_ptr,  # [M, N]
    M, N,
    stride_xm, stride_xn,
    stride_w,
    stride_ym, stride_yn,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m
    # Each program handles one row; loop over N in tiles
    for n0 in range(0, N, BLOCK_N):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        x = tl.load(X_ptr + offs_m * stride_xm + offs_n * stride_xn, mask=offs_n < N, other=0.0)
        # Compute RMS: sqrt(mean(x^2) + eps)
        sq = x * x
        mean = tl.sum(sq, axis=0) / N
        eps = 1e-6  # same as in original code
        inv_rms = 1.0 / tl.sqrt(mean + eps)
        w = tl.load(W_ptr + offs_n * stride_w, mask=offs_n < N, other=1.0)  # per-head scale
        x_norm = x * inv_rms * w
        tl.store(Y_ptr + offs_m * stride_ym + offs_n * stride_yn, x_norm, mask=offs_n < N)


# Triton kernel for Rotated Positional Embedding (RoPE) rotation on [B, S, H] per head
# We will implement per (B,S,head): rotate 128-dim vector
@triton.jit
def rotate_kernel_128(
    X_ptr, cos_ptr, sin_ptr, Y_ptr,
    B, S, H,  # not strictly needed, but can be used for bounds
    stride_xb, stride_xs, stride_xh, stride_xd,
    stride_yb, stride_ys, stride_yh, stride_yd,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    # Map program id to (b, s, h)
    # grid size should be B*S*H for this kernel
    b = pid // (S * H)
    rem = pid % (S * H)
    s = rem // H
    h = rem % H

    offs_d = tl.arange(0, BLOCK_D)  # D=128
    # Load x[:, d]
    x = tl.load(X_ptr + b * stride_xb + s * stride_xs + h * stride_xh + offs_d * stride_xd, mask=offs_d < 128, other=0.0)

    # Split into two halves
    q1 = x[:64]
    q2 = x[64:]

    # Load cos/sin for d in [0..127]
    cos_vals = tl.load(cos_ptr + offs_d, mask=offs_d < 128, other=1.0)
    sin_vals = tl.load(sin_ptr + offs_d, mask=offs_d < 128, other=0.0)

    # Rotate: q1' = q1*cos - q2*sin; q2' = q1*sin + q2*cos
    q1p = q1 * cos_vals[:64] - q2 * sin_vals[:64]
    q2p = q1 * sin_vals[:64] + q2 * cos_vals[64:]

    y = tl.zeros((128,), dtype=tl.float32)
    y[:64] = q1p
    y[64:] = q2p

    tl.store(Y_ptr + b * stride_yb + s * stride_ys + h * stride_yh + offs_d * stride_yd, y, mask=offs_d < 128)


# Triton GEMM kernel for output projection: C[M, N] = A[M, K] @ B[N, K]^T (no bias)
# Here A = attn_output [B*S, H], B = o_proj_weight [H, H], C = output [B*S, H]
@triton.jit
def out_proj_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = B_ptr + (offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk)
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_n[None, :] < N) & (offs_k[:, None] < K), other=0.0)
        acc += tl.dot(a, b)

    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


class ModelNew:
    def __init__(self,
                 num_attention_heads: int = 96,
                 head_dim: int = 128,
                 num_key_value_heads: int = 8,
                 num_key_value_groups: int = 12,
                 ):
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_key_value_groups

    def forward(self,
                hidden_states: torch.Tensor,
                q_proj_weight: torch.Tensor,
                q_norm_weight: torch.Tensor,
                k_proj_weight: torch.Tensor,
                k_norm_weight: torch.Tensor,
                v_proj_weight: torch.Tensor,
                v_norm_weight: torch.Tensor,  # not used; kept for signature symmetry
                o_proj_weight: torch.Tensor,
                q_norm_weight_heads: torch.Tensor,  # per head weight for query RMSNorm
                k_norm_weight_heads: torch.Tensor,  # per head weight for key RMSNorm
                cos: torch.Tensor,
                sin: torch.Tensor,
                rms_norm_eps: float = 1e-6,
                ):
        # Shapes
        B, S, H = hidden_states.shape
        D = self.head_dim  # 128
        # 1) Dense linear projections via Triton GEMM
        # We need to compute: query, key, value of shape [B, S, H]
        # Prepare A (M,K) where M=B*S, K=H (since inputs are [B,S,H] and weights are [H,H]).
        # For torch.contiguous: hidden_states [B, S, H] -> reshape to [M, H] then transpose to [H, M].
        # But Triton expects row-major; we can pass as [M, K] with appropriate strides.
        # Allocate outputs
        query = torch.empty((B, S, H), dtype=hidden_states.dtype, device=hidden_states.device)
        key = torch.empty((B, S, H), dtype=hidden_states.dtype, device=hidden_states.device)
        value = torch.empty((B, S, H), dtype=hidden_states.dtype, device=hidden_states.device)

        # Flatten to [M, K]
        M = B * S
        K = H
        A_q = hidden_states.reshape(M, K)
        A_k = hidden_states.reshape(M, K)
        A_v = hidden_states.reshape(M, K)

        # Launch Triton GEMM for query
        out_q = torch.empty((M, K), dtype=torch.float32, device=hidden_states.device)
        matmul_no_bias_kernel[(triton.cdiv(M, 64), triton.cdiv(K, 64))](
            A_q, q_proj_weight, out_q,
            M, K, H,
            A_q.stride(0), A_q.stride(1),
            q_proj_weight.stride(0), q_proj_weight.stride(1),
            out_q.stride(0), out_q.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=64,
        )
        query = out_q.reshape(B, S, H)

        # Launch Triton GEMM for key
        out_k = torch.empty((M, K), dtype=torch.float32, device=hidden_states.device)
        matmul_no_bias_kernel[(triton.cdiv(M, 64), triton.cdiv(K, 64))](
            A_k, k_proj_weight, out_k,
            M, K, H,
            A_k.stride(0), A_k.stride(1),
            k_proj_weight.stride(0), k_proj_weight.stride(1),
            out_k.stride(0), out_k.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=64,
        )
        key = out_k.reshape(B, S, H)

        # Launch Triton GEMM for value
        out_v = torch.empty((M, K), dtype=torch.float32, device=hidden_states.device)
        matmul_no_bias_kernel[(triton.cdiv(M, 64), triton.cdiv(K, 64))](
            A_v, v_proj_weight, out_v,
            M, K, H,
            A_v.stride(0), A_v.stride(1),
            v_proj_weight.stride(0), v_proj_weight.stride(1),
            out_v.stride(0), out_v.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=64,
        )
        value = out_v.reshape(B, S, H)

        # 2) RMSNorm per head on query and key (Triton kernel)
        # We need to reshape [B, S, H] to [B*S, num_heads, D] and apply RMSNorm per (row) last dim.
        # Prepare query_norm and key_norm outputs.
        # For each head h in 0..num_attention_heads-1:
        # X = query.view(B*S, num_attention_heads, D)[:, h, :]  -> shape [B*S, D]
        # W = q_norm_weight_heads[h, :]  -> shape [D]
        # Apply RMSNorm with eps
        # Allocate outputs
        query_norm = torch.empty_like(query)
        key_norm = torch.empty_like(key)

        # Launch Triton RMSNorm per head
        for h in range(self.num_attention_heads):
            Xq = query.view(B * S, self.num_attention_heads, D)[:, h, :]
            Xk = key.view(B * S, self.num_attention_heads, D)[:, h, :]
            # Yq and Yk
            Yq = torch.empty_like(Xq, dtype=torch.float32, device=hidden_states.device)
            Yk = torch.empty_like(Xk, dtype=torch.float32, device=hidden_states.device)
            # W per head (D vector) cast to float32
            Wq = q_norm_weight_heads[h].to(torch.float32)
            Wk = k_norm_weight_heads[h].to(torch.float32)

            rmsnorm_row_kernel[(B * S,)](
                Xq, Wq, Yq, B * S, D,
                Xq.stride(0), Xq.stride(1),
                Wq.stride(0),
                Yq.stride(0), Yq.stride(1),
                BLOCK_N=128,
            )
            rmsnorm_row_kernel[(B * S,)](
                Xk, Wk, Yk, B * S, D,
                Xk.stride(0), Xk.stride(1),
                Wk.stride(0),
                Yk.stride(0), Yk.stride(1),
                BLOCK_N=128,
            )
            # Assign back
            query_norm[:, h, :] = Yq
            key_norm[:, h, :] = Yk

        # Now we have query_norm and key_norm of shape [B, S, H]

        # 3) Apply Rotated Positional Embedding (RoPE) on query and key (Triton kernel)
        # Prepare output tensors
        query_rot = torch.empty_like(query_norm)
        key_rot = torch.empty_like(key_norm)

        # Launch Triton rotate kernel over all (B, S, H)
        total = B * S * self.num_attention_heads
        rotate_kernel_128[(total,)](
            query_norm, cos, sin, query_rot,
            B, S, self.num_attention_heads,
            query_norm.stride(0), query_norm.stride(1), query_norm.stride(2), query_norm.stride(3),
            query_rot.stride(0), query_rot.stride(1), query_rot.stride(2), query_rot.stride(3),
            BLOCK_D=128,
        )

        rotate_kernel_128[(total,)](
            key_norm, cos, sin, key_rot,
            B, S, self.num_attention_heads,
            key_norm.stride(0), key_norm.stride(1), key_norm.stride(2), key_norm.stride(3),
            key_rot.stride(0), key_rot.stride(1), key_rot.stride(2), key_rot.stride(3),
            BLOCK_D=128,
        )

        # 4) Grouped Query Attention (GQA) expansion: key/value from 8 heads to 96 heads by replication
        # We need key_rot_expanded of shape [B, 96, S, D]
        # Build mapping: for each original head k in 0..7, replicate across 12 groups -> 96 expanded heads
        key_rot_expanded = torch.empty((B, self.num_attention_heads, S, D), dtype=torch.float32, device=hidden_states.device)
        value_expanded = torch.empty((B, self.num_attention_heads, S, D), dtype=torch.float32, device=hidden_states.device)

        for k in range(self.num_key_value_heads):
            start = k * self.num_key_value_groups
            end = start + self.num_key_value_groups
            # Assign key_rot[:, k, :, :] to key_rot_expanded[:, start:end, :, :]
            key_rot_expanded[:, start:end, :, :] = key_rot[:, k, :, :]
            value_expanded[:, start:end, :, :] = value[:, k, :, :]

        # 5) Compute attention scores using torch (softmax + matmul), then output projection via Triton GEMM
        # For each (b, h in 96), compute attention over all sequence positions:
        # attn_scores[B, 96, S, S] = (query_rot[b, h, s, :] @ key_rot_expanded[b, h, :, :]^T) * scaling
        # Apply causal mask and softmax
        attn_scores = torch.empty((B, self.num_attention_heads, S, S), dtype=torch.float32, device=hidden_states.device)

        # Loop over batch and heads
        for b in range(B):
            for h in range(self.num_attention_heads):
                # query_vec [S, D], key_vec [S, D] -> attn_scores[b, h, :, :] = [S, S]
                # We need to compute dot per s_j over D: sum_d query[b, h, s_j, d] * key[b, h, s_k, d]
                # Build query_vec and key_vec for this (b, h)
                query_vec = query_rot[b, h]  # [S, D]
                key_vec = key_rot[b, h]      # [S, D]
                # Compute scores: attn_scores[b, h, s_j, s_k] = sum_d query_vec[s_j, d] * key_vec[s_k, d] * scaling
                # Use torch.bmm for efficiency: [S, 1, D] @ [1, D, S] -> [S, S]
                # Prepare [S, D] and transpose [D, S] by indexing. But torch.bmm requires 3D.
                # Construct [S, 1, D] and [1, D, S]:
                query_3 = query_vec.unsqueeze(1)  # [S, 1, D]
                key_3 = key_vec.unsqueeze(1).transpose(-1, -2)  # [1, D, S]
                scores = torch.bmm(query_3, key_3)[0] * (1.0 / math.sqrt(D))  # [S, S]
                # Causal mask: upper triangle (j > i) should be -inf
                mask = torch.triu(torch.ones((S, S), device=hidden_states.device, dtype=torch.float32), diagonal=1) * (-float('inf'))
                scores = scores + mask
                # Softmax along sequence dimension
                probs = torch.softmax(scores, dim=1)  # [S, S]
                # attn_output[b, h, s, :] = probs[s, :] @ value_expanded[b, h, s, :] -> shape [S, D]
                # Compute per s_j: dot over K
                attn_out = torch.empty((S, D), dtype=torch.float32, device=hidden_states.device)
                # Manual dot: for each j in [0..S-1], attn_out[j, :] = sum_k probs[j, k] * value_expanded[b, h, k, :]
                # This is simple because D is 128 and S is moderate.
                for j in range(S):
                    attn_out[j, :] = torch.sum(probs[j, :] * value_expanded[b, h, :, :], dim=0)
                attn_scores[b, h] = attn_out

        # 6) Output projection via Triton GEMM: output [B, S, H] = attn_scores [B*S, H] @ o_proj_weight [H, H]^T
        # Flatten attn_scores to [M, H]
        M_out = B * S
        attn_flat = attn_scores.reshape(M_out, H)

        # Output tensor
        output = torch.empty((M_out, H), dtype=torch.float32, device=hidden_states.device)

        # Launch Triton output projection kernel
        out_proj_kernel[(triton.cdiv(M_out, 64), triton.cdiv(H, 64))](
            attn_flat, o_proj_weight, output,
            M_out, H, H,
            attn_flat.stride(0), attn_flat.stride(1),
            o_proj_weight.stride(0), o_proj_weight.stride(1),
            output.stride(0), output.stride(1),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=64,
        )

        # Reshape back to [B, S, H]
        output = output.reshape(B, S, H)

        return output


# Entry point required by the evaluation environment
class Model(torch.nn.Module):
    def forward(self, *args):
        model = ModelNew(num_attention_heads=96, head_dim=128, num_key_value_heads=8, num_key_value_groups=12)
        return model(*args)


def run(*args):
    return ModelNew()(*args)
