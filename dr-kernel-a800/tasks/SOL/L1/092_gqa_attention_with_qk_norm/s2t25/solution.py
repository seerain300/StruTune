import torch
import triton
import triton.language as tl


# Triton GEMM kernel for linear without bias: C[M, N] = A[M, K] @ B[N, K]^T
@triton.jit
def linear_no_bias_kernel(
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
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = B_ptr + offs_n[None, :] * stride_bn + offs_k[:, None] * stride_bk

        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_n[None, :] < N) & (offs_k[:, None] < K), other=0.0)

        acc += tl.dot(a, b)

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


class ModelNew(torch.nn.Module):
    def __init__(self, num_attention_heads=96, head_dim=128, num_key_value_heads=8, num_key_value_groups=12):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_key_value_groups

    def forward(self, hidden_states, q_proj_weight, q_proj_bias,
                k_proj_weight, k_proj_bias,
                v_proj_weight, v_proj_bias,
                o_proj_weight, q_norm_weight, k_norm_weight,
                cos, sin, rms_norm_eps):
        # hidden_states: [B, S, H], H = num_attention_heads * head_dim
        B, S, H = hidden_states.shape
        D = self.head_dim
        H_query = self.num_attention_heads
        H_key = self.num_key_value_heads

        # 1) Dense linear layers without bias using Triton GEMM:
        # Compute query, key, value: [B, S, H]
        # Ensure weights are contiguous and float32 for Triton
        q_proj_weight_f = q_proj_weight.contiguous().to(torch.float32)
        k_proj_weight_f = k_proj_weight.contiguous().to(torch.float32)
        v_proj_weight_f = v_proj_weight.contiguous().to(torch.float32)
        hidden_f = hidden_states.contiguous().to(torch.float32)

        query = torch.empty((B, S, H), dtype=torch.float32, device=hidden_f.device)
        key = torch.empty((B, S, H), dtype=torch.float32, device=hidden_f.device)
        value = torch.empty((B, S, H), dtype=torch.float32, device=hidden_f.device)

        # Launch Triton linear_no_bias_kernel
        grid_q = (triton.cdiv(B, 1), triton.cdiv(H, 1))
        linear_no_bias_kernel[grid_q](
            hidden_f, q_proj_weight_f, query,
            B, H, H,  # M=N=H, K=H (no bias, F.linear uses weight of shape [H, H])
            hidden_f.stride(0), hidden_f.stride(2),  # stride_am, stride_ak
            q_proj_weight_f.stride(0), q_proj_weight_f.stride(1),  # stride_bn, stride_bk
            query.stride(0), query.stride(2),  # stride_cm, stride_cn
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=64,
        )

        grid_k = (triton.cdiv(B, 1), triton.cdiv(H, 1))
        linear_no_bias_kernel[grid_k](
            hidden_f, k_proj_weight_f, key,
            B, H, H,
            hidden_f.stride(0), hidden_f.stride(2),
            k_proj_weight_f.stride(0), k_proj_weight_f.stride(1),
            key.stride(0), key.stride(2),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=64,
        )

        grid_v = (triton.cdiv(B, 1), triton.cdiv(H, 1))
        linear_no_bias_kernel[grid_v](
            hidden_f, v_proj_weight_f, value,
            B, H, H,
            hidden_f.stride(0), hidden_f.stride(2),
            v_proj_weight_f.stride(0), v_proj_weight_f.stride(1),
            value.stride(0), value.stride(2),
            BLOCK_M=64, BLOCK_N=64, BLOCK_K=64,
        )

        # 2) Reshape to heads
        query_heads = query.view(B, S, H_query, D)
        key_heads = key.view(B, S, H_key, D)
        value_heads = value.view(B, S, H_key, D)

        # 3) RMSNorm per head (this code uses PyTorch for simplicity to ensure correctness)
        # Implement RMSNorm per head (learned per-head scale), per row along last dim:
        # y = w * x / sqrt(mean(x^2) + eps)
        def rmsnorm_heads(x, w, eps):
            x = x.to(torch.float32)
            # mean over last dim
            mean = x.pow(2).mean(dim=-1, keepdim=True)
            inv = torch.rsqrt(mean + eps)
            # scale with per-head weight
            w = w.to(torch.float32)
            y = x * inv * w
            return y.to(x.dtype)

        query_heads_norm = rmsnorm_heads(query_heads, q_norm_weight, rms_norm_eps)
        key_heads_norm = rmsnorm_heads(key_heads, k_norm_weight, rms_norm_eps)

        # 4) Apply Rotated Positional Embedding (RoPE) for query and key
        # Split D into two halves and apply rotation: y = [q1*c - q2*s, q1*s + q2*c]
        # We keep this in PyTorch for simplicity:
        half = D // 2

        # For query
        q1 = query_heads_norm[..., :half]
        q2 = query_heads_norm[..., half:]
        q1c = q1 * cos.unsqueeze(-1)
        q2s = q2 * sin.unsqueeze(-1)
        q_rot = torch.cat((-q2, q1), dim=-1)  # cat on last dim

        # For key
        k1 = key_heads_norm[..., :half]
        k2 = key_heads_norm[..., half:]
        k1c = k1 * cos.unsqueeze(-1)
        k2s = k2 * sin.unsqueeze(-1)
        k_rot = torch.cat((-k2, k1), dim=-1)

        # 5) Grouped Query Attention: expand key/value heads to 96 heads
        key_rot_expanded = key_rot[:, :, None, :, :].expand(B, H_key, self.num_key_value_groups, S, D).reshape(B, 96, S, D)
        value_expanded = value_heads[:, :, None, :, :].expand(B, H_key, self.num_key_value_groups, S, D).reshape(B, 96, S, D)

        # Compute attention scores: [B, 96, S, S]
        # score[b,h,i,j] = query_rot[b,h,i,:] @ key_rot_expanded[b,h,j,:]^T / sqrt(D)
        #


def run(*args):
    return ModelNew()(*args)
