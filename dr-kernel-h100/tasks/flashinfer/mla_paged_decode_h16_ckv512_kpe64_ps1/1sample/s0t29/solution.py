import math
import torch
import triton
import triton.language as tl


@triton.jit
def lse_and_output_fused_kernel(
    qn_ptr,          # *fp32, [N]
    qp_ptr,          # *fp32, [Kp_dim]
    Kc_ptr,          # *fp32, [M_total, N]
    Kp_ptr,          # *fp32, [M_total, Kp_dim]
    tok_idx_ptr,     # *int32, [M_total]
    lse_ptr,         # *fp32, scalar (per (b,h))
    out_ptr,         # *fp32, [N]
    N,               # int32, head_dim_ckv
    Kp_dim,          # int32, head_dim_kpe
    M_total,         # int32
    sm_scale,        # fp32
    BLOCK_M: tl.constexpr,
):
    # First pass: compute lse
    row_max = -float("inf")
    sum_exp = 0.0

    m = 0
    while m < M_total:
        m_offsets = m + tl.arange(0, BLOCK_M)  # [BLOCK_M]
        mask_m = m_offsets < M_total           # [BLOCK_M]
        # Load token indices for these rows
        tok_vals = tl.load(tok_idx_ptr + m_offsets, mask=mask_m, other=0)  # [BLOCK_M] int32
        # Compute per-token dot products: qn · Kc[token] + qp · Kp[token]
        # Kc_chunk: [BLOCK_M, N]
        Kc_ptrs = Kc_ptr + tok_vals[:, None] * N + tl.arange(0, N)  # [BLOCK_M, N]
        Kc_chunk = tl.load(Kc_ptrs, mask=mask_m[:, None], other=0.0)  # [BLOCK_M, N]
        qn_vec = tl.load(qn_ptr + tl.arange(0, N))                    # [N]
        dot_qn = (Kc_chunk * qn_vec[None, :]).sum(axis=1)            # [BLOCK_M]
        # Kp_chunk: [BLOCK_M, Kp_dim]
        Kp_ptrs = Kp_ptr + tok_vals[:, None] * Kp_dim + tl.arange(0, Kp_dim)
        Kp_chunk = tl.load(Kp_ptrs, mask=mask_m[:, None], other=0.0)  # [BLOCK_M, Kp_dim]
        qp_vec = tl.load(qp_ptr + tl.arange(0, Kp_dim))               # [Kp_dim]
        dot_qp = (Kp_chunk * qp_vec[None, :]).sum(axis=1)            # [BLOCK_M]
        logits_scaled = dot_qn + dot_qp                               # [BLOCK_M]
        # Scale and update row_max and sum_exp
        scaled = logits_scaled * sm_scale
        local_max = tl.max(scaled, axis=0)
        # sum_exp += sum(exp(scaled - row_max)) for valid m
        # Initialize sum_exp_local
        sum_exp_local = tl.zeros((), dtype=tl.float32)
        for i in range(BLOCK_M):
            # Only sum valid entries
            if mask_m[i]:
                sum_exp_local += tl.exp(scaled[i] - row_max)
        sum_exp += sum_exp_local
        row_max = tl.maximum(row_max, local_max)

        m += BLOCK_M

    # Compute lse
    # logsumexp over tokens = log(sum_exp) / ln(2)
    lse_val = tl.log(sum_exp) / 1.4426950408889634  # 1.0 / ln(2)
    tl.store(lse_ptr, lse_val)

    # Second pass: compute output vector y = sum_m attn[m] * Kc[m, :]
    m = 0
    out_vec = tl.zeros((N,), dtype=tl.float32)
    while m < M_total:
        m_offsets = m + tl.arange(0, BLOCK_M)
        mask_m = m_offsets < M_total
        tok_vals = tl.load(tok_idx_ptr + m_offsets, mask=mask_m, other=0)  # [BLOCK_M] int32

        Kc_ptrs = Kc_ptr + tok_vals[:, None] * N + tl.arange(0, N)      # [BLOCK_M, N]
        Kc_chunk = tl.load(Kc_ptrs, mask=mask_m[:, None], other=0.0)    # [BLOCK_M, N]
        qn_vec = tl.load(qn_ptr + tl.arange(0, N))                       # [N]

        Kp_ptrs = Kp_ptr + tok_vals[:, None] * Kp_dim + tl.arange(0, Kp_dim)
        Kp_chunk = tl.load(Kp_ptrs, mask=mask_m[:, None], other=0.0)    # [BLOCK_M, Kp_dim]
        qp_vec = tl.load(qp_ptr + tl.arange(0, Kp_dim))                  # [Kp_dim]

        # Compute logits_scaled again
        dot_qn = (Kc_chunk * qn_vec[None, :]).sum(axis=1)               # [BLOCK_M]
        dot_qp = (Kp_chunk * qp_vec[None, :]).sum(axis=1)               # [BLOCK_M]
        logits_scaled = dot_qn + dot_qp                                  # [BLOCK_M]
        scaled = logits_scaled * sm_scale
        attn = tl.exp(scaled - lse_val)                                  # [BLOCK_M]

        # Accumulate y: sum_m attn[m] * Kc[m, :]
        for i in tl.static_range(0, BLOCK_M):
            if mask_m[i]:
                out_vec += attn[i] * Kc_chunk[i, :]

        m += BLOCK_M

    # Store out_vec
    out_ptrs = out_ptr + tl.arange(0, N)
    tl.store(out_ptrs, out_vec)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # You can tune these based on hardware; 128 balances well for typical GPUs
        self.block_m = 128

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale, _unused=None):
        # Ensure all tensors are on same CUDA device
        device = q_nope.device
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda

        # Cast q_nope and q_pe to fp32 and make contiguous
        qn_fp32 = q_nope.to(torch.float32).contiguous()   # [B, H, N]
        qp_fp32 = q_pe.to(torch.float32).contiguous()     # [B, H, Kp_dim]

        # Cast ckv_cache and kpe_cache to fp32 and flatten second dim
        # Original ckv_cache shape: [num_pages, 1, N]
        Kc_fp32 = ckv_cache.to(torch.float32).contiguous().view(-1, q_nope.shape[-1])  # [num_pages, N]
        # Original kpe_cache shape: [num_pages, 1, Kp_dim]
        Kp_fp32 = kpe_cache.to(torch.float32).contiguous().view(-1, q_pe.shape[-1])    # [num_pages, Kp_dim]

        B, H, N = qn_fp32.shape
        Kp_dim = qp_fp32.shape[-1]

        # Prepare outputs
        output_fp32 = torch.empty((B, H, N), dtype=torch.float32, device=device)
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        # For each batch element, compute tok_idx per head and launch Triton kernel
        for b_idx in range(B):
            start = int(kv_indptr[b_idx].item())
            end = int(kv_indptr[b_idx + 1].item())
            M_total = end - start

            if M_total <= 0:
                # No KV tokens for this batch element: output zeros, lse = -inf
                lse[b_idx] = float("-inf")
                continue

            tok_idx = kv_indices[start:end].to(torch.int32).contiguous()  # [M_total]

            # Gather Kc and Kp rows for these tokens
            Kc_chunk = Kc_fp32[tok_idx]  # [M_total, N]
            Kp_chunk = Kp_fp32[tok_idx]  # [M_total, Kp_dim]

            # Select qn and qp for this batch
            qn_row = qn_fp32[b_idx].contiguous()         # [N]
            qp_row = qp_fp32[b_idx].contiguous()         # [Kp_dim]

            # Launch Triton kernel to compute lse and output vector for this (b, h)
            # We launch one kernel per (b, h)
            for h_idx in range(H):
                out_vec = torch.empty((N,), dtype=torch.float32, device=device)
                lse_scalar = torch.empty((), dtype=torch.float32, device=device)

                # Kernel grid: one program handles all work for (b, h)
                lse_and_output_fused_kernel[(1,)](
                    qn_row, Kp_dim, M_total, float(sm_scale),
                    Kc_chunk, Kp_chunk, tok_idx, lse_scalar, out_vec,
                    N, Kp_dim, M_total, float(sm_scale),
                    BLOCK_M=self.block_m
                )

                # Store lse and output
                lse[b_idx, h_idx] = lse_scalar.item()
                output_fp32[b_idx, h_idx, :] = out_vec

        # Cast output to bfloat16 to match original function's output dtype
        output_bf16 = output_fp32.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
