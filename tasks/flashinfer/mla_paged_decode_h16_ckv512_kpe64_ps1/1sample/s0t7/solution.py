import math
import torch
import triton
import triton.language as tl


@triton.jit
def lse_and_output_kernel(
    qn_row_ptr,      # *fp32, [N]
    qp_row_ptr,      # *fp32, [Kp_dim]
    Kc_ptr,          # *fp32, [TOTAL_PAGES, N]
    Kp_ptr,          # *fp32, [TOTAL_PAGES, Kp_dim]
    tok_idx_ptr,     # *int32, [M_total]
    lse_out_ptr,     # *fp32, scalar (per (b,h))
    output_ptr,      # *fp32, [N], output vector for this (b,h)
    N,               # int32, head_dim_ckv
    Kp_dim,          # int32, head_dim_kpe
    M_total,         # int32, number of used tokens in this batch
    sm_scale,        # fp32, scaling factor
    BLOCK_M: tl.constexpr,  # chunk size for tokens
    b, h             # indices
):
    # Pass 1: compute row-wise max and sum_exp for lse
    row_max = -float("inf")
    sum_exp = 0.0
    m = 0
    while m < M_total:
        offs = m + tl.arange(0, BLOCK_M)
        mask = offs < M_total
        tok = tl.load(tok_idx_ptr + offs, mask=mask, other=0)  # [BLOCK_M] int32

        # Compute logits for each token in the chunk and update row-wise max/sum_exp
        # Note: qn_row_ptr and qp_row_ptr are 1D vectors; Kc/Kp are row-wise 1D vectors.
        # We'll compute per-token logits by looping within the chunk using scalar 'm', because
        # Triton supports scalar loop inside kernels and masks handle boundaries.
        # Initialize chunk-wise sum_exp_chunk and track max.
        sum_exp_chunk = 0.0
        max_chunk = -float("inf")

        mm = 0
        while mm < BLOCK_M:
            mi = m + mm
            if mi >= M_total:
                break
            # Load qn row and compute dot with Kc[tok[mm]]
            qn_row = tl.load(qn_row_ptr + tl.arange(0, N), mask=False, other=0.0)  # [N]
            kc_row = tl.load(Kc_ptr + tok[mm] * N + tl.arange(0, N), mask=False, other=0.0)
            dot_qn = tl.sum(qn_row * kc_row, axis=0)

            q = tl.load(qp_row_ptr + tl.arange(0, Kp_dim), mask=False, other=0.0)   # [Kp_dim]
            k = tl.load(Kp_ptr + tok[mm] * Kp_dim + tl.arange(0, Kp_dim), mask=False, other=0.0)
            dot_qp = tl.sum(q * k, axis=0)

            logits = dot_qn + dot_qp
            logits_scaled = logits * sm_scale
            max_chunk = tl.maximum(max_chunk, logits_scaled)
            sum_exp_chunk += tl.exp(logits_scaled - max_chunk)
            mm += 1

        row_max = tl.maximum(row_max, max_chunk)
        sum_exp += sum_exp_chunk
        m += BLOCK_M

    lse_val = tl.log(sum_exp) / tl.log(2.0)
    tl.store(lse_out_ptr, lse_val)

    # Pass 2: compute output vector y = sum_m attn[m] * Kc[m, :], where attn[m] = exp((logits_scaled[m] - lse) / M_total)
    y = tl.zeros((N,), dtype=tl.float32)
    m = 0
    while m < M_total:
        offs = m + tl.arange(0, BLOCK_M)
        mask = offs < M_total
        tok = tl.load(tok_idx_ptr + offs, mask=mask, other=0)

        sum_exp_chunk = 0.0
        max_chunk = -float("inf")

        mm = 0
        while mm < BLOCK_M:
            mi = m + mm
            if mi >= M_total:
                break
            qn_row = tl.load(qn_row_ptr + tl.arange(0, N), mask=False, other=0.0)
            kc_row = tl.load(Kc_ptr + tok[mm] * N + tl.arange(0, N), mask=False, other=0.0)
            dot_qn = tl.sum(qn_row * kc_row, axis=0)

            q = tl.load(qp_row_ptr + tl.arange(0, Kp_dim), mask=False, other=0.0)
            k = tl.load(Kp_ptr + tok[mm] * Kp_dim + tl.arange(0, Kp_dim), mask=False, other=0.0)
            dot_qp = tl.sum(q * k, axis=0)

            logits = dot_qn + dot_qp
            logits_scaled = logits * sm_scale
            attn = tl.exp(logits_scaled - lse_val) / M_total
            y += attn * kc_row
            mm += 1

        m += BLOCK_M

    # Store output vector for this (b,h)
    # output_ptr is 1D [N], contiguous
    tl.store(output_ptr + tl.arange(0, N), y)


class ModelNew(torch.nn.Module):
    def __init__(self, sm_scale=1.0, block_m=128):
        super().__init__()
        self.sm_scale = float(sm_scale)
        self.block_m = int(block_m)

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices):
        # Extract shapes and assert known constants
        B = q_nope.shape[0]
        H = q_nope.shape[1]
        N = q_nope.shape[2]
        assert N == 512, "head_dim_ckv must be 512"
        Kp_dim = q_pe.shape[2]
        assert Kp_dim == 64, "head_dim_kpe must be 64"

        # Prepare inputs: cast to fp32 and make contiguous
        qn_fp32 = q_nope.to(torch.float32).contiguous()      # [B, H, N]
        qp_fp32 = q_pe.to(torch.float32).contiguous()        # [B, H, Kp_dim]

        # Flatten ckv_cache and kpe_cache: [num_pages, 1, N] -> [num_pages, N]
        Kc_fp32 = ckv_cache.to(torch.float32).squeeze(1).contiguous()  # [num_pages, N]
        Kp_fp32 = kpe_cache.to(torch.float32).squeeze(1).contiguous()  # [num_pages, Kp_dim]

        # Compute M_total per batch
        # Ensure kv_indptr and kv_indices on device
        kv_indptr = kv_indptr.to(q_nope.device)
        kv_indices = kv_indices.to(q_nope.device)

        M_total_list = []
        for b_idx in range(B):
            start = int(kv_indptr[b_idx].item())
            end = int(kv_indptr[b_idx + 1].item()) if b_idx + 1 < B else start
            M_total_list.append(end - start)

        # Allocate outputs
        output_fp32 = torch.zeros((B, H, N), dtype=torch.float32, device=q_nope.device)
        lse = torch.full((B, H), -float("inf"), dtype=torch.float32, device=q_nope.device)

        # Launch Triton kernel per (b,h)
        for b_idx in range(B):
            if b_idx + 1 >= B:
                break
            M_total = M_total_list[b_idx]
            if M_total <= 0:
                # No tokens for this batch element: output zeros and lse -inf
                lse[b_idx, :] = -float("inf")
                continue

            start = int(kv_indptr[b_idx].item())
            end = int(kv_indptr[b_idx + 1].item())
            tok_idx = kv_indices[start:end].to(torch.int32).contiguous()  # [M_total]

            # Flatten qn and qp for this (b,h)
            qn_row = qn_fp32[b_idx].reshape(H, N)[:, 0]  # [N]
            qp_row = qp_fp32[b_idx].reshape(H, 1, Kp_dim)[:, 0]  # [Kp_dim]
            # But we pass per (h) rows:
            for h_idx in range(H):
                qn_h = qn_fp32[b_idx, h_idx]       # [N]
                qp_h = qp_fp32[b_idx, h_idx]       # [Kp_dim]

                # Output vector for this (b,h)
                y = output_fp32[b_idx, h_idx]      # [N]

                # Launch Triton kernel: grid size is (1,)
                lse_and_output_kernel[(1,)](
                    qn_h.contiguous(),              # *fp32 [N]
                    qp_h.contiguous(),              # *fp32 [Kp_dim]
                    Kc_fp32,                        # *fp32 [num_pages, N]
                    Kp_fp32,                        # *fp32 [num_pages, Kp_dim]
                    tok_idx,                        # *int32 [M_total]
                    lse[b_idx, h_idx],              # *fp32 scalar
                    y,                              # *fp32 [N]
                    N, Kp_dim, M_total, self.sm_scale,
                    self.block_m,
                    b_idx, h_idx
                )

        # Cast output to bfloat16 to match original function's output dtype
        output_bf16 = output_fp32.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
