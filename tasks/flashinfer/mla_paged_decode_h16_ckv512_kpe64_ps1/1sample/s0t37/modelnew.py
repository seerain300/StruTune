import math
import torch
import triton
import triton.language as tl


@triton.jit
def compute_lse_and_output_kernel(
    qn_ptr,        # *fp32, [N]
    qp_ptr,        # *fp32, [Kp_dim]
    Kc_ptr,        # *fp32, [TOTAL_PAGES, N]
    Kp_ptr,        # *fp32, [TOTAL_PAGES, Kp_dim]
    tok_idx_ptr,   # *int32, [M_total]
    lse_ptr,       # *fp32, 1-element tensor, scalar output for lse
    out_ptr,       # *fp32, [N] output vector for current (b,h)
    N,             # int32
    Kp_dim,        # int32
    M_total,       # int32
    sm_scale,      # fp32
    BLOCK_M: tl.constexpr  # chunk size for processing tokens
):
    # First pass: compute row_max and sum_exp for logsumexp
    row_max = tl.full((), -float("inf"), tl.float32)
    sum_exp = tl.zeros((), tl.float32)

    m = 0
    while m < M_total:
        idx = m + tl.arange(0, BLOCK_M)
        mask = idx < M_total
        tok = tl.load(tok_idx_ptr + idx, mask=mask, other=0)  # int32 token indices

        # Load qn vector chunk: qn[idx] is out-of-range because idx in [m, m+BLOCK_M),
        # but we only need qn dot Kc per token -> load qn scalar per token via index.
        # However, we cannot index vectors by idx in Triton. Instead, we pre-load qn_vec.
        # We need qn_vec = qn_ptr (1D). To load per token element, we reconstruct qn_elem per idx:
        # But Triton doesn't support dynamic indexing into qn_ptr like qn_ptr[tok].
        # Workaround: since qn_ptr is 1D (N), we can load qn_elem directly by tok (int32) via tl.load(qn_ptr + tok, mask=mask).
        # Similarly for Kc_ptr and Kp_ptr.

        # Build pointers for Kc rows and Kp rows for this chunk
        Kc_chunk_ptr = Kc_ptr + tok * N  # each row in Kc is length N, contiguous
        Kp_chunk_ptr = Kp_ptr + tok * Kp_dim  # each row in Kp is length Kp_dim, contiguous

        # Load qn elements for tokens
        qn_vec = tl.load(qn_ptr + tok, mask=mask, other=0.0)  # fp32 [BLOCK_M]
        # Load qp elements for tokens
        qp_vec = tl.load(qp_ptr + tok, mask=mask, other=0.0)  # fp32 [BLOCK_M]

        # Load Kc rows and Kp rows for tokens
        Kc_rows = tl.load(Kc_chunk_ptr + tl.arange(0, N), mask=mask[:, None], other=0.0)  # [BLOCK_M, N]
        Kp_rows = tl.load(Kp_chunk_ptr + tl.arange(0, Kp_dim), mask=mask[:, None], other=0.0)  # [BLOCK_M, Kp_dim]

        # Compute dot products for this chunk
        # qn · Kc: sum over N dimension
        dot_qn = tl.sum(qn_vec[:, None] * Kc_rows, axis=1)  # [BLOCK_M]
        # qp · Kp: sum over Kp_dim dimension
        dot_qp = tl.sum(qp_vec[:, None] * Kp_rows, axis=1)  # [BLOCK_M]

        logits = dot_qn + dot_qp  # [BLOCK_M]
        # Apply scaling
        logits_scaled = logits * sm_scale

        # Update row_max and sum_exp
        chunk_max = tl.max(tl.where(mask, logits_scaled, -float("inf")), axis=0)
        new_row_max = tl.maximum(row_max, chunk_max)
        # Compute sum_exp += sum(exp(logits_scaled - new_row_max)) over valid entries
        exp_chunk = tl.exp(logits_scaled - new_row_max)
        exp_chunk = tl.where(mask, exp_chunk, 0.0)
        sum_exp += tl.sum(exp_chunk, axis=0)
        row_max = new_row_max
        m += BLOCK_M

    # Compute lse = log(sum_exp) / ln(2)
    ln2 = 0.6931471805599453
    lse_val = tl.log(sum_exp) / ln2
    tl.store(lse_ptr, lse_val)

    # Second pass: compute output vector y = sum_m exp(logits_scaled - lse) * Kc[m, :]
    m = 0
    while m < M_total:
        idx = m + tl.arange(0, BLOCK_M)
        mask = idx < M_total
        tok = tl.load(tok_idx_ptr + idx, mask=mask, other=0)  # int32

        Kc_chunk_ptr = Kc_ptr + tok * N
        Kp_chunk_ptr = Kp_ptr + tok * Kp_dim

        qn_vec = tl.load(qn_ptr + tok, mask=mask, other=0.0)
        qp_vec = tl.load(qp_ptr + tok, mask=mask, other=0.0)

        Kc_rows = tl.load(Kc_chunk_ptr + tl.arange(0, N), mask=mask[:, None], other=0.0)  # [BLOCK_M, N]
        Kp_rows = tl.load(Kp_chunk_ptr + tl.arange(0, Kp_dim), mask=mask[:, None], other=0.0)  # [BLOCK_M, Kp_dim]

        dot_qn = tl.sum(qn_vec[:, None] * Kc_rows, axis=1)  # [BLOCK_M]
        dot_qp = tl.sum(qp_vec[:, None] * Kp_rows, axis=1)  # [BLOCK_M]

        logits = dot_qn + dot_qp
        logits_scaled = logits * sm_scale
        attn = tl.exp(logits_scaled - lse_val)  # [BLOCK_M]
        attn = tl.where(mask, attn, 0.0)

        # Accumulate into output vector
        # out = sum over tokens of attn[m] * Kc[m, :]
        # We do this per N column: load Kc[m, n] and multiply by attn[m]
        # To accumulate per column, we can perform outer product reduction:
        # For each j in 0..N-1:
        #   Kc_col = tl.load(Kc_chunk_ptr + j, mask=mask)  # shape [BLOCK_M]
        #   contrib = attn * Kc_col  # [BLOCK_M]
        #   out[j] += sum(contrib, axis=0)
        # We'll loop over N columns.
        for j in range(0, N):
            Kc_col = tl.load(Kc_chunk_ptr + j, mask=mask, other=0.0)  # [BLOCK_M]
            contrib = attn * Kc_col  # [BLOCK_M]
            contrib = tl.where(mask, contrib, 0.0)
            # Reduce along chunk axis: sum over BLOCK_M
            # Since attn and Kc_col are [BLOCK_M], sum directly
            out_j_chunk = tl.sum(contrib, axis=0)
            # Atomic add to out_ptr[j]
            # Triton allows atomic_add on fp32
            # Note: out_ptr points to contiguous [N] memory
            tl.atomic_add(out_ptr + j, out_j_chunk)

        m += BLOCK_M


class ModelNew(torch.nn.Module):
    def __init__(self, block_m: int = 128, sm_scale: float = 1.0):
        super().__init__()
        self.block_m = block_m
        self.sm_scale = sm_scale

    def forward(self, *args):
        # Expect 7 args: q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale (optional)
        # Use *args unpacking to match evaluator’s signature
        # We will not use any PyTorch matmul/softmax on host; all math is in Triton kernel.
        # Unpack first 7 args
        if len(args) < 7:
            raise RuntimeError("ModelNew.forward requires at least 7 arguments")
        q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale = args[:7]
        # Ensure device consistency
        device = q_nope.device

        # Convert to fp32 and contiguous for Triton
        q_nope_fp32 = q_nope.to(torch.float32).contiguous()  # [B, H, N]
        q_pe_fp32 = q_pe.to(torch.float32).contiguous()     # [B, H, Kp_dim]

        # Flatten ckv_cache and kpe_cache to [num_pages, N] and [num_pages, Kp_dim]
        # Original tensors are [num_pages, 1, dim]; squeeze(1) is equivalent
        ckv_cache_fp32 = ckv_cache.to(torch.float32).contiguous()  # [num_pages, 1, N]
        kpe_cache_fp32 = kpe_cache.to(torch.float32).contiguous()  # [num_pages, 1, Kp_dim]
        Kc_fp32 = ckv_cache_fp32.view(-1, ckv_cache_fp32.shape[-1])  # [num_pages, N]
        Kp_fp32 = kpe_cache_fp32.view(-1, kpe_cache_fp32.shape[-1])  # [num_pages, Kp_dim]

        # Compute M_total per batch from kv_indptr
        # kv_indptr has shape [B+1]
        if kv_indptr.device != q_nope.device:
            kv_indptr = kv_indptr.to(device)
        if kv_indices.device != q_nope.device:
            kv_indices = kv_indices.to(device)
        M_total_list = (kv_indptr[1:] - kv_indptr[:-1]).tolist()  # [B], int64 -> convert to int
        B = len(M_total_list)

        # Prepare outputs
        output_fp32 = torch.empty((B, q_nope_fp32.shape[1], q_nope_fp32.shape[2]), dtype=torch.float32, device=device)
        lse = torch.empty((B, q_nope_fp32.shape[1]), dtype=torch.float32, device=device)

        # Launch Triton kernel per (b, h)
        # Note: We need to pass q_nope[b,h,:] and q_pe[b,h,:] into kernel. Build those on-the-fly per iteration.
        H = q_nope_fp32.shape[1]
        N = q_nope_fp32.shape[2]
        Kp_dim = q_pe_fp32.shape[2]

        # Triton grid: one program per (b,h)
        grid = (B * H,)

        # We need a lambda to compute per-problem grid; Triton supports calling kernel with args. But since we have loops inside kernel, grid size should match number of problems (B*H).
        for b in range(B):
            M_total_b = M_total_list[b]
            # Build tok_idx for this batch
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            tok_idx = kv_indices[start:end].to(torch.int32).contiguous()  # [M_total_b]
            if M_total_b <= 0:
                # No tokens, set output to zeros and lse to -inf
                output_fp32[b] = torch.zeros((H, N), dtype=torch.float32, device=device)
                lse[b] = -float("inf")
                continue

            for h in range(H):
                qn_vec = q_nope_fp32[b, h, :].contiguous()  # [N]
                qp_vec = q_pe_fp32[b, h, :].contiguous()    # [Kp_dim]

                # Prepare output vector and lse scalar
                out = torch.zeros((N,), dtype=torch.float32, device=device)
                lse_scalar = torch.empty((), dtype=torch.float32, device=device)

                # Launch kernel
                compute_lse_and_output_kernel[(grid,)](
                    qn_vec,                       # *fp32 [N]
                    qp_vec,                       # *fp32 [Kp_dim]
                    Kc_fp32,                      # *fp32 [num_pages, N]
                    Kp_fp32,                      # *fp32 [num_pages, Kp_dim]
                    tok_idx,                      # *int32 [M_total_b]
                    lse_scalar,                   # *fp32 scalar
                    out,                          # *fp32 [N]
                    N,                            # int32
                    Kp_dim,                       # int32
                    M_total_b,                    # int32
                    float(self.sm_scale if self.sm_scale is not None else sm_scale),  # fp32
                    BLOCK_M=self.block_m
                )

                # Store results
                lse[b, h] = lse_scalar
                output_fp32[b, h, :] = out

        # Cast output to bfloat16 to match original function's output dtype
        output_bf16 = output_fp32.to(torch.bfloat16)
        return output_bf16, lse