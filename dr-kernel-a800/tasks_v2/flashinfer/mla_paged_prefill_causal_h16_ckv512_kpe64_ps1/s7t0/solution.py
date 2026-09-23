import math
import torch
import triton
import triton.language as tl


@triton.jit
def left_matmul_kernel(A_ptr, B_ptr, C_ptr,
                        M, N, K,
                        stride_am, stride_an,
                        stride_bn, stride_bk,
                        stride_cm, stride_ck,
                        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # Each program computes a tile of C: (BLOCK_M, BLOCK_N)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in chunks
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Pointers for A and B tiles
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_an)
        b_ptrs = B_ptr + (offs_k[:, None] * stride_bn + offs_n[None, :] * stride_bk)

        # Masks for boundaries
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        # Load tiles; cast to fp32 for accumulation
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)  # [BLOCK_M, BLOCK_K]
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)  # [BLOCK_K, BLOCK_N]

        # Accumulate
        acc += tl.dot(a, b)

    # Write results
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_ck)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def softmax_with_mask_kernel(X_ptr, Mask_ptr, Out_ptr,
                              M, N,
                              stride_xm, stride_xn,
                              stride_out_m, stride_out_n,
                              BLOCK_N: tl.constexpr):
    # One program per row
    pid_m = tl.program_id(0)
    # Load row
    row_x = tl.load(X_ptr + pid_m * stride_xm + tl.arange(0, N) * stride_xn,
                    mask=(tl.arange(0, N) < N),
                    other=-float("inf")).to(tl.float32)
    # Causal mask: for each column j, if j >= query_abs_pos then set -inf
    # We receive query_abs_pos from host via Mask_ptr at index pid_m (assumes Mask[pid_m] is int32)
    query_abs_pos = tl.load(Mask_ptr + pid_m).to(tl.int32)
    cols = tl.arange(0, N)
    mask_inf = cols < query_abs_pos
    row_x = tl.where(mask_inf, row_x, -float("inf"))

    # Numerically stable softmax
    row_max = tl.max(row_x, axis=0)
    row_x = row_x - row_max
    exp_row = tl.exp(row_x)
    row_sum = tl.sum(exp_row, axis=0)
    row_soft = exp_row / row_sum

    # Store
    tl.store(Out_ptr + pid_m * stride_out_m + tl.arange(0, N) * stride_out_n, row_soft, mask=(tl.arange(0, N) < N))


@triton.jit
def lse_with_mask_base2_kernel(X_ptr, Mask_ptr, Out_ptr,
                                M, N,
                                stride_xm, stride_xn,
                                stride_out_m,
                                BLOCK_N: tl.constexpr):
    # One program per row
    pid_m = tl.program_id(0)
    row_x = tl.load(X_ptr + pid_m * stride_xm + tl.arange(0, N) * stride_xn,
                    mask=(tl.arange(0, N) < N),
                    other=-float("inf")).to(tl.float32)
    query_abs_pos = tl.load(Mask_ptr + pid_m).to(tl.int32)
    cols = tl.arange(0, N)
    mask_inf = cols < query_abs_pos
    row_x = tl.where(mask_inf, row_x, -float("inf"))

    row_max = tl.max(row_x, axis=0)
    row_x = row_x - row_max
    exp_row = tl.exp(row_x)
    row_sum = tl.sum(exp_row, axis=0)
    lse_val = row_max + math.log(2.0) * row_sum  # logsumexp base 2

    tl.store(Out_ptr + pid_m * stride_out_m, lse_val)


class ModelNew(torch.nn.Module):
    def __init__(self, num_qo_heads=16, head_dim_ckv=512, head_dim_kpe=64):
        super().__init__()
        self.num_qo_heads = num_qo_heads
        self.head_dim_ckv = head_dim_ckv
        self.head_dim_kpe = head_dim_kpe

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure device and dtype
        device = q_nope.device
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "All tensors must be on CUDA for Triton."
        # Constants and basic checks
        total_q, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]
        num_pages = ckv_cache.shape[0]
        assert num_pages == kpe_cache.shape[0], "ckv_cache and kpe_cache must have same num_pages"
        # We can use num_qo_heads from constructor; original asserts it equals 16. We'll enforce that here.
        assert num_qo_heads == self.num_qo_heads, "num_qo_heads must equal 16"
        assert head_dim_ckv == self.head_dim_ckv, "head_dim_ckv must equal 512"
        assert head_dim_kpe == self.head_dim_kpe, "head_dim_kpe must equal 64"

        # Global caches
        Kc_all = ckv_cache.squeeze(1).contiguous()  # [num_pages, head_dim_ckv]
        Kp_all = kpe_cache.squeeze(1).contiguous()  # [num_pages, head_dim_kpe]

        # Output initialization (fp32 for numerical stability)
        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Process each batch element
        len_indptr = qo_indptr.shape[0]
        batch_size = len_indptr - 1

        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            if q_start >= q_end:
                continue

            q_len = q_end - q_start

            # KV range
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            if page_beg >= page_end:
                # No KV for this batch element; lse and output remain zeros (we'll handle empty)
                # For now, just skip; since q_len would be 0, the loop won't run
                pass

            # Gather token indices for this batch element
            tok_idx = kv_indices[page_beg:page_end].to(torch.int32).contiguous()
            kv_len = tok_idx.numel()
            Kc = Kc_all[tok_idx]  # [kv_len, head_dim_ckv], fp32
            Kp = Kp_all[tok_idx]  # [kv_len, head_dim_kpe], fp32

            # Prepare batch of queries
            q_nope_batch = q_nope[q_start:q_end].contiguous()  # [q_len, num_qo_heads, head_dim_ckv]
            q_pe_batch = q_pe[q_start:q_end].contiguous()      # [q_len, num_qo_heads, head_dim_kpe]

            # We will compute per query i
            for i in range(q_len):
                qn = q_nope_batch[i]  # [num_qo_heads, head_dim_ckv], bf16 but we'll pass fp32 view
                qp = q_pe_batch[i]    # [num_qo_heads, head_dim_kpe], bf16 but we'll pass fp32 view

                # Convert to fp32 for computation
                qn_f = qn.to(torch.float32)  # [16, 512]
                qp_f = qp.to(torch.float32)  # [16, 64]

                Kc_f = Kc.contiguous()  # [kv_len, 512]
                Kp_f = Kp.contiguous()  # [kv_len, 64]

                # Compute logits = qn @ Kc.T + qp @ Kp.T, shapes [num_qo_heads, kv_len]
                # We'll call left_matmul_kernel twice
                M = num_qo_heads
                N = kv_len
                Kq = head_dim_ckv
                KpN = head_dim_kpe

                # Kernel 1: qn @ Kc.T -> logits1 [M, N]
                C1 = torch.empty((M, N), dtype=torch.float32, device=device)
                stride_am = qn_f.stride(0)  # 512
                stride_an = qn_f.stride(1)  # 1
                stride_bk = Kc_f.stride(0)  # 512
                stride_bn = Kc_f.stride(1)  # 1
                stride_cm = C1.stride(0)    # N
                stride_cn = C1.stride(1)    # 1

                # Tiling heuristics
                BLOCK_M = 16  # matches num_qo_heads
                BLOCK_N = 64 if N >= 64 else 32
                BLOCK_K = 64 if Kq >= 64 else 32

                grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
                left_matmul_kernel[grid](
                    qn_f, Kc_f, C1,
                    M, N, Kq,
                    stride_am, stride_an,
                    stride_bk, stride_bn,
                    stride_cm, stride_cn,
                    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                    num_warps=4, num_stages=2
                )

                # Kernel 2: qp @ Kp.T -> logits2 [M, N]
                C2 = torch.empty((M, N), dtype=torch.float32, device=device)
                stride_am_qp = qp_f.stride(0)  # 64
                stride_an_qp = qp_f.stride(1)  # 1
                stride_bk_qp = Kp_f.stride(0)  # 64
                stride_bn_qp = Kp_f.stride(1)  # 1
                stride_cm2 = C2.stride(0)
                stride_cn2 = C2.stride(1)

                left_matmul_kernel[grid](
                    qp_f, Kp_f, C2,
                    M, N, KpN,
                    stride_am_qp, stride_an_qp,
                    stride_bk_qp, stride_bn_qp,
                    stride_cm2, stride_cn2,
                    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                    num_warps=4, num_stages=2
                )

                # Sum and scale
                logits = C1 + C2  # [16, N]
                logits_scaled = logits * sm_scale

                # Causal mask: prefix_len = kv_len - q_len; query_abs_pos = prefix_len + i
                # Note: prefix_len may be negative, which means no masking needed; we handle it anyway.
                prefix_len = kv_len - q_len
                query_abs_pos = prefix_len + i

                # Triton softmax with mask
                out_softmax = torch.empty((M, N), dtype=torch.float32, device=device)
                # Mask_ptr holds int32 query_abs_pos for each row; here only one row, so we pass [1] and index 0.
                mask_row = torch.tensor([query_abs_pos], dtype=torch.int32, device=device)
                softmax_with_mask_kernel[(M,)](
                    logits_scaled, mask_row, out_softmax,
                    M, N,
                    out_softmax.stride(0), out_softmax.stride(1),
                    out_softmax.stride(0), out_softmax.stride(1),
                    BLOCK_N=128 if N >= 128 else (64 if N >= 64 else 32),
                    num_warps=4, num_stages=2
                )

                # Triton LSE (base-2) with mask
                lse_row = torch.empty((M,), dtype=torch.float32, device=device)
                lse_with_mask_base2_kernel[(M,)](
                    logits_scaled, mask_row, lse_row,
                    M, N,
                    logits_scaled.stride(0), logits_scaled.stride(1),
                    lse_row.stride(0),
                    BLOCK_N=128 if N >= 128 else (64 if N >= 64 else 32),
                    num_warps=4, num_stages=2
                )

                # Store per query
                # We need attn @ Kc to produce final output. Implement this matmul in Triton.
                attn = out_softmax  # [16, N]
                # Compute attn @ Kc -> [16, head_dim_ckv]
                out_vec = torch.empty((M, head_dim_ckv), dtype=torch.float32, device=device)
                stride_am_attn = attn.stride(0)  # N
                stride_an_attn = attn.stride(1)  # 1
                stride_bk_out = Kc_f.stride(0)   # head_dim_ckv
                stride_bn_out = Kc_f.stride(1)   # 1
                stride_cm_out = out_vec.stride(0)  # head_dim_ckv
                stride_cn_out = out_vec.stride(1)  # 1

                # Tiling for (M, Kq)
                BLOCK_M2 = 16
                BLOCK_N2 = 64 if head_dim_ckv >= 64 else 32
                BLOCK_K2 = 64 if head_dim_ckv >= 64 else 32

                left_matmul_kernel[(triton.cdiv(M, BLOCK_M2), triton.cdiv(head_dim_ckv, BLOCK_N2))](
                    attn, Kc_f, out_vec,
                    M, head_dim_ckv, head_dim_ckv,
                    stride_am_attn, stride_an_attn,
                    stride_bk_out, stride_bn_out,
                    stride_cm_out, stride_cn_out,
                    BLOCK_M=BLOCK_M2, BLOCK_N=BLOCK_N2, BLOCK_K=BLOCK_K2,
                    num_warps=4, num_stages=2
                )

                # Assign to output
                output[q_start + i] = out_vec  # shape [16, 512]
                # Assign to lse
                lse[q_start + i] = lse_row[0]  # scalar per query

        # Cast output to bfloat16 to match original example
        output = output.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
