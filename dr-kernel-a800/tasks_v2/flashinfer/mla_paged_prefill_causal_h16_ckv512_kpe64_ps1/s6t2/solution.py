import math
import torch
import triton
import triton.language as tl


@triton.jit
def compute_logits_kernel(
    q_nope_ptr,   # *fp32 [H, D_ckv]
    q_pe_ptr,     # *fp32 [H, D_kpe]
    Kc_ptr,       # *fp32 [L, D_ckv]
    Kp_ptr,       # *fp32 [L, D_kpe]
    out_ptr,      # *fp32 [H, L]
    H: tl.constexpr,  # num_heads, e.g., 16
    L,                # int32, number of KV rows
    D_ckv,            # int32, head_dim_ckv
    D_kpe,            # int32, head_dim_kpe
    qn_stride0,       # int32, stride between rows in q_nope
    qn_stride1,       # int32, stride between cols in q_nope
    qp_stride0,       # int32, stride between rows in q_pe
    qp_stride1,       # int32, stride between cols in q_pe
    Kc_stride0,       # int32, stride0 of Kc
    Kc_stride1,       # int32, stride1 of Kc
    Kp_stride0,       # int32, stride0 of Kp
    Kp_stride1,       # int32, stride1 of Kp
    out_stride0,      # int32, stride between rows in out
    out_stride1,      # int32, stride between cols in out
    BLOCK_N: tl.constexpr,  # tile along L
):
    # Grid: (H, cdiv(L, BLOCK_N))
    h_idx = tl.program_id(0)
    pid_n = tl.program_id(1)
    ls = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_ls = ls < L

    # Accumulator for logits for this head and tile
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Loop over combined D_q = D_ckv + D_kpe, sum contributions
    for k in range(D_ckv + D_kpe):
        # Contribution from q_nope[h, k] * Kc[ls, k]
        qn_val = tl.load(q_nope_ptr + h_idx * qn_stride0 + k * qn_stride1)
        Kc_vals = tl.load(Kc_ptr + ls * Kc_stride0 + k * Kc_stride1, mask=mask_ls, other=0.0)
        acc += qn_val * Kc_vals

        # Contribution from q_pe[h, k] * Kp[ls, k] if k >= D_ckv
        if (k >= D_ckv) and (k < D_ckv + D_kpe):
            k_kpe = k - D_ckv
            qp_val = tl.load(q_pe_ptr + h_idx * qp_stride0 + k_kpe * qp_stride1)
            Kp_vals = tl.load(Kp_ptr + ls * Kp_stride0 + k_kpe * Kp_stride1, mask=mask_ls, other=0.0)
            acc += qp_val * Kp_vals

    # Store logits for this head and tile
    out_ptrs = out_ptr + h_idx * out_stride0 + ls * out_stride1
    tl.store(out_ptrs, acc, mask=mask_ls)


@triton.jit
def lse_masked_kernel(
    logits_ptr,    # *fp32 [H, L]
    mask_ptr,      # *int32 [L], 0/1 mask indicating which positions are valid (not -inf)
    out_max_ptr,   # *fp32 [H]
    out_sumexp_ptr,# *fp32 [H]
    H,             # int32, number of heads
    L,             # int32, number of tokens
    stride_row,    # int32, row stride in logits (usually L)
    BLOCK_N: tl.constexpr,
):
    # Each program handles one head and tiles across L
    h_idx = tl.program_id(0)
    pid_n = tl.program_id(1)
    ls = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_ls = ls < L

    # Load mask for this tile
    mask_vec = tl.load(mask_ptr + ls, mask=mask_ls, other=1)  # 1 means valid, 0 means invalid

    # Load logits for this head and tile, apply mask: invalid positions become -inf
    logits_row_ptrs = logits_ptr + h_idx * stride_row + ls
    logits_row = tl.load(logits_row_ptrs, mask=mask_ls, other=0.0)
    logits_row = tl.where(mask_vec == 0, -float("inf"), logits_row)

    # Compute max and sumexp across tile
    max_val = tl.max(logits_row, axis=0)
    exp_row = tl.exp(logits_row - max_val)
    exp_row = tl.where(mask_vec == 0, 0.0, exp_row)  # zero out masked positions
    sumexp = tl.sum(exp_row, axis=0)

    # Store partial reductions
    tl.store(out_max_ptr + h_idx, max_val)
    tl.store(out_sumexp_ptr + h_idx, sumexp)


@triton.jit
def softmax_matmul_kernel(
    logits_ptr,    # *fp32 [H, L]
    Kc_ptr,        # *fp32 [L, D_ckv]
    out_ptr,       # *fp32 [H, D_ckv]
    H,             # int32
    L,             # int32
    D_ckv,         # int32
    stride_row,    # int32, row stride in logits (usually L)
    Kc_stride0,    # int32, stride0 of Kc
    Kc_stride1,    # int32, stride1 of Kc
    out_stride0,   # int32, stride between rows in out
    out_stride1,   # int32, stride between cols in out
    BLOCK_N: tl.constexpr,
):
    # Each program handles one head and writes its output vector
    h_idx = tl.program_id(0)

    # Compute row max and sumexp
    max_val = tl.max(tl.load(logits_ptr + h_idx * stride_row + tl.arange(0, L)), axis=0)
    ls = tl.arange(0, L)
    logits_row = tl.load(logits_ptr + h_idx * stride_row + ls)
    exp_row = tl.exp(logits_row - max_val)
    sumexp = tl.sum(exp_row, axis=0)

    # Output = softmax @ Kc
    out_vec = tl.zeros((D_ckv,), dtype=tl.float32)
    for l in range(0, L):
        kc_col = tl.load(Kc_ptr + l * Kc_stride0 + tl.arange(0, D_ckv) * Kc_stride1)  # [D_ckv]
        out_vec += (exp_row[l] / sumexp) * kc_col

    # Store output vector
    out_ptrs = out_ptr + h_idx * out_stride0 + tl.arange(0, D_ckv) * out_stride1
    tl.store(out_ptrs, out_vec)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        device = q_nope.device
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "All tensors must be on CUDA device"
        assert qo_indptr.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "Indptr and indices must be on CUDA device"

        # Shapes (fixed by original code)
        total_q, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]
        assert num_qo_heads == 16 and head_dim_ckv == 512 and head_dim_kpe == 64, "Fixed dims: num_qo_heads=16, head_dim_ckv=512, head_dim_kpe=64"

        # Prepare Kc_all and Kp_all: [num_pages, dim], float32, contiguous
        Kc_all = ckv_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, 64]

        # Output buffers
        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Compute batch_size and len_indptr (as in original)
        batch_size = int(kv_indptr.numel() - 1)
        len_indptr = qo_indptr.numel()

        # For each batch element, process queries and KVs
        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            if q_start >= q_end:
                continue

            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            if page_beg >= page_end:
                continue

            kv_len = page_end - page_beg
            # Token indices within this batch's KV block
            tok_idx = kv_indices[page_beg:page_end].to(torch.int64)  # [kv_len]
            # Select Kc and Kp rows for this batch
            Kc_batch = Kc_all[tok_idx]  # [kv_len, 512]
            Kp_batch = Kp_all[tok_idx]  # [kv_len, 64]

            # Process each query in this batch element
            q_len = q_end - q_start
            for i in range(q_len):
                # Load q_nope and q_pe rows for query i: [H, D] where D=D_ckv+D_kpe
                q_row_i_qn = q_nope[q_start + i].contiguous().to(torch.float32)  # [16, 512]
                q_row_i_qp = q_pe[q_start + i].contiguous().to(torch.float32)   # [16, 64]
                q_row_i = torch.cat([q_row_i_qn, q_row_i_qp], dim=1)            # [16, 576]

                # Compute logits = q_row_i @ Kc_batch.T + q_row_i @ Kp_batch.T
                logits = torch.empty((num_qo_heads, kv_len), dtype=torch.float32, device=device)
                grid = (num_qo_heads, triton.cdiv(kv_len, 128))
                compute_logits_kernel[grid](
                    q_row_i, torch.empty(0, device=device), torch.empty(0, device=device), Kc_batch, Kp_batch, logits,
                    H=num_qo_heads, L=kv_len, D_ckv=head_dim_ckv, D_kpe=head_dim_kpe,
                    qn_stride0=q_row_i.stride(0), qn_stride1=q_row_i.stride(1),
                    qp_stride0=0, qp_stride1=0,  # placeholders; q_row_i is both inputs
                    Kc_stride0=Kc_batch.stride(0), Kc_stride1=Kc_batch.stride(1),
                    Kp_stride0=Kp_batch.stride(0), Kp_stride1=Kp_batch.stride(1),
                    out_stride0=logits.stride(0), out_stride1=logits.stride(1),
                    BLOCK_N=128,
                    num_warps=4, num_stages=2,
                )

                # Scale logits
                logits_scaled = logits * sm_scale

                # Apply causal mask: positions l > (kv_len - q_len + i) are set to -inf
                abs_pos = kv_len - q_len + i
                arange_L = torch.arange(kv_len, device=device)
                mask_bool = arange_L > abs_pos  # [kv_len], bool
                mask_int = mask_bool.to(torch.int32)  # [kv_len], 0/1

                # Triton lse kernel: per-head max and sumexp
                out_max = torch.empty((num_qo_heads,), dtype=torch.float32, device=device)
                out_sumexp = torch.empty((num_qo_heads,), dtype=torch.float32, device=device)
                grid_lse = (num_qo_heads, triton.cdiv(kv_len, 128))
                lse_masked_kernel[grid_lse](
                    logits_scaled, mask_int, out_max, out_sumexp,
                    H=num_qo_heads, L=kv_len,
                    stride_row=logits_scaled.stride(1),
                    BLOCK_N=128,
                    num_warps=4, num_stages=2,
                )

                # lse = (log(sumexp) - max) / ln(2) (we already have log(sumexp) via Triton)
                # Note: Triton kernel computed sumexp directly; here we compute final lse using host math, but since Triton doesn't return log, we recompute using PyTorch:
                # We can instead compute lse in Triton by taking log inside the kernel. To strictly avoid host log, we recompute with PyTorch:
                ln2 = math.log(2.0)
                lse_i = (torch.log(out_sumexp) - out_max) / ln2  # [16]
                lse[q_start + i] = lse_i  # [16]

                # Compute output for each head: softmax(logits_scaled) @ Kc_batch
                for h in range(num_qo_heads):
                    out_row = torch.empty((head_dim_ckv,), dtype=torch.float32, device=device)
                    grid_out = (h,)  # one program per head
                    softmax_matmul_kernel[grid_out](
                        logits_scaled[h], Kc_batch, out_row,
                        H=num_qo_heads, L=kv_len, D_ckv=head_dim_ckv,
                        stride_row=logits_scaled[h].stride(0),  # for 1D row: stride is L
                        Kc_stride0=Kc_batch.stride(0), Kc_stride1=Kc_batch.stride(1),
                        out_stride0=out_row.stride(0), out_stride1=1,
                        BLOCK_N=128,
                        num_warps=4, num_stages=2,
                    )
                    output[q_start + i, h] = out_row.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
