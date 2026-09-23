import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernels: invoked from forward, no torch ops inside.

if TRITON_AVAILABLE:
    @triton.jit
    def row_gemv_kernel_512(y_ptr,  # *float32, output [M]
                            q_ptr,  # *float32, row vector [512]
                            K_ptr,  # *float32, matrix [M, 512]
                            M: tl.constexpr,
                            D_in: tl.constexpr):
        pid = tl.program_id(0)  # program id for each output element along M
        # Each program computes y[pid] = sum_j q[j] * K[pid, j]
        acc = 0.0
        for j in range(0, D_in):
            qj = tl.load(q_ptr + j)  # q[j]
            Kj = tl.load(K_ptr + pid * D_in + j)  # K[pid, j]
            acc += qj * Kj
        tl.store(y_ptr + pid, acc)

    @triton.jit
    def row_gemv_kernel_64(y_ptr,  # *float32, output [M]
                           q_ptr,  # *float32, row vector [64]
                           K_ptr,  # *float32, matrix [M, 64]
                           M: tl.constexpr,
                           D_in: tl.constexpr):
        pid = tl.program_id(0)  # program id for each output element along M
        acc = 0.0
        for j in range(0, D_in):
            qj = tl.load(q_ptr + j)  # q[j]
            Kj = tl.load(K_ptr + pid * D_in + j)  # K[pid, j]
            acc += qj * Kj
        tl.store(y_ptr + pid, acc)

    @triton.jit
    def scale_1d_kernel(x_ptr,  # *float32, input vector [N]
                        y_ptr,  # *float32, output vector [N]
                        scale: tl.float32,
                        N: tl.constexpr):
        pid = tl.program_id(0)
        val = tl.load(x_ptr + pid)
        tl.store(y_ptr + pid, val * scale)

    @triton.jit
    def mask_neg_inf_1d_kernel(x_ptr,  # *float32, input vector [M]
                               y_ptr,  # *float32, output vector [M]
                               M: tl.constexpr,
                               pos: tl.int32):
        # Set j <= pos to -inf, keep others
        for j in range(0, M):
            xj = tl.load(x_ptr + j)
            keep = j > pos
            outj = tl.where(keep, xj, -float('inf'))
            tl.store(y_ptr + j, outj)

    @triton.jit
    def lse_row_kernel(x_ptr,  # *float32, input vector [M]
                        out_ptr,  # *float32, scalar output
                        M: tl.constexpr):
        # lse = log(sum(exp(x - max))) / ln(2)
        maxv = -float('inf')
        # find max
        for j in range(0, M):
            xj = tl.load(x_ptr + j)
            if xj > maxv:
                maxv = xj
        sum_exp = 0.0
        for j in range(0, M):
            xj = tl.load(x_ptr + j)
            sum_exp += tl.exp(xj - maxv)
        lse_val = tl.log(sum_exp) / 0.6931471805599453  # 1 / ln(2)
        tl.store(out_ptr, lse_val)

    @triton.jit
    def softmax_masked_row_kernel(x_ptr,  # *float32, input vector [M]
                                  y_ptr,  # *float32, output vector [M]
                                  M: tl.constexpr):
        # softmax(x) = exp(x - max) / sum(exp(x - max))
        maxv = -float('inf')
        for j in range(0, M):
            xj = tl.load(x_ptr + j)
            if xj > maxv:
                maxv = xj
        sum_exp = 0.0
        for j in range(0, M):
            xj = tl.load(x_ptr + j)
            sum_exp += tl.exp(xj - maxv)
        for j in range(0, M):
            xj = tl.load(x_ptr + j)
            yj = tl.exp(xj - maxv) / sum_exp
            tl.store(y_ptr + j, yj)

    @triton.jit
    def gemv_row_kernel(y_ptr,   # *float32, input vector [M]
                        K_ptr,   # *float32, matrix [M, 512]
                        out_ptr, # *float32, output vector [512]
                        M: tl.constexpr,
                        Dn: tl.constexpr):
        # out[d] = sum_j y[j] * K[j, d]
        for d in range(0, Dn):
            acc = 0.0
            for j in range(0, M):
                yj = tl.load(y_ptr + j)
                Kjd = tl.load(K_ptr + j * Dn + d)
                acc += yj * Kjd
            tl.store(out_ptr + d, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA and contiguous tensors
        device = q_nope.device
        if not q_nope.is_cuda:
            q_nope = q_nope.cuda(non_blocking=True)
        if not q_pe.is_cuda:
            q_pe = q_pe.cuda(non_blocking=True)
        if not ckv_cache.is_cuda:
            ckv_cache = ckv_cache.cuda(non_blocking=True)
        if not kpe_cache.is_cuda:
            kpe_cache = kpe_cache.cuda(non_blocking=True)
        if not qo_indptr.is_cuda:
            qo_indptr = qo_indptr.cuda(non_blocking=True)
        if not kv_indptr.is_cuda:
            kv_indptr = kv_indptr.cuda(non_blocking=True)
        if not kv_indices.is_cuda:
            kv_indices = kv_indices.cuda(non_blocking=True)

        # Cast caches to float32 for kernel math
        Kc_all = ckv_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, 64]

        total_q = int(qo_indptr[-1].item())
        num_qo_heads = q_nope.shape[1]
        head_dim_ckv = q_nope.shape[2]
        head_dim_kpe = q_pe.shape[2]
        # Constants from asserts
        assert num_qo_heads == 16
        assert head_dim_ckv == 512
        assert head_dim_kpe == 64

        # Output allocation
        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Process batches
        for b in range(qo_indptr.shape[0] - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            if q_start >= q_end:
                continue

            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())
            if kv_start >= kv_end:
                # no KV for this batch element, fill zeros and proceed
                # For each head h:
                for h in range(num_qo_heads):
                    # logits zeros -> lse = -inf (unchanged), out zeros
                    lse[q_start, h] = -float('inf')
                    output[q_start, h, :] = torch.zeros((head_dim_ckv,), dtype=torch.bfloat16, device=device)
                continue

            kv_len = kv_end - kv_start
            tok_idx = kv_indices[kv_start:kv_end].to(torch.int32)  # [M]
            M = tok_idx.numel()

            # Gather Kc and Kp rows for this batch element
            Kc_rows = Kc_all[tok_idx].contiguous().to(torch.float32)  # [M, 512]
            Kp_rows = Kp_all[tok_idx].contiguous().to(torch.float32)  # [M, 64]

            # For each query i in this batch
            for i in range(q_end - q_start):
                # For each head h
                for h in range(num_qo_heads):
                    q_row_nope = q_nope[q_start + i, h, :].contiguous().to(torch.float32)  # [512]
                    q_row_pe = q_pe[q_start + i, h, :].contiguous().to(torch.float32)      # [64]

                    # Compute qn_logits[M] = q_row_nope @ Kc_rows.T
                    qn_logits = torch.empty((M,), dtype=torch.float32, device=device)
                    grid = (M,)
                    row_gemv_kernel_512[grid](qn_logits, q_row_nope, Kc_rows, M= M, D_in=512)

                    # Compute qp_logits[M] = q_row_pe @ Kp_rows.T
                    qp_logits = torch.empty((M,), dtype=torch.float32, device=device)
                    grid = (M,)
                    row_gemv_kernel_64[grid](qp_logits, q_row_pe, Kp_rows, M= M, D_in=64)

                    logits = qn_logits + qp_logits  # [M]

                    # Scale logits
                    logits_scaled = torch.empty((M,), dtype=torch.float32, device=device)
                    grid = (M,)
                    scale_1d_kernel[grid](logits, logits_scaled, sm_scale, N=M)

                    # Apply causal mask: j <= (prefix_len + i) -> -inf
                    prefix_len = kv_len - (q_end - q_start)  # number of previously cached tokens
                    query_abs_pos = prefix_len + i
                    logits_masked = torch.empty((M,), dtype=torch.float32, device=device)
                    grid = (M,)
                    mask_neg_inf_1d_kernel[grid](logits_scaled, logits_masked, M, query_abs_pos)

                    # Compute lse for this head
                    lse_val = torch.empty((1,), dtype=torch.float32, device=device)
                    grid = (1,)
                    lse_row_kernel[grid](logits_masked, lse_val, M=M)
                    lse[q_start + i, h] = lse_val[0]

                    # Compute softmax over masked logits
                    softmax_out = torch.empty((M,), dtype=torch.float32, device=device)
                    grid = (M,)
                    softmax_masked_row_kernel[grid](logits_masked, softmax_out, M)

                    # Compute out[h, :] = softmax_out @ Kc_rows.T -> [512]
                    out_row = torch.empty((head_dim_ckv,), dtype=torch.float32, device=device)
                    grid = (head_dim_ckv,)
                    gemv_row_kernel[grid](softmax_out, Kc_rows, out_row, M=M, Dn=512)

                    # Store results
                    output[q_start + i, h, :] = out_row.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
