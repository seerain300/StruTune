import torch
import math
import triton
import triton.language as tl


@triton.jit
def row_gemv_kernel_512(y_ptr, q_ptr, K_ptr, M: tl.constexpr, D: tl.constexpr):
    """
    Compute y[j] = sum_{d=0..D-1} q[d] * K[j, d]
    y_ptr: *float32, output vector [M]
    q_ptr: *float32, row vector [D], D=512
    K_ptr: *float32, matrix [M, D], contiguous
    """
    pid_m = tl.program_id(0)  # each program handles one j
    acc = tl.zeros((), dtype=tl.float32)
    # D is constexpr, so a for-loop is valid
    for d in range(D):
        qd = tl.load(q_ptr + d)
        K_jd = tl.load(K_ptr + pid_m * D + d)
        acc += qd * K_jd
    tl.store(y_ptr + pid_m, acc)


@triton.jit
def row_gemv_kernel_64(y_ptr, q_ptr, K_ptr, M: tl.constexpr, D: tl.constexpr):
    """
    Compute y[j] = sum_{d=0..D-1} q[d] * K[j, d]
    y_ptr: *float32, output vector [M]
    q_ptr: *float32, row vector [D], D=64
    K_ptr: *float32, matrix [M, D], contiguous
    """
    pid_m = tl.program_id(0)
    acc = tl.zeros((), dtype=tl.float32)
    for d in range(D):
        qd = tl.load(q_ptr + d)
        K_jd = tl.load(K_ptr + pid_m * D + d)
        acc += qd * K_jd
    tl.store(y_ptr + pid_m, acc)


@triton.jit
def scale_1d_kernel(x_ptr, y_ptr, scale: tl.float32, N: tl.constexpr):
    """
    y[i] = x[i] * scale for i in [0..N)
    """
    pid = tl.program_id(0)
    xi = tl.load(x_ptr + pid)
    tl.store(y_ptr + pid, xi * scale)


@triton.jit
def mask_neg_inf_1d_kernel(x_ptr, y_ptr, pos: tl.int32, N: tl.constexpr):
    """
    Apply causal mask: keep x[i] if i > pos else -inf
    """
    pid = tl.program_id(0)
    xi = tl.load(x_ptr + pid)
    keep = pid > pos
    yi = tl.where(keep, xi, -float("inf"))
    tl.store(y_ptr + pid, yi)


@triton.jit
def lse_row_kernel(x_ptr, lse_ptr, N: tl.constexpr):
    """
    Compute logsumexp(x) across N elements, divided by ln(2).
    """
    # compute max
    max_val = -float("inf")
    for i in range(N):
        xi = tl.load(x_ptr + i)
        if xi > max_val:
            max_val = xi
    # sum exp(x - max)
    sum_exp = 0.0
    for i in range(N):
        xi = tl.load(x_ptr + i)
        sum_exp += tl.exp(xi - max_val)
    lse = tl.log(sum_exp) + max_val
    ln2 = 1.4426950408889634  # 1 / ln(2)
    tl.store(lse_ptr, lse / ln2)


@triton.jit
def softmax_masked_row_kernel(x_ptr, y_ptr, pos: tl.int32, N: tl.constexpr):
    """
    Compute softmax over x with causal mask: if i <= pos, x[i] = -inf before softmax.
    y[i] = exp(x[i] - max) / sum_j exp(x[j] - max)
    """
    # compute max
    max_val = -float("inf")
    for i in range(N):
        xi = tl.load(x_ptr + i)
        keep = i > pos
        xi = tl.where(keep, xi, -float("inf"))
        if xi > max_val:
            max_val = xi
    # compute sum of exp
    sum_exp = 0.0
    for i in range(N):
        xi = tl.load(x_ptr + i)
        keep = i > pos
        xi = tl.where(keep, xi, -float("inf"))
        sum_exp += tl.exp(xi - max_val)
    # write normalized softmax
    for i in range(N):
        xi = tl.load(x_ptr + i)
        keep = i > pos
        xi = tl.where(keep, xi, -float("inf"))
        yi = tl.exp(xi - max_val) / sum_exp
        tl.store(y_ptr + i, yi)


@triton.jit
def gemv_row_kernel(y_ptr, K_ptr, out_ptr, M: tl.constexpr, Dn: tl.constexpr):
    """
    out[d] = sum_{j=0..M-1} y[j] * K[j, d] where K is [M, Dn]
    y_ptr: *float32, input vector [M]
    K_ptr: *float32, matrix [M, Dn], contiguous
    out_ptr: *float32, output vector [Dn]
    """
    pid_d = tl.program_id(0)  # each program handles one feature d
    acc = tl.zeros((), dtype=tl.float32)
    for j in range(M):
        yj = tl.load(y_ptr + j)
        K_jd = tl.load(K_ptr + j * Dn + pid_d)
        acc += yj * K_jd
    tl.store(out_ptr + pid_d, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure on CUDA
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "All inputs must be on CUDA"
        assert qo_indptr.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "Index tensors must be on CUDA"
        device = q_nope.device

        total_q = int(qo_indptr[-1].item())
        num_qo_heads = q_nope.shape[1]
        head_dim_ckv = q_nope.shape[2]  # 512
        head_dim_kpe = q_pe.shape[2]    # 64

        # Constants from reference
        assert num_qo_heads == 16
        assert head_dim_ckv == 512
        assert head_dim_kpe == 64

        # Prepare cached Ks as float32, squeeze the "1" dim
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, 64]

        # Output and lse allocation
        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Process each batch element
        for b in range(qo_indptr.shape[0] - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())
            if q_start >= q_end or kv_start >= kv_end:
                continue

            # Gather token indices and corresponding rows from caches
            tok_idx = kv_indices[kv_start:kv_end]  # [M]
            Kc_rows = Kc_all[tok_idx]              # [M, 512]
            Kp_rows = Kp_all[tok_idx]              # [M, 64]
            kv_len = Kc_rows.shape[0]
            Dn = Kc_rows.shape[1]
            Dp = Kp_rows.shape[1]

            # Loop over queries in this batch
            for i in range(q_start, q_end):
                # For each head h
                for h in range(num_qo_heads):
                    # Load qn_row[h, :] and qp_row[h, :] as float32 vectors
                    qn_vec = q_nope[i, h, :].to(torch.float32).contiguous()  # [512]
                    qp_vec = q_pe[i, h, :].to(torch.float32).contiguous()   # [64]

                    # Compute logits components via Triton GEMV
                    qn_logits = torch.empty(kv_len, dtype=torch.float32, device=device)
                    # Launch GEMV for qn
                    grid_qn = (kv_len,)
                    row_gemv_kernel_512[grid_qn](
                        qn_logits, qn_vec, Kc_rows, M=kv_len, D=512
                    )

                    qp_logits = torch.empty(kv_len, dtype=torch.float32, device=device)
                    # Launch GEMV for qp
                    grid_qp = (kv_len,)
                    row_gemv_kernel_64[grid_qp](
                        qp_logits, qp_vec, Kp_rows, M=kv_len, D=64
                    )

                    logits = qn_logits + qp_logits  # [M]
                    # Scale logits
                    logits_scaled = torch.empty_like(logits)
                    grid_scale = (kv_len,)
                    scale_1d_kernel[grid_scale](logits, logits_scaled, sm_scale, N=kv_len)

                    # Apply causal mask: j > (prefix_len + i)
                    prefix_len = kv_len - (q_end - q_start)
                    query_abs_pos = prefix_len + (i - q_start)  # absolute position of this query in sequence
                    logits_masked = torch.empty_like(logits_scaled)
                    grid_mask = (kv_len,)
                    mask_neg_inf_1d_kernel[grid_mask](logits_scaled, logits_masked, pos=query_abs_pos, N=kv_len)

                    # Compute LSE per head
                    lse_row = torch.empty(1, dtype=torch.float32, device=device)
                    grid_lse = (1,)
                    lse_row_kernel[grid_lse](logits_masked, lse_row, N=kv_len)
                    lse[i, h] = lse_row[0]

                    # Compute softmax (masked) for this head
                    softmax_out = torch.empty(kv_len, dtype=torch.float32, device=device)
                    grid_softmax = (kv_len,)
                    softmax_masked_row_kernel[grid_softmax](logits_masked, softmax_out, pos=query_abs_pos, N=kv_len)

                    # Compute final output: out[h, :] = softmax @ Kc_rows
                    out_row = torch.empty(Dn, dtype=torch.float32, device=device)
                    grid_gemv = (Dn,)
                    gemv_row_kernel[grid_gemv](softmax_out, Kc_rows, out_row, M=kv_len, Dn=Dn)

                    # Store output row as bfloat16
                    output[i, h, :] = out_row.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
