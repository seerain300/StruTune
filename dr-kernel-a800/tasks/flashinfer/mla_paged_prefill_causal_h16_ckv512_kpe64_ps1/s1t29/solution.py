import math
import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M: tl.int32, K: tl.int32, N: tl.int32,
    stride_am: tl.int32, stride_ak: tl.int32,
    stride_bk: tl.int32, stride_bn: tl.int32,
    stride_cm: tl.int32, stride_cn: tl.int32,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D grid over output tiles
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over K dimension in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # Compute pointers for A and B tiles
        A_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        B_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn

        # Load tiles (masked for out-of-bounds)
        a = tl.load(A_ptrs, mask=(m_offsets[:, None] < M) & (k_offsets[None, :] < K), other=0.0)
        b = tl.load(B_ptrs, mask=(k_offsets[:, None] < K) & (n_offsets[None, :] < N), other=0.0)

        # Accumulate
        acc += tl.dot(a, b)

    # Write back results
    C_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    tl.store(C_ptrs, acc, mask=(m_offsets[:, None] < M) & (n_offsets[None, :] < N))


@triton.jit
def softmax_row_causal_kernel(
    X_ptr, Out_ptr,
    N: tl.int32,
    scale: tl.float32,           # currently 1.0
    absolute_pos: tl.int32,      # prefix_len + i
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    idx = tl.arange(0, BLOCK)
    # Load row
    x = tl.load(X_ptr + row_id * N + idx, mask=idx < N, other=-float("inf"))
    # Apply causal mask: positions j > absolute_pos -> -inf
    x = tl.where(idx <= absolute_pos, x, -float("inf"))
    # Stable softmax
    m = tl.max(x, axis=0)
    x = x - m
    exp_x = tl.exp(x)
    sum_exp = tl.sum(exp_x, axis=0)
    out = exp_x / sum_exp
    # Store
    tl.store(Out_ptr + row_id, out, mask=idx < N)


@triton.jit
def lse_row_causal_kernel(
    X_ptr, Out_ptr,
    N: tl.int32,
    scale: tl.float32,           # 1 / ln(2)
    absolute_pos: tl.int32,      # prefix_len + i
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    idx = tl.arange(0, BLOCK)
    x = tl.load(X_ptr + row_id * N + idx, mask=idx < N, other=-float("inf"))
    # Apply causal mask
    x = tl.where(idx <= absolute_pos, x, -float("inf"))
    m = tl.max(x, axis=0)
    x = x - m
    sum_exp = tl.sum(tl.exp(x), axis=0)
    lse = tl.log(sum_exp) * scale
    tl.store(Out_ptr + row_id, lse)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Convert caches to float32 and gather per batch
        device = q_nope.device
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "Tensors must be on CUDA"
        total_q = q_nope.shape[0]
        num_qo_heads = q_nope.shape[1]
        head_dim_ckv = q_nope.shape[2]
        head_dim_kpe = q_pe.shape[2]
        num_pages = ckv_cache.shape[0]
        assert head_dim_ckv == 512 and head_dim_kpe == 64

        Kc_all = ckv_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, 64]

        # Output tensors
        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Process each batch element
        for b in range(qo_indptr.shape[0] - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            if q_start >= q_end:
                continue
            q_len = q_end - q_start

            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            if page_beg >= page_end:
                continue
            kv_len = page_end - page_beg

            # Gather tokens
            tok_idx = kv_indices[page_beg:page_end].to(torch.long).to(device)  # [kv_len]
            Kc = Kc_all[tok_idx]  # [kv_len, 512]
            Kp = Kp_all[tok_idx]  # [kv_len, 64]

            # Per-query loop
            for i in range(q_len):
                abs_q = q_start + i
                # Load qn, qp
                qn = q_nope[abs_q].contiguous().to(torch.float32)  # [16, 512]
                qp = q_pe[abs_q].contiguous().to(torch.float32)   # [16, 64]

                # scores_n = qn @ Kc.T, scores_p = qp @ Kp.T
                scores_n = torch.empty((num_qo_heads, kv_len), dtype=torch.float32, device=device)
                scores_p = torch.empty((num_qo_heads, kv_len), dtype=torch.float32, device=device)

                # Launch Triton matmul for scores_n
                # A = qn [16,512], B = Kc.T [512,kv_len]
                grid_n = (triton.cdiv(num_qo_heads, 16), triton.cdiv(kv_len, 64))
                matmul_kernel[grid_n](
                    qn, Kc.t(), scores_n,
                    num_qo_heads, 512, kv_len,
                    qn.stride(0), qn.stride(1),
                    Kc.t().stride(0), Kc.t().stride(1),
                    scores_n.stride(0), scores_n.stride(1),
                    BLOCK_M=16, BLOCK_N=64, BLOCK_K=32,
                    num_warps=4, num_stages=2
                )

                # scores_p = [16, kv_len]
                grid_p = (triton.cdiv(num_qo_heads, 16), triton.cdiv(kv_len, 64))
                matmul_kernel[grid_p](
                    qp, Kp.t(), scores_p,
                    num_qo_heads, 64, kv_len,
                    qp.stride(0), qp.stride(1),
                    Kp.t().stride(0), Kp.t().stride(1),
                    scores_p.stride(0), scores_p.stride(1),
                    BLOCK_M=16, BLOCK_N=64, BLOCK_K=32,
                    num_warps=4, num_stages=2
                )

                scores = scores_n + scores_p  # [16, kv_len]

                # Causal mask: j > (prefix_len + i)
                prefix_len = kv_len - q_len
                absolute_pos = prefix_len + i

                # Compute softmax and lse row-wise via Triton
                # Softmax
                attn = torch.empty((num_qo_heads, kv_len), dtype=torch.float32, device=device)
                softmax_row_causal_kernel[(num_qo_heads,)](
                    scores, attn,
                    kv_len, 1.0, absolute_pos,
                    BLOCK=kv_len
                )

                # Output = attn @ Kc
                out_row = torch.empty((num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
                matmul_kernel[(triton.cdiv(num_qo_heads, 16), triton.cdiv(head_dim_ckv, 64),)](
                    attn, Kc, out_row,
                    num_qo_heads, kv_len, head_dim_ckv,
                    attn.stride(0), attn.stride(1),
                    Kc.stride(0), Kc.stride(1),
                    out_row.stride(0), out_row.stride(1),
                    BLOCK_M=16, BLOCK_N=64, BLOCK_K=32,
                    num_warps=4, num_stages=2
                )
                output[abs_q] = out_row

                # LSE (base-2)
                lse_row = torch.empty((num_qo_heads,), dtype=torch.float32, device=device)
                scale = 1.0 / math.log(2.0)
                lse_row_causal_kernel[(num_qo_heads,)](
                    scores, lse_row,
                    kv_len, scale, absolute_pos,
                    BLOCK=kv_len
                )
                lse[abs_q] = lse_row

        # Cast output to bfloat16 as original model
        output = output.to(torch.bfloat16)
        return output, lse


# Optional: entry point helpers if needed by harness
class Model(torch.nn.Module):
    def forward(self, *args):
        return ModelNew().forward(*args)


def run(*args):
    return ModelNew()(*args)
