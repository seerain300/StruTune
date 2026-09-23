import math
import torch
import triton
import triton.language as tl


# Matmul kernel: C[M, N] = A[M, K] @ B[K, N]
# A: [MA, K], B: [K, NB], C: [MA, NB]
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    MA: tl.int32, NA: tl.int32, NB: tl.int32,  # NA is not used directly but kept for signature
    stride_am: tl.int32, stride_ak: tl.int32,
    stride_bk: tl.int32, stride_bn: tl.int32,
    stride_cm: tl.int32, stride_cn: tl.int32,
    K: tl.int32,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D launch over M and N tiles
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # A tile: [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        A_mask = (m_offsets[:, None] < MA) & (k_offsets[None, :] < K)
        A_tile = tl.load(A_ptrs, mask=A_mask, other=0.0)

        # B tile: [BLOCK_K, BLOCK_N]
        B_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn
        B_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < NB)
        B_tile = tl.load(B_ptrs, mask=B_mask, other=0.0)

        # Accumulate
        acc += tl.dot(A_tile, B_tile)

    # Write back C tile
    C_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    C_mask = (m_offsets[:, None] < MA) & (n_offsets[None, :] < NB)
    tl.store(C_ptrs, acc, mask=C_mask)


# Triton kernel: softmax per row with optional causal mask (positions beyond 'absolute_pos' are -inf)
@triton.jit
def softmax_mask_row_kernel(
    X_ptr, Out_ptr,
    N: tl.int32,
    scale: tl.float32,  # for multiplication, typically 1.0
    absolute_pos: tl.int32,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    # Load row into vector
    x = tl.load(X_ptr + row_id * N + tl.arange(0, BLOCK), mask=tl.arange(0, BLOCK) < N, other=-float("inf"))
    x = x * scale
    pos = tl.arange(0, BLOCK)
    mask_inf = pos > absolute_pos
    x = tl.where(mask_inf & (pos < N), -float("inf"), x)
    # stable softmax
    x_max = tl.max(x, axis=0)
    x = x - x_max
    x_exp = tl.exp(x)
    x_sum = tl.sum(x_exp, axis=0)
    x = x_exp / x_sum
    tl.store(Out_ptr + row_id * N + tl.arange(0, BLOCK), x, mask=tl.arange(0, BLOCK) < N)


# Triton kernel: compute lse per row with optional causal mask
@triton.jit
def lse_mask_row_kernel(
    X_ptr, Out_ptr,
    N: tl.int32,
    scale: tl.float32,  # for multiplication, typically 1.0
    absolute_pos: tl.int32,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    x = tl.load(X_ptr + row_id * N + tl.arange(0, BLOCK), mask=tl.arange(0, BLOCK) < N, other=-float("inf"))
    x = x * scale
    pos = tl.arange(0, BLOCK)
    mask_inf = pos > absolute_pos
    x = tl.where(mask_inf & (pos < N), -float("inf"), x)
    x_max = tl.max(x, axis=0)
    x = x - x_max
    x_exp = tl.exp(x)
    x_sum = tl.sum(x_exp, axis=0)
    ln2 = 0.6931471805599453  # log(2)
    lse_val = tl.log(x_sum) / ln2  # base-2 logsumexp
    tl.store(Out_ptr + row_id, lse_val)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure on CUDA for Triton execution
        device = q_nope.device
        if device.type != "cuda":
            q_nope = q_nope.cuda()
            q_pe = q_pe.cuda()
            ckv_cache = ckv_cache.cuda()
            kpe_cache = kpe_cache.cuda()
            qo_indptr = qo_indptr.cuda()
            kv_indptr = kv_indptr.cuda()
            kv_indices = kv_indices.cuda()

        # Dimensions
        total_q, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]
        num_pages = ckv_cache.shape[0]
        len_indptr = qo_indptr.shape[0]
        batch_size = len_indptr - 1

        # Ensure contiguity
        q_nope = q_nope.contiguous()
        q_pe = q_pe.contiguous()
        Kc_all = ckv_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, 64]
        qo_indptr = qo_indptr.contiguous()
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        # Output buffers
        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Process batches
        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            q_len = q_end - q_start
            kv_len = kv_end - kv_start

            # Select token IDs from cache
            tok_idx = kv_indices[kv_start:kv_end].to(torch.int64)  # indices into Kc_all/Kp_all
            Kc = Kc_all[tok_idx].contiguous().to(torch.float32)   # [kv_len, 512]
            Kp = Kp_all[tok_idx].contiguous().to(torch.float32)   # [kv_len, 64]

            # Prepare output per absolute position
            for i in range(q_len):
                abs_q = q_start + i  # absolute query position in overall sequence

                # Load qn, qp: [16, 512], [16, 64]
                qn = q_nope[abs_q].contiguous().to(torch.float32)  # [16, 512]
                qp = q_pe[abs_q].contiguous().to(torch.float32)    # [16, 64]

                # scores_n = qn @ Kc.T → [16, kv_len]
                scores_n = torch.empty((16, kv_len), dtype=torch.float32, device=device)
                matmul_kernel[(triton.cdiv(16, 16), triton.cdiv(kv_len, 64),)](
                    qn, Kc.T, scores_n,
                    16, kv_len, kv_len,
                    qn.stride(0), qn.stride(1),
                    Kc.T.stride(0), Kc.T.stride(1),
                    scores_n.stride(0), scores_n.stride(1),
                    kv_len,
                    BLOCK_M=16, BLOCK_N=64, BLOCK_K=32,
                    num_warps=4, num_stages=2
                )

                # scores_p = qp @ Kp.T → [16, kv_len]
                scores_p = torch.empty((16, kv_len), dtype=torch.float32, device=device)
                matmul_kernel[(triton.cdiv(16, 16), triton.cdiv(kv_len, 64),)](
                    qp, Kp.T, scores_p,
                    16, kv_len, kv_len,
                    qp.stride(0), qp.stride(1),
                    Kp.T.stride(0), Kp.T.stride(1),
                    scores_p.stride(0), scores_p.stride(1),
                    kv_len,
                    BLOCK_M=16, BLOCK_N=64, BLOCK_K=32,
                    num_warps=4, num_stages=2
                )

                # scores = scores_n + scores_p
                scores = scores_n + scores_p

                # Apply scale
                scores_scaled = scores * sm_scale

                # Causal absolute position: prefix_len = kv_len - q_len
                prefix_len = kv_len - q_len
                absolute_pos = prefix_len + i  # query abs position

                # 1) Compute lse per row
                lse_row = torch.empty((16,), dtype=torch.float32, device=device)
                lse_mask_row_kernel[(16,)](
                    scores_scaled, lse_row,
                    kv_len, sm_scale, absolute_pos,
                    BLOCK=kv_len,
                    num_warps=1, num_stages=1
                )
                lse[abs_q] = lse_row  # [16]

                # 2) Compute attn per row
                attn = torch.empty((16, kv_len), dtype=torch.float32, device=device)
                softmax_mask_row_kernel[(16,)](
                    scores_scaled, attn,
                    kv_len, sm_scale, absolute_pos,
                    BLOCK=kv_len,
                    num_warps=1, num_stages=1
                )

                # 3) out = attn @ Kc → [16, 512]
                out_row = torch.empty((16, 512), dtype=torch.float32, device=device)
                matmul_kernel[(triton.cdiv(16, 16), triton.cdiv(512, 64),)](
                    attn, Kc, out_row,
                    16, kv_len, 512,
                    attn.stride(0), attn.stride(1),
                    Kc.stride(0), Kc.stride(1),
                    out_row.stride(0), out_row.stride(1),
                    kv_len,
                    BLOCK_M=16, BLOCK_N=64, BLOCK_K=32,
                    num_warps=4, num_stages=2
                )
                output[abs_q] = out_row.to(torch.bfloat16)

        return output, lse


# Entry point class required by evaluator
class Model(ModelNew):
    pass


def run(*args):
    return ModelNew()(*args)
