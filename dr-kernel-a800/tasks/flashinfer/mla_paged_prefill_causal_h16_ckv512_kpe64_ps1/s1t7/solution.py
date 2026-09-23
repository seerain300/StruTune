import math
import torch
import triton
import triton.language as tl


# Matmul kernel: C[M, N] = A[M, K] @ B[K, N]
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M: tl.int32, K: tl.int32, N: tl.int32,
    stride_am: tl.int32, stride_ak: tl.int32,
    stride_bk: tl.int32, stride_bn: tl.int32,
    stride_cm: tl.int32, stride_cn: tl.int32,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D grid: tiles over M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # A tile: [BLOCK_M, BLOCK_K]
        A_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        A_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        A_tile = tl.load(A_ptrs, mask=A_mask, other=0.0)

        # B tile: [BLOCK_K, BLOCK_N]
        B_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn
        B_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        B_tile = tl.load(B_ptrs, mask=B_mask, other=0.0)

        acc += tl.dot(A_tile, B_tile)

    # Store C tile
    C_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    C_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(C_ptrs, acc, mask=C_mask)


# Triton kernel: per-row stable softmax with causal mask
# X: [N] vector, Out: [N] vector, N: length, scale: multiplier (usually 1.0), absolute_pos: int, BLOCK >= N
@triton.jit
def softmax_row_causal_kernel(
    X_ptr, Out_ptr,
    N: tl.int32, scale: tl.float32, absolute_pos: tl.int32,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    idx = tl.arange(0, BLOCK)
    mask = idx < N
    x = tl.load(X_ptr + row_id * N + idx, mask=mask, other=-float("inf"))
    # Apply causal mask: positions j > absolute_pos -> -inf
    causal_mask = idx > absolute_pos
    x = tl.where(causal_mask, -float("inf"), x)

    # Stable softmax
    x_max = tl.max(x, axis=0)
    v = x - x_max
    e = tl.exp(v)
    e = tl.where(causal_mask, 0.0, e)  # exclude masked positions from sum
    sum_e = tl.sum(e, axis=0)
    softmax = e / sum_e

    tl.store(Out_ptr + row_id * N + idx, softmax, mask=mask)


# Triton kernel: per-row logsumexp with causal mask (base-2), returns scalar per row
@triton.jit
def lse_row_causal_kernel(
    X_ptr, Out_ptr,
    N: tl.int32, scale: tl.float32, absolute_pos: tl.int32,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    idx = tl.arange(0, BLOCK)
    mask = idx < N
    x = tl.load(X_ptr + row_id * N + idx, mask=mask, other=-float("inf"))

    # Apply causal mask
    causal_mask = idx > absolute_pos
    x = tl.where(causal_mask, -float("inf"), x)

    # Stable logsumexp
    x_max = tl.max(x, axis=0)
    v = x - x_max
    e = tl.exp(v)
    e = tl.where(causal_mask, 0.0, e)
    sum_e = tl.sum(e, axis=0)
    lse_val = tl.log(sum_e)  # log base e
    lse_val = lse_val / tl.log(2.0)  # base-2
    tl.store(Out_ptr + row_id, lse_val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure tensors are on CUDA for Triton
        device = q_nope.device
        assert device.type == "cuda", "ModelNew expects CUDA tensors for Triton kernels"

        total_q, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]
        assert num_qo_heads == 16
        assert head_dim_ckv == 512
        assert head_dim_kpe == 64

        num_pages = ckv_cache.shape[0]
        # Kc_all, Kp_all as float32
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, 64]

        # Output buffers (compute in fp32, cast to bf16 at the end)
        output = torch.empty((total_q, 16, 512), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, 16), dtype=torch.float32, device=device)

        # Process each batch element b
        for b in range(qo_indptr.shape[0] - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            if q_start >= q_end:
                continue

            q_len = q_end - q_start

            # KV tokens range
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            if page_beg >= page_end:
                continue

            kv_len = page_end - page_beg
            tok_idx = kv_indices[page_beg:page_end].to(torch.int32).to(device)  # [kv_len]

            # Fetch Kc, Kp
            Kc = Kc_all[tok_idx]  # [kv_len, 512]
            Kp = Kp_all[tok_idx]  # [kv_len, 64]

            # Process each query position i in this batch
            for i in range(q_len):
                abs_q = q_start + i

                # qn, qp: convert to fp32 for compute
                qn = q_nope[abs_q].to(torch.float32)  # [16, 512]
                qp = q_pe[abs_q].to(torch.float32)   # [16, 64]

                # scores_n = qn @ Kc.T → [16, kv_len]
                scores_n = torch.empty((16, kv_len), dtype=torch.float32, device=device)
                matmul_kernel[(triton.cdiv(16, 16), triton.cdiv(kv_len, 64),)](
                    qn, Kc.transpose(0, 1), scores_n,
                    16, 512, kv_len,
                    512, 512,
                    kv_len, 512,
                    BLOCK_M=16, BLOCK_N=64, BLOCK_K=32,
                    num_warps=4, num_stages=2
                )

                # scores_p = qp @ Kp.T → [16, kv_len]
                scores_p = torch.empty((16, kv_len), dtype=torch.float32, device=device)
                matmul_kernel[(triton.cdiv(16, 16), triton.cdiv(kv_len, 64),)](
                    qp, Kp.transpose(0, 1), scores_p,
                    16, 64, kv_len,
                    64, 64,
                    kv_len, 64,
                    BLOCK_M=16, BLOCK_N=64, BLOCK_K=32,
                    num_warps=4, num_stages=2
                )

                scores = scores_n + scores_p
                # Causal mask: positions j > (prefix_len + i) should be -inf
                prefix_len = kv_len - q_len
                absolute_pos = prefix_len + i
                attn = torch.empty((16, kv_len), dtype=torch.float32, device=device)
                softmax_row_causal_kernel[(16,)](
                    scores, attn,
                    kv_len, 1.0, absolute_pos,
                    BLOCK=kv_len,
                    num_warps=4, num_stages=2
                )

                # out = attn @ Kc → [16, 512]
                out_row = torch.empty((16, 512), dtype=torch.float32, device=device)
                matmul_kernel[(triton.cdiv(16, 16), triton.cdiv(512, 64),)](
                    attn, Kc, out_row,
                    16, kv_len, 512,
                    kv_len, 512,
                    512, 512,
                    BLOCK_M=16, BLOCK_N=64, BLOCK_K=32,
                    num_warps=4, num_stages=2
                )
                output[abs_q] = out_row

                # lse per head
                lse_row = torch.empty((16,), dtype=torch.float32, device=device)
                lse_row_causal_kernel[(16,)](
                    scores, lse_row,
                    kv_len, 1.0, absolute_pos,
                    BLOCK=kv_len,
                    num_warps=4, num_stages=2
                )
                lse[abs_q] = lse_row

        # Cast output to bfloat16 as in original
        output = output.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
