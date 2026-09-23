import math
import torch
import triton
import triton.language as tl


# Matmul kernel: C[M, N] = A[M, K] @ B[K, N]
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    MA: tl.int32, NA: tl.int32, NB: tl.int32,
    stride_am: tl.int32, stride_ak: tl.int32,
    stride_bk: tl.int32, stride_bn: tl.int32,
    stride_cm: tl.int32, stride_cn: tl.int32,
    K: tl.int32,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

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

    # Write back C
    C_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    C_mask = (m_offsets[:, None] < MA) & (n_offsets[None, :] < NB)
    tl.store(C_ptrs, acc, mask=C_mask)


# Triton kernel to compute per-row softmax with causal mask
# Input X: [M], output OUT: [M]
# prefix_len = kv_len - q_len, absolute_pos = prefix_len + i
@triton.jit
def softmax_mask_row_kernel(
    X_ptr, OUT_ptr,
    M: tl.int32,
    scale: tl.float32,
    prefix_len: tl.int32,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    m = offs < M
    x = tl.load(X_ptr + offs, mask=m, other=-float('inf'))

    # Build causal mask: j <= (prefix_len + i)
    j = offs
    abs_pos = prefix_len
    causal_mask = j <= abs_pos
    # Apply -inf where not causal
    x = tl.where(causal_mask, x, -float('inf'))

    # Stable softmax
    x_max = tl.max(x, axis=0)
    x_shift = x - x_max
    e = tl.exp(scale * x_shift)
    denom = tl.sum(e, axis=0)
    out = e / denom
    tl.store(OUT_ptr + offs, out, mask=m)


# Triton kernel to compute per-row logsumexp with causal mask
@triton.jit
def lse_mask_row_kernel(
    X_ptr, OUT_ptr,
    M: tl.int32,
    scale: tl.float32,
    prefix_len: tl.int32,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    m = offs < M
    x = tl.load(X_ptr + offs, mask=m, other=-float('inf'))

    j = offs
    abs_pos = prefix_len
    causal_mask = j <= abs_pos
    x = tl.where(causal_mask, x, -float('inf'))

    x_max = tl.max(x, axis=0)
    x_shift = x - x_max
    e = tl.exp(scale * x_shift)
    sum_e = tl.sum(e, axis=0)
    lse_val = tl.log(sum_e) + x_max
    # scale by 1/ln(2)
    lse_val = lse_val / math.log(2.0)
    tl.store(OUT_ptr + offs, lse_val, mask=m)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA
        device = q_nope.device
        assert device.type == "cuda", "ModelNew requires CUDA device for Triton kernels."

        # Cast to float32 for compute
        q_nope_f = q_nope.contiguous().to(torch.float32)
        q_pe_f = q_pe.contiguous().to(torch.float32)

        # Squeeze caches
        Kc_all = ckv_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, 64]

        total_q = int(qo_indptr[-1].item())
        output = torch.empty((total_q, 16, 512), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, 16), dtype=torch.float32, device=device)

        batch_size = qo_indptr.shape[0] - 1

        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            q_len = q_end - q_start
            if q_len == 0:
                continue

            # KV tokens
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            kv_len = page_end - page_beg
            if kv_len == 0:
                continue

            tok_idx = kv_indices[page_beg:page_end].to(torch.int64).contiguous()  # [kv_len]
            Kc = Kc_all[tok_idx].contiguous().to(torch.float32)  # [kv_len, 512]
            Kp = Kp_all[tok_idx].contiguous().to(torch.float32)  # [kv_len, 64]

            # Loop over i in [0, q_len)
            for i in range(q_len):
                abs_q = q_start + i

                # Load qn and qp
                qn = q_nope_f[abs_q].contiguous().to(torch.float32)  # [16, 512]
                qp = q_pe_f[abs_q].contiguous().to(torch.float32)   # [16, 64]

                # scores_n = qn @ Kc.T
                scores_n = torch.empty((16, kv_len), dtype=torch.float32, device=device)
                matmul_kernel[(triton.cdiv(16, 16), triton.cdiv(kv_len, 64),)](
                    qn, Kc.transpose(0, 1), scores_n,
                    16, 512, kv_len,
                    16, 512,
                    512, kv_len,
                    512,
                    BLOCK_M=16, BLOCK_N=64, BLOCK_K=32
                )

                # scores_p = qp @ Kp.T
                scores_p = torch.empty((16, kv_len), dtype=torch.float32, device=device)
                matmul_kernel[(triton.cdiv(16, 16), triton.cdiv(kv_len, 64),)](
                    qp, Kp.transpose(0, 1), scores_p,
                    16, 64, kv_len,
                    16, 64,
                    64, kv_len,
                    64,
                    BLOCK_M=16, BLOCK_N=64, BLOCK_K=32
                )

                scores = scores_n + scores_p  # [16, kv_len]

                # Softmax with causal mask
                prefix_len = kv_len - q_len
                absolute_pos = prefix_len + i

                attn = torch.empty((16, kv_len), dtype=torch.float32, device=device)
                # No original mask: use only causal mask
                softmax_mask_row_kernel[(16,)](
                    scores, attn,
                    kv_len, 1.0, absolute_pos,
                    BLOCK=kv_len
                )

                # out = attn @ Kc
                out_row = torch.empty((16, 512), dtype=torch.float32, device=device)
                matmul_kernel[(triton.cdiv(16, 16), triton.cdiv(512, 64),)](
                    attn, Kc, out_row,
                    16, kv_len, 512,
                    kv_len, 512,
                    512, 512,
                    512,
                    BLOCK_M=16, BLOCK_N=64, BLOCK_K=32
                )
                output[abs_q] = out_row.to(torch.bfloat16)

                # lse per head
                lse_row = torch.empty((16,), dtype=torch.float32, device=device)
                lse_mask_row_kernel[(16,)](
                    scores, lse_row,
                    kv_len, 1.0, absolute_pos,
                    BLOCK=kv_len
                )
                lse[abs_q] = lse_row

        return output, lse


# Optional: keep a Model class for environments that expect 'Model' as entry point
class Model(ModelNew):
    pass


def run(*args):
    return ModelNew()(*args)
