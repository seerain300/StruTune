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
    # Tile ids
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        A_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        B_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn

        a = tl.load(A_ptrs, mask=(m_offsets[:, None] < M) & (k_offsets[None, :] < K), other=0.0)
        b = tl.load(B_ptrs, mask=(k_offsets[:, None] < K) & (n_offsets[None, :] < N), other=0.0)
        acc += tl.dot(a, b)

    C_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    tl.store(C_ptrs, acc, mask=(m_offsets[:, None] < M) & (n_offsets[None, :] < N))


@triton.jit
def softmax_row_causal_kernel(
    X_ptr, Out_ptr,
    N: tl.int32,
    scale: tl.float32,  # not used
    absolute_pos: tl.int32,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    x = tl.load(X_ptr + row_id * N + offs, mask=offs < N, other=-float("inf"))
    mask_causal = offs <= absolute_pos
    x = tl.where(mask_causal, x, -float("inf"))
    x_max = tl.max(x, axis=0)
    x = x - x_max
    x = tl.exp(x)
    denom = tl.sum(x, axis=0)
    out = x / denom
    tl.store(Out_ptr + row_id * N + offs, out, mask=offs < N)


@triton.jit
def lse_row_causal_kernel(
    X_ptr, Out_ptr,
    N: tl.int32,
    scale: tl.float32,  # not used
    absolute_pos: tl.int32,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    x = tl.load(X_ptr + row_id * N + offs, mask=offs < N, other=-float("inf"))
    mask_causal = offs <= absolute_pos
    x = tl.where(mask_causal, x, -float("inf"))
    x_max = tl.max(x, axis=0)
    x = x - x_max
    e = tl.exp(x)
    denom = tl.sum(e, axis=0)
    lse_val = tl.log(denom) / tl.log(2.0)  # base-2 logsumexp
    tl.store(Out_ptr + row_id, lse_val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Tunable blocks
        self.BLOCK_M = 16
        self.BLOCK_N = 64
        self.BLOCK_K = 32
        self.BLOCK_ROW = 128

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Shapes
        total_q, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]
        num_pages = ckv_cache.shape[0]
        len_indptr = qo_indptr.shape[0]
        batch_size = len_indptr - 1

        assert num_qo_heads == 16
        assert head_dim_ckv == 512
        assert head_dim_kpe == 64

        # Prepare Kc_all and Kp_all: [num_pages, dim]
        Kc_all = ckv_cache[:, 0].contiguous().to(torch.float32)  # [num_pages, 512]
        Kp_all = kpe_cache[:, 0].contiguous().to(torch.float32)  # [num_pages, 64]

        # Output tensors
        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=q_nope.device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=q_nope.device)

        # Process batches
        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            if q_start >= q_end:
                continue

            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            if page_beg >= page_end:
                continue

            tok_idx = kv_indices[page_beg:page_end].to(torch.int32)  # [kv_len]
            Kc = Kc_all[tok_idx]  # [kv_len, 512], float32
            Kp = Kp_all[tok_idx]  # [kv_len, 64], float32

            q_len = q_end - q_start
            Kc = Kc.contiguous()
            Kp = Kp.contiguous()

            for i in range(q_len):
                abs_q = q_start + i
                qn = q_nope[abs_q].to(torch.float32).contiguous()  # [16, 512]
                qp = q_pe[abs_q].to(torch.float32).contiguous()   # [16, 64]

                # Compute scores_n = qn @ Kc.T
                scores_n = torch.empty((num_qo_heads, Kc.shape[0]), dtype=torch.float32, device=q_nope.device)
                grid_n = (triton.cdiv(num_qo_heads, self.BLOCK_M), triton.cdiv(Kc.shape[0], self.BLOCK_N))
                matmul_kernel[grid_n](
                    qn, Kc.transpose(0, 1).contiguous(), scores_n,
                    num_qo_heads, Kc.shape[1], Kc.shape[0],
                    qn.stride(0), qn.stride(1),
                    Kc.transpose(0, 1).stride(0), Kc.transpose(0, 1).stride(1),
                    scores_n.stride(0), scores_n.stride(1),
                    BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N, BLOCK_K=self.BLOCK_K,
                    num_warps=4, num_stages=2
                )

                # Compute scores_p = qp @ Kp.T
                scores_p = torch.empty((num_qo_heads, Kp.shape[0]), dtype=torch.float32, device=q_nope.device)
                grid_p = (triton.cdiv(num_qo_heads, self.BLOCK_M), triton.cdiv(Kp.shape[0], self.BLOCK_N))
                matmul_kernel[grid_p](
                    qp, Kp.transpose(0, 1).contiguous(), scores_p,
                    num_qo_heads, Kp.shape[1], Kp.shape[0],
                    qp.stride(0), qp.stride(1),
                    Kp.transpose(0, 1).stride(0), Kp.transpose(0, 1).stride(1),
                    scores_p.stride(0), scores_p.stride(1),
                    BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N, BLOCK_K=self.BLOCK_K,
                    num_warps=4, num_stages=2
                )

                scores = scores_n + scores_p  # [16, kv_len]

                # prefix_len = number of previously cached tokens for this batch
                prefix_len = Kc_all.shape[0] - (Kc.shape[0] if Kc.numel() > 0 else 0) - (Kp_all.shape[0] - (Kp.shape[0] if Kp.numel() > 0 else 0))
                absolute_pos = prefix_len + i

                # lse per head (base-2 logsumexp with causal mask)
                lse_row = torch.empty((num_qo_heads,), dtype=torch.float32, device=q_nope.device)
                lse_row_causal_kernel[(num_qo_heads,)](
                    scores, lse_row,
                    scores.shape[1], 1.0, absolute_pos,
                    BLOCK=self.BLOCK_ROW
                )
                lse[abs_q] = lse_row

                # attention with causal mask
                attn = torch.empty_like(scores, dtype=torch.float32, device=q_nope.device)
                softmax_row_causal_kernel[(num_qo_heads,)](
                    scores, attn,
                    scores.shape[1], 1.0, absolute_pos,
                    BLOCK=self.BLOCK_ROW
                )

                # out = attn @ Kc
                out_row = torch.empty((num_qo_heads, head_dim_ckv), dtype=torch.float32, device=q_nope.device)
                grid_out = (triton.cdiv(num_qo_heads, self.BLOCK_M), triton.cdiv(head_dim_ckv, self.BLOCK_N))
                matmul_kernel[grid_out](
                    attn, Kc, out_row,
                    num_qo_heads, Kc.shape[1], head_dim_ckv,
                    attn.stride(0), attn.stride(1),
                    Kc.stride(0), Kc.stride(1),
                    out_row.stride(0), out_row.stride(1),
                    BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N, BLOCK_K=self.BLOCK_K,
                    num_warps=4, num_stages=2
                )
                output[abs_q] = out_row

        # Cast output to bfloat16 as required by original
        output = output.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
