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
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        a_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        b_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn
        a = tl.load(a_ptrs, mask=(m_offsets[:, None] < M) & (k_offsets[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(k_offsets[:, None] < K) & (n_offsets[None, :] < N), other=0.0)
        acc += tl.dot(a, b)

    c_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(m_offsets[:, None] < M) & (n_offsets[None, :] < N))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.BLOCK_M = 128
        self.BLOCK_N = 128
        self.BLOCK_K = 64
        self.num_warps = 4
        self.num_stages = 2
        self.SM_SCALE = 1.0  # default scale, may be overridden

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        device = q_nope.device
        total_q, num_qo_heads, head_dim_ckv = q_nope.shape
        _, _, head_dim_kpe = q_pe.shape

        # Asserts
        assert num_qo_heads == 16
        assert head_dim_ckv == 512
        assert head_dim_kpe == 64

        # Prepare caches
        num_pages = ckv_cache.shape[0]
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, 64]

        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
        lse = torch.full((total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

        batch_size = qo_indptr.numel() - 1
        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            if q_start >= q_end:
                continue

            q_len = q_end - q_start

            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            if page_beg >= page_end:
                for i in range(q_len):
                    lse[q_start + i] = -float("inf")
                output[q_start:q_end] = 0.0
                continue

            kv_len = page_end - page_beg
            tok_idx = kv_indices[page_beg:page_end].to(torch.long)  # [kv_len]
            Kc = Kc_all[tok_idx]  # [kv_len, 512]
            Kp = Kp_all[tok_idx]  # [kv_len, 64]

            # Prepare q_nope batch and q_pe batch
            q_batch_nope = q_nope[q_start:q_end].to(torch.float32)  # [q_len, 16, 512]
            q_batch_pe = q_pe[q_start:q_end].to(torch.float32)     # [q_len, 16, 64]

            # Compute q_nope contribution: Aq @ Kc.T -> [q_len, 16, kv_len]
            # Aq: [M, K] where M = q_len * num_qo_heads, K = 512
            M = q_len * num_qo_heads
            Kq = head_dim_ckv
            Mt = M
            Nt = kv_len
            grid = (triton.cdiv(Mt, self.BLOCK_M), triton.cdiv(Nt, self.BLOCK_N))
            Cq_nope = torch.empty((Mt, Nt), dtype=torch.float32, device=device)

            Aq = q_batch_nope.reshape(Mt, Kq)  # [M, 512]
            KcT = Kc.T  # [512, kv_len]

            matmul_kernel[grid](
                Aq, KcT, Cq_nope,
                Mt, Kq, Nt,
                Aq.stride(0), Aq.stride(1),
                KcT.stride(0), KcT.stride(1),
                Cq_nope.stride(0), Cq_nope.stride(1),
                BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N, BLOCK_K=self.BLOCK_K,
                num_warps=self.num_warps, num_stages=self.num_stages
            )

            # Now add q_pe contribution per head
            # Compute per-head q_pe @ Kp.T and add to Cq_nope rows corresponding to that head
            for h in range(num_qo_heads):
                # rows for this head: h*q_len : (h+1)*q_len
                rows = h * q_len + torch.arange(q_len, device=device)
                # q_pe for head h
                qpe_h = q_batch_pe[:, h]  # [q_len, 64]
                KpT = Kp.T  # [64, kv_len]
                # Build Aq_pe_h: [q_len, 64] => [Mh, Kpe] with Mh=q_len, Kpe=64
                Aq_pe_h = qpe_h  # already [q_len, 64]
                grid_h = (triton.cdiv(q_len, self.BLOCK_M), triton.cdiv(Nt, self.BLOCK_N))
                Cq_pe = torch.empty((q_len, Nt), dtype=torch.float32, device=device)
                matmul_kernel[grid_h](
                    Aq_pe_h, KpT, Cq_pe,
                    q_len, 64, Nt,
                    Aq_pe_h.stride(0), Aq_pe_h.stride(1),
                    KpT.stride(0), KpT.stride(1),
                    Cq_pe.stride(0), Cq_pe.stride(1),
                    BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N, BLOCK_K=64,
                    num_warps=self.num_warps, num_stages=self.num_stages
                )
                Cq_nope[rows] += Cq_pe  # add per-head q_pe contribution

            # Now Cq_nope has shape [M, Nt] = [(q_len*16), kv_len]
            # Compute per-head per-query logsumexp and output
            for h in range(num_qo_heads):
                rows = h * q_len + torch.arange(q_len, device=device)
                scores_h = Cq_nope[rows]  # [q_len, kv_len]
                for i in range(q_len):
                    row_vec = scores_h[i]  # [kv_len]
                    # Apply causal mask: positions j > (prefix_len + i) => -inf
                    prefix_len = kv_len - q_len
                    abs_pos = prefix_len + i
                    # torch implementation for simplicity
                    m = torch.max(row_vec)
                    e = torch.exp(row_vec - m)
                    s = torch.sum(e)
                    lse_row = torch.log(s) / math.log(2.0)  # base-2
                    lse[q_start + i, h] = lse_row
                    # attention
                    attn = torch.softmax(row_vec * self.SM_SCALE, dim=-1)  # [kv_len]
                    # out = attn @ Kc  -> [512]
                    out_row = attn @ Kc  # [512]
                    output[q_start + i, h] = out_row.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
