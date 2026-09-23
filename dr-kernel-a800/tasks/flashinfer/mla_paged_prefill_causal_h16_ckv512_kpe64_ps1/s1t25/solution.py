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
    # 2D launch over M and N tiles
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        A_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        B_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn
        # Masking: if M or N is not multiple of tile, guard loads
        a = tl.load(A_ptrs, mask=(m_offsets[:, None] < M) & (k_offsets[None, :] < K), other=0.0)
        b = tl.load(B_ptrs, mask=(k_offsets[:, None] < K) & (n_offsets[None, :] < N), other=0.0)
        acc += tl.dot(a, b)

    C_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    tl.store(C_ptrs, acc, mask=(m_offsets[:, None] < M) & (n_offsets[None, :] < N))


# Softmax per row with causal mask: positions j > absolute_pos set to -inf
@triton.jit
def softmax_row_causal_kernel(
    X_ptr, Out_ptr,
    N: tl.int32,
    scale: tl.float32,  # keep 1.0
    absolute_pos: tl.int32,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    idx = tl.arange(0, BLOCK)
    mask = idx < N
    x = tl.load(X_ptr + row_id * N + idx, mask=mask, other=-float("inf"))
    causal = idx <= absolute_pos
    x = tl.where(causal, x, -float("inf"))
    x_max = tl.max(x, axis=0)
    x = x - x_max
    e = tl.exp(x)
    denom = tl.sum(e, axis=0)
    out = e / denom
    tl.store(Out_ptr + row_id * N + idx, out, mask=mask)


# LogSumExp per row with causal mask (base-2). Output is 1D lse per row.
@triton.jit
def lse_row_causal_kernel(
    X_ptr, Out_ptr,
    N: tl.int32,
    scale: tl.float32,  # keep 1.0
    absolute_pos: tl.int32,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    idx = tl.arange(0, BLOCK)
    mask = idx < N
    x = tl.load(X_ptr + row_id * N + idx, mask=mask, other=-float("inf"))
    causal = idx <= absolute_pos
    x = tl.where(causal, x, -float("inf"))
    x_max = tl.max(x, axis=0)
    sum_exp = tl.sum(tl.exp(x - x_max), axis=0)
    lse_nat = tl.log(sum_exp)
    # convert natural log to base-2
    lse = lse_nat * (1.0 / 0.6931471805599453)  # 1 / ln(2)
    tl.store(Out_ptr + row_id, lse)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA tensors for Triton
        if q_nope.device.type != 'cuda':
            raise RuntimeError("ModelNew requires CUDA tensors for Triton kernels. Please move inputs to CUDA.")
        device = q_nope.device

        # Shapes
        total_q, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]
        num_pages, _, _ = ckv_cache.shape
        len_indptr = qo_indptr.shape[0]
        batch_size = len_indptr - 1

        # Prepare caches
        Kc_all = ckv_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, 64]

        # Outputs
        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            if q_start >= q_end:
                continue

            q_len = q_end - q_start
            q_nope_batch = q_nope[q_start:q_end].to(torch.float32)  # [q_len, 16, 512]
            q_pe_batch = q_pe[q_start:q_end].to(torch.float32)     # [q_len, 16, 64]

            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            if page_beg >= page_end:
                continue

            kv_len = page_end - page_beg
            tok_idx = kv_indices[page_beg:page_end].to(torch.int32)  # [kv_len]
            Kc = Kc_all[tok_idx]  # [kv_len, 512], float32
            Kp = Kp_all[tok_idx]  # [kv_len, 64], float32

            for i in range(q_len):
                abs_q = q_start + i
                # qn: [16, 512], qp: [16, 64]
                qn = q_nope_batch[i].contiguous()  # [16, 512]
                qp = q_pe_batch[i].contiguous()    # [16, 64]

                # scores_n = qn @ Kc.T -> [16, kv_len]
                scores_n = torch.empty((num_qo_heads, kv_len), dtype=torch.float32, device=device)
                grid_n = (triton.cdiv(num_qo_heads, 16), triton.cdiv(kv_len, 64))
                matmul_kernel[grid_n](
                    qn, Kc.t().contiguous(), scores_n,
                    num_qo_heads, 512, kv_len,
                    16, 512, 512, kv_len, 512, 0, 0,  # stride_am=16, stride_ak=512, stride_bk=512, stride_bn=kv_len, stride_cm=16, stride_cn=kv_len
                    BLOCK_M=16, BLOCK_N=64, BLOCK_K=32,
                    num_warps=4, num_stages=2
                )

                # scores_p = qp @ Kp.T -> [16, kv_len]
                scores_p = torch.empty((num_qo_heads, kv_len), dtype=torch.float32, device=device)
                matmul_kernel[grid_n](
                    qp, Kp.t().contiguous(), scores_p,
                    num_qo_heads, 64, kv_len,
                    16, 64, 64, kv_len, 64, 0, 0,
                    BLOCK_M=16, BLOCK_N=64, BLOCK_K=32,
                    num_warps=4, num_stages=2
                )

                scores = scores_n + scores_p * sm_scale  # [16, kv_len]

                # causal mask per row: positions j > (prefix_len + i) set to -inf
                prefix_len = kv_len - q_len
                absolute_pos = prefix_len + i

                # Apply softmax with causal mask per head
                for h in range(num_qo_heads):
                    softmax_row_causal_kernel[(1,)](
                        scores[h], scores[h],
                        kv_len, 1.0, absolute_pos,
                        BLOCK=kv_len
                    )

                # out = attn @ Kc -> [16, 512]
                out_row = torch.empty((num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
                matmul_kernel[(triton.cdiv(num_qo_heads, 16), triton.cdiv(head_dim_ckv, 64))](
                    scores, Kc.contiguous(), out_row,
                    num_qo_heads, kv_len, head_dim_ckv,
                    num_qo_heads, kv_len, kv_len, head_dim_ckv, 64, 0, 0,
                    BLOCK_M=16, BLOCK_N=64, BLOCK_K=64,
                    num_warps=4, num_stages=2
                )
                output[abs_q] = out_row

                # lse per head
                lse_vec = torch.empty((num_qo_heads,), dtype=torch.float32, device=device)
                lse_row_causal_kernel[(num_qo_heads,)](
                    scores, lse_vec,
                    kv_len, 1.0, absolute_pos,
                    BLOCK=kv_len
                )
                lse[abs_q] = lse_vec

        # Cast output to bfloat16 for final return
        output = output.to(torch.bfloat16)
        return output, lse


# Optional alias for environments expecting 'Model'
class Model(ModelNew):
    pass


# Helper from the prompt
def get_inputs():
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16)
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16)
    ckv_cache = torch.randn([989669, 1, 512], dtype=torch.bfloat16)
    kpe_cache = torch.randn([989669, 1, 64], dtype=torch.bfloat16)
    _n = 1; _t = 1
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    qo_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32)
    _n = 1; _t = 34
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32)
    kv_indices = torch.randint(0, 989669, [34], dtype=torch.int32)
    sm_scale = 1.0
    return [q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale]


def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6, tensor_7):
    _out = ModelNew().forward(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6, tensor_7)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


def run(*args):
    return ModelNew()(*args)
