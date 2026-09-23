import math
import torch
import triton
import triton.language as tl


# Triton matmul: C[M, N] = A[M, K] @ B[K, N]
# Arguments: positional (A_ptr, B_ptr, C_ptr, M, N, K)
#            keyword (stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn)
# Compile-time constants: BLOCK_M, BLOCK_N, BLOCK_K (must be provided as keyword args only)
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M: tl.int32, N: tl.int32, K: tl.int32,
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

        A_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak  # [BM, BK]
        B_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn  # [BK, BN]

        a = tl.load(A_ptrs, mask=(m_offsets[:, None] < M) & (k_offsets[None, :] < K), other=0.0)
        b = tl.load(B_ptrs, mask=(k_offsets[:, None] < K) & (n_offsets[None, :] < N), other=0.0)

        acc += tl.dot(a, b)

    C_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    tl.store(C_ptrs, acc, mask=(m_offsets[:, None] < M) & (n_offsets[None, :] < N))


# Row-wise softmax with causal mask: C = softmax(A) where positions j > absolute_pos -> -inf
# A: [N], returns C: [N] in same dtype as A
@triton.jit
def softmax_row_causal_kernel(
    A_ptr, C_ptr,
    N: tl.int32,
    absolute_pos: tl.int32,
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(A_ptr + offs, mask=mask, other=-float("inf"))
    causal = (offs > absolute_pos) & mask
    a = tl.where(causal, -float("inf"), a)
    m = tl.max(a, axis=0)
    a = a - m
    exp_a = tl.exp(a)
    denom = tl.sum(exp_a, axis=0)
    c = exp_a / denom
    tl.store(C_ptr + offs, c, mask=mask)


# Row-wise logsumexp (base-2) with causal mask: C = logsumexp(A) / ln(2)
@triton.jit
def lse_row_causal_kernel(
    A_ptr, C_ptr,
    N: tl.int32,
    absolute_pos: tl.int32,
    scale: tl.float32,  # 1.0 / ln(2)
    BLOCK: tl.constexpr,
):
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    a = tl.load(A_ptr + offs, mask=mask, other=-float("inf"))
    causal = (offs > absolute_pos) & mask
    a = tl.where(causal, -float("inf"), a)
    m = tl.max(a, axis=0)
    a = a - m
    sum_e = tl.sum(tl.exp(a), axis=0)
    lse = tl.log(sum_e) * scale
    tl.store(C_ptr, lse)


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
    total_q, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    num_pages = ckv_cache.shape[0]
    len_indptr = qo_indptr.shape[0]
    batch_size = len_indptr - 1

    # Cast caches to float32 for compute
    Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, 512]
    Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, 64]

    output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=q_nope.device)
    lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=q_nope.device)

    for b in range(batch_size):
        q_start = int(qo_indptr[b].item())
        q_end = int(qo_indptr[b + 1].item())
        q_len = q_end - q_start

        page_beg = int(kv_indptr[b].item())
        page_end = int(kv_indptr[b + 1].item())
        kv_len = page_end - page_beg

        if q_len <= 0 or kv_len <= 0:
            continue

        tok_idx = kv_indices[page_beg:page_end].to(torch.int64)
        Kc = Kc_all[tok_idx]  # [kv_len, 512]
        Kp = Kp_all[tok_idx]  # [kv_len, 64]

        for i in range(q_len):
            abs_q = q_start + i

            qn = q_nope[abs_q].to(torch.float32).contiguous()  # [16, 512]
            qp = q_pe[abs_q].to(torch.float32).contiguous()   # [16, 64]

            # scores_n = qn @ Kc.T -> [16, kv_len]
            M = qn.shape[0]
            N = kv_len
            K = 512

            scores_n = torch.empty((M, N), dtype=torch.float32, device=qn.device)
            grid_n = (triton.cdiv(M, 16), triton.cdiv(N, 64))
            matmul_kernel[grid_n](
                qn, Kc, scores_n,
                M, N, K,
                qn.stride(0), qn.stride(1),
                Kc.stride(0), Kc.stride(1),
                scores_n.stride(0), scores_n.stride(1),
                BLOCK_M=16, BLOCK_N=64, BLOCK_K=64,
                num_warps=4, num_stages=2
            )

            # scores_p = qp @ Kp.T -> [16, kv_len]
            M2, K2 = qp.shape[0], Kp.shape[1]
            N2 = kv_len
            scores_p = torch.empty((M2, N2), dtype=torch.float32, device=qp.device)
            grid_p = (triton.cdiv(M2, 16), triton.cdiv(N2, 64))
            matmul_kernel[grid_p](
                qp, Kp, scores_p,
                M2, N2, K2,
                qp.stride(0), qp.stride(1),
                Kp.stride(0), Kp.stride(1),
                scores_p.stride(0), scores_p.stride(1),
                BLOCK_M=16, BLOCK_N=64, BLOCK_K=64,
                num_warps=4, num_stages=2
            )

            scores = scores_n + scores_p  # [16, kv_len]
            prefix_len = kv_len - q_len
            absolute_pos = prefix_len + i

            # Softmax with causal mask
            attn = torch.empty((M, N), dtype=torch.float32, device=scores.device)
            softmax_row_causal_kernel[(1,)](
                scores, attn, N, absolute_pos,
                BLOCK=128,
                num_warps=1
            )

            # lse base-2
            lse_row = torch.empty((), dtype=torch.float32, device=attn.device)
            lse_row_causal_kernel[(1,)](
                scores, lse_row, N, absolute_pos, 1.4426950408889634,  # 1 / ln(2)
                BLOCK=128,
                num_warps=1
            )
            lse[abs_q] = lse_row

            # output: attn @ Kc -> [16, 512]
            out_row = torch.empty((M, 512), dtype=torch.float32, device=attn.device)
            grid_out = (triton.cdiv(M, 16), triton.cdiv(512, 64))
            matmul_kernel[grid_out](
                attn, Kc, out_row,
                M, 512, N,
                attn.stride(0), attn.stride(1),
                Kc.stride(0), Kc.stride(1),
                out_row.stride(0), out_row.stride(1),
                BLOCK_M=16, BLOCK_N=64, BLOCK_K=64,
                num_warps=4, num_stages=2
            )
            # Store row per head
            for h in range(num_qo_heads):
                output[abs_q, h] = out_row[h]

    return output, lse


def get_inputs():
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16, device='cuda')
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16, device='cuda')
    ckv_cache = torch.randn([989669, 1, 512], dtype=torch.bfloat16, device='cuda')
    kpe_cache = torch.randn([989669, 1, 64], dtype=torch.bfloat16, device='cuda')
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
    _out = run(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6, tensor_7)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


class Model(torch.nn.Module):
    def forward(self, *args):
        return run(*args)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure device is CUDA
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "Inputs must be on CUDA device"
        return run(q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale)


def run(*args):
    return ModelNew()(*args)
