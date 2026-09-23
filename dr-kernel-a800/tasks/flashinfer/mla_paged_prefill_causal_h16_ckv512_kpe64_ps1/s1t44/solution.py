import math
import torch
import triton
import triton.language as tl


# Triton matmul: C[M, K] = A[M, N] @ B[N, K]
@triton.jit
def matmul_MNK(
    A_ptr, B_ptr, C_ptr,
    M: tl.int32, N: tl.int32, K: tl.int32,
    stride_am: tl.int32, stride_an: tl.int32,
    stride_bn: tl.int32, stride_bk: tl.int32,
    stride_cm: tl.int32, stride_ck: tl.int32,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # One program per row m
    pid = tl.program_id(0)
    m = pid
    acc = tl.zeros((1,), dtype=tl.float32)  # scalar accumulator
    # Loop over N dimension in blocks
    for n0 in range(0, N, BLOCK_N):
        n_offsets = n0 + tl.arange(0, BLOCK_N)
        # For each K block, compute partial dot products
        for k0 in range(0, K, BLOCK_K):
            k_offsets = k0 + tl.arange(0, BLOCK_K)
            # A[m, n_offsets] -> vector
            a_vec = tl.load(A_ptr + m * stride_am + n_offsets * stride_an,
                            mask=n_offsets < N, other=0.0)
            # B[n_offsets, k_offsets] -> [BLOCK_N, BLOCK_K]
            b_block = tl.load(B_ptr + n_offsets[:, None] * stride_bn + k_offsets[None, :] * stride_bk,
                              mask=(n_offsets[:, None] < N) & (k_offsets[None, :] < K),
                              other=0.0)
            # Accumulate: dot(a_vec, b_block along axis=0) -> scalar
            acc += tl.sum(a_vec[:, None] * b_block, axis=0)
    # Store result: C[m, :]
    k_offsets = tl.arange(0, K)
    tl.store(C_ptr + m * stride_cm + k_offsets * stride_ck, acc, mask=True)


# Triton softmax with causal mask (row-wise)
@triton.jit
def softmax_row_causal(
    X_ptr, Y_ptr,
    M: tl.int32, N: tl.int32,
    absolute_pos: tl.int32,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(X_ptr + row * N + offsets, mask=mask, other=-float("inf"))
    j = offsets
    causal = j > absolute_pos
    x = tl.where(causal, -float("inf"), x)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    y = e / s
    tl.store(Y_ptr + row, y)


# Triton logsumexp (base-2) with causal mask (row-wise)
@triton.jit
def lse_row_causal(
    X_ptr, Y_ptr,
    M: tl.int32, N: tl.int32,
    absolute_pos: tl.int32,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < N
    x = tl.load(X_ptr + row * N + offsets, mask=mask, other=-float("inf"))
    j = offsets
    causal = j > absolute_pos
    x = tl.where(causal, -float("inf"), x)
    m = tl.max(x, axis=0)
    e = tl.exp(x - m)
    s = tl.sum(e, axis=0)
    lse = tl.log(s) / tl.log(2.0)
    tl.store(Y_ptr + row, lse)


@torch.no_grad()
def run(q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
    # Ensure CUDA device
    assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "Inputs must be CUDA tensors"

    total_q, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    assert num_qo_heads == 16
    assert head_dim_ckv == 512
    assert head_dim_kpe == 64

    # Extract cached Ks
    Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, 512]
    Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, 64]

    device = q_nope.device
    output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
    lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

    batch_size = qo_indptr.shape[0] - 1
    for b in range(batch_size):
        q_start = int(qo_indptr[b].item())
        q_end = int(qo_indptr[b + 1].item())
        if q_start >= q_end:
            continue

        q_len = q_end - q_start
        if q_len == 0:
            continue

        page_beg = int(kv_indptr[b].item())
        page_end = int(kv_indptr[b + 1].item())
        if page_beg >= page_end:
            continue
        kv_len = page_end - page_beg
        tok_idx = kv_indices[page_beg:page_end].to(torch.int32).to(device)
        Kc = Kc_all[tok_idx].contiguous()  # [kv_len, 512]
        Kp = Kp_all[tok_idx].contiguous()  # [kv_len, 64]

        for i in range(q_len):
            abs_q = q_start + i

            # Load qn, qp (fp32 compute)
            qn = q_nope[abs_q].to(torch.float32).contiguous()  # [16, 512]
            qp = q_pe[abs_q].to(torch.float32).contiguous()   # [16, 64]

            # scores_n = qn @ Kc.T -> [16, kv_len]
            scores_n = torch.empty((qn.shape[0], Kc.shape[0]), dtype=torch.float32, device=device)
            grid_n = (qn.shape[0],)
            matmul_MNK[grid_n](
                qn, Kc.transpose(0, 1), scores_n,  # B^T is [512, kv_len]
                qn.shape[0], Kc.shape[1], Kc.shape[0],
                qn.stride(0), qn.stride(1),
                Kc.transpose(0, 1).stride(0), Kc.transpose(0, 1).stride(1),
                scores_n.stride(0), scores_n.stride(1),
                BLOCK_N=128, BLOCK_K=128, num_warps=4, num_stages=2,
            )

            # scores_p = qp @ Kp.T -> [16, kv_len]
            scores_p = torch.empty((qp.shape[0], Kp.shape[0]), dtype=torch.float32, device=device)
            grid_p = (qp.shape[0],)
            matmul_MNK[grid_p](
                qp, Kp.transpose(0, 1), scores_p,
                qp.shape[0], Kp.shape[1], Kp.shape[0],
                qp.stride(0), qp.stride(1),
                Kp.transpose(0, 1).stride(0), Kp.transpose(0, 1).stride(1),
                scores_p.stride(0), scores_p.stride(1),
                BLOCK_N=128, BLOCK_K=64, num_warps=4, num_stages=2,
            )

            scores = scores_n + scores_p  # [16, kv_len]

            # Causal mask: positions j > (prefix_len + i) => -inf
            prefix_len = kv_len - q_len
            absolute_pos = prefix_len + i

            # Softmax (row-wise) into attn
            attn = torch.empty((scores.shape[0], scores.shape[1]), dtype=torch.float32, device=device)
            grid_s = (scores.shape[0],)
            softmax_row_causal[grid_s](
                scores, attn,
                scores.shape[0], scores.shape[1],
                absolute_pos,
                BLOCK=256 if scores.shape[1] >= 256 else 128,
                num_warps=4,
            )

            # LSE per head (base-2), one scalar per row
            lse_row = torch.empty((scores.shape[0],), dtype=torch.float32, device=device)
            grid_l = (scores.shape[0],)
            lse_row_causal[grid_l](
                scores, lse_row,
                scores.shape[0], scores.shape[1],
                absolute_pos,
                BLOCK=256 if scores.shape[1] >= 256 else 128,
                num_warps=4,
            )
            lse[abs_q] = lse_row  # [16]

            # Output: attn @ Kc -> [16, 512], cast to bfloat16 and store
            out_row = torch.empty((attn.shape[0], head_dim_ckv), dtype=torch.float32, device=device)
            grid_out = (attn.shape[0],)
            matmul_MNK[grid_out](
                attn, Kc, out_row,
                attn.shape[0], Kc.shape[1], head_dim_ckv,
                attn.stride(0), attn.stride(1),
                Kc.stride(0), Kc.stride(1),
                out_row.stride(0), out_row.stride(1),
                BLOCK_N=128, BLOCK_K=128, num_warps=4, num_stages=2,
            )
            output[abs_q] = out_row.to(torch.bfloat16)

    return output, lse


# Original helpers (kept for evaluation harness)
def get_inputs():
    device = 'cuda'
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16, device=device)
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16, device=device)
    ckv_cache = torch.randn([989669, 1, 512], dtype=torch.bfloat16, device=device)
    kpe_cache = torch.randn([989669, 1, 64], dtype=torch.bfloat16, device=device)
    _n = 1; _t = 1
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    qo_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32).to(device)
    _n = 1; _t = 34
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32).to(device)
    kv_indices = torch.randint(0, 989669, [34], dtype=torch.int32).to(device)
    sm_scale = 1.0  # float32 scalar
    return [q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale]


def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6, tensor_7):
    _out = run(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6, tensor_7)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        return run(q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale)


def run(*args):
    return ModelNew()(*args)
