import math
import torch
import triton
import triton.language as tl


# Matmul kernel: C[M, N] = A[M, K] @ B[K, N]
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M: tl.int32, N: tl.int32, K: tl.int32,
    stride_am: tl.int32, stride_ak: tl.int32,
    stride_bk: tl.int32, stride_bn: tl.int32,
    stride_cm: tl.int32, stride_cn: tl.int32,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D tiling over output (M, N)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # loop over K
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # pointers to A and B tiles
        A_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        B_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn

        # load tiles
        A = tl.load(A_ptrs, mask=(m_offsets[:, None] < M) & (k_offsets[None, :] < K), other=0.0)
        B = tl.load(B_ptrs, mask=(k_offsets[:, None] < K) & (n_offsets[None, :] < N), other=0.0)

        # accumulate
        acc += tl.dot(A, B)

    # write back
    C_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    tl.store(C_ptrs, acc, mask=(m_offsets[:, None] < M) & (n_offsets[None, :] < N))


# Softmax with causal mask (row-wise)
@triton.jit
def softmax_row_causal_kernel(
    X_ptr, Y_ptr,
    N: tl.int32,  # row length
    scale: tl.float32,
    absolute_pos: tl.int32,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask_valid = offs < N

    # load row
    X = tl.load(X_ptr + row * N + offs, mask=mask_valid, other=-float("inf"))
    # apply scale
    X = X * scale

    # causal mask: positions > absolute_pos -> -inf
    causal = (offs > absolute_pos) & mask_valid
    X = tl.where(causal, -float("inf"), X)

    # stable softmax
    m = tl.max(X, axis=0)
    X = X - m
    exps = tl.exp(X)
    sum_exps = tl.sum(exps, axis=0)
    Y = exps / sum_exps

    # store
    tl.store(Y_ptr + row * N + offs, Y, mask=mask_valid)


# Row-wise logsumexp with causal mask (base-2), returns vector length N
@triton.jit
def lse_row_causal_kernel(
    X_ptr, Y_ptr,
    N: tl.int32,
    scale: tl.float32,
    absolute_pos: tl.int32,
    base_log: tl.float32,  # log(2)
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask_valid = offs < N

    X = tl.load(X_ptr + row * N + offs, mask=mask_valid, other=-float("inf"))
    X = X * scale
    causal = (offs > absolute_pos) & mask_valid
    X = tl.where(causal, -float("inf"), X)

    m = tl.max(X, axis=0)
    X = X - m
    sum_exps = tl.sum(tl.exp(X), axis=0)
    lse = tl.log(sum_exps) / base_log  # base-2 logsumexp
    tl.store(Y_ptr + row, lse)  # store scalar per row


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure tensors are on CUDA for Triton
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "All tensors must be on CUDA"
        device = q_nope.device

        # Shape constraints: q_nope: [T, 16, 512], q_pe: [T, 16, 64]
        T, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]
        assert num_qo_heads == 16, "num_qo_heads must be 16"
        assert head_dim_ckv == 512, "head_dim_ckv must be 512"
        assert head_dim_kpe == 64, "head_dim_kpe must be 64"

        total_q = qo_indptr[-1].item()
        batch_size = qo_indptr.numel() - 1
        num_pages = kv_indices.numel()

        # Cache all tokens
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)   # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)   # [num_pages, 64]

        # Output buffers
        output = torch.empty((total_q, 16, 512), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, 16), dtype=torch.float32, device=device)

        # Process batches
        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            # If degenerate, skip
            if q_start >= q_end or kv_start >= kv_end:
                continue

            q_len = q_end - q_start
            kv_len = kv_end - kv_start
            tok_idx = kv_indices[kv_start:kv_end]  # [kv_len]

            # Gather keys
            Kc = Kc_all[tok_idx]  # [kv_len, 512]
            Kp = Kp_all[tok_idx]  # [kv_len, 64]

            # For Triton matmul, we'll handle one query per iteration
            # Compute per query i
            for i in range(q_len):
                abs_q = q_start + i
                # Gather qn, qp: [16, 512], [16, 64]
                qn = q_nope[abs_q]            # [16, 512]
                qp = q_pe[abs_q]              # [16, 64]

                # Make contiguous [M, K] for matmul
                # Note: M=16 is fixed by the model definition; we pass as M into Triton
                A_qn = qn.to(torch.float32).contiguous().view(16, kv_len)  # [M=16, K=kv_len]
                A_qp = qp.to(torch.float32).contiguous().view(16, kv_len)  # [M=16, K=kv_len]

                # Compute scores_n = qn @ Kc.T -> [16, kv_len]
                scores_n = torch.empty((16, kv_len), dtype=torch.float32, device=device)
                matmul_kernel[(1,)](
                    A_qn, Kc.transpose(0, 1).contiguous(), scores_n,
                    16, kv_len, 512,
                    A_qn.stride(0), A_qn.stride(1),
                    Kc.transpose(0, 1).stride(0), Kc.transpose(0, 1).stride(1),
                    scores_n.stride(0), scores_n.stride(1),
                    BLOCK_M=16, BLOCK_N=64, BLOCK_K=64,
                    num_warps=4, num_stages=2
                )

                # Compute scores_p = qp @ Kp.T -> [16, kv_len]
                scores_p = torch.empty((16, kv_len), dtype=torch.float32, device=device)
                matmul_kernel[(1,)](
                    A_qp, Kp.transpose(0, 1).contiguous(), scores_p,
                    16, kv_len, 64,
                    A_qp.stride(0), A_qp.stride(1),
                    Kp.transpose(0, 1).stride(0), Kp.transpose(0, 1).stride(1),
                    scores_p.stride(0), scores_p.stride(1),
                    BLOCK_M=16, BLOCK_N=64, BLOCK_K=64,
                    num_warps=4, num_stages=2
                )

                scores = scores_n + scores_p  # [16, kv_len]
                scale = sm_scale
                absolute_pos = (kv_len - q_len) + i

                # Softmax with causal mask
                attn = torch.empty((16, kv_len), dtype=torch.float32, device=device)
                softmax_row_causal_kernel[(16,)](
                    scores, attn,
                    kv_len, scale, absolute_pos,
                    BLOCK=128,
                    num_warps=4, num_stages=2
                )

                # Output projection: attn @ Kc -> [16, 512]
                out_row = torch.empty((16, 512), dtype=torch.float32, device=device)
                matmul_kernel[(1,)](
                    attn, Kc, out_row,
                    16, 512, kv_len,
                    attn.stride(0), attn.stride(1),
                    Kc.stride(0), Kc.stride(1),
                    out_row.stride(0), out_row.stride(1),
                    BLOCK_M=16, BLOCK_N=64, BLOCK_K=64,
                    num_warps=4, num_stages=2
                )

                output[abs_q] = out_row

                # LSE per head
                lse_row = torch.empty((16,), dtype=torch.float32, device=device)
                lse_row_causal_kernel[(16,)](
                    scores, lse_row,
                    kv_len, scale, absolute_pos, 1.0 / math.log(2.0),
                    BLOCK=128,
                    num_warps=4, num_stages=2
                )
                lse[abs_q] = lse_row

        # Cast output to bfloat16 as per original
        return output.to(torch.bfloat16), lse


# Original get_inputs stays the same, for evaluation harness
def get_inputs():
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16, device="cuda")
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16, device="cuda")
    ckv_cache = torch.randn([989669, 1, 512], dtype=torch.bfloat16, device="cuda")
    kpe_cache = torch.randn([989669, 1, 64], dtype=torch.bfloat16, device="cuda")
    _n = 1; _t = 1
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    qo_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32)
    _n = 1; _t = 34
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32)
    kv_indices = torch.randint(0, 989669, [34], dtype=torch.int32, device="cuda")
    sm_scale = 1.0  # float32 scalar
    return [q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale]


# Fused wrapper to mimic original API
def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6, tensor_7):
    _out = ModelNew().forward(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6, tensor_7)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


def run(*args):
    return ModelNew()(*args)
