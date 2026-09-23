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
    # 2D grid over M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in tiles
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # Compute pointers for A and B tiles
        A_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        B_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn

        A_tile = tl.load(A_ptrs, mask=(m_offsets[:, None] < M) & (k_offsets[None, :] < K), other=0.0)
        B_tile = tl.load(B_ptrs, mask=(k_offsets[:, None] < K) & (n_offsets[None, :] < N), other=0.0)

        acc += tl.dot(A_tile, B_tile)

    # Write results to C
    C_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    tl.store(C_ptrs, acc, mask=(m_offsets[:, None] < M) & (n_offsets[None, :] < N))


# Softmax over a row (1D), with causal mask j > absolute_pos -> -inf
@triton.jit
def softmax_row_causal_kernel(
    X_ptr, Out_ptr,
    N: tl.int32,
    scale: tl.float32,  # dummy; not used in this kernel
    absolute_pos: tl.int32,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    # Load row and apply causal mask: j > (absolute_pos) -> -inf
    x = tl.load(X_ptr + row_id * N + offs, mask=mask, other=0.0)
    causal = offs > absolute_pos
    x = tl.where(causal & mask, -float("inf"), x)

    # Stable softmax
    max_x = tl.max(x, axis=0)
    x = x - max_x
    exp_x = tl.exp(x)
    sum_exp = tl.sum(exp_x, axis=0)
    out = exp_x / sum_exp

    tl.store(Out_ptr + row_id * N + offs, out, mask=mask)


# Logsumexp over a row (1D), with causal mask, base-2
@triton.jit
def lse_row_causal_kernel(
    X_ptr, Out_ptr,
    N: tl.int32,
    scale: tl.float32,  # dummy; not used in this kernel
    absolute_pos: tl.int32,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N

    x = tl.load(X_ptr + row_id * N + offs, mask=mask, other=0.0)
    causal = offs > absolute_pos
    x = tl.where(causal & mask, -float("inf"), x)

    max_x = tl.max(x, axis=0)
    x = x - max_x
    exp_x = tl.exp(x)
    sum_exp = tl.sum(exp_x, axis=0)
    lse_val = tl.log(sum_exp) / tl.log(2.0)  # base-2 logsumexp

    tl.store(Out_ptr + row_id, lse_val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # constants for blocks
        self.BLOCK_M = 16
        self.BLOCK_N = 64
        self.BLOCK_K = 32
        self.BLOCK_ROW = 256  # for row kernels (softmax/logsumexp)

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # device and shapes
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "Tensors must be on CUDA device for Triton kernels"
        device = q_nope.device
        dtype_compute = torch.float32

        total_q = q_nope.shape[0]
        num_qo_heads = q_nope.shape[1]
        head_dim_ckv = q_nope.shape[2]
        head_dim_kpe = q_pe.shape[2]

        # Prepare key caches
        Kc_all = ckv_cache.squeeze(1).to(dtype_compute)  # [num_pages, head_dim_ckv]
        Kp_all = kpe_cache.squeeze(1).to(dtype_compute)  # [num_pages, head_dim_kpe]

        # Output buffers
        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)  # compute in fp32, cast to bf16 at end
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Number of batches
        batch_size = qo_indptr.shape[0] - 1
        assert kv_indptr.shape[0] - 1 == batch_size, "Batch sizes from qo_indptr and kv_indptr must match"

        # Iterate over batches
        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            q_len = q_end - q_start
            if q_len <= 0:
                continue

            # Get kv indices for this batch
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            kv_len = page_end - page_beg
            if kv_len <= 0:
                # No KV for this batch: produce zeros
                for i in range(q_len):
                    out_row = torch.zeros((num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
                    output[q_start + i] = out_row
                    lse[q_start + i] = torch.tensor(0.0, dtype=torch.float32, device=device)
                continue

            # Gather Kc and Kp based on kv_indices
            tok_idx = kv_indices[page_beg:page_end]  # [kv_len], int32
            Kc = Kc_all[tok_idx]  # [kv_len, head_dim_ckv]
            Kp = Kp_all[tok_idx]  # [kv_len, head_dim_kpe]

            # Process each query in this batch
            for i in range(q_len):
                abs_q = q_start + i

                # Load qn, qp
                qn = q_nope[abs_q].to(dtype_compute)  # [num_heads=16, Kc_dim=512]
                qp = q_pe[abs_q].to(dtype_compute)   # [num_heads=16, Kp_dim=64]

                # Compute scores_n = qn @ Kc.T and scores_p = qp @ Kp.T
                scores_n = torch.empty((num_qo_heads, kv_len), dtype=torch.float32, device=device)
                scores_p = torch.empty((num_qo_heads, kv_len), dtype=torch.float32, device=device)

                # A = qn [M=16,K=512], B = Kc.T [K=512,N=kv_len]
                grid_n = (triton.cdiv(num_qo_heads, self.BLOCK_M), triton.cdiv(kv_len, self.BLOCK_N))
                matmul_kernel[grid_n](
                    qn, Kc.transpose(0, 1), scores_n,  # A, B, C
                    num_qo_heads, head_dim_ckv, kv_len,
                    qn.stride(0), qn.stride(1),       # strides for A
                    Kc.stride(0), Kc.stride(1),       # strides for B (Kc.T)
                    scores_n.stride(0), scores_n.stride(1),
                    BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N, BLOCK_K=self.BLOCK_K,
                    num_warps=4, num_stages=2
                )

                # A = qp [M=16,K=64], B = Kp.T [K=64,N=kv_len]
                grid_p = (triton.cdiv(num_qo_heads, self.BLOCK_M), triton.cdiv(kv_len, self.BLOCK_N))
                scores_p[...] = 0.0  # ensure initialized
                matmul_kernel[grid_p](
                    qp, Kp.transpose(0, 1), scores_p,  # A, B, C
                    num_qo_heads, head_dim_kpe, kv_len,
                    qp.stride(0), qp.stride(1),
                    Kp.stride(0), Kp.stride(1),
                    scores_p.stride(0), scores_p.stride(1),
                    BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N, BLOCK_K=self.BLOCK_K,
                    num_warps=4, num_stages=2
                )

                scores = scores_n + scores_p  # [16, kv_len]
                # Apply causal mask: positions j > (prefix_len + i) -> -inf
                prefix_len = kv_len - q_len  # number of previously cached tokens
                absolute_pos = prefix_len + i

                # Softmax over kv_len for each head
                softmax_out = torch.empty((num_qo_heads, kv_len), dtype=torch.float32, device=device)
                lse_row = torch.empty((num_qo_heads,), dtype=torch.float32, device=device)

                grid_row = (num_qo_heads,)
                softmax_row_causal_kernel[grid_row](
                    scores, softmax_out,
                    kv_len, 1.0, absolute_pos,
                    BLOCK=self.BLOCK_ROW
                )

                # out = softmax @ Kc
                out_row = torch.empty((num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
                matmul_kernel[(triton.cdiv(num_qo_heads, self.BLOCK_M), triton.cdiv(head_dim_ckv, self.BLOCK_N))](
                    softmax_out, Kc, out_row,
                    num_qo_heads, kv_len, head_dim_ckv,
                    softmax_out.stride(0), softmax_out.stride(1),
                    Kc.stride(0), Kc.stride(1),
                    out_row.stride(0), out_row.stride(1),
                    BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N, BLOCK_K=self.BLOCK_K,
                    num_warps=4, num_stages=2
                )

                # Store outputs
                output[abs_q] = out_row
                # lse is per-head, base-2 logsumexp (already in scores, per head)
                # Use kernel to compute:
                lse_row_causal_kernel[(num_qo_heads,)](
                    scores, lse_row,
                    kv_len, 1.0, absolute_pos,
                    BLOCK=self.BLOCK_ROW
                )
                lse[abs_q] = lse_row

        # Cast output to bfloat16 as required
        output = output.to(torch.bfloat16)
        return output, lse


# Optional: original helper and Model (if needed by harness)
def get_inputs():
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16, device='cuda')
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16, device='cuda')
    ckv_cache = torch.randn([989669, 1, 512], dtype=torch.bfloat16, device='cuda')
    kpe_cache = torch.randn([989669, 1, 64], dtype=torch.bfloat16, device='cuda')
    _n = 1; _t = 1
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32, device='cuda')
    _lens[: _t % _n] += 1
    qo_indptr = torch.cat([torch.zeros(1, dtype=torch.int32, device='cuda'), torch.cumsum(_lens, 0)]).to(torch.int32)
    _n = 1; _t = 34
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32, device='cuda')
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32, device='cuda'), torch.cumsum(_lens, 0)]).to(torch.int32)
    kv_indices = torch.randint(0, 989669, [34], dtype=torch.int32, device='cuda')
    sm_scale = 1.0
    return [q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale]


class Model(torch.nn.Module):
    def forward(self, *args):
        return ModelNew().forward(*args)


def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6, tensor_7):
    _out = ModelNew().forward(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6, tensor_7)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


def run(*args):
    return ModelNew()(*args)
