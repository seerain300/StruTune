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
    # 2D tiling over M and N
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over K in tiles
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        A_ptrs = A_ptr + m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        B_ptrs = B_ptr + k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn

        a = tl.load(A_ptrs, mask=(m_offsets[:, None] < M) & (k_offsets[None, :] < K), other=0.0)
        b = tl.load(B_ptrs, mask=(k_offsets[:, None] < K) & (n_offsets[None, :] < N), other=0.0)

        acc += tl.dot(a, b)

    C_ptrs = C_ptr + m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn
    tl.store(C_ptrs, acc, mask=(m_offsets[:, None] < M) & (n_offsets[None, :] < N))


# Row-wise softmax with causal mask: Out[M] = softmax(X[M] * scale, causal j > absolute_pos)
@triton.jit
def softmax_row_causal_kernel(
    X_ptr, Out_ptr,
    M: tl.int32,  # number of columns (elements in row)
    scale: tl.float32,  # multiply before softmax
    absolute_pos: tl.int32,  # positions > absolute_pos are masked
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    x = tl.load(X_ptr + row_id * M + cols, mask=cols < M, other=-float("inf"))
    # Scale
    x = x * scale
    # Causal mask: positions j > absolute_pos -> -inf
    x = tl.where(cols <= absolute_pos, x, -float("inf"))
    # Stable softmax
    m = tl.max(x, axis=0)
    x = x - m
    e = tl.exp(x)
    s = tl.sum(e, axis=0)
    out = e / s
    tl.store(Out_ptr + row_id * M + cols, out, mask=cols < M)


# Row-wise logsumexp (base-2) with causal mask: Out[M] = logsumexp(X[M] * scale, causal) / ln(2)
@triton.jit
def lse_row_causal_kernel(
    X_ptr, Out_ptr,
    M: tl.int32,
    scale: tl.float32,
    absolute_pos: tl.int32,
    BLOCK: tl.constexpr,
):
    row_id = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    x = tl.load(X_ptr + row_id * M + cols, mask=cols < M, other=-float("inf"))
    x = x * scale
    x = tl.where(cols <= absolute_pos, x, -float("inf"))
    m = tl.max(x, axis=0)
    x = x - m
    e = tl.exp(x)
    s = tl.sum(e, axis=0)
    lse = tl.log(s) / tl.log(2.0)
    tl.store(Out_ptr + row_id, lse)  # one scalar per row


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.SM_SCALE = 1.0
        # Triton tiling defaults (can be tuned per run)
        self.BLOCK_M = 64
        self.BLOCK_N = 64
        self.BLOCK_K = 32
        self.num_warps = 4
        self.num_stages = 2

    def _matmul_triton(self, A, B):
        """
        Compute A @ B with Triton matmul kernel. A: [M,K], B: [K,N] -> C: [M,N]
        Returns C as float32 tensor on device.
        """
        assert A.dim() == 2 and B.dim() == 2
        M, K = A.shape
        Kb, N = B.shape
        assert K == Kb
        C = torch.empty((M, N), dtype=torch.float32, device=A.device)
        grid = (triton.cdiv(M, self.BLOCK_M), triton.cdiv(N, self.BLOCK_N))
        matmul_kernel[grid](
            A, B, C,
            M, K, N,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=self.BLOCK_M, BLOCK_N=self.BLOCK_N, BLOCK_K=self.BLOCK_K,
            num_warps=self.num_warps, num_stages=self.num_stages
        )
        return C

    def _softmax_row_causal_triton(self, X):
        """
        X: [1, N] float32 tensor on device. Return softmax(X) with causal mask j <= absolute_pos (abs_pos=0).
        """
        M, N = X.shape
        assert M == 1
        Out = torch.empty((1, N), dtype=torch.float32, device=X.device)
        grid = (1,)
        softmax_row_causal_kernel[grid](X, Out, N, 1.0, 0, BLOCK=N)
        return Out[0]  # [N] tensor

    def _lse_row_causal_triton(self, X):
        """
        X: [1, N] float32 tensor on device. Return logsumexp(X) with causal mask j <= absolute_pos (abs_pos=0).
        """
        M, N = X.shape
        assert M == 1
        Out = torch.empty((1,), dtype=torch.float32, device=X.device)
        grid = (1,)
        lse_row_causal_kernel[grid](X, Out, N, 1.0, 0, BLOCK=N)
        return Out[0]


    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Input tensors are on CUDA device (eval environment supplies CUDA tensors)
        device = q_nope.device
        total_q = q_nope.shape[0]
        num_qo_heads = q_nope.shape[1]
        head_dim_ckv = q_nope.shape[2]
        head_dim_kpe = q_pe.shape[2]

        # Constants (from original)
        assert num_qo_heads == 16
        assert head_dim_ckv == 512
        assert head_dim_kpe == 64

        # Kc_all: [num_pages, 512], Kp_all: [num_pages, 64]
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, 64]

        # Output buffers
        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        batch_size = qo_indptr.numel() - 1
        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            if q_start >= q_end:
                continue
            q_len = q_end - q_start

            # kv indices for this batch
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            kv_len = page_end - page_beg
            tok_idx = kv_indices[page_beg:page_end].to(torch.int32)
            # Gather Kc and Kp
            Kc = Kc_all[tok_idx].to(torch.float32)  # [kv_len, 512]
            Kp = Kp_all[tok_idx].to(torch.float32)  # [kv_len, 64]

            # Iterate over queries in this batch
            for i in range(q_len):
                abs_q = q_start + i
                qn = q_nope[abs_q].to(torch.float32)  # [16, 512]
                qp = q_pe[abs_q].to(torch.float32)   # [16, 64]

                # Compute scores_n = qn @ Kc.T  -> [16, kv_len]
                scores_n = self._matmul_triton(qn, Kc.T)  # A [16,512], B [512,kv_len]
                # Compute scores_p = qp @ Kp.T  -> [16, kv_len]
                scores_p = self._matmul_triton(qp, Kp.T)  # A [16,64], B [64,kv_len]
                scores = scores_n + scores_p  # [16, kv_len]

                # Apply causal mask: positions j > (prefix_len + i) -> -inf
                prefix_len = kv_len - q_len  # number of previously cached tokens
                absolute_pos = prefix_len + i

                # Triton softmax with causal mask (1 row per head), scale=1.0
                attn = torch.empty((16, scores.shape[1]), dtype=torch.float32, device=device)
                for h in range(16):
                    x = scores[h].view(1, -1).to(torch.float32)  # [1, kv_len]
                    attn[h] = self._softmax_row_causal_triton(x)  # [kv_len]

                # Now compute output = attn @ Kc -> [16, 512]
                out_row = self._matmul_triton(attn, Kc)  # A [16,kv_len], B [kv_len,512] -> [16,512]
                output[abs_q] = out_row.to(torch.bfloat16)

                # Compute LSE per head: logsumexp(scores) / ln(2) with causal mask
                lse_row = torch.empty((16,), dtype=torch.float32, device=device)
                for h in range(16):
                    x = scores[h].view(1, -1).to(torch.float32)
                    lse_row[h] = self._lse_row_causal_triton(x) / math.log(2.0)  # base-2
                lse[abs_q] = lse_row

        # Return as original types
        output = output.to(torch.bfloat16)
        lse = lse  # float32
        return output, lse


def run(*args):
    return ModelNew()(*args)
