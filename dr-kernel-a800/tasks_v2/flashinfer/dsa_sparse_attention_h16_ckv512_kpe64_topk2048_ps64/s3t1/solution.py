import math
import torch
import triton
import triton.language as tl


# Triton kernel: row-wise matmul C = A_row @ B, where:
# A_row is a 1D vector of length M.
# B is a 2D matrix of shape (N, M). Output C is a 1D vector of length N.
@triton.jit
def matmul_row(A_ptr, B_ptr, C_ptr,
               M: tl.constexpr, N: tl.constexpr,
               BLOCK_N: tl.constexpr):
    # Single-program reduction over N in chunks
    # For each chunk [j_start, j_start+BLOCK_N), compute dot(A, B[j, :]) for j in chunk
    for j_start in range(0, N, BLOCK_N):
        j = j_start + tl.arange(0, BLOCK_N)
        mask_j = j < N
        # Load A vector
        a = tl.load(A_ptr)  # A_ptr points to [M], contiguous
        # Load B block: shape [BLOCK_N, M]
        b_ptrs = B_ptr + j[:, None] * M + tl.arange(0, M)
        b = tl.load(b_ptrs, mask=mask_j[:, None], other=0.0)
        # Accumulate dot products over M
        acc = tl.zeros([BLOCK_N], dtype=tl.float32)
        for m in range(0, M):
            acc += b[:, m] * a[m]
        # Store results
        tl.store(C_ptr + j, acc, mask=mask_j)


# Triton kernel: row-wise softmax (numerically stable) and base-2 logsumexp.
# X is 1D vector of length TOPK (up to MAX_BLOCK_SIZE).
# Valid is 1D mask (int32) of length TOPK: 1 for valid, 0 for padded.
# Outputs:
#   Out: softmax probabilities (float32) of length TOPK (only valid positions written via mask logic).
#   LSE_ptr: scalar (float32) storing logsumexp base-2 for this row.
@triton.jit
def softmax_logsumexp2_row(X_ptr, Valid_ptr, Out_ptr, LSE_ptr,
                            TOPK: tl.constexpr, NUM_VALID: tl.constexpr):
    # Compute max
    m = -float("inf")
    for j in range(0, TOPK):
        is_valid = tl.load(Valid_ptr + j)  # int32 0/1
        xj = tl.load(X_ptr + j)
        xj = tl.where(is_valid != 0, xj, -float("inf"))
        m = tl.maximum(m, xj)
    # Compute sum of exp
    s = 0.0
    for j in range(0, TOPK):
        is_valid = tl.load(Valid_ptr + j)
        xj = tl.load(X_ptr + j)
        xj = tl.where(is_valid != 0, xj, -float("inf"))
        s += tl.exp(xj - m)
    # Write softmax into Out (valid positions only)
    for j in range(0, TOPK):
        is_valid = tl.load(Valid_ptr + j)
        xj = tl.load(X_ptr + j)
        xj = tl.where(is_valid != 0, xj, -float("inf"))
        prob = tl.exp(xj - m) / s
        # We store all entries but invalid positions will be zeroed on host side. Here we can skip store if invalid.
        # Triton doesn't support conditional store via mask in for loop easily; we rely on host to zero invalid after.
        tl.store(Out_ptr + j, prob)
    # Compute base-2 logsumexp: lse = m + log(s) then divide by ln(2)
    lse_val = m + tl.log(s)
    tl.store(LSE_ptr, lse_val / 1.4426950408889634)  # 1 / ln(2)


# Triton kernel: compute out = attn @ Kc_row
# attn is a 1D vector of length N (NUM_VALID).
# Kc is a 2D matrix of shape (N, OUT), OUT is a constexpr (512).
# out is a 1D vector of length OUT, computed as sum_i attn[i] * Kc[i, :].
@triton.jit
def reduction_row(Attn_ptr, Kc_ptr, Out_ptr,
                  NUM_VALID: tl.constexpr, OUT: tl.constexpr, BLOCK_N: tl.constexpr):
    acc = tl.zeros([OUT], dtype=tl.float32)
    for i_start in range(0, NUM_VALID, BLOCK_N):
        i = i_start + tl.arange(0, BLOCK_N)
        mask_i = i < NUM_VALID
        attn_chunk = tl.load(Attn_ptr + i, mask=mask_i, other=0.0)
        kc_ptrs = Kc_ptr + i[:, None] * OUT + tl.arange(0, OUT)
        kc = tl.load(kc_ptrs, mask=mask_i[:, None], other=0.0)
        # Accumulate over the chunk
        for ii in range(0, BLOCK_N):
            ii_mask = mask_i[ii]
            # attn_chunk[ii] is zero when ii_mask is False
            acc += attn_chunk[ii] * kc[ii, :]
    tl.store(Out_ptr, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_qo_heads = 16
        self.head_dim_ckv = 512
        self.head_dim_kpe = 64
        self.topk = 2048  # length of sparse_indices per token
        self.sm_scale = 1.0
        self.BLOCK_N = 128
        self.MAX_BLOCK_SIZE = 2048  # matches topk

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
        # Validate inputs
        num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
        assert num_qo_heads == self.num_qo_heads
        assert head_dim_ckv == self.head_dim_ckv
        assert q_pe.shape[1] == self.num_qo_heads
        assert q_pe.shape[-1] == self.head_dim_kpe
        assert ckv_cache.shape[1] == 64 and ckv_cache.shape[-1] == self.head_dim_ckv
        assert kpe_cache.shape[1] == 64 and kpe_cache.shape[-1] == self.head_dim_kpe

        # Ensure CUDA tensors and dtype for compute
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "All tensors must be on CUDA."
        device = q_nope.device

        # Flatten paged KV cache to token-level selections
        Kc_all = ckv_cache.reshape(-1, self.head_dim_ckv).to(torch.float32)  # [num_pages*64, 512]
        Kp_all = kpe_cache.reshape(-1, self.head_dim_kpe).to(torch.float32)  # [num_pages*64, 64]

        output = torch.empty((num_tokens, num_qo_heads, self.head_dim_ckv), dtype=torch.float32, device=device)
        lse = torch.empty((num_tokens, num_qo_heads), dtype=torch.float32, device=device)

        for t in range(num_tokens):
            indices = sparse_indices[t]  # [topk]
            valid_mask = indices != -1
            valid_indices = indices[valid_mask].to(torch.long)
            num_valid = valid_indices.numel()
            # Prepare mask for Triton softmax (pad to TOPK)
            valid_int = torch.ones(self.topk, dtype=torch.int32, device=device)
            valid_int[num_valid:] = 0  # invalid positions

            # Gather Kc and Kp rows for this token
            Kc = Kc_all[valid_indices]  # [num_valid, 512]
            Kp = Kp_all[valid_indices]  # [num_valid, 64]

            # q_nope and q_pe for this token
            qn = q_nope[t].to(torch.float32)  # [num_qo_heads, 512]
            qp = q_pe[t].to(torch.float32)    # [num_qo_heads, 64]

            for h in range(num_qo_heads):
                # Compute qn[h] @ Kc.T and qp[h] @ Kp.T
                A_qn = qn[h]  # [


def run(*args):
    return ModelNew()(*args)
