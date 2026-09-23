import math
import torch
import triton
import triton.language as tl


# Triton kernel: row-wise matmul C = A_row @ B, where:
# A_row is a 1D vector of length M (constexpr).
# B is a 2D matrix of shape (N, M). We use TOPK (constexpr) for vectorization and mask j < N to handle variable N.
@triton.jit
def matmul_row(A_ptr, B_ptr, C_ptr,
               M: tl.constexpr, N: tl.constexpr, TOPK: tl.constexpr):
    # j over TOPK (constexpr)
    j = tl.arange(0, TOPK)
    mask_j = j < N  # only first N columns are valid
    # Load A row (length M), contiguous
    a = tl.load(A_ptr)
    # Load B block: shape [TOPK, M]
    b_ptrs = B_ptr + j[:, None] * M + tl.arange(0, M)
    b = tl.load(b_ptrs, mask=mask_j[:, None], other=0.0)
    # Accumulate dot products over M
    acc = tl.zeros([TOPK], dtype=tl.float32)
    for m in range(0, M):
        acc += b[:, m] * a[m]
    # Store results only for valid j
    tl.store(C_ptr + j, acc, mask=mask_j)


# Triton kernel: row-wise softmax (numerically stable) and base-2 logsumexp.
# X: 1D vector of length TOPK.
# NUM_VALID: number of valid elements (runtime scalar).
# Outputs:
#   Out: softmax probabilities (float32) for valid positions.
#   LSE_ptr: scalar (float32) storing logsumexp base-2.
@triton.jit
def softmax_logsumexp2_row(X_ptr, Out_ptr, LSE_ptr,
                            NUM_VALID: tl.constexpr, TOPK: tl.constexpr):
    # Compute max over valid entries; set invalid to -inf
    m = -float("inf")
    for j in range(0, TOPK):
        xj = tl.load(X_ptr + j)
        if j >= NUM_VALID:
            xj = -float("inf")
        m = tl.maximum(m, xj)
    # Compute sum of exp over valid entries
    s = 0.0
    for j in range(0, TOPK):
        xj = tl.load(X_ptr + j)
        if j >= NUM_VALID:
            xj = -float("inf")
        s += tl.exp(xj - m)
    # Write softmax
    for j in range(0, TOPK):
        xj = tl.load(X_ptr + j)
        if j >= NUM_VALID:
            xj = -float("inf")
        prob = tl.exp(xj - m) / s
        tl.store(Out_ptr + j, prob)
    # Base-2 logsumexp
    lse_val = m + tl.log(s)
    tl.store(LSE_ptr, lse_val / 1.4426950408889634)  # 1 / ln(2)


# Triton kernel: compute out = Attn @ Kc_row
# Attn: 1D vector of length NUM_VALID (constexpr).
# Kc: 2D matrix [NUM_VALID, OUT] (OUT=512, constexpr).
# Out: 1D vector of length OUT, computed as sum_i Attn[i] * Kc[i, :].
@triton.jit
def reduction_row(Attn_ptr, Kc_ptr, Out_ptr,
                  NUM_VALID: tl.constexpr, OUT: tl.constexpr):
    acc = tl.zeros([OUT], dtype=tl.float32)
    # Loop over columns (constexpr)
    for o in range(0, OUT):
        sum_val = 0.0
        # Accumulate across NUM_VALID (constexpr)
        for i in range(0, NUM_VALID):
            sum_val += tl.load(Attn_ptr + i) * tl.load(Kc_ptr + i * OUT + o)
        acc[o] = sum_val
    tl.store(Out_ptr, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_qo_heads = 16
        self.head_dim_ckv = 512
        self.head_dim_kpe = 64
        self.topk = 2048  # constexpr candidate count

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
        # Validate shapes
        num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
        assert num_qo_heads == self.num_qo_heads
        assert head_dim_ckv == self.head_dim_ckv
        assert q_pe.shape[1] == self.num_qo_heads
        assert q_pe.shape[-1] == self.head_dim_kpe
        assert ckv_cache.shape[1] == 64 and ckv_cache.shape[-1] == self.head_dim_ckv
        assert kpe_cache.shape[1] == 64 and kpe_cache.shape[-1] == self.head_dim_kpe

        # Ensure CUDA tensors and compute dtype
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "All tensors must be on CUDA."
        device = q_nope.device

        # Flatten paged KV cache to token-level selections
        Kc_all = ckv_cache.reshape(-1, self.head_dim_ckv).to(torch.float32)  # [num_pages*64, 512]
        Kp_all = kpe_cache.reshape(-1, self.head_dim_kpe).to(torch.float32)  # [num_pages*64


def run(*args):
    return ModelNew()(*args)
