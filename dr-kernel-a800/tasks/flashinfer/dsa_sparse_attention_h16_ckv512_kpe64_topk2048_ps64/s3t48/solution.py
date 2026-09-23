import math
import torch
import triton
import triton.language as tl


# Triton kernel: compute softmax of X (length TOPK) and base-2 logsumexp,
# writing out probabilities to Out_ptr and lse to LSE_ptr. Assumes TOPK is a constexpr.
@triton.jit
def softmax_logsumexp2_row(X_ptr, Valid_ptr, Out_ptr, LSE_ptr,
                            TOPK: tl.constexpr):
    # Compute m = max(X_valid)
    m = -float("inf")
    for j in range(0, TOPK):
        xj = tl.load(X_ptr + j)
        is_valid = tl.load(Valid_ptr + j)  # 0/1
        xj = tl.where(is_valid != 0, xj, -float("inf"))
        m = tl.maximum(m, xj)

    # Compute sum of exp(X - m)
    s = 0.0
    for j in range(0, TOPK):
        xj = tl.load(X_ptr + j)
        is_valid = tl.load(Valid_ptr + j)
        xj = tl.where(is_valid != 0, xj, -float("inf"))
        s += tl.exp(xj - m)

    # Write softmax probabilities for valid entries
    for j in range(0, TOPK):
        xj = tl.load(X_ptr + j)
        is_valid = tl.load(Valid_ptr + j)
        xj = tl.where(is_valid != 0, xj, -float("inf"))
        prob = tl.exp(xj - m) / s
        tl.store(Out_ptr + j, prob)

    # Base-2 logsumexp: lse = m + log(s), divide by ln(2)
    lse_val = m + tl.log(s)
    tl.store(LSE_ptr, lse_val / 1.4426950408889634)  # 1 / ln(2)


# Triton kernel: compute out = Attn @ Kc, where:
# Attn is a 1D vector of length NUM_VALID (compile-time; pass NUM_VALID=TOPK here).
# Kc is a 1D flattened vector of length NUM_VALID * OUT (e.g., 2048 * 512).
# OUT is tl.constexpr (e.g., 512). Produces a 1D Out vector of length OUT.
@triton.jit
def reduction_row(Attn_ptr, Kc_ptr, Out_ptr,
                  NUM_VALID: tl.constexpr, OUT: tl.constexpr):
    acc = tl.zeros([OUT], dtype=tl.float32)
    for o in range(0, OUT):
        sum_val = 0.0
        for j in range(0, NUM_VALID):
            sum_val += tl.load(Attn_ptr + j) * tl.load(Kc_ptr + j * OUT + o)
        acc[o] = sum_val
    tl.store(Out_ptr, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Dimensions as in the original code
        self.num_qo_heads = 16
        self.head_dim_ckv = 512
        self.head_dim_kpe = 64
        self.topk = 2048  # candidates per token

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
        # Ensure tensors are on CUDA
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "All tensors must be on CUDA for Triton kernels"
        device = q_nope.device

        # Shapes
        num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
        assert num_qo_heads == self.num_qo_heads
        assert head_dim_ckv == self.head


def run(*args):
    return ModelNew()(*args)
