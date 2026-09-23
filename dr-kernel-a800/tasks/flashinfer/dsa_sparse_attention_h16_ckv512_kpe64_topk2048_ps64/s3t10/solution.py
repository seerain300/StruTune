import math
import torch
import triton
import triton.language as tl


# Triton kernel: row-wise softmax (numerically stable) and base-2 logsumexp.
# X is 1D vector of length TOPK (e.g., 2048).
# Valid is 1D mask (int32) of length TOPK: 1 for valid, 0 for padded.
# Outputs:
#   Out: softmax probabilities (float32) of length TOPK (only valid positions meaningful).
#   LSE_ptr: scalar (float32) storing logsumexp base-2 for this row.
@triton.jit
def softmax_logsumexp2_row(X_ptr, Valid_ptr, Out_ptr, LSE_ptr,
                            TOPK: tl.constexpr):
    # Compute max over valid entries
    m = -float("inf")
    for j in range(0, TOPK):
        is_valid = tl.load(Valid_ptr + j)  # 0/1
        xj = tl.load(X_ptr + j)
        xj = tl.where(is_valid != 0, xj, -float("inf"))
        m = tl.maximum(m, xj)
    # Compute sum of exp over valid entries
    s = 0.0
    for j in range(0, TOPK):
        is_valid = tl.load(Valid_ptr + j)
        xj = tl.load(X_ptr + j)
        xj = tl.where(is_valid != 0, xj, -float("inf"))
        s += tl.exp(xj - m)
    # Write softmax into Out (valid positions only; invalid entries are -inf)
    for j in range(0, TOPK):
        is_valid = tl.load(Valid_ptr + j)
        xj = tl.load(X_ptr + j)
        xj = tl.where(is_valid != 0, xj, -float("inf"))
        prob = tl.exp(xj - m) / s
        tl.store(Out_ptr + j, prob)
    # Base-2 logsumexp: lse = m + log(s), then divide by ln(2)
    lse_val = m + tl.log(s)
    tl.store(LSE_ptr, lse_val / 1.4426950408889634)  # 1 / ln(2)


# Triton kernel: compute out = Attn @ Kc_row
# Attn is a 1D vector of length NUM_VALID (constexpr, e.g., 2048).
# Kc is a 2D matrix of shape (NUM_VALID, OUT), OUT is tl.constexpr (e.g., 512).
# Out is a 1D vector of length OUT, computed as sum_i Attn[i] * Kc[i, :].
@triton.jit
def reduction_row(Attn_ptr, Kc_ptr, Out_ptr,
                  NUM_VALID: tl.constexpr, OUT: tl.constexpr):
    acc = tl.zeros([OUT], dtype=tl.float32)
    # Loop over columns of Kc (OUT is constexpr)
    for o in range(0, OUT):
        sum_val = 0.0
        # j_vec across NUM_VALID
        j_vec = tl.arange(0, NUM_VALID)
        # Load Kc[j, o] for all j
        kc_col = tl.load(Kc_ptr + j_vec * OUT + o)
        # Multiply and reduce: sum over j
        for jj in range(0, NUM_VALID):
            sum_val += tl.load(Attn_ptr + jj) * kc_col[jj]
        acc[o] = sum_val
    tl.store(Out_ptr, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_qo_heads = 16
        self.head_dim_ckv = 512
        self.head_dim_kpe = 64
        self.topk = 2048  # maximum number of candidates per token (constexpr)

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
        # Validate shapes
        num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
        assert num_qo_heads == self.num_qo_heads
        assert head_dim_ckv == self.head_dim_ckv
        assert q_pe.shape[1] == self.num_qo_heads
        assert q_pe.shape[-1] == self.head_dim_kpe
        assert ckv_cache.shape[1] == 64 and ckv_cache.shape[-1] == self.head_dim_ckv
        assert kpe_cache.shape[1] == 64 and kpe_cache.shape[-1] == self.head_dim_kpe

        # Ensure


def run(*args):
    return ModelNew()(*args)
