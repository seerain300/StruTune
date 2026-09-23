import torch
import triton
import triton.language as tl

# Kernel 1: Compute C[j] = sum_i A[i] * B[j, i], for j in 0..TOPK-1.
# A_ptr: 1D tensor of length M
# B_ptr: 2D tensor layout as contiguous [TOPK, M]
# C_ptr: output vector of length TOPK
@triton.jit
def matmul_row(A_ptr, B_ptr, C_ptr,
               M: tl.constexpr,       # e.g., 512 or 64
               TOPK: tl.constexpr):   # e.g., 2048
    j = tl.program_id(0)  # which output index
    acc = 0.0
    for i in range(0, M):
        a = tl.load(A_ptr + i)
        b = tl.load(B_ptr + j * M + i)
        acc += a * b
    tl.store(C_ptr + j, acc, mask=j < TOPK)

# Kernel 2: Softmax over TOPK using base-2 logsumexp; requires Valid mask (int32 0/1).
@triton.jit
def softmax_logsumexp2_row(X_ptr, Valid_ptr, Out_ptr, LSE_ptr,
                            TOPK: tl.constexpr):
    # Stable softmax (base-2). Use first pass to compute max and sum over valid entries.
    max_x = -float('inf')
    for i in range(0, TOPK):
        xi = tl.load(X_ptr + i)
        valid = tl.load(Valid_ptr + i)
        include = valid != 0
        xi = tl.where(include, xi, -float('inf'))
        max_x = tl.maximum(max_x, xi)

    sum_exp = 0.0
    for i in range(0, TOPK):
        xi = tl.load(X_ptr + i)
        valid = tl.load(Valid_ptr + i)
        include = valid != 0
        xi = tl.where(include, xi, -float('inf'))
        e = tl.exp(xi - max_x)
        sum_exp += tl.where(include, e, 0.0)

    inv_ln2 = 1.4426950408889634  # 1 / ln(2)
    lse = max_x + tl.log(sum_exp) * inv_ln2
    tl.store(LSE_ptr, lse)

    # Second pass: write attn
    for i in range(0, TOPK):
        xi = tl.load(X_ptr + i)
        valid = tl.load(Valid_ptr + i)
        include = valid != 0
        xi = tl.where(include, xi, -float('inf'))
        e = tl.exp(xi - max_x)
        attn_i = e / sum_exp
        attn_i = tl.where(include, attn_i, 0.0)
        tl.store(Out_ptr + i, attn_i)

# Kernel 3: Compute Out[OUT] = sum_i Attn[i] * Kc[i, :]
# Attn_ptr: 1D vector (length NUM_VALID padded to TOPK with zeros in host before launch)
# Kc_ptr:   2D tensor [NUM_VALID, OUT], contiguous
@triton.jit
def reduction_row(Attn_ptr, Kc_ptr, Out_ptr,
                  NUM_VALID: tl.constexpr,  # actual number of valid rows
                  OUT: tl.constexpr):       # e.g., 512
    acc = tl.zeros((OUT,), dtype=tl.float32)
    for i in range(0, NUM_VALID):
        attn_i = tl.load(Attn_ptr + i)
        for j in range(0, OUT):
            kc = tl.load(Kc_ptr + i * OUT + j)
            acc[j] += attn_i * kc
    for j in range(0, OUT):
        tl.store(Out_ptr + j, acc[j])


class ModelNew(torch.nn.Module):
    def __init__(self, num_tokens, num_qo_heads, head_dim_ckv, head_dim_kpe, num_pages, page_size, topk, sm_scale):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self


def run(*args):
    return ModelNew()(*args)
