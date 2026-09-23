import math
import torch
import triton
import triton.language as tl


# Triton核：行级矩阵乘法 C = A_row @ B，其中A_row长度为M，B为N×M矩阵，输出C长度为N。
# 使用TOPK作为constexpr，并通过mask j < N来处理动态N <= TOPK。
@triton.jit
def matmul_row(A_ptr, B_ptr, C_ptr,
               M: tl.constexpr, N: tl.constexpr, TOPK: tl.constexpr):
    # 向量化j
    j = tl.arange(0, TOPK)
    mask_j = j < N  # 仅前N列为有效
    # 加载A行（长度M），连续
    a = tl.load(A_ptr)
    # 加载B块：形状 [TOPK, M]
    b_ptrs = B_ptr + j[:, None] * M + tl.arange(0, M)
    b = tl.load(b_ptrs, mask=mask_j[:, None], other=0.0)
    # 计算点积
    acc = tl.zeros([TOPK], dtype=tl.float32)
    for m in range(0, M):
        acc += b[:, m] * a[m]
    # 存储结果
    tl.store(C_ptr + j, acc, mask=mask_j)


# Triton核：行级softmax（数值稳定）和base-2 logsumexp。
# X: 长度TOPK的向量
# Valid: 长度TOPK的有效性掩码（int32 0/1）
# 输出：
#   Out: 概率向量（float32），有效位置有意义
#   LSE_ptr: 该行的logsumexp (base-2) 标量
@triton.jit
def softmax_logsumexp2_row(X_ptr, Valid_ptr, Out_ptr, LSE_ptr,
                            TOPK: tl.constexpr):
    # 计算最大值
    m = -float("inf")
    for j in range(0, TOPK):
        is_valid = tl.load(Valid_ptr + j)  # 0/1
        xj = tl.load(X_ptr + j)
        xj = tl.where(is_valid != 0, xj, -float("inf"))
        m = tl.maximum(m, xj)
    # 计算exp和sum
    s = 0.0
    for j in range(0, TOPK):
        is_valid = tl.load(Valid_ptr + j)
        xj = tl.load(X_ptr + j)
        xj = tl.where(is_valid != 0, xj, -float("inf"))
        s += tl.exp(xj - m)
    # 写出softmax
    for j in range(0, TOPK):
        is_valid = tl.load(Valid_ptr + j)
        xj = tl.load(X_ptr + j)
        xj = tl.where(is_valid != 0, xj, -float("inf"))
        prob = tl.exp(xj - m) / s
        tl.store(Out_ptr + j, prob)
    # base-2 logsumexp
    lse_val = m + tl.log(s)
    tl.store(LSE_ptr, lse_val / 1.4426950408889634)  # 1 / ln(2)


# Triton核：生成有效性掩码 Valid [TOPK] (int32): 1 for j < NUM_VALID, else 0
@triton.jit
def create_valid_mask(Valid_ptr, NUM_VALID: tl.constexpr, TOPK: tl.constexpr):
    j = tl.arange(0, TOPK)
    mask = j < NUM_VALID
    ones = tl.full([TOPK], 1, tl.int32)
    zeros = tl.zeros([TOPK], tl.int32)
    val = tl.where(mask, ones, zeros)
    tl.store(Valid_ptr + j, val)


# Triton核：计算 out = Attn @ Kc_row
# Attn: 1D向量长度NUM_VALID (constexpr)
# Kc: 2D矩阵 [NUM_VALID, OUT] (OUT=512, constexpr)
# Out: 1D向量长度OUT, 计算 sum_i Attn[i] * Kc[i, :]
@triton.jit
def reduction_row(Attn_ptr, Kc_ptr, Out_ptr,
                  NUM_VALID: tl.constexpr, OUT: tl.constexpr):
    acc = tl.zeros([OUT], dtype=tl.float32)
    # 循环输出列
    for o in range(0, OUT):
        sum_val = 0.0
        # 对每个j累加 attn[j] * Kc[j, o]
        for j in range(0, NUM_VALID):
            kj = tl.load(Kc_ptr + j * OUT + o)
            sum_val += tl.load(Attn_ptr + j) * kj
        acc[o] = sum_val
    tl.store(Out_ptr, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_qo_heads = 16
        self.head_dim_ckv = 512
        self.head_dim_kpe = 64
        self.topk = 2048  # 最大候选数，作为constexpr

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
        # 验证形状
        num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
        assert num_qo_heads == self.num_qo_heads
        assert head_dim_ckv == self.head_dim_ckv
        assert q_pe.shape[1] == self.num_qo_heads
        assert q_pe.shape[-1] == self.head_dim_kpe
        assert ckv_cache.shape[1] == 64 and ckv_cache.shape[-1] == self.head_dim_ckv
        assert kpe_cache.shape[1] == 64 and kpe_cache.shape[-1] == self.head_dim_kpe

        # 确保CUDA张量
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda


def run(*args):
    return ModelNew()(*args)
