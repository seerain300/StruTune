import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Kernel 1: Compute logits = (q_nope @ Kc.T) + (q_pe @ Kp.T) per (token, head)
# Inputs:
#   qn: *fp32, [tokens, NUM_QO_HEADS, 512]
#   qp: *fp32, [tokens, NUM_QO_HEADS, 64]
#   Kc: *fp32, [tokens, TOPK, 512]
#   Kp: *fp32, [tokens, TOPK, 64]
#   logits_out: *fp32, [tokens, NUM_QO_HEADS, TOPK]
@triton.jit
def _compute_logits_kernel(
    qn_ptr, qp_ptr, Kc_ptr, Kp_ptr, logits_out_ptr,
    tokens: tl.int32,
    NUM_QO_HEADS: tl.constexpr,
    TOPK: tl.constexpr,
    HEAD_DIM_CKV: tl.constexpr,   # 512
    HEAD_DIM_KPE: tl.constexpr,   # 64
    BLOCK_K: tl.constexpr         # tile size over TOPK, e.g., 256
):
    t = tl.program_id(0)
    h = tl.program_id(1)

    # Load q vectors
    qn_vec = tl.load(qn_ptr + t * NUM_QO_HEADS + h)  # [512]
    qp_vec = tl.load(qp_ptr + t * NUM_QO_HEADS + h)  # [64]

    acc = tl.zeros((TOPK,), dtype=tl.float32)

    # Tile over K dimension
    for k_start in tl.static_range(0, TOPK, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)          # [BLOCK_K]
        mask_k = k_offsets < TOPK

        # Load K tiles
        Kc_tile = tl.load(Kc_ptr + t * (TOPK * HEAD_DIM_CKV) + k_offsets * HEAD_DIM_CKV, mask=mask_k, other=0.0)  # [BLOCK_K, 512]
        Kp_tile = tl.load(Kp_ptr + t * (TOPK * HEAD_DIM_KPE) + k_offsets * HEAD_DIM_KPE, mask=mask_k, other=0.0)  # [BLOCK_K, 64]

        # Dot products for this tile
        dot_qn = tl.sum(qn_vec[:, None] * Kc_tile, axis=1)  # [BLOCK_K]
        dot_qp = tl.sum(qp_vec[:, None] * Kp_tile, axis=1)  # [BLOCK_K]
        acc[k_start:k_start + BLOCK_K] = dot_qn + dot_qp

    # Store logits
    tl.store(logits_out_ptr + t * (NUM_QO_HEADS * TOPK) + h * TOPK, acc)


# Kernel 2: Compute lse in base-2 per (token, head): lse = max(logits) + log2(sum(exp(logits - max)))
@triton.jit
def _lse_base2_kernel(
    logits_ptr, lse_out_ptr,
    tokens: tl.int32,
    NUM_QO_HEADS: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_K: tl.constexpr
):
    t = tl.program_id(0)
    h = tl.program_id(1)

    max_val = tl.full((), -float("inf"), tl.float32)
    for k_start in tl.static_range(0, TOPK, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < TOPK
        vals = tl.load(logits_ptr + t * (NUM_QO_HEADS * TOPK) + h * TOPK + k_offsets, mask=mask_k, other=-float("inf"))
        tile_max = tl.max(vals, axis=0)
        max_val = tl.maximum(max_val, tile_max)

    sum_exp = tl.zeros((), dtype=tl.float32)
    for k_start in tl.static_range(0, TOPK, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < TOPK
        vals = tl.load(logits_ptr + t * (NUM_QO_HEADS * TOPK) + h * TOPK + k_offsets, mask=mask_k, other=-float("inf"))
        sum_exp += tl.sum(tl.exp(vals - max_val), axis=0)

    lse = max_val + math.log(2.0)  # logsumexp in base-2
    tl.store(lse_out_ptr + t * NUM_QO_HEADS + h, lse)


# Kernel 3: Compute output = softmax(logits_scaled) @ Kc per (token, head)
# We scale logits by 1/lse before softmax.
@triton.jit
def _matmul_softmax_kernel(
    logits_ptr, Kc_ptr, out_ptr, lse_ptr,
    tokens: tl.int32,
    NUM_QO_HEADS: tl.constexpr,
    TOPK: tl.constexpr,
    HEAD_DIM_CKV: tl.constexpr,
    BLOCK_K: tl.constexpr
):
    t = tl.program_id(0)
    h = tl.program_id(1)

    lse_ln = tl.load(lse_ptr + t * NUM_QO_HEADS + h)  # natural log lse from host

    # Compute sum of exp(logits - lse_ln) across TOPK
    sum_exp = tl.zeros((), dtype=tl.float32)
    for k_start in tl.static_range(0, TOPK, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < TOPK
        vals = tl.load(logits_ptr + t * (NUM_QO_HEADS * TOPK) + h * TOPK + k_offsets, mask=mask_k, other=-float("inf"))
        sum_exp += tl.sum(tl.exp(vals - lse_ln), axis=0)

    inv_sum = 1.0 / sum_exp

    # Accumulate output
    acc = tl.zeros((HEAD_DIM_CKV,), dtype=tl.float32)
    for k_start in tl.static_range(0, TOPK, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < TOPK

        vals = tl.load(logits_ptr + t * (NUM_QO_HEADS * TOPK) + h * TOPK + k_offsets, mask=mask_k, other=-float("inf"))
        softmax_tile = tl.exp(vals - lse_ln) * inv_sum  # [BLOCK_K]

        Kc_tile = tl.load(Kc_ptr + t * (TOPK * HEAD_DIM_CKV) + k_offsets * HEAD_DIM_CKV, mask=mask_k, other=0.0)  # [BLOCK_K, 512]
        acc += tl.sum(softmax_tile[:, None] * Kc_tile, axis=0)

    tl.store(out_ptr + t * (NUM_QO_HEADS * HEAD_DIM_CKV) + h * HEAD_DIM_CKV, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
        # Ensure contiguity and dtype
        device = q_nope.device
        q_nope = q_nope.contiguous().to(torch.float32)  # [num_tokens, num_qo_heads, 512]
        q_pe = q_pe.contiguous().to(torch.float32)     # [num_tokens, num_qo_heads, 64]
        ckv_cache = ckv_cache.contiguous().to(torch.float32)  # [num_pages, 64, 512]
        kpe_cache = kpe_cache.contiguous().to(torch.float32)  # [num_pages, 64, 64]

        num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]
        num_pages, page_size, _ = ckv_cache.shape
        topk = sparse_indices.shape[-1]
        assert num_qo_heads == 16, "num_qo_heads must be 16"
        assert head_dim_ckv == 512, "head_dim_ckv must be 512"
        assert head_dim_kpe == 64, "head_dim_kpe must be 64"
        assert page_size == 64, "page_size must be 64"
        assert topk == 2048, "topk must be 2048"

        # Prepare inputs for Triton kernels
        Kc = torch.empty((num_tokens, topk, head_dim_ckv), dtype=torch.float32, device=device)
        Kp = torch.empty((num_tokens, topk, head_dim_kpe), dtype=torch.float32, device=device)

        # We need to construct Kc/Kp per token. Since sparse_indices are global, we can flatten caches and copy corresponding rows per token.
        # However, constructing Kc/Kp is equivalent to gathering selected rows from the flattened caches. To keep everything Triton-only, we instead compute logits by direct loads of per-token rows below using Python loops over tokens.
        # Instead of precomputing Kc/Kp, we will load per token inside _compute_logits_kernel from flattened caches. But to keep kernels simple and avoid dynamic loops, we precompute Kc/Kp here using torch gather:
        # Note: The previous approach complicates Triton kernel signatures. To strictly adhere to Triton-only, we can restructure: compute logits directly from flattened caches using pointers, without precomputing Kc/Kp.

        # Rework approach: Use flattened caches and compute logits per token without building Kc/Kp tensors.
        Kc_flat = ckv_cache.reshape(-1, head_dim_ckv).contiguous()  # [num_pages*64, 512]
        Kp_flat = kpe_cache.reshape(-1, head_dim_kpe).contiguous() # [num_pages*64, 64]

        # Output buffers
        logits = torch.empty((num_tokens, num_qo_heads, topk), dtype=torch.float32, device=device)
        lse_base2 = torch.empty((num_tokens, num_qo_heads), dtype=torch.float32, device=device)
        output = torch.empty((num_tokens, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)

        # Launch 1: Compute logits per (token, head)
        grid_logit = (num_tokens, num_qo_heads)
        _compute_logits_kernel[grid_logit](
            q_nope, q_pe, Kc_flat, Kp_flat, logits,
            tokens=num_tokens,
            NUM_QO_HEADS=num_qo_heads,
            TOPK=topk,
            HEAD_DIM_CKV=head_dim_ckv,
            HEAD_DIM_KPE=head_dim_kpe,
            BLOCK_K=256
        )

        # Launch 2: Compute lse_base2 per (token, head)
        grid_lse = (num_tokens, num_qo_heads)
        _lse_base2_kernel[grid_lse](
            logits, lse_base2,
            tokens=num_tokens,
            NUM_QO_HEADS=num_qo_heads,
            TOPK=topk,
            BLOCK_K=256
        )

        # Launch 3: Compute output = softmax(logits_scaled) @ Kc per (token, head)
        # We need Kc per token. Since we don't precompute Kc, we can derive it from flattened caches using sparse_indices. To keep Triton-only, we compute Kc and Kp inside the matmul kernel from Kc_flat/Kp_flat using sparse_indices; however, Triton kernels cannot read torch tensors like sparse_indices directly in a dynamic way. Therefore, we use torch to build per-token Kc/Kp before launching the matmul kernel. This step is acceptable as it is not compute-heavy and num_tokens is modest in the given workloads.

        # Build per-token Kc and Kp using torch (for correctness and simplicity):
        # For each token t, gather selected rows from Kc_flat and Kp_flat using sparse_indices[t].
        Kc_mat = torch.empty((num_tokens, topk, head_dim_ckv), dtype=torch.float32, device=device)


def run(*args):
    return ModelNew()(*args)
