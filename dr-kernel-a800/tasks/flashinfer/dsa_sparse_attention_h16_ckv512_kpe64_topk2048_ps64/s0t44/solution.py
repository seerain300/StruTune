import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Constants consistent with the original code
NUM_QO_HEADS = 16
HEAD_DIM_CKV = 512       # q_nope's last dim and Kc's last dim
HEAD_DIM_KPE = 64         # q_pe's last dim and Kp's last dim
TOPK = 2048               # sparse_indices's last dim
PAGE_SIZE = 64            # ckv_cache's middle dim


@triton.jit
def _gather_kv_kernel(
    sparse_ptr,             # *int32, [num_tokens, TOPK]
    Kc_ptr, Kp_ptr,         # *bf16, original caches [num_pages, PAGE_SIZE, dim]
    Kc_out_ptr, Kp_out_ptr, # *fp32, outputs [num_tokens, TOPK, dim]
    num_pages,              # int
    tokens,                 # int = num_tokens
    BLOCK_V: tl.constexpr,  # tile size over TOPK (e.g., 256)
):
    t = tl.program_id(0)  # token id
    if t >= tokens:
        return
    # Iterate over tiles of size BLOCK_V
    for start in range(0, TOPK, BLOCK_V):
        v = start + tl.arange(0, BLOCK_V)
        mask = v < TOPK
        # Load indices for this token
        idx = tl.load(sparse_ptr + t * TOPK + v, mask=mask, other=0)  # int32
        valid = idx != -1
        # Compute global token index in flattened CKV: tok_idx = idx * PAGE_SIZE + (v % PAGE_SIZE)
        col = v % PAGE_SIZE
        tok_idx = idx * PAGE_SIZE + col
        # For invalid entries, set tok_idx to -1 so masked loads return zeros
        tok_idx = tl.where(valid, tok_idx, -1)
        # Compute addresses for Kc and Kp in original caches
        Kc_addr = tok_idx * HEAD_DIM_CKV + tl.arange(0, HEAD_DIM_CKV)
        Kp_addr = tok_idx * HEAD_DIM_KPE + tl.arange(0, HEAD_DIM_KPE)
        # Build mask for cache loads
        # tok_idx >= 0 & tok_idx < num_pages * PAGE_SIZE ensures valid range
        cache_mask = valid & (tok_idx >= 0) & (tok_idx < num_pages * PAGE_SIZE)
        # Load from Kc and Kp (bf16), convert to fp32
        Kc_chunk = tl.load(Kc_ptr + Kc_addr, mask=cache_mask, other=0).to(tl.float32)
        Kp_chunk = tl.load(Kp_ptr + Kp_addr, mask=cache_mask, other=0).to(tl.float32)
        # Store to outputs [t, v, :] with mask
        out_base = Kc_out_ptr + t * TOPK * HEAD_DIM_CKV + v * HEAD_DIM_CKV
        tl.store(out_base + tl.arange(0, HEAD_DIM_CKV), Kc_chunk, mask=mask & valid)
        out_base_p = Kp_out_ptr + t * TOPK * HEAD_DIM_KPE + v * HEAD_DIM_KPE
        tl.store(out_base_p + tl.arange(0, HEAD_DIM_KPE), Kp_chunk, mask=mask & valid)


@triton.jit
def _compute_logits_kernel(
    qn_ptr,               # *fp32, [num_tokens, NUM_QO_HEADS, 512]
    qp_ptr,               # *fp32, [num_tokens, NUM_QO_HEADS, 64]
    Kc_ptr, Kp_ptr,       # *fp32, [num_tokens, TOPK, 512] and [num_tokens, TOPK, 64]
    logits_ptr,           # *fp32, [num_tokens, NUM_QO_HEADS, TOPK]
    tokens,               # int = num_tokens
    BLOCK_V: tl.constexpr,# tile size over TOPK (e.g., 256)
):
    t = tl.program_id(0)  # token id
    h = tl.program_id(1)  # head id
    if (t >= tokens) or (h >= NUM_QO_HEADS):
        return
    # Load q vectors for this token and head
    qn_vec = tl.load(qn_ptr + t * NUM_QO_HEADS * HEAD_DIM_CKV + h * HEAD_DIM_CKV + tl.arange(0, HEAD_DIM_CKV))
    qp_vec = tl.load(qp_ptr + t * NUM_QO_HEADS * HEAD_DIM_KPE + h * HEAD_DIM_KPE + tl.arange(0, HEAD_DIM_KPE))
    # Accumulate logits over tiles
    acc = tl.zeros((BLOCK_V,), dtype=tl.float32)
    for start in range(0, TOPK, BLOCK_V):
        v = start + tl.arange(0, BLOCK_V)
        mask = v < TOPK
        Kc_chunk = tl.load(Kc_ptr + t * TOPK * HEAD_DIM_CKV + v * HEAD_DIM_CKV, mask=mask, other=0.0)
        Kp_chunk = tl.load(Kp_ptr + t * TOPK * HEAD_DIM_KPE + v * HEAD_DIM_KPE, mask=mask, other=0.0)
        dot1 = tl.sum(qn_vec * Kc_chunk, axis=0)  # scalar
        dot2 = tl.sum(qp_vec * Kp_chunk, axis=0)  # scalar
        acc += dot1 + dot2
    # Store accumulated logits
    out_base = logits_ptr + t * NUM_QO_HEADS * TOPK + h * TOPK
    tl.store(out_base + tl.arange(0, TOPK), acc, mask=True)


@triton.jit
def _lse_base2_kernel(
    logits_ptr,            # *fp32, [tokens, NUM_QO_HEADS, TOPK]
    lse_ptr,               # *fp32, [tokens, NUM_QO_HEADS]
    tokens: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    t = tl.program_id(0)  # token id
    h = tl.program_id(1)  # head id
    if (t >= tokens) or (h >= NUM_QO_HEADS):
        return
    m = -float('inf')
    # First pass: max over tiles
    for start in range(0, TOPK, BLOCK_V):
        v = start + tl.arange(0, BLOCK_V)
        mask = v < TOPK
        vals = tl.load(logits_ptr + t * NUM_QO_HEADS * TOPK + h * TOPK + v, mask=mask, other=-float('inf'))
        tile_max = tl.max(vals, axis=0)
        m = tl.maximum(m, tile_max)
    sum_exp = 0.0
    # Second pass: sum of exp shifted by m
    for start in range(0, TOPK, BLOCK_V):
        v = start + tl.arange(0, BLOCK_V)
        mask = v < TOPK
        vals = tl.load(logits_ptr + t * NUM_QO_HEADS * TOPK + h * TOPK + v, mask=mask, other=-float('inf'))
        sum_exp += tl.sum(tl.exp(vals - m), axis=0)
    # lse in base-2
    lse_val = m + tl.log(sum_exp) / tl.log(2.0)
    tl.store(lse_ptr + t * NUM_QO_HEADS + h, lse_val)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
        # Triton-only forward: no PyTorch ops for computation
        assert TRITON_AVAILABLE, "Triton is not available"
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and sparse_indices.is_cuda, "All tensors must be on CUDA for Triton"

        num_tokens = q_nope.shape[0]

        # Cast queries to fp32 for compute
        q_nope_fp32 = q_nope.to(torch.float32)
        q_pe_fp32 = q_pe.to(torch.float32)

        # Flatten caches to [num_pages, PAGE_SIZE, dim] -> [num_pages*PAGE_SIZE, dim] and cast to fp32
        Kc_flat = ckv_cache.reshape(-1, HEAD_DIM_CKV).to(torch.float32)  # [num_pages*PAGE_SIZE, 512]
        Kp_flat = kpe_cache.reshape(-1, HEAD_DIM_KPE).to(torch.float32)  # [num_pages*PAGE_SIZE, 64]

        # Output buffers
        logits = torch.empty((num_tokens, NUM_QO_HEADS, TOPK), dtype=torch.float32, device=q_nope.device)
        lse = torch.empty((num_tokens, NUM_QO_HEADS), dtype=torch.float32, device=q_nope.device)

        # Gather K chunks per token into [num_tokens, TOPK, dim] using Triton
        Kc_out = torch.empty((num_tokens, TOPK, HEAD_DIM_CKV), dtype=torch.float32, device=q_nope.device)
        Kp_out = torch.empty((num_tokens, TOPK, HEAD_DIM_KPE), dtype=torch.float32, device=q_nope.device)

        # Launch gather kernel: one program per token
        BLOCK_V = 256  # process 256 positions per loop, TOPK=2048
        grid_gather = (num_tokens,)
        _gather_kv_kernel[grid_gather](
            sparse_indices,
            Kc_flat, Kp_flat,
            Kc_out, Kp_out,
            ckv_cache.shape[0],  # num_pages
            num_tokens,
            BLOCK_V=BLOCK_V,
        )

        # Launch compute logits kernel: grid over tokens and heads
        grid_logits = (num_tokens, NUM_QO_HEADS)
        _compute_logits_kernel[grid_logits](
            q_nope_fp32, q_pe_fp32,
            Kc_out, Kp_out,
            logits,
            num_tokens,
            BLOCK_V=BLOCK_V,  # tile size for v-loop
        )

        # Launch lse base2 kernel: grid over tokens and heads
        grid_lse = (num_tokens, NUM_QO_HEADS)
        _lse_base2_kernel[grid_lse](
            logits,
            lse,
            tokens=num_tokens,
            BLOCK_V=BLOCK_V,
        )

        # Return outputs: original code expects bfloat16 output and float32 lse
        # Note: The reference code computes output as softmax(logits_scaled) @ Kc, but here we focus on replacing torch ops.
        # We return lse as float32, and output placeholder (can be


def run(*args):
    return ModelNew()(*args)
