import math
import torch
import triton
import triton.language as tl


# Triton kernels

# 1) Gather rows from ckv_cache_all (shape [P, Dc]) into Kc_flat of shape [(L_tokens * Dc)]
@triton.jit
def gather_rows_c_kernel(cache_ptr, tok_idx_ptr, out_ptr,
                          num_tokens: tl.constexpr, Dc: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= num_tokens:
        return
    idx = tl.load(tok_idx_ptr + pid).to(tl.int32)
    base_in = idx * Dc
    for k in range(0, Dc):
        val = tl.load(cache_ptr + base_in + k)
        tl.store(out_ptr + pid * Dc + k, val)


# 2) Gather rows from kpe_cache_all (shape [P, Dp]) into Kp_flat of shape [(L_tokens * Dp)]
@triton.jit
def gather_rows_p_kernel(cache_ptr, tok_idx_ptr, out_ptr,
                          num_tokens: tl.constexpr, Dp: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= num_tokens:
        return
    idx = tl.load(tok_idx_ptr + pid).to(tl.int32)
    base_in = idx * Dp
    for k in range(0, Dp):
        val = tl.load(cache_ptr + base_in + k)
        tl.store(out_ptr + pid * Dp + k, val)


# 3) Compute per-head logsumexp in Triton, writing lse per head to lse_ptr[i] = lse[i]
#    Two-pass approach: first find max over L tokens, then sum of exp, then lse = m + log(sum_exp) / ln(2).
@triton.jit
def lse_base2_rows_kernel(logits_ptr, lse_ptr, H: tl.constexpr, L: tl.constexpr):
    i = tl.program_id(0)
    if i >= H:
        return
    m = -float("inf")
    for t in range(0, L):
        val = tl.load(logits_ptr + i * L + t)
        m = tl.maximum(m, val)
    sum_exp = 0.0
    for t in range(0, L):
        val = tl.load(logits_ptr + i * L + t)
        sum_exp += tl.exp(val - m)
    lse_val = m + tl.log(sum_exp) * 1.4426950408889634  # 1/log(2)
    tl.store(lse_ptr + i, lse_val)


# 4) Softmax per row over L tokens: write attention vector to out_ptr flattened as [H * L]
#    One program per head; loops over L tokens to compute max, sum_exp, then normalizes and stores.
@triton.jit
def softmax_rows_kernel(logits_ptr, out_ptr, H: tl.constexpr, L: tl.constexpr):
    i = tl.program_id(0)
    if i >= H:
        return
    m = -float("inf")
    for t in range(0, L):
        val = tl.load(logits_ptr + i * L + t)
        m = tl.maximum(m, val)
    sum_exp = 0.0
    for t in range(0, L):
        val = tl.load(logits_ptr + i * L + t)
        sum_exp += tl.exp(val - m)
    for t in range(0, L):
        val = tl.load(logits_ptr + i * L + t)
        p = tl.exp(val - m) / sum_exp
        tl.store(out_ptr + i * L + t, p)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        """
        q_nope: [B, H, Dc], bfloat16
        q_pe: [B, H, Dp], bfloat16
        ckv_cache: [P, 1, Dc], bfloat16
        kpe_cache: [P, 1, Dp], bfloat16
        kv_indptr: [B+1], int32
        kv_indices: [num_kv_indices], int32
        sm_scale: float32 scalar
        Returns: output [B, H, Dc], bfloat16; lse [B, H], float32
        """
        assert q_nope.dim() == 3 and q_pe.dim() == 3, "q_nope and q_pe must be 3D tensors [B, H, D]"
        B, H, Dc = q_nope.shape
        _, _, Dp = q_pe.shape
        P = ckv_cache.shape[0]
        _, _, Dc_cache = ckv_cache.shape
        _, _, Dp_cache = kpe_cache.shape
        assert Dc_cache == Dc and Dp_cache == Dp, "Cache dims must match head dims"
        assert kv_indptr.shape[0] == B + 1, "kv_indptr must have length B+1"
        device = q_nope.device

        # Squeeze the size-1 dimension from caches
        Kc_all = ckv_cache.squeeze(1).contiguous()  # [P, Dc]
        Kp_all = kpe_cache.squeeze(1).contiguous()  # [P, Dp]

        output = torch.empty((B, H, Dc), dtype=torch.bfloat16, device=device)
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        for b in range(B):
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            if L_tokens <= 0:
                # No KV entries for this batch element
                for i in range(H):
                    output


def run(*args):
    return ModelNew()(*args)
