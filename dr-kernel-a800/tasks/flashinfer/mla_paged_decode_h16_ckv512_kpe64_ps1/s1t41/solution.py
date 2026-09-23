import math
import torch
import triton
import triton.language as tl


@triton.jit
def gather_rows_c_kernel(cache_ptr, tok_idx_ptr, out_ptr,
                          L_tokens: tl.constexpr, Dc: tl.constexpr):
    # Each program handles one token row for CKV
    pid = tl.program_id(0)
    if pid >= L_tokens:
        return
    idx = tl.load(tok_idx_ptr + pid).to(tl.int32)
    base_in = idx * Dc
    for k in range(0, Dc):
        val = tl.load(cache_ptr + base_in + k)
        tl.store(out_ptr + pid * Dc + k, val)


@triton.jit
def matvec_kernel(attn_ptr, Kc_ptr, out_ptr,
                  H: tl.constexpr, Dc: tl.constexpr, L: tl.constexpr,
                  BLOCK_D: tl.constexpr):
    # One program per head (H programs), compute out_vec[i] = attn[i, :] @ Kc
    i = tl.program_id(0)
    if i >= H:
        return
    acc = tl.zeros((Dc,), dtype=tl.float32)
    for t0 in range(0, L, BLOCK_D):
        offs = t0 + tl.arange(0, BLOCK_D)
        mask = offs < L
        attn_slice = tl.load(attn_ptr + i * L + offs, mask=mask, other=0.0)  # [BLOCK_D]
        k_ptrs = Kc_ptr + offs[:, None] * Dc + tl.arange(0, Dc)[None, :]     # [BLOCK_D, Dc]
        k_vals = tl.load(k_ptrs, mask=mask[:, None], other=0.0)              # [BLOCK_D, Dc]
        for kk in range(0, BLOCK_D):
            if not mask[kk]:
                continue
            a = attn_slice[kk]
            k_row = k_vals[kk, :]  # [Dc]
            acc += a * k_row
    out_base = i * Dc
    for d in range(0, Dc):
        tl.store(out_ptr + out_base + d, acc[d])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Shapes
        batch_size = q_nope.shape[0]
        num_qo_heads = q_nope.shape[1]
        head_dim_ckv = q_nope.shape[2]
        head_dim_kpe = q_pe.shape[2]
        num_pages = ckv_cache.shape[0]
        device = q_nope.device

        # Constants
        assert num_qo_heads == 16
        assert head_dim_ckv == 512
        assert head_dim_kpe == 64

        # Prepare caches: squeeze size-1 dim
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, Dc]
        K


def run(*args):
    return ModelNew()(*args)
