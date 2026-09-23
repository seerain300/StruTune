import math
import torch

import triton
import triton.language as tl


@triton.jit
def fused_logits_kernel(
    q_nope_ptr,       # *f32 [H, Dq]
    q_pe_ptr,         # *f32 [H, Dp]
    Kc_ptr,           # *f32 [T, Dq]
    Kp_ptr,           # *f32 [T, Dp]
    logits_ptr,       # *f32 [H, T]
    H: tl.constexpr,
    Dq: tl.constexpr,
    Dp: tl.constexpr,
    T: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    h = tl.program_id(0)  # one program per head
    # Preload q vectors once
    qn = tl.load(q_nope_ptr + h * Dq + tl.arange(0, Dq))  # [Dq]
    qp = tl.load(q_pe_ptr + h * Dp + tl.arange(0, Dp))    # [Dp]
    offs_t = tl.arange(0, BLOCK_T)
    for t_start in range(0, T, BLOCK_T):
        t_idx = t_start + offs_t
        mask_t = t_idx < T
        sum_vec = tl.zeros([BLOCK_T], dtype=tl.float32)
        # dot over Dq for Kc
        for k in range(0, Dq):
            k_vec = tl.load(Kc_ptr + k * T + t_idx, mask=mask_t, other=0.0)  # [BLOCK_T]
            sum_vec += qn[k] * k_vec
        # dot over Dp for Kp
        for k in range(0, Dp):
            p_vec = tl.load(Kp_ptr + k * T + t_idx, mask=mask_t, other=0.0)  # [BLOCK_T]
            sum_vec += qp[k] * p_vec
        tl.store(logits_ptr + h * T + t_idx, sum_vec, mask=mask_t)


@triton.jit
def attn_matmul_kernel(
    attn_ptr,   # *f32 [H, T]
    Kc_ptr,     # *f32 [T, D]
    out_ptr,    # *f32 [H, D]
    H: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    h = tl.program_id(0)  # one program per head
    offs_t = tl.arange(0, BLOCK_T)
    offs_d = tl.arange(0, BLOCK_D)
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)
    for d_start in range(0, D, BLOCK_D):
        d_idx = d_start + offs_d
        mask_d = d_idx < D
        for t_start in range(0, T, BLOCK_T):
            t_idx = t_start + offs_t
            mask_t = t_idx < T
            # Load attn chunk [BLOCK_T]
            attn_chunk = tl.load(attn_ptr + h * T + t_idx, mask=mask_t, other=0.0)  # [BLOCK_T]
            # Load Kc chunk [BLOCK_T, BLOCK_D]
            Kc_chunk = tl.load(
                Kc_ptr + t_idx[:, None] * D + d_idx[None, :],
                mask=mask_t[:, None] & mask_d[None, :],
                other=0.0,
            )
            # Accumulate over T tiles
            acc += tl.sum(attn_chunk[:, None] * Kc_chunk, axis=0)
        tl.store(out_ptr + h * D + d_idx, acc, mask=mask_d)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Move to CUDA if needed
        original_device = q_nope.device
        use_cuda = original_device.type == 'cuda'
        if not use_cuda:
            device = torch.device('cuda')
            q_nope = q_nope.to(device)
            q_pe = q_pe.to(device)
            ckv_cache = ckv_cache.to(device)
            kpe_cache = kpe_cache.to(device)
            kv_indptr = kv_indptr.to(device)
            kv_indices = kv_indices.to(device)
        else:
            device = original_device

        # Ensure inputs are contiguous and float32 for compute
        q_nope_f32 = q_nope.contiguous().to(torch.float32)
        q_pe_f32 = q_pe.contiguous().to(torch.float32)
        Kc_all = ckv_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, 64]

        batch_size = q_nope_f32.shape[0]
        num_qo_heads = q_nope_f32.shape[1]
        head_dim_ckv = q_nope_f32.shape[2]
        head_dim_kpe = q_pe_f32.shape[2]

        # Output tensors (float32 for compute, will cast to bfloat16 at end)
        output = torch.empty((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
        lse = torch.full((batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

        for b in range(batch_size):
            # Compute range from kv_indptr
            page_beg = int


def run(*args):
    return ModelNew()(*args)
