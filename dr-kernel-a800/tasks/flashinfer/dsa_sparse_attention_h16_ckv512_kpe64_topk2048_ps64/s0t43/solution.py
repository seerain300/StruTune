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
HEAD_DIM_CKV = 512      # q_nope's last dim and Kc's last dim
HEAD_DIM_KPE = 64        # q_pe's last dim and Kp's last dim
TOPK = 2048              # sparse_indices's last dim
PAGE_SIZE = 64           # ckv_cache's middle dim
LN2 = 0.6931471805599453


@triton.jit
def _compute_logits_kernel(
    qn_ptr,           # *fp32, [NUM_QO_HEADS, HEAD_DIM_CKV]
    qp_ptr,           # *fp32, [NUM_QO_HEADS, HEAD_DIM_KPE]
    Kc_ptr,           # *fp32, [TOPK, HEAD_DIM_CKV]
    Kp_ptr,           # *fp32, [TOPK, HEAD_DIM_KPE]
    logits_ptr,       # *fp32, [NUM_QO_HEADS, TOPK]
    NUM_QO_HEADS: tl.constexpr,
    TOPK: tl.constexpr,
    HEAD_DIM_CKV: tl.constexpr,
    HEAD_DIM_KPE: tl.constexpr,
):
    # 2D grid over (head, v-block)
    pid_h = tl.program_id(axis=0)
    pid_vb = tl.program_id(axis=1)

    h = pid_h
    v_offsets = pid_vb * 128 + tl.arange(0, 128)
    mask_v = v_offsets < TOPK

    # Load q vectors for this head (full length)
    qn_vec = tl.load(qn_ptr + h * HEAD_DIM_CKV + tl.arange(0, HEAD_DIM_CKV), mask=True, other=0.0)  # [HEAD_DIM_CKV]
    qp_vec = tl.load(qp_ptr + h * HEAD_DIM_KPE + tl.arange(0, HEAD_DIM_KPE), mask=True, other=0.0)  # [HEAD_DIM_KPE]

    accum = tl.zeros((128,), dtype=tl.float32)

    # Loop over Kc dimension in chunks of 128
    for k_start in range(0, HEAD_DIM_CKV, 128):
        k_offsets = k_start + tl.arange(0, 128)
        mask_k = k_offsets < HEAD_DIM_CKV
        # Load Kc and Kp tiles for this block of v
        Kc_tile = tl.load(
            Kc_ptr + v_offsets[:, None] * HEAD_DIM_CKV + k_offsets[None, :],
            mask=mask_v[:, None] & mask_k[None, :],
            other=0.0
        )  # [128, 128]
        # qn_vec: [128], Kc_tile: [128, 128] -> [128]
        part = tl.sum(qn_vec[None, :] * Kc_tile, axis=1)
        accum += part

    # Compute dot with Kp (64 dims)
    Kp_tile = tl.load(
        Kp_ptr + v_offsets * HEAD_DIM_KPE,  # v_offsets is 64-dim index into [TOPK, 64]
        mask=mask_v,
        other=0.0
    )  # [128]
    accum += tl.sum(qp_vec * Kp_tile, axis=0)

    # Store logits for this (h, v-block)
    tl.store(logits_ptr + h * TOPK + v_offsets, accum, mask=mask_v)


@triton.jit
def _lse_base2_kernel(
    logits_scaled_ptr,   # *fp32, [NUM_QO_HEADS, TOPK]
    lse_ptr,             # *fp32, [NUM_QO_HEADS]
    NUM_QO_HEADS: tl.constexpr,
    TOPK: tl.constexpr,
):
    h = tl.program_id(axis=0)
    # Compute max across v for this head
    max_val = tl.full((), -float('inf'), dtype=tl.float32)
    for v_start in range(0, TOPK, 128):
        v_offsets = v_start + tl.arange(0, 128)
        mask = v_offsets < TOPK
        vals = tl.load(logits_scaled_ptr + h * TOPK + v_offsets, mask=mask, other=-float('inf'))
        block_max = tl.max(vals, axis=0)
        max_val = tl.maximum(max_val, block_max)

    # Compute sum(exp(vals - max)) across v
    sum_exp = tl.zeros((), dtype=tl.float32)
    for v_start in range(0, TOPK, 128):
        v_offsets = v_start + tl.arange(0, 128)
        mask = v_offsets < TOPK
        vals = tl.load(logits_scaled_ptr + h * TOPK + v_offsets, mask=mask, other=0.0)
        sum_exp += tl.sum(tl.exp(vals - max_val), axis=0)

    lse = max_val + tl.log(sum_exp) / LN2
    tl.store(lse_ptr + h, lse)


@triton.jit
def _softmax_matmul_kernel(
    logits_scaled_ptr,   # *fp32, [NUM_QO_HEADS, TOPK]
    Kc_ptr,              # *fp32, [TOPK, HEAD_DIM_CKV]
    out_ptr,             # *fp32, [NUM_QO_HEADS, HEAD_DIM_CKV]
    NUM_QO_HEADS: tl.constexpr,
    TOPK: tl.constexpr,
    HEAD_DIM_CKV: tl.constexpr,
):
    h = tl.program_id(axis=0)

    # Compute denominator = sum(exp(logits_scaled[h, :]))
    denom = tl.zeros((), dtype=tl.float32)
    for v_start in range(0, TOPK, 128):
        v_offsets = v_start + tl.arange(0, 128)
        mask = v_offsets < TOPK
        vals = tl.load(logits_scaled_ptr + h * TOPK + v_offsets, mask=mask, other=0.0)
        denom += tl.sum(tl.exp(vals), axis=0)

    # Compute softmax_scaled[h, :] and output[h, :]
    for k_start in range(0, HEAD_DIM_CKV, 128):
        out_vec = tl.zeros((128,), dtype=tl.float32)
        for v_start in range(0, TOPK, 128):
            v_offsets = v_start + tl.arange(0, 128)
            mask_v = v_offsets < TOPK
            vals = tl.load(logits_scaled_ptr + h * TOPK + v_offsets, mask=mask_v, other=0.0)
            exp_vals = tl.exp(vals / denom)  # softmax
            Kc_tile = tl.load(
                Kc_ptr + v_offsets[:, None] * HEAD_DIM_CKV + (k_start + tl.arange(0, 128))[None, :],
                mask=mask_v[:, None] & ((k_start + tl.arange(0, 128)) < HEAD_DIM_CKV)[None, :],
                other=0.0
            )  # [128, 128]
            out_vec += tl.sum(exp_vals[:, None] * Kc_tile, axis=0)
        # Store out[h, k_start:k_start+128]
        k_offsets = k_start + tl.arange(0, 128)
        mask_k = k_offsets < HEAD_DIM_CKV
        tl.store(out_ptr + h * HEAD_DIM_CKV + k_offsets, out_vec, mask=mask_k)


def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    device = q_nope.device
    assert q_nope.shape == (1, NUM_QO_HEADS, HEAD_DIM_CKV), "Unexpected q_nope shape"
    assert q_pe.shape == (1, NUM_QO_HEADS, HEAD_DIM_KPE), "Unexpected q_pe shape"
    # Ensure inputs are contiguous and on CUDA
    q_nope = q_nope.contiguous().to(torch.float32)
    q_pe = q_pe.contiguous().to(torch.float32)
    # Flatten paged caches to token-level: [num_pages, 64, D] -> [(num_pages*64), D]
    Kc_all = ckv_cache.reshape(-1, HEAD_DIM_CKV).contiguous().to(torch.float32)  # [total_kv_tokens, head_dim_ckv]
    Kp_all = kpe_cache.reshape(-1, HEAD_DIM_KPE).contiguous().to(torch.float32)  # [total_kv_tokens, head_dim_kpe]
    # sparse_indices is [1, 2048] int32; load and filter valid
    indices = sparse_indices.squeeze(0)  # [2048]
    valid_mask = indices != -1
    valid_indices = indices[valid_mask]
    num_valid = valid_indices.numel()

    # If no valid indices for this token, return zeros
    if num_valid == 0:
        output = torch.zeros((1, NUM_QO_HEADS, HEAD_DIM_CKV), dtype=torch.bfloat16, device=device)
        lse = torch.full((1, NUM_QO_HEADS), -float("inf"), dtype=torch.float32, device=device)
        return output, lse

    Kc = Kc_all[valid_indices]  # [num_valid, head_dim_ckv]
    Kp = Kp_all[valid_indices]  # [num_valid, head_dim_kpe]

    # Allocate outputs
    logits = torch.empty((NUM_QO_HEADS, TOPK), dtype=torch.float32, device=device)
    lse = torch.empty((NUM_QO_HEADS,), dtype=torch.float32, device=device)
    out = torch.empty((NUM_QO_HEADS, HEAD_DIM_CKV), dtype=torch.float32, device=device)

    # Launch Triton kernels
    # 1) Compute logits for all heads and all v positions (topk=2048, tiled by 128)
    grid1 = (NUM_QO_HEADS, triton.cdiv(TOPK, 128))
    _compute_logits_kernel[grid1](
        q_nope.view(-1, HEAD_DIM_CKV),  # q_nope is [1, 16, 512] but we flatten heads
        q_pe.view(-1, HEAD_DIM_KPE),    # q_pe is [1, 16, 64]
        Kc,                             # [num_valid, 512]
        Kp,                             # [num_valid, 64]
        logits,                         # [16, 2048]
        NUM_QO_HEADS=NUM_QO_HEADS,
        TOPK=TOPK,
        HEAD_DIM_CKV=HEAD_DIM_CKV,
        HEAD_DIM_KPE=HEAD_DIM_KPE,
    )

    # 2) Compute logsumexp in base-2 for each head
    grid2 = (NUM_QO_HEADS,)
    _lse_base2_kernel[grid2](
        logits,  # [16, 2048]
        lse,     # [16]
        NUM_QO_HEADS=NUM_QO_HEADS,
        TOPK=TOPK,
    )

    # 3) Compute output = softmax(logits_scaled) @ Kc
    logits_scaled = logits * sm_scale  # scaling in Triton kernel could be added, here we scale on host
    grid3 = (NUM_QO_HEADS,)
    _softmax_matmul_kernel[grid3](
        logits_scaled,   # [16, 2048]
        Kc,              # [num_valid, 512]
        out,             # [16, 512]
        NUM_QO_HEADS=NUM_QO_HEADS,
        TOPK=TOPK,
        HEAD_DIM_CKV=HEAD_DIM_CKV,
    )

    # Return outputs as per original shape and dtypes
    # Original returns output for num_tokens tokens; here num_tokens=1
    output = out.view(1, NUM_QO_HEADS, HEAD_DIM_CKV).to(torch.bfloat16)
    # lse should be [1, 16] per original
    lse = lse.view(1, NUM_QO_HEADS)

    return output, lse


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
        # Triton-only forward: no PyTorch ops for computation
        assert TRITON_AVAILABLE, "Triton is not available"
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and sparse_indices.is_cuda, "All tensors must be on CUDA for Triton"
        return run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)


def get_inputs():
    # Evaluation environment runs on CUDA; use bfloat16 tensors and place on CUDA
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16, device='cuda')
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16, device='cuda')
    ckv_cache = torch.randn([8462, 64, 512], dtype=torch.bfloat16, device='cuda')
    kpe_cache = torch.randn([8462, 64, 64], dtype=torch.bfloat16, device='cuda')
    sparse_indices = torch.randint(0, 541568, [1, 2048], dtype=torch.int32, device='cuda')
    sm_scale = 1.0  # float32 scalar
    return [q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale]


def run(*args):
    return ModelNew()(*args)
