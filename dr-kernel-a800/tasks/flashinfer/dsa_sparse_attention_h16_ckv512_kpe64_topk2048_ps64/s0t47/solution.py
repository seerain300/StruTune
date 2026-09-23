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


@triton.jit
def _compute_logits_kernel(
    qn_ptr,           # *fp32, [NUM_QO_HEADS, HEAD_DIM_CKV]
    qp_ptr,           # *fp32, [NUM_QO_HEADS, HEAD_DIM_KPE]
    Kc_ptr,           # *fp32, [TOPK, HEAD_DIM_CKV]
    Kp_ptr,           # *fp32, [TOPK, HEAD_DIM_KPE]
    logits_ptr,       # *fp32, [NUM_QO_HEADS, TOPK]
    NUM_QO_HEADS: tl.constexpr,
    HEAD_DIM_CKV: tl.constexpr,
    HEAD_DIM_KPE: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_V: tl.constexpr = 128,
    BLOCK_C: tl.constexpr = 128,
    BLOCK_P: tl.constexpr = 64,
):
    # Each program handles one head h; we launch grid=(NUM_QO_HEADS,)
    h = tl.program_id(axis=0)

    # Pass 1: compute logits[h, v] for all v tiles
    for v_start in tl.static_range(0, TOPK, BLOCK_V):
        v_offsets = v_start + tl.arange(0, BLOCK_V)
        mask_v = v_offsets < TOPK

        accum = tl.zeros((BLOCK_V,), dtype=tl.float32)

        # Sum over Kc chunks: accum += sum_c qn[h, c] * Kc[v, c]
        for c_start in tl.static_range(0, HEAD_DIM_CKV, BLOCK_C):
            c_offsets = c_start + tl.arange(0, BLOCK_C)
            mask_c = c_offsets < HEAD_DIM_CKV
            qn_vec = tl.load(qn_ptr + h * HEAD_DIM_CKV + c_offsets, mask=mask_c, other=0.0)  # [BLOCK_C]
            Kc_block = tl.load(
                Kc_ptr + v_offsets[:, None] * HEAD_DIM_CKV + c_offsets[None, :],
                mask=mask_v[:, None] & mask_c[None, :],
                other=0.0
            )  # [BLOCK_V, BLOCK_C]
            accum += tl.sum(Kc_block * qn_vec[None, :], axis=1)

        # Sum over Kp chunks: accum += sum_p qp[h, p] * Kp[v, p]
        for p_start in tl.static_range(0, HEAD_DIM_KPE, BLOCK_P):
            p_offsets = p_start + tl.arange(0, BLOCK_P)
            mask_p = p_offsets < HEAD_DIM_KPE
            qp_vec = tl.load(qp_ptr + h * HEAD_DIM_KPE + p_offsets, mask=mask_p, other=0.0)  # [BLOCK_P]
            Kp_block = tl.load(
                Kp_ptr + v_offsets[:, None] * HEAD_DIM_KPE + p_offsets[None, :],
                mask=mask_v[:, None] & mask_p[None, :],
                other=0.0
            )  # [BLOCK_V, BLOCK_P]
            accum += tl.sum(Kp_block * qp_vec[None, :], axis=1)

        # Store logits for this tile
        tl.store(logits_ptr + h * TOPK + v_offsets, accum, mask=mask_v)


@triton.jit
def _lse_base2_kernel(
    logits_ptr,       # *fp32, [NUM_QO_HEADS, TOPK]
    lse_ptr,          # *fp32, [NUM_QO_HEADS]
    TOPK: tl.constexpr,
    BLOCK_V: tl.constexpr = 128,
):
    # Each program handles one head h; grid=(NUM_QO_HEADS,)
    h = tl.program_id(axis=0)
    m = -float('inf')

    # Compute max across all v tiles
    for v_start in tl.static_range(0, TOPK, BLOCK_V):
        v_offsets = v_start + tl.arange(0, BLOCK_V)
        mask_v = v_offsets < TOPK
        vals = tl.load(logits_ptr + h * TOPK + v_offsets, mask=mask_v, other=-float('inf'))
        tile_max = tl.max(vals, axis=0)
        m = tl.maximum(m, tile_max)

    # Compute sum_exp = sum(exp(vals - m)) across all v
    sum_exp = tl.zeros((), dtype=tl.float32)
    for v_start in tl.static_range(0, TOPK, BLOCK_V):
        v_offsets = v_start + tl.arange(0, BLOCK_V)
        mask_v = v_offsets < TOPK
        vals = tl.load(logits_ptr + h * TOPK + v_offsets, mask=mask_v, other=-float('inf'))
        exp_vals = tl.exp(vals - m)
        exp_vals = tl.where(mask_v, exp_vals, 0.0)
        sum_exp += tl.sum(exp_vals, axis=0)

    lse_val = m + math.log(sum_exp) / math.log(2.0)
    tl.store(lse_ptr + h, lse_val)


@triton.jit
def _compute_output_kernel(
    Kc_ptr,           # *fp32, [TOPK, HEAD_DIM_CKV]
    logits_ptr,       # *fp32, [NUM_QO_HEADS, TOPK]
    lse_ptr,          # *fp32, [NUM_QO_HEADS]
    out_ptr,          # *fp32, [NUM_QO_HEADS, HEAD_DIM_CKV]
    sm_scale,         # fp32 scalar
    NUM_QO_HEADS: tl.constexpr,
    HEAD_DIM_CKV: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_V: tl.constexpr = 128,
    BLOCK_C: tl.constexpr = 128,
):
    h = tl.program_id(axis=0)
    lse_val = tl.load(lse_ptr + h)

    # Pass 2: compute out[h, c] = sum_v exp((logits[h, v] - lse_val) * sm_scale) * Kc[v, c]
    for c_start in tl.static_range(0, HEAD_DIM_CKV, BLOCK_C):
        c_offsets = c_start + tl.arange(0, BLOCK_C)
        mask_c = c_offsets < HEAD_DIM_CKV
        out_vec = tl.zeros((BLOCK_C,), dtype=tl.float32)

        for v_start in tl.static_range(0, TOPK, BLOCK_V):
            v_offsets = v_start + tl.arange(0, BLOCK_V)
            mask_v = v_offsets < TOPK

            logits_vec = tl.load(
                logits_ptr + h * TOPK + v_offsets,
                mask=mask_v,
                other=-float('inf')
            )  # [BLOCK_V]
            Kc_block = tl.load(
                Kc_ptr + v_offsets[:, None] * HEAD_DIM_CKV + c_offsets[None, :],
                mask=mask_v[:, None] & mask_c[None, :],
                other=0.0
            )  # [BLOCK_V, BLOCK_C]

            scaled = tl.exp((logits_vec - lse_val) * sm_scale)
            scaled = tl.where(mask_v, scaled, 0.0)

            out_vec += tl.sum(scaled[:, None] * Kc_block, axis=0)

        tl.store(out_ptr + h * HEAD_DIM_CKV + c_offsets, out_vec, mask=mask_c)


def run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    # Ensure Triton is available and tensors are on CUDA
    assert TRITON_AVAILABLE, "Triton is not available"
    device = q_nope.device
    assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and sparse_indices.is_cuda, "All tensors must be on CUDA for Triton"

    # Flatten paged KV cache to [num_pages * 64, dim] in fp32
    Kc_all = ckv_cache.reshape(-1, HEAD_DIM_CKV).to(torch.float32)  # [num_pages * 64, HEAD_DIM_CKV]
    Kp_all = kpe_cache.reshape(-1, HEAD_DIM_KPE).to(torch.float32)  # [num_pages * 64, HEAD_DIM_KPE]

    num_tokens = q_nope.shape[0]
    output = torch.empty((num_tokens, NUM_QO_HEADS, HEAD_DIM_CKV), dtype=torch.float32, device=device)
    lse = torch.empty((num_tokens, NUM_QO_HEADS), dtype=torch.float32, device=device)

    # Process one token at a time to keep kernel definitions simple and avoid recursion in JIT
    for t in range(num_tokens):
        # Allocate temporary buffers
        logits = torch.empty((NUM_QO_HEADS, TOPK), dtype=torch.float32, device=device)
        lse_row = torch.empty((NUM_QO_HEADS,), dtype=torch.float32, device=device)
        out = torch.empty((NUM_QO_HEADS, HEAD_DIM_CKV), dtype=torch.float32, device=device)

        # Compute logits[h, v] for this token
        grid = (NUM_QO_HEADS,)
        _compute_logits_kernel[grid](
            qn_ptr=q_nope[t].to(torch.float32),  # [NUM_QO_HEADS, HEAD_DIM_CKV]
            qp_ptr=q_pe[t].to(torch.float32),    # [NUM_QO_HEADS, HEAD_DIM_KPE]
            Kc_ptr=Kc_all,                       # [TOPK, HEAD_DIM_CKV]
            Kp_ptr=Kp_all,                       # [TOPK, HEAD_DIM_KPE]
            logits_ptr=logits,                   # [NUM_QO_HEADS, TOPK]
            NUM_QO_HEADS=NUM_QO_HEADS,
            HEAD_DIM_CKV=HEAD_DIM_CKV,
            HEAD_DIM_KPE=HEAD_DIM_KPE,
            TOPK=TOPK,
            BLOCK_V=128,
            BLOCK_C=128,
            BLOCK_P=64,
        )

        # Compute lse per head (base-2)
        _lse_base2_kernel[grid](
            logits_ptr=logits,                   # [NUM_QO_HEADS, TOPK]
            lse_ptr=lse_row,                    # [NUM_QO_HEADS]
            TOPK=TOPK,
            BLOCK_V=128,
        )

        # Compute final output
        _compute_output_kernel[grid](
            Kc_ptr=Kc_all,                      # [TOPK, HEAD_DIM_CKV]
            logits_ptr=logits,                  # [NUM_QO_HEADS, TOPK]
            lse_ptr=lse_row,                   # [NUM_QO_HEADS]
            out_ptr=out,                       # [NUM_QO_HEADS, HEAD_DIM_CKV]
            sm_scale=float(sm_scale),
            NUM_QO_HEADS=NUM_QO_HEADS,
            HEAD_DIM_CKV=HEAD_DIM_CKV,
            TOPK=TOPK,
            BLOCK_V=128,
            BLOCK_C=128,
        )

        output[t] = out
        lse[t] = lse_row

    # Match original output dtype: output bfloat16, lse float32
    output_bf16 = output.to(torch.bfloat16)
    return output_bf16, lse


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
        return run(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)


# Helper for local testing; evaluator may provide its own inputs
def get_inputs():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16, device=device)
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16, device=device)
    ckv_cache = torch.randn([8462, 64, 512], dtype=torch.bfloat16, device=device)
    kpe_cache = torch.randn([8462, 64, 64], dtype=torch.bfloat16, device=device)
    sparse_indices = torch.randint(0, 541568, [1, 2048], dtype=torch.int32, device=device)
    sm_scale = 1.0
    return [q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale]


def run(*args):
    return ModelNew()(*args)
