import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Constants consistent with the original code (used in kernel launch; host dims are dynamic)
NUM_QO_HEADS = 16
HEAD_DIM_CKV = 512      # q_nope's last dim and Kc's last dim
HEAD_DIM_KPE = 64        # q_pe's last dim and Kp's last dim
TOPK = 2048              # sparse_indices's last dim


@triton.jit
def _fused_attention_kernel(
    qn_ptr,           # *fp32, [NUM_QO_HEADS, HEAD_DIM_CKV]
    qp_ptr,           # *fp32, [NUM_QO_HEADS, HEAD_DIM_KPE]
    Kc_ptr,           # *fp32, [TOPK, HEAD_DIM_CKV]
    Kp_ptr,           # *fp32, [TOPK, HEAD_DIM_KPE]
    output_ptr,       # *fp32, [NUM_QO_HEADS, HEAD_DIM_CKV]
    lse_ptr,          # *fp32, [NUM_QO_HEADS]
    sm_scale: tl.constexpr,  # scaling for logits
    NUM_QO_HEADS: tl.constexpr,
    HEAD_DIM_CKV: tl.constexpr,
    HEAD_DIM_KPE: tl.constexpr,
    TOPK: tl.constexpr,
):
    # One program per head; tile over v and c to keep compute bounded
    h = tl.program_id(axis=0)

    # Vector of valid positions per tile
    v_tile = 128
    for v_start in range(0, TOPK, v_tile):
        v_offsets = v_start + tl.arange(0, v_tile)
        mask_v = v_offsets < TOPK

        # Accumulator for logits for this head across v tile
        acc_logits = tl.zeros((v_tile,), dtype=tl.float32)

        # Compute logits[h, v_offsets] = dot(qn[h], Kc[v_offsets]) + dot(qp[h], Kp[v_offsets])
        # Reduction over c and p
        for c in tl.static_range(0, HEAD_DIM_CKV):
            qn_val = tl.load(qn_ptr + h * HEAD_DIM_CKV + c)  # scalar
            kc_vec = tl.load(Kc_ptr + v_offsets * HEAD_DIM_CKV + c, mask=mask_v, other=0.0)  # [v_tile]
            acc_logits += qn_val * kc_vec

        for p in tl.static_range(0, HEAD_DIM_KPE):
            qp_val = tl.load(qp_ptr + h * HEAD_DIM_KPE + p)  # scalar
            kp_vec = tl.load(Kp_ptr + v_offsets * HEAD_DIM_KPE + p, mask=mask_v, other=0.0)  # [v_tile]
            acc_logits += qp_val * kp_vec

        scaled = acc_logits * sm_scale

        # Compute lse per tile (numerically stable)
        m = -float("inf")
        for i in range(0, v_tile):
            if (v_start + i) < TOPK:
                m = tl.maximum(m, scaled[i])

        sum_exp = 0.0
        for i in range(0, v_tile):
            if (v_start + i) < TOPK:
                sum_exp += tl.exp(scaled[i] - m)

        lse_h = m + tl.log(sum_exp) / 0.6931471805599453  # ln(2)
        # Store lse for this head (scalar)
        tl.store(lse_ptr + h, lse_h)

        # Compute output[h, c] using tiled reduction over v
        c_tile = 128
        for c_start in range(0, HEAD_DIM_CKV, c_tile):
            c_offsets = c_start + tl.arange(0, c_tile)
            mask_c = c_offsets < HEAD_DIM_CKV

            acc_out = tl.zeros((c_tile,), dtype=tl.float32)
            for v in range(0, TOPK):
                # prob_v = exp((logits_scaled[v] - lse_h)) / ln(2), but since we scaled by ln(2), it's simply exp(logits_scaled - lse_h)
                # However, we already have scaled = logits * sm_scale; and we stored lse_h; recompute per v to avoid storing full logits
                val = tl.load(Kc_ptr + v * HEAD_DIM_CKV + c_offsets, mask=mask_c, other=0.0)  # [c_tile]
                # We need probability for each v; compute per v scalar by loading scaled at that v
                # Since we don't store scaled per v, recompute: dot with Kc? No, we need scaled for that v.
                # Instead, keep scaled vector for the tile:
                # But to avoid recomputing huge memory, we can compute scaled from acc_logits and v index
                # For each v within v_tile, we can access scaled[i] where i=v_start + v_index; outside v_tile, masked.
                # Implement by recomputation: For each v loop, we don't have acc_logits for that v; therefore, recompute per v is not feasible.
                # So, we need to store scaled for each v; we'll do that in registers for v_tile, and reuse across c tile.
                # Here we'll compute scaled for each v by dotting qn[h] with Kc[v] and qp[h] with Kp[v], which is costly. Better approach:
                # Re-store scaled vector for this v tile and then use it in the next stage.

            # We can't recompute scaled per v here without reloading Kc and Kp for each v; hence we store scaled as needed.
            # To simplify, we recompute per v: it's acceptable given TOPK is moderate, and Triton supports scalar math.
            # However, Triton kernels don't have dynamic Python loops over TOPK easily without excessive code. Therefore, for robustness,
            # we compute scaled per v by re-dotting qn[h] and qp[h] with Kc[v] and Kp[v]. This is fine for TOPK=2048.
            for v in range(0, TOPK):
                kc_v = tl.load(Kc_ptr + v * HEAD_DIM_CKV + c_offsets, mask=mask_c, other=0.0)  # [c_tile]
                # Compute scaled for this v: we need logits for this v.
                # logits_v = dot(qn[h], Kc[v]) + dot(qp[h], Kp[v])
                logits_v = 0.0
                for c2 in tl.static_range(0, HEAD_DIM_CKV):
                    qn_val = tl.load(qn_ptr + h * HEAD_DIM_CKV + c2)
                    kc_elem = tl.load(Kc_ptr + v * HEAD_DIM_CKV + c2)
                    logits_v += qn_val * kc_elem
                for p2 in tl.static_range(0, HEAD_DIM_KPE):
                    qp_val = tl.load(qp_ptr + h * HEAD_DIM_KPE + p2)
                    kp_elem = tl.load(Kp_ptr + v * HEAD_DIM_KPE + p2)
                    logits_v += qp_val * kp_elem
                scaled_v = logits_v * sm_scale
                prob_v = tl.exp(scaled_v - lse_h)
                acc_out += prob_v * kc_v  # kc_v is [c_tile]
            tl.store(output_ptr + h * HEAD_DIM_CKV + c_offsets, acc_out, mask=mask_c)


# Triton-only forward function
def _run_triton(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    assert TRITON_AVAILABLE, "Triton is not available"
    assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and sparse_indices.is_cuda, "All tensors must be on CUDA for Triton"

    num_tokens = q_nope.shape[0]
    output = torch.empty(
        (num_tokens, NUM_QO_HEADS, HEAD_DIM_CKV), dtype=torch.bfloat16, device=q_nope.device
    )
    lse = torch.empty((num_tokens, NUM_QO_HEADS), dtype=torch.float32, device=q_nope.device)

    # Flatten paged KV cache to token-level: [num_pages, page_size, dim] -> [num_tokens, dim]
    Kc_all = ckv_cache.reshape(-1, HEAD_DIM_CKV).to(torch.float32)  # [num_tokens * num_pages * page_size, HEAD_DIM_CKV]
    Kp_all = kpe_cache.reshape(-1, HEAD_DIM_KPE).to(torch.float32)  # [num_tokens * num_pages * page_size, HEAD_DIM_KPE]

    for t in range(num_tokens):
        indices = sparse_indices[t]  # [TOPK]
        valid_mask = indices != -1
        valid_indices = indices[valid_mask].to(torch.long)  # [num_valid]
        if valid_indices.numel() == 0:
            output[t].zero_()
            continue

        # Gather rows for this token
        Kc_sel = Kc_all[valid_indices]  # [num_valid, HEAD_DIM_CKV]
        Kp_sel = Kp_all[valid_indices]  # [num_valid, HEAD_DIM_KPE]

        # Cast q to fp32
        qn = q_nope[t].to(torch.float32).contiguous()  # [NUM_QO_HEADS, HEAD_DIM_CKV]
        qp = q_pe[t].to(torch.float32).contiguous()   # [NUM_QO_HEADS, HEAD_DIM_KPE]

        # Allocate output for this token
        output_t = torch.empty((NUM_QO_HEADS, HEAD_DIM_CKV), dtype=torch.float32, device=q_nope.device)

        # Launch fused kernel: one program per head
        grid = (NUM_QO_HEADS,)
        _fused_attention_kernel[grid](
            qn, qp, Kc_sel, Kp_sel, output_t, lse[t],
            sm_scale=sm_scale,
            NUM_QO_HEADS=NUM_QO_HEADS, HEAD_DIM_CKV=HEAD_DIM_CKV, HEAD_DIM_KPE=HEAD_DIM_KPE, TOPK=TOPK,
        )

        # Store result for this token
        output[t] = output_t.to(torch.bfloat16)

    return output, lse


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
        # Triton-only forward: no PyTorch ops for computation
        return _run_triton(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale)


def get_inputs():
    # Use bfloat16 tensors and place on CUDA
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16, device='cuda')
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16, device='cuda')
    ckv_cache = torch.randn([8462, 64, 512], dtype=torch.bfloat16, device='cuda')
    kpe_cache = torch.randn([8462, 64, 64], dtype=torch.bfloat16, device='cuda')
    sparse_indices = torch.randint(0, 541568, [1, 2048], dtype=torch.int32, device='cuda')
    sm_scale = 1.0  # float32 scalar
    return [q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale]


def run(*args):
    return ModelNew()(*args)
