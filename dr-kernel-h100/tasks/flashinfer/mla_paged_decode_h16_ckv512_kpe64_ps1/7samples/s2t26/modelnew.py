import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: compute output and lse for a single batch element (b) and head (h).
# Grid: (B, H). Each program handles one (b, h).
@triton.jit
def _compute_single_head_kernel(
    q_nope_ptr, q_pe_ptr,
    Kc_all_ptr, Kp_all_ptr,
    out_ptr, lse_ptr,
    B: tl.int32, H: tl.int32, Dc: tl.int32, Dp: tl.int32,
    L_tokens: tl.int32,
    sm_scale: tl.float32,
    inv_ln2: tl.float32,
    # strides (in elements)
    q_nope_stride_b: tl.int32, q_nope_stride_h: tl.int32, q_nope_stride_d: tl.int32,
    q_pe_stride_b: tl.int32, q_pe_stride_h: tl.int32, q_pe_stride_d: tl.int32,
    Kc_stride_t: tl.int32, Kc_stride_d: tl.int32,
    Kp_stride_t: tl.int32, Kp_stride_d: tl.int32,
    out_stride_b: tl.int32, out_stride_h: tl.int32, out_stride_d: tl.int32,
    lse_stride_b: tl.int32, lse_stride_h: tl.int32,
    BLOCK_D: tl.constexpr  # vectorization across Dc
):
    b = tl.program_id(0)
    h = tl.program_id(1)

    # Prepare pointers to qn and qp for this (b, h)
    qn_ptr = q_nope_ptr + b * q_nope_stride_b + h * q_nope_stride_h
    qp_ptr = q_pe_ptr + b * q_pe_stride_b + h * q_pe_stride_h

    # Vector of Dc for dot products
    d_offsets = tl.arange(0, BLOCK_D)

    # Pass 1: compute logits, track max, and sumexp for stable logsumexp
    max_val = -float('inf')
    sumexp = 0.0
    # Loop over tokens (L_tokens is runtime int, smallish, but we keep loop for generality)
    for t in range(0, L_tokens):
        # qn[d] for this head
        qn_vals = tl.load(qn_ptr + d_offsets * q_nope_stride_d, mask=d_offsets < Dc, other=0.0).to(tl.float32)
        # Kc[t, d]
        Kc_row_ptr = Kc_all_ptr + t * Kc_stride_t + d_offsets * Kc_stride_d
        Kc_vals = tl.load(Kc_row_ptr, mask=d_offsets < Dc, other=0.0).to(tl.float32)
        dot_qn_Kc = tl.sum(qn_vals * Kc_vals, axis=0)

        # qp[d] for this head
        qp_vals = tl.load(qp_ptr + d_offsets * q_pe_stride_d, mask=d_offsets < Dp, other=0.0).to(tl.float32)
        # Kp[t, d]
        Kp_row_ptr = Kp_all_ptr + t * Kp_stride_t + d_offsets * Kp_stride_d
        Kp_vals = tl.load(Kp_row_ptr, mask=d_offsets < Dp, other=0.0).to(tl.float32)
        dot_qp_Kp = tl.sum(qp_vals * Kp_vals, axis=0)

        logits_t = dot_qn_Kc + dot_qp_Kp
        logits_scaled = logits_t * sm_scale
        max_val = tl.maximum(max_val, logits_scaled)
        # sum of exp(logits_scaled - max)
        sumexp += tl.exp(logits_scaled - max_val)

    # lse (base-2) = (max + log(sumexp)) * (1/ln2)
    lse_b_h = (max_val + tl.log(sumexp)) * inv_ln2
    # store lse to [b, h]
    tl.store(lse_ptr + b * lse_stride_b + h * lse_stride_h, lse_b_h)

    # Pass 2: compute attn and accumulate output vector
    out_vec = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for t in range(0, L_tokens):
        # qn[d]
        qn_vals = tl.load(qn_ptr + d_offsets * q_nope_stride_d, mask=d_offsets < Dc, other=0.0).to(tl.float32)
        Kc_row_ptr = Kc_all_ptr + t * Kc_stride_t + d_offsets * Kc_stride_d
        Kc_vals = tl.load(Kc_row_ptr, mask=d_offsets < Dc, other=0.0).to(tl.float32)

        # qp[d]
        qp_vals = tl.load(qp_ptr + d_offsets * q_pe_stride_d, mask=d_offsets < Dp, other=0.0).to(tl.float32)
        Kp_row_ptr = Kp_all_ptr + t * Kp_stride_t + d_offsets * Kp_stride_d
        Kp_vals = tl.load(Kp_row_ptr, mask=d_offsets < Dp, other=0.0).to(tl.float32)

        dot_qn_Kc = tl.sum(qn_vals * Kc_vals, axis=0)
        dot_qp_Kp = tl.sum(qp_vals * Kp_vals, axis=0)
        logits_t = dot_qn_Kc + dot_qp_Kp
        logits_scaled = logits_t * sm_scale

        # attn_t = exp(logits_scaled - lse) / ln2
        attn_t = tl.exp(logits_scaled - lse_b_h) * inv_ln2
        out_vec += attn_t * Kc_vals

    # Store output as bfloat16
    out_ptr_row = out_ptr + b * out_stride_b + h * out_stride_h
    out_store_ptr = out_ptr_row + d_offsets * out_stride_d
    tl.store(out_store_ptr, out_vec.to(tl.bfloat16), mask=d_offsets < Dc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        """
        q_nope: [B, H, Dc], bfloat16
        q_pe: [B, H, Dp], bfloat16
        ckv_cache: [N, 1, Dc], bfloat16
        kpe_cache: [N, 1, Dp], bfloat16
        kv_indptr: [len_indptr], int32
        kv_indices: [M], int32
        sm_scale: float32 scalar
        returns: out [B, H, Dc] bfloat16, lse [B, H] float32
        """
        # Ensure CUDA tensors
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All tensors must be on CUDA for Triton."

        device = q_nope.device
        B, H, Dc = q_nope.shape
        _, _, Dp = q_pe.shape
        N = ckv_cache.shape[0]
        assert H == 16, "num_qo_heads must be 16"
        assert Dc == 512 and Dp == 64, "head dims must be 512 and 64"
        assert kv_indptr.shape[0] == B + 1, "kv_indptr must have length B+1"

        # Prepare output and lse tensors
        out = torch.empty((B, H, Dc), dtype=torch.bfloat16, device=device)
        lse = torch.full((B, H), -float("inf"), dtype=torch.float32, device=device)

        # Process each batch element
        for b in range(B):
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            L_tokens = page_end - page_beg

            # If no tokens for this batch element, zero output and skip
            if L_tokens == 0:
                # output[b] already zero
                lse[b, :] = torch.tensor([-float("inf")], dtype=torch.float32, device=device).expand(H).contiguous()
                continue

            tok_idx = kv_indices[page_beg:page_end].to(torch.long)  # [L_tokens]
            Kc_all = ckv_cache[tok_idx].contiguous()  # [L_tokens, Dc], bfloat16
            Kp_all = kpe_cache[tok_idx].contiguous()  # [L_tokens, Dp], bfloat16

            # Strides (elements)
            q_nope_stride_b = q_nope.stride(0)
            q_nope_stride_h = q_nope.stride(1)
            q_nope_stride_d = q_nope.stride(2)

            q_pe_stride_b = q_pe.stride(0)
            q_pe_stride_h = q_pe.stride(1)
            q_pe_stride_d = q_pe.stride(2)

            Kc_stride_t = Kc_all.stride(0)
            Kc_stride_d = Kc_all.stride(1)

            Kp_stride_t = Kp_all.stride(0)
            Kp_stride_d = Kp_all.stride(1)

            out_stride_b = out.stride(0)
            out_stride_h = out.stride(1)
            out_stride_d = out.stride(2)

            lse_stride_b = lse.stride(0)
            lse_stride_h = lse.stride(1)

            # Launch Triton kernel: one program per (b, h)
            grid = (B, H)
            BLOCK_D = 128  # vectorize across Dc; we loop over Dc in chunks, but set BLOCK_D to a multiple of 64 for good warp utilization
            _compute_single_head_kernel[grid](
                q_nope, q_pe,
                Kc_all, Kp_all,
                out, lse,
                B, H, Dc, Dp,
                L_tokens,
                sm_scale,  # 1.0
                1.0 / math.log(2.0),  # inv_ln2
                q_nope_stride_b, q_nope_stride_h, q_nope_stride_d,
                q_pe_stride_b, q_pe_stride_h, q_pe_stride_d,
                Kc_stride_t, Kc_stride_d,
                Kp_stride_t, Kp_stride_d,
                out_stride_b, out_stride_h, out_stride_d,
                lse_stride_b, lse_stride_h,
                BLOCK_D=BLOCK_D,
                num_warps=4,
                num_stages=2
            )

        return out, lse


# Original helper for generating inputs (ensure on CUDA for Triton):
def get_inputs():
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16, device='cuda')
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16, device='cuda')
    ckv_cache = torch.randn([989669, 1, 512], dtype=torch.bfloat16, device='cuda')
    kpe_cache = torch.randn([989669, 1, 64], dtype=torch.bfloat16, device='cuda')
    _n = 1; _t = 8
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32, device='cuda')
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32, device='cuda'), torch.cumsum(_lens, 0)]).to(torch.int32)
    kv_indices = torch.randint(0, 989669, [8], dtype=torch.int32, device='cuda')
    sm_scale = 1.0  # float32 scalar
    return [q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale]


def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6):
    _out = ModelNew()(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]