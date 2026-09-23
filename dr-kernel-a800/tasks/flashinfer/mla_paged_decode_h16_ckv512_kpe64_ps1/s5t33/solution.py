import math
import torch
import triton
import triton.language as tl


@triton.jit
def compute_logits_kernel(
    qn_ptr,   # *f32 [H, CK]
    qp_ptr,   # *f32 [H, KP]
    Kc_ptr,   # *f32 [L_tokens, CK]
    Kp_ptr,   # *f32 [L_tokens, KP]
    logits_ptr,  # *f32 [H, L_tokens]
    sm_scale: tl.constexpr,  # scalar float
    H: tl.constexpr,
    L_tokens: tl.constexpr,
    CK: tl.constexpr,
    KP: tl.constexpr,
):
    h = tl.program_id(0)
    t = tl.program_id(1)
    if h >= H or t >= L_tokens:
        return

    # Load qn[h, :] and qp[h, :]
    qn_vec = tl.load(qn_ptr + h * CK + tl.arange(0, CK))
    qp_vec = tl.load(qp_ptr + h * KP + tl.arange(0, KP))

    # Load Kc[t, :] and Kp[t, :]
    Kc_row = tl.load(Kc_ptr + t * CK + tl.arange(0, CK))
    Kp_row = tl.load(Kp_ptr + t * KP + tl.arange(0, KP))

    # Dot products
    dot_qn_Kc = 0.0
    dot_qp_Kp = 0.0
    for i in range(CK):
        dot_qn_Kc += qn_vec[i] * Kc_row[i]
    for i in range(KP):
        dot_qp_Kp += qp_vec[i] * Kp_row[i]

    scaled = sm_scale * (dot_qn_Kc + dot_qp_Kp)
    tl.store(logits_ptr + h * L_tokens + t, scaled)


@triton.jit
def lse_kernel(
    logits_ptr,  # *f32 [H, L_tokens]
    lse_ptr,     # *f32 [H]
    H: tl.constexpr,
    L_tokens: tl.constexpr,
):
    h = tl.program_id(0)
    if h >= H:
        return

    max_val = -float("inf")
    for t in range(L_tokens):
        val = tl.load(logits_ptr + h * L_tokens + t)
        if val > max_val:
            max_val = val

    sumexp = 0.0
    for t in range(L_tokens):
        val = tl.load(logits_ptr + h * L_tokens + t)
        sumexp += tl.exp(val - max_val)

    lse = tl.log(sumexp)
    tl.store(lse_ptr + h, lse)


@triton.jit
def compute_output_kernel(
    qn_ptr,     # *f32 [H, CK]
    qp_ptr,     # *f32 [H, KP]
    Kc_ptr,     # *f32 [L_tokens, CK]
    logits_ptr, # *f32 [H, L_tokens]
    lse_ptr,    # *f32 [H]
    out_ptr,    # *f32 [H, CK] (host allocated, kernel writes h slice)
    H: tl.constexpr,
    L_tokens: tl.constexpr,
    CK: tl.constexpr,
    sm_scale: tl.constexpr,  # not used here, kept for signature symmetry
):
    h = tl.program_id(0)
    if h >= H:
        return

    # Load qn[h, :] and qp[h, :]
    qn_vec = tl.load(qn_ptr + h * CK + tl.arange(0, CK))
    # We don't need qp here for output accumulation, but signature keeps it.
    # Accumulator
    out_vec = tl.zeros((CK,), dtype=tl.float32)

    # Load lse[h]
    lse_h = tl.load(lse_ptr + h)

    # Loop over tokens t and accumulate softmax-weighted Kc[t, :]
    for t in range(L_tokens):
        val = tl.load(logits_ptr + h * L_tokens + t)
        p = tl.exp(val - lse_h)
        Kc_row = tl.load(Kc_ptr + t * CK + tl.arange(0, CK))
        out_vec += p * Kc_row

    # Store out[h, :]
    tl.store(out_ptr + h * CK + tl.arange(0, CK), out_vec)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA tensors and float32 compute
        device = q_nope.device
        assert device.type == "cuda", "ModelNew requires CUDA tensors"
        q_nope = q_nope.to(torch.float32).contiguous()
        q_pe = q_pe.to(torch.float32).contiguous()
        ckv_cache = ckv_cache.to(torch.float32).contiguous()
        kpe_cache = kpe_cache.to(torch.float32).contiguous()

        batch_size, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]
        assert num_qo_heads == 16 and head_dim_ckv == 512 and head_dim_kpe == 64, "Shape assertions must hold"
        num_pages = ckv_cache.shape[0]
        assert kpe_cache.shape == (num_pages, 1, 64)
        assert ckv_cache.shape == (num_pages, 1, 512)
        assert kv_indptr.shape[0] == batch_size + 1

        # Allocate outputs (host-side). No torch ops on tensors inside kernels.
        output = torch.empty((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        sm_scale_val = float(sm_scale.item())  # ensure Python float, not NoneType

        for b in range(batch_size):
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            L_tokens = max(0, page_end - page_beg)

            if L_tokens == 0:
                # No tokens for this batch element
                lse[b] = float("-inf")
                output[b] = torch.zeros((num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
                continue

            # Allocate intermediate logits [H, L_tokens]
            logits = torch.empty((num_qo_heads, L_tokens), dtype=torch.float32, device=device)

            # Launch compute_logits_kernel: grid (H, L_tokens)
            compute_logits_kernel[(num_qo_heads, L_tokens)](
                q_nope[b], q_pe[b], ckv_cache[page_beg:page_end], kpe_cache[page_beg:page_end], logits,
                sm_scale=sm_scale_val, H=num_qo_heads, L_tokens=L_tokens, CK=512, KP=64
            )

            # Launch lse_kernel: grid (H,)
            lse_kernel[(num_qo_heads,)](
                logits, lse[b],
                H=num_qo_heads, L_tokens=L_tokens
            )

            # Launch compute_output_kernel: grid (H,)
            compute_output_kernel[(num_qo_heads,)](
                q_nope[b], q_pe[b], ckv_cache[page_beg:page_end], logits, lse[b], output[b],
                H=num_qo_heads, L_tokens=L_tokens, CK=512, sm_scale=sm_scale_val
            )

        # Return (output, lse) matching original signature; cast output to bfloat16 as in the original code
        output_bf16 = output.to(torch.bfloat16)
        return (output_bf16, lse)


def run(*args):
    return ModelNew()(*args)
