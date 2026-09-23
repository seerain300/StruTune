import torch
import triton
import triton.language as tl

# Shapes and constants from the original code
# We don't hardcode batch size/num_qo_heads here; forward will pass sizes to kernels.


@triton.jit
def compute_logits_kernel(
    qn_ptr,     # [H, CK], H=num_qo_heads, CK=512
    qp_ptr,     # [H, KP], KP=64
    Kc_all_ptr, # [P, CK], but we only use tok_idx to pick rows; P can be ignored here
    Kp_all_ptr, # [P, KP]
    tok_idx_ptr,    # [L_tokens], int32
    logits_ptr,     # [H, L_tokens], float32
    H: tl.constexpr,
    CK: tl.constexpr,
    KP: tl.constexpr,
    L_tokens: tl.constexpr,
):
    # 2D grid: (h, t)
    h = tl.program_id(0)
    t = tl.program_id(1)
    # Bounds guard
    if h >= H or t >= L_tokens:
        return

    # Load token index
    idx = tl.load(tok_idx_ptr + t)  # int32

    # Load q vectors for this head
    dim_offsets_qn = tl.arange(0, CK)
    qn_vec = tl.load(qn_ptr + h * CK + dim_offsets_qn)  # [CK]
    dim_offsets_qp = tl.arange(0, KP)
    qp_vec = tl.load(qp_ptr + h * KP + dim_offsets_qp)  # [KP]

    # Load Kc and Kp rows
    Kc_row = tl.load(Kc_all_ptr + idx * CK + dim_offsets_qn)  # [CK]
    Kp_row = tl.load(Kp_all_ptr + idx * KP + dim_offsets_qp)  # [KP]

    # Dot products
    dot_qn_Kc = tl.sum(qn_vec * Kc_row, axis=0)  # float32
    dot_qp_Kp = tl.sum(qp_vec * Kp_row, axis=0)  # float32

    logit = dot_qn_Kc + dot_qp_Kp
    tl.store(logits_ptr + h * L_tokens + t, logit)


@triton.jit
def lse_kernel(
    logits_ptr,  # [H, L_tokens], float32
    lse_out_ptr, # [H], float32
    H: tl.constexpr,
    L_tokens: tl.constexpr,
):
    h = tl.program_id(0)
    if h >= H:
        return
    # Pass 1: max
    m = -1.0e30
    for t in range(L_tokens):
        val = tl.load(logits_ptr + h * L_tokens + t)
        m = tl.maximum(m, val)
    # Pass 2: sum of exp shifted
    sum_exp = 0.0
    for t in range(L_tokens):
        val = tl.load(logits_ptr + h * L_tokens + t) * 1.0  # already scaled outside if needed
        # Here logits are raw; we need to apply scaling in host code before calling this kernel.
        # Since we pass sm_scale separately, we should compute scaled here if needed.
        # But for simplicity, we assume logits_ptr already contains scaled values. If not, host must pre-scale.
        # To keep correctness, host code will scale logits_ptr before this call.
        val_scaled = val  # already scaled
        sum_exp += tl.exp(val_scaled - m)
    lse_val = m + tl.log(sum_exp)
    tl.store(lse_out_ptr + h, lse_val)


@triton.jit
def compute_output_kernel(
    logits_ptr,   # [H, L_tokens], float32
    lse_ptr,      # [H], float32
    Kc_all_ptr,   # [L_tokens, CK] (we will index by tok_idx to get Kc[t, :] for each t)
    out_ptr,      # [H, CK], float32 accumulator
    H: tl.constexpr,
    CK: tl.constexpr,
    L_tokens: tl.constexpr,
):
    h = tl.program_id(0)
    t = tl.program_id(1)
    if h >= H or t >= L_tokens:
        return

    # Load softmax(h, t) = exp(logits[h, t] - lse[h])
    lse_h = tl.load(lse_ptr + h)
    logit_scaled = tl.load(logits_ptr + h * L_tokens + t)  # logits already scaled in host before calling
    soft = tl.exp(logit_scaled - lse_h)

    # Gather Kc[t, :]
    idx = t  # since tok_idx is 0..L_tokens-1 and contiguous in our setup? Not generally true; we need tok_idx.
    # We don't have tok_idx here; we can't gather Kc_row from Kc_all_ptr using idx directly.
    # Fix: we must pass tok_idx to this kernel. Let's redefine output kernel to take tok_idx_ptr.
    pass  # Placeholder to show structure; actual code will be below.


# Redefine kernels with tok_idx handling in compute_output_kernel


@triton.jit
def compute_output_kernel_with_idx(
    logits_ptr,   # [H, L_tokens], float32
    lse_ptr,      # [H], float32
    Kc_all_ptr,   # [P, CK], we will index by tok_idx[t]
    tok_idx_ptr,  # [L_tokens], int32
    out_ptr,      # [H, CK], float32 accumulator
    H: tl.constexpr,
    CK: tl.constexpr,
    L_tokens: tl.constexpr,
):
    h = tl.program_id(0)
    t = tl.program_id(1)
    if h >= H or t >= L_tokens:
        return

    # Load softmax(h, t)
    lse_h = tl.load(lse_ptr + h)
    logit_scaled = tl.load(logits_ptr + h * L_tokens + t)
    soft = tl.exp(logit_scaled - lse_h)

    # Load tok_idx[t] and gather Kc_row
    idx = tl.load(tok_idx_ptr + t)  # int32
    dim_offsets = tl.arange(0, CK)
    Kc_row = tl.load(Kc_all_ptr + idx * CK + dim_offsets)  # [CK], float32

    # Accumulate into out[h, :]
    out_row_ptr = out_ptr + h * CK  # vector base
    out_row = tl.load(out_row_ptr + dim_offsets)  # [CK], float32
    out_row += soft * Kc_row
    tl.store(out_row_ptr + dim_offsets, out_row)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        device = q_nope.device
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All tensors must be on CUDA for Triton."

        # Shapes
        batch_size, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]
        num_pages = ckv_cache.shape[0]
        P_ckv, _, _ = ckv_cache.shape
        P_kpe, _, _ = kpe_cache.shape
        assert P_ckv == num_pages and P_kpe == num_pages, "ckv_cache and kpe_cache first dim must equal num_pages."

        # Prepare outputs
        output = torch.empty((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        # Convert caches to float32 for compute; Triton will read float32
        Kc_all = ckv_cache.to(torch.float32).contiguous()  # [num_pages, 512]
        Kp_all = kpe_cache.to(torch.float32).contiguous()  # [num_pages, 64]

        # Flatten q_nope and q_pe to [B*H, D] for pointer slicing convenience
        qn_flat = q_nope.to(torch.float32).contiguous().view(batch_size * num_qo_heads, head_dim_ckv)  # [B*H, CK]
        qp_flat = q_pe.to(torch.float32).contiguous().view(batch_size * num_qo_heads, head_dim_kpe)  # [B*H, KP]

        for b in range(batch_size):
            # Compute L_tokens for this batch element
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            if L_tokens <= 0:
                output[b].zero_()
                lse[b] = -float("inf")
                continue

            tok_idx = kv_indices[kv_indptr[b]: kv_indptr[b + 1]].to(torch.int32).contiguous()  # [L_tokens]

            # Base pointers for qn and qp for this batch
            qn_batch_ptr = qn_flat[b * num_qo_heads: (b + 1) * num_qo_heads]  # [H, CK]
            qp_batch_ptr = qp_flat[b * num_qo_heads: (b + 1) * num_qo_heads]  # [H, KP]

            # Allocate logits buffer [H, L_tokens], float32
            logits = torch.empty((num_qo_heads, L_tokens), dtype=torch.float32, device=device)

            # Launch compute_logits_kernel: grid (H, L_tokens), vectorizing across tokens
            grid_logits = (num_qo_heads, L_tokens)
            compute_logits_kernel[grid_logits](
                qn_batch_ptr, qp_batch_ptr,
                Kc_all, Kp_all,
                tok_idx,
                logits,
                H=num_qo_heads,
                CK=head_dim_ckv,
                KP=head_dim_kpe,
                L_tokens=L_tokens,
                num_warps=4,
            )

            # Scale logits by sm_scale (host-side scaling to avoid extra kernel)
            # However, to keep Triton-only computation minimal, we can scale in-kernel or here.
            # We'll scale here in float32 to keep kernels simple.
            scaled_logits = logits * sm_scale

            # Compute lse per head using reduction kernel
            lse_b = torch.empty((num_qo_heads,), dtype=torch.float32, device=device)
            grid_lse = (num_qo_heads,)
            lse_kernel[grid_lse](
                scaled_logits,
                lse_b,
                H=num_qo_heads,
                L_tokens=L_tokens,
                num_warps=1,
            )

            # Initialize output accumulator to zeros (float32)
            out_accum = torch.zeros((num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)

            # Launch compute_output_kernel_with_idx: grid (H, L_tokens)
            grid_out = (num_qo_heads, L_tokens)
            compute_output_kernel_with_idx[grid_out](
                scaled_logits, lse_b,
                Kc_all,
                tok_idx,
                out_accum,
                H=num_qo_heads,
                CK=head_dim_ckv,
                L_tokens=L_tokens,
                num_warps=4,
            )

            # Store outputs: cast to bfloat16 and place in output[b]
            output[b] = out_accum.to(torch.bfloat16)
            lse[b] = lse_b

        return output, lse


def run(*args):
    return ModelNew()(*args)
