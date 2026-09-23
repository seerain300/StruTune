import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: compute base-2 logsumexp for a given (b, h) vector scale_logits.
# scale_logits_ptr: pointer to float32 vector of length L_b
# lse_ptr: pointer to output scalar float32 at lse[b*H + h]
# L_b: number of tokens (runtime passed as tl.constexpr)
@triton.jit
def compute_lse_base2_kernel(scale_logits_ptr, lse_ptr, L_b: tl.constexpr):
    # This kernel runs in a single program instance; grid is (1,) and we index by global offset.
    # We read the vector via pointer arithmetic in static loops.
    # 1) compute max for numerical stability
    max_val = -float('inf')
    for i in tl.static_range(0, L_b):
        val = tl.load(scale_logits_ptr + i)
        if val > max_val:
            max_val = val
    # 2) compute sumexp
    sumexp = 0.0
    for i in tl.static_range(0, L_b):
        val = tl.load(scale_logits_ptr + i)
        sumexp += tl.exp(val - max_val)
    # 3) compute lse = log(sumexp) / ln(2)
    ln2 = 0.6931471805599453  # math.log(2.0)
    lse_val = tl.log(sumexp) / ln2
    tl.store(lse_ptr, lse_val)


# Triton kernel: compute softmax of scale_logits into attn[b*H + h, :] for given L_b
# scale_logits_ptr: input pointer to float32 vector of length L_b
# lse_ptr: pointer to precomputed lse[b*H + h] as float32
# attn_ptr: output pointer to float32 vector of length L_b
# L_b: tl.constexpr
@triton.jit
def compute_softmax_kernel(scale_logits_ptr, lse_ptr, attn_ptr, L_b: tl.constexpr):
    ln2 = 0.6931471805599453
    lse_val = tl.load(lse_ptr)  # scalar
    # Write attn = exp(scale_logits - lse) / sum(exp(scale_logits - lse))
    sumexp = 0.0
    for i in tl.static_range(0, L_b):
        val = tl.load(scale_logits_ptr + i)
        sumexp += tl.exp(val - lse_val)
    for i in tl.static_range(0, L_b):
        val = tl.load(scale_logits_ptr + i)
        attn_i = tl.exp(val - lse_val) / sumexp
        tl.store(attn_ptr + i, attn_i)


# Triton kernel: compute out[b, h, :] = attn[b, h, :] @ Kc_b where Kc_b is the gathered Kc for this (b, h)
# attn_ptr: pointer to float32 vector attn[b*H + h, L_b]
# Kc_ptr: pointer to float32 matrix Kc_b with shape [L_b, Dc]
# out_ptr: pointer to float32 vector output[b*H + h, Dc]
# L_b: tl.constexpr (number of tokens)
# Dc: tl.constexpr (hidden dimension)
@triton.jit
def compute_out_kernel(attn_ptr, Kc_ptr, out_ptr,
                       L_b: tl.constexpr, Dc: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_T: tl.constexpr):
    # Accumulate output vector of length Dc
    acc = tl.zeros((Dc,), dtype=tl.float32)
    for t0 in tl.static_range(0, L_b, BLOCK_T):
        # Load a chunk of attn values: vector of length BLOCK_T
        t_idx = t0 + tl.arange(0, BLOCK_T)
        mask_t = t_idx < L_b
        attn_vec = tl.load(attn_ptr + t_idx, mask=mask_t, other=0.0)  # [BLOCK_T]
        # For each dim in Dc, accumulate attn_vec[k] * Kc[k, d] across k
        for d0 in tl.static_range(0, Dc, BLOCK_D):
            d_idx = d0 + tl.arange(0, BLOCK_D)  # [BLOCK_D]
            mask_d = d_idx < Dc
            # Compute contribution for this (d0, t0) block: acc[d] += sum_k attn_vec[k] * Kc[k, d]
            # We'll loop k across BLOCK_T and map to 2D addressing by constructing pointers.
            # Note: Triton supports 2D pointer arithmetic via row-major addressing.
            # We need Kc[k, d] where k in [t0, t0+BLOCK_T), d in [d0, d0+BLOCK_D).
            # For each k, load Kc[k, d] vector across d, multiply with attn_vec[k], and accumulate.
            # To do this, we use nested loop over k in the block and add contributions to acc.
            for k in tl.static_range(0, BLOCK_T):
                k_idx = t0 + k
                k_valid = k_idx < L_b
                # attn_k = attn_vec[k] if valid else 0
                attn_k = tl.load(attn_ptr + k_idx, mask=k_valid, other=0.0)
                # Load Kc[k, d0:d0+BLOCK_D] vector
                # Address: Kc_ptr + k_idx * Dc + d_idx
                # Note: Kc_ptr points to a 2D matrix of shape [L_b, Dc] (contiguous).
                Kc_vec = tl.load(Kc_ptr + k_idx * Dc + d_idx, mask=mask_d, other=0.0)  # [BLOCK_D]
                # Accumulate: acc[d] += attn_k * Kc_vec[d] for all d in block
                # We do this by adding to each acc element indexed by d_idx.
                # Since Triton does not allow vectorized update across d directly here, we loop over d in the block.
                for dd in tl.static_range(0, BLOCK_D):
                    d_pos = d0 + dd
                    d_valid = d_pos < Dc
                    contrib = attn_k * tl.load(Kc_ptr + k_idx * Dc + (d0 + dd), mask=d_valid, other=0.0)
                    acc[d_pos] += contrib
    # Store the accumulated output
    for d in tl.static_range(0, Dc):
        tl.store(out_ptr + d, acc[d])


def _compute_output_triton(qn_f32, qp_f32, Kc_b_f32, Kp_b_f32, sm_scale, batch_size, num_qo_heads, device):
    """
    Compute outputs for a single batch element using Triton kernels.
    Returns:
      output: [num_qo_heads, Dc] float32
      lse: [num_qo_heads] float32
    """
    Dc = Kc_b_f32.shape[1]
    L_b = Kc_b_f32.shape[0]

    # Prepare scale logits buffer: [num_qo_heads, L_b] float32
    scale_logits = torch.empty((num_qo_heads, L_b), dtype=torch.float32, device=device)

    # Output buffers
    out_b = torch.empty((num_qo_heads, Dc), dtype=torch.float32, device=device)
    lse_b = torch.empty((num_qo_heads,), dtype=torch.float32, device=device)

    # Launch Triton kernels per (b, h)
    for h in range(num_qo_heads):
        qn = qn_f32[h]  # [Dc]
        qp = qp_f32[h]  # [Dp]
        # Compute logits (PyTorch) and scale: [L_b]
        logits = torch.matmul(qn, Kc_b_f32.T) + torch.matmul(qp, Kp_b_f32.T)  # [L_b]
        scale_logits[h] = logits * sm_scale

        # Launch lse kernel for this (b, h)
        lse_offset = batch_size * num_qo_heads + h  # but we only need 1 batch element, so lse_b[h] is fine
        grid = (1,)
        compute_lse_base2_kernel[grid](scale_logits[h], lse_b[h], L_b=L_b)

        # Launch softmax kernel for this (b, h)
        attn_vec = torch.empty((L_b,), dtype=torch.float32, device=device)
        compute_softmax_kernel[grid](scale_logits[h], lse_b[h], attn_vec, L_b=L_b)

        # Launch out matmul kernel for this (b, h)
        compute_out_kernel[grid](attn_vec, Kc_b_f32, out_b[h], L_b=L_b, Dc=Dc, BLOCK_D=128, BLOCK_T=64)

    return out_b, lse_b


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        """
        Triton-only implementation:
        - Computes logits in PyTorch for robustness, then uses Triton kernels for lse, softmax, and final out.
        - Ensures sm_scale is passed to Triton kernels as a parameter to avoid 'unrecognised keyword' errors.
        """
        assert TRITON_AVAILABLE, "Triton is not available"
        batch_size = q_nope.shape[0]
        num_qo_heads = q_nope.shape[1]
        device = q_nope.device

        # Ensure dtype is float32 for compute, bfloat16 for final output
        q_nope_f32 = q_nope.to(torch.float32)
        q_pe_f32 = q_pe.to(torch.float32)

        # Process each batch element
        output = torch.empty((batch_size, num_qo_heads, 512), dtype=torch.float32, device=device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        for b in range(batch_size):
            # Compute token indices for this batch
            # Note: kv_indptr is [B+1]; kv_indices is [L_tot]
            # For b=0..B-1, tokens = kv_indices[kv_indptr[b]:kv_indptr[b+1]]
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            L_b = end - start
            if L_b <= 0:
                # No tokens for this batch element
                lse[b].zero_()
                continue

            tok_idx = kv_indices[start:end]  # [L_b], int32
            # Gather Kc and Kp for this batch: ckv_cache[tok_idx, 0], kpe_cache[tok_idx, 0]
            Kc_b = ckv_cache[tok_idx, 0].to(torch.float32)  # [L_b, 512]
            Kp_b = kpe_cache[tok_idx, 0].to(torch.float32)  # [L_b, 64]

            # Compute outputs for this batch using Triton kernels
            out_b, lse_b = _compute_output_triton(q_nope_f32[b], q_pe_f32[b], Kc_b, Kp_b, sm_scale, batch_size, num_qo_heads, device)
            output[b] = out_b
            lse[b] = lse_b

        # Cast final output to bfloat16 to match original behavior
        output = output.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
