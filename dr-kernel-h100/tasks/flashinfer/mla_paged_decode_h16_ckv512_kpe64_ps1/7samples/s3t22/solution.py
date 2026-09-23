import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: compute base-2 logsumexp of a float32 vector x (length L), write to out[0]
# Assumes x_ptr points to a contiguous vector of length L, all float32.
@triton.jit
def compute_lse_kernel(x_ptr, out_ptr, L: tl.constexpr, inv_ln2: tl.float32, BLOCK_L: tl.constexpr):
    # One program instance: reduce over tokens in chunks
    m = -float("inf")
    for l_off in tl.static_range(0, L, BLOCK_L):
        idx = l_off + tl.arange(0, BLOCK_L)  # [BLOCK_L]
        mask = idx < L
        x = tl.load(x_ptr + idx, mask=mask, other=-float("inf"))
        # Reduce to max across the chunk
        chunk_max = -float("inf")
        for i in tl.static_range(0, BLOCK_L):
            vi = x[i]
            chunk_max = tl.maximum(chunk_max, vi)
        m = tl.maximum(m, chunk_max)

    # Compute sum(exp(x - m)) across all tokens
    sumexp = 0.0
    for l_off in tl.static_range(0, L, BLOCK_L):
        idx = l_off + tl.arange(0, BLOCK_L)
        mask = idx < L
        x = tl.load(x_ptr + idx, mask=mask, other=-float("inf"))
        sumexp += tl.sum(tl.exp(x - m))

    lse = m + tl.log(sumexp) * inv_ln2
    tl.store(out_ptr, lse)


# Triton kernel: compute softmax of a float32 vector x (length L), write to out[0..L-1]
@triton.jit
def compute_softmax_kernel(x_ptr, out_ptr, L: tl.constexpr, BLOCK_L: tl.constexpr):
    # Pass 1: max for numerical stability
    m = -float("inf")
    for l_off in tl.static_range(0, L, BLOCK_L):
        idx = l_off + tl.arange(0, BLOCK_L)
        mask = idx < L
        x = tl.load(x_ptr + idx, mask=mask, other=-float("inf"))
        chunk_max = -float("inf")
        for i in tl.static_range(0, BLOCK_L):
            vi = x[i]
            chunk_max = tl.maximum(chunk_max, vi)
        m = tl.maximum(m, chunk_max)

    # Pass 2: compute sum(exp(x - m))
    sumexp = 0.0
    for l_off in tl.static_range(0, L, BLOCK_L):
        idx = l_off + tl.arange(0, BLOCK_L)
        mask = idx < L
        x = tl.load(x_ptr + idx, mask=mask, other=-float("inf"))
        sumexp += tl.sum(tl.exp(x - m))

    inv_sumexp = 1.0 / sumexp

    # Pass 3: write softmax normalized values
    for l_off in tl.static_range(0, L, BLOCK_L):
        idx = l_off + tl.arange(0, BLOCK_L)
        mask = idx < L
        x = tl.load(x_ptr + idx, mask=mask, other=-float("inf"))
        # Compute softmax for each element in the chunk
        for i in tl.static_range(0, BLOCK_L):
            # For masked positions, skip by setting to 0 below; or rely on mask via compute
            pass  # We'll do per-element write using tl.store in a loop
    # Implement per-element write using chunked loops
    for l_off in tl.static_range(0, L, BLOCK_L):
        idx = l_off + tl.arange(0, BLOCK_L)
        mask = idx < L
        x = tl.load(x_ptr + idx, mask=mask, other=-float("inf"))
        y = tl.exp(x - m) * inv_sumexp
        # Store y to out_ptr with mask
        for i in tl.static_range(0, BLOCK_L):
            tl.store(out_ptr + (l_off + i), y[i], mask=(l_off + i) < L)


# Triton kernel: compute out[h, :] = attn @ Kc_b (Kc_b: [L, Dc], attn: [L], output vector [Dc])
@triton.jit
def compute_out_kernel(attn_ptr, Kc_ptr, out_ptr,
                       L: tl.constexpr, Dc: tl.constexpr, BLOCK_D: tl.constexpr):
    # One program instance; reduce over tokens in chunks
    out_vec = tl.zeros((Dc,), dtype=tl.float32)
    for l_off in tl.static_range(0, L, BLOCK_D):
        idx_l = l_off + tl.arange(0, BLOCK_D)  # [BLOCK_D], tokens
        mask_l = idx_l < L
        attn_chunk = tl.load(attn_ptr + idx_l, mask=mask_l, other=0.0)  # [BLOCK_D]
        # Accumulate over tokens in this chunk
        for i in tl.static_range(0, BLOCK_D):
            # For each token in chunk, add attn_chunk[i] * Kc[:, idx_l[i]] to out_vec
            l_val = idx_l[i]
            # Only proceed if l_val < L
            if l_val < L:
                k_ptr = Kc_ptr + l_val * Dc  # pointer to row l_val in Kc
                # Load Kc row slice of length Dc in chunks
                for d_off in tl.static_range(0, Dc, BLOCK_D):
                    idx_d = d_off + tl.arange(0, BLOCK_D)
                    mask_d = idx_d < Dc
                    k = tl.load(k_ptr + idx_d, mask=mask_d, other=0.0)  # [BLOCK_D]
                    out_vec += attn_chunk[i] * k
    tl.store(out_ptr, out_vec)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure device is CUDA; convert inputs to float32 for compute
        device = q_nope.device
        assert device.type == "cuda", "ModelNew requires CUDA device for Triton kernels"

        batch_size, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]
        num_pages = ckv_cache.shape[0]

        # Sanity checks (as in original)
        assert num_qo_heads == 16
        assert head_dim_ckv == 512
        assert head_dim_kpe == 64
        assert num_pages == 989669  # consistent with provided inputs

        # Derive per-batch token indices
        B = batch_size
        H = num_qo_heads
        Dc = head_dim_ckv
        Dp = head_dim_kpe

        # Compute L_b for each batch: len(kv_indptr) should be B+1
        L_vec = kv_indptr[1:] - kv_indptr[:-1]  # [B]
        L_tot = L_vec.sum().item()
        # Output tensors
        output = torch.empty((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        for b in range(batch_size):
            L_b = int(L_vec[b].item())
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            tok_idx = kv_indices[start:end].to(torch.long)  # [L_b]

            # Gather Kc and Kp for this batch
            Kc_b = ckv_cache[tok_idx].to(torch.float32)  # [L_b, Dc]
            Kp_b = kpe_cache[tok_idx].to(torch.float32)  # [L_b, Dp]

            # Precompute inv_ln2 for base-2 logsumexp
            inv_ln2 = 1.0 / math.log(2.0)

            # For each head h
            for h in range(H):
                # Compute qn and qp (float32), vectors
                qn = q_nope[b, h].to(torch.float32).contiguous()  # [Dc]
                qp = q_pe[b, h].to(torch.float32).contiguous()   # [Dp]

                # Compute logits vector in PyTorch: logits[l] = qn @ Kc_b[l] + qp @ Kp_b[l]
                # We'll use vectorized PyTorch ops for this small vector; Triton can do this too if desired,
                # but this keeps kernels minimal and avoids dynamic indexing issues.
                logits = (qn.unsqueeze(1) * Kc_b).sum(dim=1) + (qp.unsqueeze(1) * Kp_b).sum(dim=1)  # [L_b]
                # Scale logits by sm_scale (positional scalar)
                logits_scaled = logits * sm_scale

                # lse[h] = logsumexp(logits_scaled, base=2)
                lse_scalar = torch.empty((), dtype=torch.float32, device=device)
                compute_lse_kernel[(1,)](logits_scaled, lse_scalar, L_b, inv_ln2, 1024)  # BLOCK_L=1024
                lse[b, h] = lse_scalar  # keep as tensor to avoid host sync

                # Softmax of scaled logits: attn[l] = softmax(logits_scaled[l])
                attn = torch.empty((L_b,), dtype=torch.float32, device=device)
                compute_softmax_kernel[(1,)](logits_scaled, attn, L_b, 1024)

                # Compute out[h, :] = attn @ Kc_b using Triton reduction kernel over L_b
                out_vec = torch.empty((Dc,), dtype=torch.float32, device=device)
                compute_out_kernel[(1,)](attn, Kc_b, out_vec, L_b, Dc, 128)

                # Store to output
                output[b, h] = out_vec.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
