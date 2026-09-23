import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: compute logits = qn @ Kc.T + qp @ Kp.T for a single head h
# Inputs:
#   qn_ptr: [Hc] float32 (row vector for this head)
#   qp_ptr: [Hp] float32 (row vector for this head)
#   Kc_ptr: [L, Hc] float32
#   Kp_ptr: [L, Hp] float32
#   out_ptr: [L] float32 (logits)
# Params:
#   Hc, Hp, L: int32 constexpr
#   sm_scale: float32
#   BLOCK_K: tl.constexpr
@triton.jit
def matmul_add_row_kernel(
    qn_ptr, qp_ptr, Kc_ptr, Kp_ptr, out_ptr,
    Hc: tl.constexpr, Hp: tl.constexpr, L: tl.constexpr,
    sm_scale: tl.constexpr,
    BLOCK_K: tl.constexpr
):
    # Initialize output vector
    offs = tl.arange(0, BLOCK_K)
    # Accumulator for logits
    logits = tl.zeros((L,), dtype=tl.float32)
    # Loop over Kc/Kp in chunks
    for k in range(0, L, BLOCK_K):
        k_idx = k + offs
        mask = k_idx < L
        # Load qn and qp slices (scalar per k since Hc/Hp are constexpr; we use vectorized loads by 1-element per k)
        # We iterate k and load per element, which is fine for Triton constexpr loops.
        # For Kc: load row k_idx, all Hc columns; for Kp: load row k_idx, all Hp columns
        # Accumulate: qn[j] * Kc[k_idx, j] + qp[i] * Kp[k_idx, i]
        # We implement two loops over j and i respectively.
        acc1 = tl.zeros((), dtype=tl.float32)  # dot(qn, Kc[k_idx, :])
        acc2 = tl.zeros((), dtype=tl.float32)  # dot(qp, Kp[k_idx, :])
        # Loop over j in Hc and accumulate qn[j] * Kc[k_idx, j]
        for j in range(0, Hc):
            qnj = tl.load(qn_ptr + j)
            kc_j = tl.load(Kc_ptr + k_idx * Hc + j, mask=mask, other=0.0)
            acc1 += qnj * kc_j
        # Loop over i in Hp and accumulate qp[i] * Kp[k_idx, i]
        for i in range(0, Hp):
            qpj = tl.load(qp_ptr + i)
            kp_i = tl.load(Kp_ptr + k_idx * Hp + i, mask=mask, other=0.0)
            acc2 += qpj * kp_i
        # Store partial sum
        # We want to store acc1 + acc2 at position k_idx. Use masked assignment.
        # Triton doesn't have direct scatter; we can assign per index via loop, or keep a vector and store chunk.
        # Here, we reconstruct vector with zeros and add at k_idx via loop (BLOCK_K is small).
        # However, since we have a vector 'logits' initialized, we can add acc1+acc2 for each k.
        # But Triton requires static loop. Compute contribution vector and add.
        contrib = tl.zeros((L,), dtype=tl.float32)
        # For each kk in chunk, set logits[kk] += acc1+acc2
        for kk in range(0, BLOCK_K):
            kk_idx = k + kk
            if kk_idx < L:
                contrib[kk_idx] = acc1 + acc2
        logits += contrib
    # Scale
    logits = logits * sm_scale
    # Store results
    for k in range(0, L):
        tl.store(out_ptr + k, logits[k])


# Triton kernel: compute lse = logsumexp(x) / log(2) for a single row (logits) of length L
# Pass 1: compute max
# Pass 2: compute sum of exp(x - max)
# Output: lse written to out_lse_ptr[0]
@triton.jit
def softmax_logsumexp_row_kernel(
    x_ptr, out_lse_ptr,
    L: tl.constexpr,
    inv_log2: tl.constexpr  # 1 / log(2)
):
    # Pass 1: compute max
    max_val = -1.0e20
    for i in range(0, L):
        vi = tl.load(x_ptr + i)
        if vi > max_val:
            max_val = vi
    # Pass 2: sum of exp(x - max)
    sum_exp = 0.0
    for i in range(0, L):
        vi = tl.load(x_ptr + i)
        sum_exp += tl.exp(vi - max_val)
    lse_val = tl.log(sum_exp) * inv_log2
    tl.store(out_lse_ptr, lse_val)


# Triton kernel: compute out_row = softmax(x) @ Kc, where x is logits_scaled of length L and Kc is [L, Hc]
# We vectorize across output columns (Hc) in chunks of BLOCK_N and accumulate over tokens.
# x_ptr: [L] float32 (logits_scaled)
# Kc_ptr: [L, Hc] float32
# out_ptr: [Hc] float32
@triton.jit
def matvec_row_kernel(
    x_ptr, Kc_ptr, out_ptr,
    L: tl.constexpr, Hc: tl.constexpr, BLOCK_N: tl.constexpr
):
    # First pass: compute max of x for numerical stability
    max_val = -1.0e20
    for i in range(0, L):
        vi = tl.load(x_ptr + i)
        if vi > max_val:
            max_val = vi
    # Second pass: compute sum of exp(x - max)
    sum_exp = 0.0
    for i in range(0, L):
        vi = tl.load(x_ptr + i)
        sum_exp += tl.exp(vi - max_val)
    # Third pass: compute probabilities and accumulate into out
    for n in range(0, Hc, BLOCK_N):
        offs = n + tl.arange(0, BLOCK_N)
        mask_n = offs < Hc
        acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
        # Loop over tokens to accumulate dot product with Kc
        for i in range(0, L):
            pi = tl.exp(tl.load(x_ptr + i) - max_val) / sum_exp  # scalar
            row_ptr = Kc_ptr + i * Hc
            k_vec = tl.load(row_ptr + offs, mask=mask_n, other=0.0)
            acc += pi * k_vec
        tl.store(out_ptr + offs, acc, mask=mask_n)


# Entry point ModelNew
class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Shapes
        batch_size, num_qo_heads, head_dim_ckv = q_nope.shape
        _, _, head_dim_kpe = q_pe.shape
        device = q_nope.device
        # Prepare K_all by squeezing cache dimension (as in original)
        Kc_all = ckv_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, Hc]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, Hp]

        # Output and lse
        output = torch.empty((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        for b in range(batch_size):
            # Compute token indices for this batch element
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            L_tokens = page_end - page_beg
            if L_tokens <= 0:
                output[b].zero_()
                lse[b] = -float("inf")
                continue

            tok_idx = kv_indices[page_beg:page_end].to(torch.int32).contiguous()  # [L_tokens]
            # Gather Kc and Kp for this batch element
            Kc = Kc_all[tok_idx].contiguous()  # [L_tokens, Hc]
            Kp = Kp_all[tok_idx].contiguous()  # [L_tokens, Hp]

            # Per-head loop
            for h in range(num_qo_heads):
                # Load qn[h] and qp[h]
                qn = q_nope[b, h, :].to(torch.float32).contiguous()  # [Hc]
                qp = q_pe[b, h, :].to(torch.float32).contiguous()   # [Hp]

                # Buffer for logits
                logits = torch.empty((L_tokens,), dtype=torch.float32, device=device)

                # Kernel 1: compute logits = qn @ Kc.T + qp @ Kp.T
                # Choose BLOCK_K (constexpr); loop handles L_tokens by chunks
                BLOCK_K = 64
                matmul_add_row_kernel[(1,)](
                    qn, qp, Kc, Kp, logits,
                    Hc=head_dim_ckv, Hp=head_dim_kpe, L=L_tokens,
                    sm_scale=sm_scale,
                    BLOCK_K=BLOCK_K,
                    num_warps=2, num_stages=2
                )

                # Kernel 2: compute lse = logsumexp(logits_scaled) / log(2)
                lse_kernel_out = torch.empty((1,), dtype=torch.float32, device=device)
                inv_log2 = 1.0 / math.log(2.0)
                softmax_logsumexp_row_kernel[(1,)](
                    logits, lse_kernel_out,
                    L=L_tokens,
                    inv_log2=inv_log2,
                    num_warps=1, num_stages=1
                )
                lse[b, h] = lse_kernel_out[0]

                # Kernel 3: compute output[b, h, :] = softmax(logits_scaled) @ Kc
                BLOCK_N = 128
                out_row = torch.empty((head_dim_ckv,), dtype=torch.float32, device=device)
                matvec_row_kernel[(triton.cdiv(head_dim_ckv, BLOCK_N),)](
                    logits, Kc, out_row,
                    L=L_tokens, Hc=head_dim_ckv, BLOCK_N=BLOCK_N,
                    num_warps=4, num_stages=2
                )
                output[b, h, :] = out_row

        # Return output (bfloat16) and lse (float32), matching original behavior
        return output.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
