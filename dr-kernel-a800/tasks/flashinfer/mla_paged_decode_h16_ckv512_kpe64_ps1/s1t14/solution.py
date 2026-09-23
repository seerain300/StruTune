import math
import torch
import triton
import triton.language as tl


@triton.jit
def gather_rows_c_kernel(cache_ptr, tok_idx_ptr, out_ptr,
                          num_tokens: tl.int32, Dc: tl.constexpr):
    # Each program handles one token row: copy cache[row, :] into out[i*Dc:(i+1)*Dc]
    pid = tl.program_id(0)
    if pid >= num_tokens:
        return
    idx = tl.load(tok_idx_ptr + pid).to(tl.int32)
    base_in = idx * Dc
    for k in range(0, Dc):
        val = tl.load(cache_ptr + base_in + k)
        tl.store(out_ptr + pid * Dc + k, val)


@triton.jit
def gather_rows_p_kernel(cache_ptr, tok_idx_ptr, out_ptr,
                          num_tokens: tl.int32, Dp: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= num_tokens:
        return
    idx = tl.load(tok_idx_ptr + pid).to(tl.int32)
    base_in = idx * Dp
    for k in range(0, Dp):
        val = tl.load(cache_ptr + base_in + k)
        tl.store(out_ptr + pid * Dp + k, val)


@triton.jit
def softmax_kernel(x_ptr, out_ptr, L: tl.int32, H: tl.constexpr):
    # Compute softmax for each row of length L across the whole x_ptr (size H*L).
    # One program per head i; loop over tokens t.
    i = tl.program_id(0)
    if i >= H:
        return
    row_base = i * L
    m = -float("inf")
    # Pass 1: find max
    for t in range(0, L):
        m = tl.maximum(m, tl.load(x_ptr + row_base + t))
    # Pass 2: compute sum of exp(x - m)
    sum_exp = 0.0
    for t in range(0, L):
        val = tl.load(x_ptr + row_base + t)
        sum_exp += tl.exp(val - m)
    inv_sum = 1.0 / sum_exp
    # Pass 3: write normalized softmax
    for t in range(0, L):
        val = tl.load(x_ptr + row_base + t)
        tl.store(out_ptr + row_base + t, tl.exp(val - m) * inv_sum)


@triton.jit
def lse_base2_kernel(x_ptr, out_ptr, L: tl.int32, H: tl.constexpr):
    # Compute logsumexp base-2 for each row: out[i] = log(sum(exp(x_i))) / ln(2).
    i = tl.program_id(0)
    if i >= H:
        return
    row_base = i * L
    m = -float("inf")
    # Pass 1: find max
    for t in range(0, L):
        m = tl.maximum(m, tl.load(x_ptr + row_base + t))
    # Pass 2: compute sum of exp(x - m)
    sum_exp = 0.0
    for t in range(0, L):
        val = tl.load(x_ptr + row_base + t)
        sum_exp += tl.exp(val - m)
    lse = tl.log(sum_exp) * 1.4426950408889634  # 1 / ln(2)
    tl.store(out_ptr + i, lse)


@triton.jit
def gemv_kernel(a_ptr, b_ptr, out_ptr,
                M: tl.int32, K: tl.int32, BLOCK_K: tl.constexpr):
    # Compute out[j] = sum_k a[j, k] * b[k], where a is [M, K], b is [K], out is [M].
    j = tl.program_id(0)
    if j >= M:
        return
    acc = 0.0
    for k0 in range(0, K, BLOCK_K):
        offs = k0 + tl.arange(0, BLOCK_K)
        mask = offs < K
        a = tl.load(a_ptr + j * K + offs, mask=mask, other=0.0)     # [BLOCK_K]
        b = tl.load(b_ptr + offs, mask=mask, other=0.0)             # [BLOCK_K]
        acc += tl.sum(a * b, axis=0)
    tl.store(out_ptr + j, acc)


@triton.jit
def matvec_kernel(attn_ptr, K_ptr, out_ptr,
                   H: tl.constexpr, Dc: tl.constexpr, L: tl.int32, BLOCK_D: tl.constexpr):
    # One program computes out[i] for a given head i
    i = tl.program_id(0)
    if i >= H:
        return
    acc = tl.zeros((Dc,), dtype=tl.float32)
    # Loop over tokens in chunks
    for t0 in range(0, L, BLOCK_D):
        offs = t0 + tl.arange(0, BLOCK_D)
        mask = offs < L
        attn_chunk = tl.load(attn_ptr + i * L + offs, mask=mask, other=0.0)  # [BLOCK_D]
        K_chunk = tl.load(K_ptr + offs * Dc + tl.arange(0, Dc), mask=mask[:, None], other=0.0)  # [BLOCK_D, Dc]
        # Reduce across tokens for each dimension: [BLOCK_D, Dc] -> [Dc]
        prod = attn_chunk[:, None] * K_chunk                    # [BLOCK_D, Dc]
        acc += tl.sum(prod, axis=0)
    tl.store(out_ptr + i * Dc + tl.arange(0, Dc), acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Shapes
        B = q_nope.shape[0]
        H = q_nope.shape[1]
        Dc = q_nope.shape[2]
        Dp = q_pe.shape[2]
        # Squeeze the size-1 cache dim
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [P, Dc]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [P, Dp]

        # Output and lse buffers
        output = torch.empty((B, H, Dc), dtype=torch.bfloat16, device=q_nope.device)
        lse = torch.empty((B, H), dtype=torch.float32, device=q_nope.device)

        # Precompute inv_ln2 for base-2 logsumexp
        inv_ln2 = 1.4426950408889634  # 1 / ln(2)

        for b in range(B):
            # Number of tokens for this batch element
            L = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            if L <= 0:
                # No tokens, zero outputs and lse
                for i in range(H):
                    output[b, i] = torch.zeros((Dc,), dtype=torch.bfloat16, device=q_nope.device)
                lse[b] = torch.zeros((H,), dtype=torch.float32, device=q_nope.device)
                continue

            # Gather token indices for this batch element
            tok_idx = kv_indices[int(kv_indptr[b].item()):int(kv_indptr[b + 1].item())].contiguous()  # [L]

            # 1) Gather rows from caches into Kc_flat and Kp_flat (float32)
            Kc_flat = torch.empty((L * Dc,), dtype=torch.float32, device=q_nope.device)
            Kp_flat = torch.empty((L * Dp,), dtype=torch.float32, device=q_nope.device)

            grid_gather = (L,)
            gather_rows_c_kernel[grid_gather](Kc_all, tok_idx, Kc_flat, L, Dc)
            Kc = Kc_flat.view(L, Dc)  # [L, Dc]

            gather_rows_p_kernel[grid_gather](Kp_all, tok_idx, Kp_flat, L, Dp)
            Kp = Kp_flat.view(L, Dp)  # [L, Dp]

            # For each head i: compute logits_scaled per token, softmax, and output
            for i in range(H):
                # qn[i] and qp[i]
                qn = q_nope[b, i].to(torch.float32).contiguous()  # [Dc]
                qp = q_pe[b, i].to(torch.float32).contiguous()   # [Dp]

                # GEMV for qn @ Kc.T -> [L] and qp @ Kp.T -> [L]
                attn_qn = torch.empty((L,), dtype=torch.float32, device=q_nope.device)
                gemv_kernel[(L,)](qn, Kc, attn_qn, Dc, L, 128)

                attn_qp = torch.empty((L,), dtype=torch.float32, device=q_nope.device)
                gemv_kernel[(L,)](qp, Kp, attn_qp, Dp, L, 128)

                attn = attn_qn + attn_qp  # [L]

                # Scale
                logits_scaled = attn * sm_scale  # [L]

                # 2) Softmax in Triton: attn is stored in logits_scaled_ptr for Triton kernel
                # Create a buffer for attention weights: [H, L], but we only use i-th row
                attn_buf = torch.empty((H, L), dtype=torch.float32, device=q_nope.device)
                attn_buf[i] = logits_scaled  # fill only the i-th row
                attn_out = torch.empty((H, L), dtype=torch.float32, device=q_nope.device)
                softmax_kernel[(H,)](attn_buf.view(-1), attn_out.view(-1), L, H)
                attn_i = attn_out[i]  # [L]

                # 3) Compute out_vec[i] = attn_i @ Kc using Triton matvec
                out_vec_flat = torch.empty((Dc,), dtype=torch.float32, device=q_nope.device)
                matvec_kernel[(1,)](attn_i, Kc, out_vec_flat, H, Dc, L, 128)

                # Store to output[b, i, :] as bfloat16
                output[b, i] = out_vec_flat.to(torch.bfloat16)

                # 4) lse (base-2) for head i
                # Compute lse per head using Triton lse_base2_kernel
                lse_i = torch.empty((1,), dtype=torch.float32, device=q_nope.device)
                lse_base2_kernel[(1,)](logits_scaled, lse_i, L, H)
                lse[b, i] = lse_i[0]

        return output, lse


def run(*args):
    return ModelNew()(*args)
