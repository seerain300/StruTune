import math
import torch
import triton
import triton.language as tl


@triton.jit
def gather_rows_c_kernel(cache_ptr, tok_idx_ptr, out_ptr,
                          L_tokens: tl.constexpr, Dc: tl.constexpr):
    # Each program handles one token row for CKV
    pid = tl.program_id(0)
    if pid >= L_tokens:
        return
    idx = tl.load(tok_idx_ptr + pid).to(tl.int32)
    base_in = idx * Dc
    # Copy the row of length Dc into out_ptr at offset pid*Dc
    for k in range(0, Dc):
        val = tl.load(cache_ptr + base_in + k)
        tl.store(out_ptr + pid * Dc + k, val)


@triton.jit
def gather_rows_p_kernel(cache_ptr, tok_idx_ptr, out_ptr,
                          L_tokens: tl.constexpr, Dp: tl.constexpr):
    # Each program handles one token row for KPE
    pid = tl.program_id(0)
    if pid >= L_tokens:
        return
    idx = tl.load(tok_idx_ptr + pid).to(tl.int32)
    base_in = idx * Dp
    for k in range(0, Dp):
        val = tl.load(cache_ptr + base_in + k)
        tl.store(out_ptr + pid * Dp + k, val)


@triton.jit
def lse_base2_row_kernel(logits_ptr, lse_ptr, L: tl.constexpr, scale: tl.float32):
    # One program per head; compute lse for that head in base-2
    i = tl.program_id(0)
    m = -float("inf")
    sum_exp = 0.0
    # Pass 1: find max over the row
    for t in range(0, L):
        val = tl.load(logits_ptr + i * L + t)
        m = tl.maximum(m, val)
    # Pass 2: compute sum of exp(logits - m)
    for t in range(0, L):
        val = tl.load(logits_ptr + i * L + t)
        sum_exp += tl.exp(val - m)
    lse_val = tl.log(sum_exp) + m  # logsumexp in natural log
    # Scale by log(2)
    lse_val = lse_val / math.log(2.0)
    tl.store(lse_ptr + i, lse_val)


@triton.jit
def softmax_row_kernel(logits_ptr, attn_ptr, L: tl.constexpr):
    # One program per head; compute softmax over L tokens into attn
    i = tl.program_id(0)
    m = -float("inf")
    for t in range(0, L):
        val = tl.load(logits_ptr + i * L + t)
        m = tl.maximum(m, val)
    sum_exp = 0.0
    for t in range(0, L):
        val = tl.load(logits_ptr + i * L + t)
        sum_exp += tl.exp(val - m)
    denom = sum_exp
    for t in range(0, L):
        val = tl.load(logits_ptr + i * L + t)
        p = tl.exp(val - m) / denom
        tl.store(attn_ptr + i * L + t, p)


@triton.jit
def matvec_kernel(attn_ptr, Kc_ptr, out_ptr,
                  H: tl.constexpr, Dc: tl.constexpr, L: tl.constexpr,
                  BLOCK_D: tl.constexpr):
    # One program per head (H programs), compute out_vec[i] = attn[i, :] @ Kc
    i = tl.program_id(0)
    if i >= H:
        return
    acc = tl.zeros((Dc,), dtype=tl.float32)
    for t0 in range(0, L, BLOCK_D):
        offs = t0 + tl.arange(0, BLOCK_D)
        mask = offs < L
        attn_slice = tl.load(attn_ptr + i * L + offs, mask=mask, other=0.0)  # [BLOCK_D]
        # Kc rows for these tokens: shape [BLOCK_D, Dc]
        k_ptrs = Kc_ptr + offs[:, None] * Dc + tl.arange(0, Dc)[None, :]
        k_vals = tl.load(k_ptrs, mask=mask[:, None], other=0.0)              # [BLOCK_D, Dc]
        # Accumulate dot products
        for kk in range(0, BLOCK_D):
            if not mask[kk]:
                continue
            a = attn_slice[kk]  # scalar
            k_row = k_vals[kk, :]  # [Dc]
            acc += a * k_row
    out_base = i * Dc
    for d in range(0, Dc):
        tl.store(out_ptr + out_base + d, acc[d])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Shapes
        batch_size = q_nope.shape[0]
        num_qo_heads = q_nope.shape[1]
        head_dim_ckv = q_nope.shape[2]
        head_dim_kpe = q_pe.shape[2]
        num_pages = ckv_cache.shape[0]
        device = q_nope.device

        # Constants
        assert num_qo_heads == 16
        assert head_dim_ckv == 512
        assert head_dim_kpe == 64

        # Prepare caches: squeeze size-1 dim and cast to float32 for compute
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, Dc]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, Dp]

        # Output buffers
        output = torch.empty((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        for b in range(batch_size):
            # Derive number of tokens
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            if L_tokens <= 0:
                for i in range(num_qo_heads):
                    output[b, i] = torch.zeros((head_dim_ckv,), dtype=torch.bfloat16, device=device)
                lse[b] = torch.zeros((num_qo_heads,), dtype=torch.float32, device=device)
                continue

            # Gather token indices for this batch element
            tok_idx = kv_indices[int(kv_indptr[b].item()):int(kv_indptr[b + 1].item())].contiguous()  # [L_tokens]

            # 1) Gather rows from caches into Kc_flat and Kp_flat (float32)
            Kc_flat = torch.empty((L_tokens * head_dim_ckv,), dtype=torch.float32, device=device)
            Kp_flat = torch.empty((L_tokens * head_dim_kpe,), dtype=torch.float32, device=device)

            grid_gather = (L_tokens,)
            gather_rows_c_kernel[grid_gather](Kc_all, tok_idx, Kc_flat, L_tokens, head_dim_ckv)
            Kc = Kc_flat.view(L_tokens, head_dim_ckv)  # [L_tokens, Dc]

            gather_rows_p_kernel[grid_gather](Kp_all, tok_idx, Kp_flat, L_tokens, head_dim_kpe)
            Kp = Kp_flat.view(L_tokens, head_dim_kpe)  # [L_tokens, Dp]

            # 2) For each head i: compute logits_scaled = qn[i] @ Kc.T + qp[i] @ Kp.T
            for i in range(num_qo_heads):
                qn = q_nope[b, i].to(torch.float32).contiguous()  # [Dc]
                qp = q_pe[b, i].to(torch.float32).contiguous()   # [Dp]

                # GEMV: qn @ Kc.T -> [1, L_tokens]
                attn = torch.empty((L_tokens,), dtype=torch.float32, device=device)
                gemv_kernel[(1,)](qn, Kc, attn, Dc=head_dim_ckv, L=L_tokens)

                # GEMV: qp @ Kp.T -> [1, L_tokens]
                attn_p = torch.empty((L_tokens,), dtype=torch.float32, device=device)
                gemv_kernel[(1,)](qp, Kp, attn_p, Dc=head_dim_kpe, L=L_tokens)

                logits = attn[0] + attn_p[0]  # [L_tokens]
                logits_scaled = logits * sm_scale  # [L_tokens]

                # 3) Compute lse per head in base-2 via Triton
                lse[b, i] = torch.zeros((), dtype=torch.float32, device=device)
                lse_base2_row_kernel[(1,)](logits_scaled, lse[b, i], L_tokens, sm_scale)

                # 4) Compute attention weights via Triton softmax over L_tokens
                attn_out = torch.empty((L_tokens,), dtype=torch.float32, device=device)
                softmax_row_kernel[(1,)](logits_scaled, attn_out, L_tokens)

                # 5) Final projection: attn_out @ Kc -> [Dc], use Triton matvec
                out_vec = torch.empty((head_dim_ckv,), dtype=torch.float32, device=device)
                matvec_kernel[(1,)](attn_out, Kc.contiguous().view(-1), out_vec,
                                    H=1, Dc=head_dim_ckv, L=L_tokens, BLOCK_D=128)

                # Store to output[b, i] as bfloat16
                output[b, i] = out_vec.to(torch.bfloat16)

        return output, lse


# Triton GEMV kernel for demonstration: computes y = x @ A^T (x: [D], A: [L, D], y: [L])
@triton.jit
def gemv_kernel(x_ptr, A_ptr, y_ptr,
                 D: tl.constexpr, L: tl.constexpr):
    # One program computes one output vector of length L
    for t0 in range(0, L, 128):
        offs = t0 + tl.arange(0, 128)
        mask = offs < L
        acc = tl.zeros((128,), dtype=tl.float32)
        # Accumulate dot products
        for d in range(0, D):
            a = tl.load(x_ptr + d)  # scalar
            row_ptrs = A_ptr + d * L + offs  # [128]
            row_vals = tl.load(row_ptrs, mask=mask, other=0.0)  # [128]
            acc += a * row_vals
        # Store partial results
        tl.store(y_ptr + offs, acc, mask=mask)


def run(*args):
    return ModelNew()(*args)
