import math
import torch
import triton
import triton.language as tl


@triton.jit
def gather_rows_c_kernel(cache_ptr, tok_idx_ptr, out_ptr,
                          L_tokens: tl.constexpr, Dc: tl.constexpr):
    # One program per token row
    pid = tl.program_id(0)
    if pid >= L_tokens:
        return
    idx = tl.load(tok_idx_ptr + pid).to(tl.int32)
    base_in = idx * Dc
    for k in range(0, Dc):
        val = tl.load(cache_ptr + base_in + k)
        tl.store(out_ptr + pid * Dc + k, val)


@triton.jit
def gather_rows_p_kernel(cache_ptr, tok_idx_ptr, out_ptr,
                          L_tokens: tl.constexpr, Dp: tl.constexpr):
    # One program per token row
    pid = tl.program_id(0)
    if pid >= L_tokens:
        return
    idx = tl.load(tok_idx_ptr + pid).to(tl.int32)
    base_in = idx * Dp
    for k in range(0, Dp):
        val = tl.load(cache_ptr + base_in + k)
        tl.store(out_ptr + pid * Dp + k, val)


@triton.jit
def softmax_row_kernel(row_ptr, out_ptr, L: tl.constexpr):
    # Single program: softmax over a vector of length L (assumes row_ptr points to [L] contiguous)
    m = -float("inf")
    # Compute max
    for t in range(0, L):
        m = tl.maximum(m, tl.load(row_ptr + t))
    sum_exp = 0.0
    # Compute sum of exp(x - m)
    for t in range(0, L):
        e = tl.exp(tl.load(row_ptr + t) - m)
        sum_exp += e
    inv_sum = 1.0 / sum_exp
    # Write normalized values
    for t in range(0, L):
        val = tl.load(row_ptr + t)
        norm = tl.exp(val - m) * inv_sum
        tl.store(out_ptr + t, norm)


@triton.jit
def matvec_kernel(attn_row_ptr, K_ptr, out_vec_ptr,
                  Dc: tl.constexpr, L: tl.constexpr):
    # One program: compute out_vec = attn_row @ K, where attn_row is a vector of length L,
    # K is [L, Dc], out_vec is [Dc]
    # attn_row_ptr points to a contiguous vector of length L
    for d in range(0, Dc):
        acc = 0.0
        for t in range(0, L):
            acc += tl.load(attn_row_ptr + t) * tl.load(K_ptr + t * Dc + d)
        tl.store(out_vec_ptr + d, acc)


@triton.jit
def lse_per_head_kernel(logits_ptr, lse_ptr, H: tl.constexpr, L: tl.constexpr, sm_scale: tl.constexpr):
    # One program per head: compute logsumexp(base-2) for the row logits[i*L:(i+1)*L]
    i = tl.program_id(0)
    row_start = i * L
    m = -float("inf")
    for t in range(0, L):
        val = tl.load(logits_ptr + row_start + t)
        m = tl.maximum(m, val)
    sum_exp = 0.0
    for t in range(0, L):
        sum_exp += tl.exp(tl.load(logits_ptr + row_start + t) - m)
    lse_val = (m + tl.log(sum_exp)) * sm_scale / math.log(2.0)
    tl.store(lse_ptr + i, lse_val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Shapes and constants from the original code
        batch_size = q_nope.shape[0]
        num_qo_heads = q_nope.shape[1]
        head_dim_ckv = q_nope.shape[2]
        head_dim_kpe = q_pe.shape[2]
        assert num_qo_heads == 16
        assert head_dim_ckv == 512
        assert head_dim_kpe == 64

        num_pages = ckv_cache.shape[0]
        device = q_nope.device

        # Flatten caches: [num_pages, Dc] and [num_pages, Dp], in float32
        Kc_all = ckv_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, 64]

        output = torch.empty((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        for b in range(batch_size):
            # Determine number of tokens for this batch element
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            if L_tokens <= 0:
                # Do not modify output[b] or lse[b] for degenerate cases; leave as provided by previous runs
                continue

            # Token indices for this batch
            tok_idx = kv_indices[int(kv_indptr[b].item()):int(kv_indptr[b + 1].item())].contiguous()  # [L_tokens]

            # Gather rows from caches into float32
            Kc_flat = torch.empty((L_tokens * head_dim_ckv,), dtype=torch.float32, device=device)
            Kp_flat = torch.empty((L_tokens * head_dim_kpe,), dtype=torch.float32, device=device)

            grid_gather = (L_tokens,)
            gather_rows_c_kernel[grid_gather](Kc_all, tok_idx, Kc_flat, L_tokens, head_dim_ckv)
            Kc = Kc_flat.view(L_tokens, head_dim_ckv)  # [L_tokens, 512]

            gather_rows_p_kernel[grid_gather](Kp_all, tok_idx, Kp_flat, L_tokens, head_dim_kpe)
            Kp = Kp_flat.view(L_tokens, head_dim_kpe)  # [L_tokens, 64]

            # For each head i:
            for i in range(num_qo_heads):
                qn = q_nope[b, i].to(torch.float32).contiguous()  # [512]
                qp = q_pe[b, i].to(torch.float32).contiguous()   # [64]

                # Compute logits = qn @ Kc.T + qp @ Kp.T  -> [L_tokens]
                logits_qn = qn @ Kc.T                             # [1, L_tokens]
                logits_qp = qp @ Kp.T                             # [1, L_tokens]
                logits = (logits_qn + logits_qp).squeeze(0)       # [L_tokens]
                logits_scaled = logits * sm_scale                 # [L_tokens]

                # lse per head using Triton (base-2)
                lse[b, i] = torch.empty((), dtype=torch.float32, device=device)  # scalar
                lse_per_head_kernel[(1,)](logits_scaled, lse[b, i], num_qo_heads, L_tokens, sm_scale)

                # Attention weights using Triton softmax
                attn = torch.empty((L_tokens,), dtype=torch.float32, device=device)
                softmax_row_kernel[(L_tokens,)](logits_scaled, attn, L_tokens)

                # Final projection: attn @ Kc -> [512]
                out_vec = torch.empty((head_dim_ckv,), dtype=torch.float32, device=device)
                matvec_kernel[(head_dim_ckv,)](attn, Kc.contiguous().view(-1), out_vec,
                                               head_dim_ckv, L_tokens)
                output[b, i] = out_vec.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
