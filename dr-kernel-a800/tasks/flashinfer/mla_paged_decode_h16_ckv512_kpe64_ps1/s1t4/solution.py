import math
import torch
import triton
import triton.language as tl


@triton.jit
def gather_rows_c_kernel(cache_ptr, tok_idx_ptr, out_ptr,
                          num_tokens: tl.int32, Dc: tl.int32):
    # Each program handles one token row
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
                          num_tokens: tl.int32, Dp: tl.int32):
    # Each program handles one token row
    pid = tl.program_id(0)
    if pid >= num_tokens:
        return
    idx = tl.load(tok_idx_ptr + pid).to(tl.int32)
    base_in = idx * Dp
    for k in range(0, Dp):
        val = tl.load(cache_ptr + base_in + k)
        tl.store(out_ptr + pid * Dp + k, val)


@triton.jit
def lse_base2_rowwise_kernel(logits_ptr, lse_ptr, L: tl.int32, sm_scale: tl.float32):
    # One program per head (we assume H is implicit from logits_ptr layout, but here we only do per-row)
    i = tl.program_id(0)  # head index
    m = -float("inf")
    # First pass: compute max over L tokens
    for t in range(0, L):
        val = tl.load(logits_ptr + i * L + t)
        m = tl.maximum(m, val)
    # Second pass: compute sum_exp
    sum_exp = 0.0
    for t in range(0, L):
        val = tl.load(logits_ptr + i * L + t)
        sum_exp += tl.exp(val - m)
    # lse = m + log(sum_exp) / ln(2)
    lse_val = m + tl.log(sum_exp) / 0.6931471805599453  # 1 / ln(2)
    tl.store(lse_ptr + i, lse_val)


@triton.jit
def softmax_rowwise_kernel(logits_ptr, out_ptr, L: tl.int32):
    # One program per head, compute softmax over L tokens
    i = tl.program_id(0)
    m = -float("inf")
    for t in range(0, L):
        val = tl.load(logits_ptr + i * L + t)
        m = tl.maximum(m, val)
    sum_exp = 0.0
    for t in range(0, L):
        val = tl.load(logits_ptr + i * L + t)
        sum_exp += tl.exp(val - m)
    for t in range(0, L):
        val = tl.load(logits_ptr + i * L + t)
        out_val = tl.exp(val - m) / sum_exp
        tl.store(out_ptr + i * L + t, out_val)


@triton.jit
def matvec_kernel(attn_ptr, K_ptr, out_ptr,
                  H: tl.int32, Dc: tl.int32, L: tl.int32,
                  BLOCK_D: tl.constexpr):
    # One program handles one head i and one D-block
    pid = tl.program_id(0)
    # Grid is (H, ceil_div(Dc, BLOCK_D))
    i = pid // tl.cdiv(Dc, BLOCK_D)
    if i >= H:
        return
    d_block = pid % tl.cdiv(Dc, BLOCK_D)
    d_start = d_block * BLOCK_D

    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for l in range(0, L):
        attn_val = tl.load(attn_ptr + i * L + l)  # scalar
        for dd in range(0, BLOCK_D):
            d = d_start + dd
            if d < Dc:
                K_val = tl.load(K_ptr + l * Dc + d)
                acc[dd] += attn_val * K_val
    out_base = i * Dc + d_start
    for dd in range(0, BLOCK_D):
        d = d_start + dd
        if d < Dc:
            tl.store(out_ptr + d, acc[dd])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        device = q_nope.device
        head_dim_ckv = 512
        head_dim_kpe = 64
        num_qo_heads = 16
        batch_size = q_nope.shape[0]

        # Squeeze size-1 cache dim and flatten
        Kc_all = ckv_cache.squeeze(1).contiguous()  # [P, Dc]
        Kp_all = kpe_cache.squeeze(1).contiguous()  # [P, Dp]
        Kc_all_flat = Kc_all.view(-1)               # [P * Dc]
        Kp_all_flat = Kp_all.view(-1)               # [P * Dp]

        output = torch.empty((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        for b in range(batch_size):
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            if L_tokens <= 0:
                lse[b] = torch.zeros((num_qo_heads,), dtype=torch.float32, device=device)
                for i in range(num_qo_heads):
                    output[b, i] = torch.zeros((head_dim_ckv,), dtype=torch.bfloat16, device=device)
                continue

            # Token indices for this batch element
            tok_idx = kv_indices[int(kv_indptr[b].item()):int(kv_indptr[b + 1].item())].contiguous()  # [L_tokens]

            # 1) Gather rows from caches into Kc_flat and Kp_flat (float32)
            Kc_flat = torch.empty((L_tokens * head_dim_ckv,), dtype=torch.float32, device=device)
            Kp_flat = torch.empty((L_tokens * head_dim_kpe,), dtype=torch.float32, device=device)

            grid_gather = (L_tokens,)
            gather_rows_c_kernel[grid_gather](Kc_all_flat, tok_idx, Kc_flat, L_tokens, head_dim_ckv)
            Kc = Kc_flat.view(L_tokens, head_dim_ckv)  # [L_tokens, Dc]

            gather_rows_p_kernel[grid_gather](Kp_all_flat, tok_idx, Kp_flat, L_tokens, head_dim_kpe)
            Kp = Kp_flat.view(L_tokens, head_dim_kpe)  # [L_tokens, Dp]

            # 2) For each head i: compute logits_scaled
            logits_scaled_list = []
            for i in range(num_qo_heads):
                qn = q_nope[b, i].to(torch.float32).contiguous()  # [Dc]
                qp = q_pe[b, i].to(torch.float32).contiguous()   # [Dp]
                logits_qn = qn @ Kc.T                           # [1, L_tokens]
                logits_qp = qp @ Kp.T                           # [1, L_tokens]
                logits = (logits_qn + logits_qp).squeeze(0)     # [L_tokens]
                logits_scaled = logits * sm_scale               # [L_tokens]
                logits_scaled_list.append(logits_scaled)

            # 3) Compute lse per head using Triton rowwise kernel
            logits_stack = torch.stack(logits_scaled_list, dim=0).contiguous()  # [H, L]
            lse[b] = torch.empty((num_qo_heads,), dtype=torch.float32, device=device)
            grid_lse = (num_qo_heads,)
            lse_base2_rowwise_kernel[grid_lse](logits_stack.view(-1), lse[b], L_tokens, sm_scale)

            # 4) Compute attn via Triton softmax rowwise kernel
            attn = torch.empty((num_qo_heads, L_tokens), dtype=torch.float32, device=device)
            grid_softmax = (num_qo_heads,)
            softmax_rowwise_kernel[grid_softmax](logits_stack.view(-1), attn.view(-1), L_tokens)

            # 5) Final projection: out_vec[i] = attn[i] @ Kc -> [Dc], Triton matvec
            out_flat = torch.empty((num_qo_heads * head_dim_ckv,), dtype=torch.float32, device=device)
            BLOCK_D = 128
            grid_matvec = (num_qo_heads * tl.cdiv(head_dim_ckv, BLOCK_D),)
            matvec_kernel[grid_matvec](attn.contiguous().view(-1), Kc.contiguous().view(-1), out_flat, num_qo_heads, head_dim_ckv, L_tokens, BLOCK_D=BLOCK_D)
            out_vec = out_flat.view(num_qo_heads, head_dim_ckv)  # [H, Dc]

            # Store output[b, i] as bfloat16
            for i in range(num_qo_heads):
                output[b, i] = out_vec[i].to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
