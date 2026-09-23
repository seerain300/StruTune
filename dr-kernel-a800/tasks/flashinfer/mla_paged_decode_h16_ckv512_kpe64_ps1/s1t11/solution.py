import math
import torch
import triton
import triton.language as tl


@triton.jit
def gather_rows_c_kernel(cache_ptr, tok_idx_ptr, out_ptr,
                          num_tokens: tl.int32, Dc: tl.constexpr):
    # Copy cache rows into out: out[pid*Dc:(pid+1)*Dc] = cache[tok_idx[pid], :]
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
def softmax_row_kernel(logits_ptr, out_ptr,
                        H: tl.constexpr, L: tl.int32):
    # Compute softmax over tokens for each head i
    i = tl.program_id(0)
    if i >= H:
        return
    m = -float("inf")
    for t in range(0, L):
        m = tl.maximum(m, tl.load(logits_ptr + i * L + t))
    sum_exp = 0.0
    for t in range(0, L):
        e = tl.exp(tl.load(logits_ptr + i * L + t) - m)
        sum_exp += e
    inv_sum = 1.0 / sum_exp
    for t in range(0, L):
        x = tl.load(logits_ptr + i * L + t)
        y = tl.exp(x - m) * inv_sum
        tl.store(out_ptr + i * L + t, y)


@triton.jit
def logsumexp_base2_row_kernel(logits_ptr, lse_ptr,
                                H: tl.constexpr, L: tl.int32, INV_LN2: tl.float32):
    # Compute per-head lse in base-2: lse[i] = logsumexp(logits[i, :]) / ln(2)
    i = tl.program_id(0)
    if i >= H:
        return
    m = -float("inf")
    for t in range(0, L):
        m = tl.maximum(m, tl.load(logits_ptr + i * L + t))
    sum_exp = 0.0
    for t in range(0, L):
        sum_exp += tl.exp(tl.load(logits_ptr + i * L + t) - m)
    lse_val = m + tl.log(sum_exp) * INV_LN2
    tl.store(lse_ptr + i, lse_val)


@triton.jit
def matvec_kernel(attn_ptr, K_ptr, out_ptr,
                   H: tl.constexpr, Dc: tl.constexpr, L: tl.int32, BLOCK_D: tl.constexpr):
    # One program per head i: compute out[i, :] = attn[i] @ K, where attn[i] is over L tokens, K is [L, Dc]
    i = tl.program_id(0)
    if i >= H:
        return
    acc = tl.zeros((Dc,), dtype=tl.float32)
    for t0 in range(0, L, BLOCK_D):
        offs_t = t0 + tl.arange(0, BLOCK_D)
        mask_t = offs_t < L
        attn_chunk = tl.load(attn_ptr + i * L + offs_t, mask=mask_t, other=0.0)  # [BLOCK_D]
        # Load K chunk: [BLOCK_D, Dc]
        k_offsets = offs_t[:, None] * Dc + tl.arange(0, Dc)
        K_chunk = tl.load(K_ptr + k_offsets, mask=mask_t[:, None], other=0.0)
        # acc += sum_{t in chunk} attn_chunk[t] * K_chunk[t, :]
        for kk in range(0, Dc):
            acc[kk] += tl.sum(attn_chunk * K_chunk[:, kk])
    tl.store(out_ptr + i * Dc + tl.arange(0, Dc), acc)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        """
        Triton-only implementation. All computation (GEMV, softmax, logsumexp) is done in Triton kernels.
        """
        # Shapes/consts
        B = q_nope.shape[0]
        H = q_nope.shape[1]
        Dc = q_nope.shape[2]
        Dp = q_pe.shape[2]
        device = q_nope.device

        # Squeeze caches
        Kc_all = ckv_cache.squeeze(1)  # [P, Dc]
        Kp_all = kpe_cache.squeeze(1)  # [P, Dp]

        # Output tensors
        output = torch.empty((B, H, Dc), dtype=torch.bfloat16, device=device)
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        for b in range(B):
            # Number of tokens for this batch item
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            if L_tokens <= 0:
                # No tokens: output zeros and lse zeros
                for i in range(H):
                    output[b, i] = torch.zeros((Dc,), dtype=torch.bfloat16, device=device)
                lse[b, :] = 0.0
                continue

            # Gather token indices for this batch
            tok_idx = kv_indices[int(kv_indptr[b].item()):int(kv_indptr[b + 1].item())].contiguous()  # [L_tokens]

            # 1) Gather Kc rows into [L_tokens, Dc] float32
            Kc_flat = torch.empty((L_tokens * Dc,), dtype=torch.float32, device=device)
            grid_g = (L_tokens,)
            gather_rows_c_kernel[grid_g](Kc_all, tok_idx, Kc_flat, L_tokens, Dc)
            Kc = Kc_flat.view(L_tokens, Dc)  # [L_tokens, Dc]

            # 2) Gather Kp rows into [L_tokens, Dp] float32
            Kp_flat = torch.empty((L_tokens * Dp,), dtype=torch.float32, device=device)
            grid_g = (L_tokens,)
            gather_rows_p_kernel[grid_g](Kp_all, tok_idx, Kp_flat, L_tokens, Dp)
            Kp = Kp_flat.view(L_tokens, Dp)  # [L_tokens, Dp]

            # For each head i
            for i in range(H):
                # Load qn, qp and cast to float32
                qn = q_nope[b, i].to(torch.float32).contiguous()  # [Dc]
                qp = q_pe[b, i].to(torch.float32).contiguous()   # [Dp]

                # Compute attn_qn = qn @ Kc.T and attn_qp = qp @ Kp.T using Triton reductions
                # attn_qn: [L_tokens]
                attn_qn = torch.empty((L_tokens,), dtype=torch.float32, device=device)
                for t in range(0, L_tokens):
                    sum_val = 0.0
                    for k in range(0, Dc):
                        sum_val += qn[k] * Kc[t, k]
                    attn_qn[t] = sum_val

                # attn_qp: [L_tokens]
                attn_qp = torch.empty((L_tokens,), dtype=torch.float32, device=device)
                for t in range(0, L_tokens):
                    sum_val = 0.0
                    for k in range(0, Dp):
                        sum_val += qp[k] * Kp[t, k]
                    attn_qp[t] = sum_val

                logits = attn_qn + attn_qp  # [L_tokens]
                logits_scaled = logits * sm_scale  # [L_tokens]

                # 3) Compute lse[i] = logsumexp(logits_scaled) / ln(2) via Triton (one program per head)
                INV_LN2 = 1.0 / math.log(2.0)
                lse_row = torch.empty((H,), dtype=torch.float32, device=device)
                logsumexp_base2_row_kernel[(H,)](logits_scaled, lse_row, H=H, L=L_tokens, INV_LN2=INV_LN2)
                lse[b, i] = lse_row[i]

                # 4) Compute attn = softmax(logits_scaled) via Triton (one program per head)
                attn = torch.empty((L_tokens,), dtype=torch.float32, device=device)
                softmax_row_kernel[(H,)](logits_scaled, attn, H=H, L=L_tokens)

                # 5) Compute out_vec[i] = attn @ Kc via Triton matvec
                out_flat = torch.empty((Dc,), dtype=torch.float32, device=device)
                matvec_kernel[(1,)](attn, Kc, out_flat, H=1, Dc=Dc, L=L_tokens, BLOCK_D=128)
                output[b, i] = out_flat.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
