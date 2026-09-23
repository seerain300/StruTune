import math
import torch
import triton
import triton.language as tl


@triton.jit
def gather_rows_c_kernel(cache_ptr, tok_idx_ptr, out_ptr,
                          num_tokens: tl.constexpr, Dc: tl.constexpr):
    # Each program handles one token row; copy a single row [Dc] from cache into out
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
                          num_tokens: tl.constexpr, Dp: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= num_tokens:
        return
    idx = tl.load(tok_idx_ptr + pid).to(tl.int32)
    base_in = idx * Dp
    for k in range(0, Dp):
        val = tl.load(cache_ptr + base_in + k)
        tl.store(out_ptr + pid * Dp + k, val)


@triton.jit
def matmul_hxK_to_vec_kernel(Q_ptr, K_ptr, Out_ptr,
                             H: tl.constexpr, Dc: tl.constexpr, L: tl.constexpr):
    # Compute Out[i, l] = sum_k Q[i, k] * K[l, k] for i in [0..H), l in [0..L)
    # Q_ptr: [H*Dc] row-major (contiguous)
    # K_ptr: [L*Dc] row-major (contiguous)
    # Out_ptr: [H*L] row-major (contiguous)
    for i in range(0, H):
        for l in range(0, L):
            acc = 0.0
            for k in range(0, Dc):
                qk = tl.load(Q_ptr + i * Dc + k)
                kl = tl.load(K_ptr + l * Dc + k)
                acc += qk * kl
            tl.store(Out_ptr + i * L + l, acc)


@triton.jit
def matvec_kernel(Out_ptr, K_ptr, Out2_ptr,
                  H: tl.constexpr, L: tl.constexpr, Dc: tl.constexpr):
    # Out_ptr: flattened [H*L] row-major
    # K_ptr: flattened [L*Dc] row-major (K as [L, Dc])
    # Out2_ptr: flattened [H*Dc] row-major
    # Compute Out2[i, :] = sum_l Out[i, l] * K[l, :]
    for i in range(0, H):
        for k in range(0, Dc):
            acc = 0.0
            for l in range(0, L):
                outil = tl.load(Out_ptr + i * L + l)
                kl = tl.load(K_ptr + l * Dc + k)
                acc += outil * kl
            tl.store(Out2_ptr + i * Dc + k, acc)


@triton.jit
def softmax_row_kernel(inp_ptr, out_ptr,
                        L: tl.constexpr):
    # Softmax over a row of length L: inp_ptr and out_ptr are length-L vectors
    # We process one row per program; inp_ptr is [start, start+1, ..., start+L-1]
    # For simplicity, assume single vector passed; use a loop to load into a vector
    # Compute softmax: m = max, sum_exp = sum(exp(x - m)), out_j = exp(x_j - m) / sum_exp
    # We will read inp_ptr row by row using a fixed stride L by launching with grid=(H,).
    # Implementation: pass base pointer per program, here assume inp_ptr is already the row.
    row_id = tl.program_id(0)
    m = -float("inf")
    # First pass: max
    for t in range(0, L):
        x = tl.load(inp_ptr + row_id * L + t)
        m = tl.maximum(m, x)
    # Second pass: sum of exp
    sum_exp = 0.0
    for t in range(0, L):
        x = tl.load(inp_ptr + row_id * L + t)
        sum_exp += tl.exp(x - m)
    inv_sum = 1.0 / sum_exp
    # Third pass: write normalized
    for t in range(0, L):
        x = tl.load(inp_ptr + row_id * L + t)
        y = tl.exp(x - m) * inv_sum
        tl.store(out_ptr + row_id * L + t, y)


@triton.jit
def lse_row_kernel(inp_ptr, out_ptr,
                   L: tl.constexpr):
    # Compute per-token lse in base-2: out_ptr[t] = log(sum(exp(inp_ptr[t])))/ln(2)
    row_id = tl.program_id(0)
    sum_exp = 0.0
    for t in range(0, L):
        x = tl.load(inp_ptr + row_id * L + t)
        sum_exp += tl.exp(x)
    lse_t = math.log(sum_exp) / math.log(2.0)
    for t in range(0, L):
        tl.store(out_ptr + row_id * L + t, lse_t)


@triton.jit
def reduce_sum_l_kernel(vec_ptr, out_ptr,
                        L: tl.constexpr):
    # Reduce sum across L tokens for a single row and store to out_ptr[0]
    row_id = tl.program_id(0)
    acc = 0.0
    for t in range(0, L):
        x = tl.load(vec_ptr + row_id * L + t)
        acc += x
    tl.store(out_ptr + 0, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Constants
        H = 16
        Dc = 512
        Dp = 64
        Bsz = q_nope.shape[0]
        device = q_nope.device

        # Squeeze caches
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [P, Dc]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [P, Dp]

        # Output and lse
        output = torch.zeros((Bsz, H, Dc), dtype=torch.bfloat16, device=device)
        lse = torch.empty((Bsz, H), dtype=torch.float32, device=device)

        for b in range(Bsz):
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            if L_tokens <= 0:
                # No work for this batch element
                lse[b] = torch.zeros((H,), dtype=torch.float32, device=device)
                continue

            # Gather token indices for this batch element
            tok_idx = kv_indices[int(kv_indptr[b].item()):int(kv_indptr[b + 1].item())].contiguous()  # [L_tokens]

            # 1) Gather rows from caches into Kc_flat and Kp_flat (float32)
            Kc_flat = torch.empty((L_tokens * Dc,), dtype=torch.float32, device=device)
            Kp_flat = torch.empty((L_tokens * Dp,), dtype=torch.float32, device=device)

            grid_g = (L_tokens,)
            gather_rows_c_kernel[grid_g](Kc_all, tok_idx, Kc_flat, L_tokens, Dc)
            Kc = Kc_flat.view(L_tokens, Dc)  # [L_tokens, Dc]

            grid_g2 = (L_tokens,)
            gather_rows_p_kernel[grid_g2](Kp_all, tok_idx, Kp_flat, L_tokens, Dp)
            Kp = Kp_flat.view(L_tokens, Dp)  # [L_tokens, Dp]

            # 2) For each head i: compute logits_scaled = qn[i] @ Kc.T + qp[i] @ Kp.T
            for i in range(H):
                qn_vec = q_nope[b, i].to(torch.float32).contiguous()  # [Dc]
                qp_vec = q_pe[b, i].to(torch.float32).contiguous()   # [Dp]

                # Out_qn: [1, L_tokens], Out_qp: [1, L_tokens]
                Out_qn = torch.empty((1 * L_tokens,), dtype=torch.float32, device=device)
                Out_qp = torch.empty((1 * L_tokens,), dtype=torch.float32, device=device)

                grid_m = (1,)  # H=1 for this kernel call
                matmul_hxK_to_vec_kernel[grid_m](qn_vec, Kc, Out_qn, 1, Dc, L_tokens)
                matmul_hxK_to_vec_kernel[grid_m](qp_vec, Kp, Out_qp, 1, Dp, L_tokens)

                # Sum to get logits for this head
                logits = (Out_qn + Out_qp)  # [L_tokens]
                logits_scaled = logits * sm_scale  # [L_tokens]

                # 3) Compute attn = softmax(logits_scaled) per head using Triton
                attn = torch.empty((L_tokens,), dtype=torch.float32, device=device)
                grid_soft = (1,)  # one row (head)
                softmax_row_kernel[grid_soft](logits_scaled, attn, L_tokens)

                # 4) Compute per-token lse in base-2 and reduce to per-head lse
                lse_tokens = torch.empty((L_tokens,), dtype=torch.float32, device=device)
                grid_lse = (1,)
                lse_row_kernel[grid_lse](logits_scaled, lse_tokens, L_tokens)

                per_head_lse = torch.empty((1,), dtype=torch.float32, device=device)
                reduce_sum_l_kernel[grid_lse](lse_tokens, per_head_lse, L_tokens)
                # Note: lse_tokens contains equal values per-token; per_head_lse is sum. We want logsumexp.
                # Correct approach: compute per-token lse then reduce; but per_token lse is uniform -> sum * L_tokens != logsumexp.
                # Fix: compute per-head lse via logsumexp correctly:
                # We need lse = log(sum(exp(logits))) / log(2). We have sum_exp computed earlier in lse_row_kernel as sum(exp(x)).
                # However, we only stored per-token lse. Let's recompute sum_exp correctly using Triton:
                sum_exp = torch.zeros((1,), dtype=torch.float32, device=device)
                for t in range(0, L_tokens):
                    x = logits_scaled[t]
                    sum_exp += torch.exp(x)  # scalar accumulation; Triton does not support atomics here; use torch reduction instead
                # Use PyTorch to compute per-head lse:
                per_head_lse = torch.log(sum_exp) / math.log(2.0)  # scalar tensor [1]
                lse[b, i] = per_head_lse.item()

                # 5) Compute out_vec[i, :] = attn @ Kc using Triton matvec
                # attn is [L_tokens]; Kc is [L_tokens, Dc] flattened
                out_vec_flat = torch.empty((H * Dc,), dtype=torch.float32, device=device)
                grid_mv = (H,)
                matvec_kernel[grid_mv](attn, Kc_flat, out_vec_flat, H, L_tokens, Dc)
                out_vec = out_vec_flat.view(H, Dc)
                output[b, i] = out_vec[i].to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
