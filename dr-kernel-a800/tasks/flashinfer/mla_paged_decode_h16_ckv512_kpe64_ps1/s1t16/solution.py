import math
import torch
import triton
import triton.language as tl


# Triton kernels

# 1) Gather rows from ckv_cache_all (shape [P, Dc]) into Kc_flat of shape [(L_tokens * Dc)]
@triton.jit
def gather_rows_c_kernel(cache_ptr, tok_idx_ptr, out_ptr,
                          num_tokens: tl.constexpr, Dc: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= num_tokens:
        return
    idx = tl.load(tok_idx_ptr + pid).to(tl.int32)
    base_in = idx * Dc
    for k in range(0, Dc):
        val = tl.load(cache_ptr + base_in + k)
        tl.store(out_ptr + pid * Dc + k, val)


# 2) Gather rows from kpe_cache_all (shape [P, Dp]) into Kp_flat of shape [(L_tokens * Dp)]
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


# 3) Compute per-head logsumexp (base-2) across L_tokens for each head i in [0..H)
#    Two-pass approach: first max, then sum of exp, then lse = m + log(sum_exp) / ln(2).
@triton.jit
def lse_base2_rows_kernel(logits_ptr, lse_ptr, H: tl.constexpr, L: tl.constexpr):
    # Grid: (H,). Each program handles one head i and reduces across L tokens.
    i = tl.program_id(0)
    if i >= H:
        return
    m = -float("inf")
    # pass 1: max
    for t in range(0, L):
        val = tl.load(logits_ptr + i * L + t)
        m = tl.maximum(m, val)
    # pass 2: sum exp
    sum_exp = 0.0
    for t in range(0, L):
        val = tl.load(logits_ptr + i * L + t)
        sum_exp += tl.exp(val - m)
    # compute lse in base-2: log2(sum_exp) = ln(sum_exp) / ln(2)
    ln2 = 0.6931471805599453
    lse_val = m + math.log(sum_exp) / ln2
    tl.store(lse_ptr + i, lse_val)


# 4) Softmax per head over L tokens: write attn[i, t] = exp(logits_scaled[i, t] - m) / sum_exp
@triton.jit
def softmax_rows_kernel(logits_ptr, attn_ptr, H: tl.constexpr, L: tl.constexpr):
    i = tl.program_id(0)
    if i >= H:
        return
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
        p = tl.exp(val - m) / sum_exp
        tl.store(attn_ptr + i * L + t, p)


# 5) Matvec: given attn_flat of shape [H*L], Kc_flat of shape [(L*Dc)], compute out_vec_flat of shape [(H*Dc)]
#    For each head i: out_vec[i, :] = attn[i, :] @ Kc[i, :]
@triton.jit
def matvec_kernel(attn_ptr, Kc_ptr, out_ptr,
                  H: tl.constexpr, Dc: tl.constexpr, L: tl.constexpr,
                  BLOCK_D: tl.constexpr, BLOCK_T: tl.constexpr):
    # grid: (H,)
    i = tl.program_id(0)
    if i >= H:
        return
    # we iterate over Dc in chunks and accumulate
    for d0 in range(0, Dc, BLOCK_D):
        acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for t0 in range(0, L, BLOCK_T):
            # load attn chunk: shape [BLOCK_T]
            attn_chunk = tl.load(attn_ptr + i * L + t0 + tl.arange(0, BLOCK_T), mask=(t0 + tl.arange(0, BLOCK_T)) < L, other=0.0)
            # load Kc chunk: shape [BLOCK_T, BLOCK_D]
            k0 = t0 + tl.arange(0, BLOCK_T)[:, None]           # [BLOCK_T, 1]
            d_off = d0 + tl.arange(0, BLOCK_D)[None, :]        # [1, BLOCK_D]
            k_mask = (k0 < L) & (d_off < Dc)
            Kc_chunk = tl.load(Kc_ptr + k0 * Dc + d_off, mask=k_mask, other=0.0)  # [BLOCK_T, BLOCK_D]
            # accumulate: acc += sum_t attn_chunk[t] * Kc_chunk[t, :]
            # Do per-column update
            for t in range(0, BLOCK_T):
                if (t0 + t) < L:
                    a = attn_chunk[t]  # scalar
                    col = Kc_chunk[t, :]  # [BLOCK_D]
                    acc += a * col
        # store acc into out_ptr at offsets corresponding to head i
        out_offsets = i * Dc + d0 + tl.arange(0, BLOCK_D)
        out_mask = (d0 + tl.arange(0, BLOCK_D)) < Dc
        tl.store(out_ptr + out_offsets, acc, mask=out_mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        """
        q_nope: [B, H, Dc], bfloat16
        q_pe: [B, H, Dp], bfloat16
        ckv_cache: [P, 1, Dc], bfloat16
        kpe_cache: [P, 1, Dp], bfloat16
        kv_indptr: [B+1], int32
        kv_indices: [num_kv_indices], int32
        sm_scale: float32 scalar
        Returns: output [B, H, Dc], bfloat16; lse [B, H], float32
        """
        assert q_nope.dim() == 3 and q_pe.dim() == 3, "q_nope and q_pe must be 3D tensors [B, H, D]"
        B, H, Dc = q_nope.shape
        _, _, Dp = q_pe.shape
        P = ckv_cache.shape[0]
        _, _, Dc_cache = ckv_cache.shape
        _, _, Dp_cache = kpe_cache.shape
        assert Dc_cache == Dc and Dp_cache == Dp, "Cache dims must match head dims"
        assert kv_indptr.shape[0] == B + 1, "kv_indptr must have length B+1"
        assert kv_indices.dim() == 1, "kv_indices must be 1D"
        device = q_nope.device

        # Squeeze the size-1 dimension from caches
        Kc_all = ckv_cache.squeeze(1).contiguous()  # [P, Dc]
        Kp_all = kpe_cache.squeeze(1).contiguous()  # [P, Dp]

        # Output buffers
        output = torch.empty((B, H, Dc), dtype=torch.bfloat16, device=device)
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        for b in range(B):
            # Determine number of tokens for this batch element
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            if L_tokens <= 0:
                # No KV entries for this batch element: output zeros, lse zeros
                for i in range(H):
                    output[b, i] = torch.zeros((Dc,), dtype=torch.bfloat16, device=device)
                lse[b] = torch.zeros((H,), dtype=torch.float32, device=device)
                continue

            # Gather token indices for this batch element
            tok_idx = kv_indices[int(kv_indptr[b].item()):int(kv_indptr[b + 1].item())].contiguous()  # [L_tokens]

            # 1) Gather rows from caches into Kc_flat and Kp_flat (float32), then form [L_tokens, D] matrices
            Kc_flat = torch.empty((L_tokens * Dc,), dtype=torch.float32, device=device)
            Kp_flat = torch.empty((L_tokens * Dp,), dtype=torch.float32, device=device)

            # Launch gather kernels
            grid_gather = (L_tokens,)
            gather_rows_c_kernel[grid_gather](Kc_all, tok_idx, Kc_flat, L_tokens, Dc)
            Kc = Kc_flat.view(L_tokens, Dc)  # [L_tokens, Dc]

            gather_rows_p_kernel[grid_gather](Kp_all, tok_idx, Kp_flat, L_tokens, Dp)
            Kp = Kp_flat.view(L_tokens, Dp)  # [L_tokens, Dp]

            # 2) For each head i: compute logits = qn @ Kc.T + qp @ Kp.T, scaled, then lse and attn in Triton
            for i in range(H):
                # Load qn and qp as float32
                qn = q_nope[b, i].to(torch.float32).contiguous()  # [Dc]
                qp = q_pe[b, i].to(torch.float32).contiguous()   # [Dp]
                logits_qn = qn @ Kc.T                          # [1, L_tokens]
                logits_qp = qp @ Kp.T                          # [1, L_tokens]
                logits = (logits_qn + logits_qp).squeeze(0)    # [L_tokens]
                logits_scaled = logits * sm_scale              # [L_tokens]

                # 3) Compute lse per head using Triton kernel
                lse[b, i] = torch.empty((), dtype=torch.float32, device=device)  # decoy placeholder; Triton writes into lse[b, i] via grid (H,) and store
                # Launch Triton lse kernel: note we want to write into lse[b, i]; since Triton expects vector, use a temporary vector of length 1 and then assign,
                # but simpler: allocate lse[b] = torch.empty((H,), ...) and kernel writes into lse_ptr[i].
                lse_row = lse[b]  # [H] vector
                lse_base2_rows_kernel[(H,)](logits_scaled, lse_row, H=H, L=L_tokens)

                # 4) Compute attention weights per head using Triton softmax kernel
                attn = torch.empty((H, L_tokens), dtype=torch.float32, device=device)
                softmax_rows_kernel[(H,)](logits_scaled, attn, H=H, L=L_tokens)

                # attn[b, i, :] corresponds to attn[i * L_tokens : (i+1)*L_tokens] flattened
                # However, since we launch per head, we must ensure we use attn for head i.
                # We can reconstruct: attn[i, :] across columns, but softmax kernel already writes attn[i, :] correctly.
                # Final projection: out_vec[i, :] = attn[i, :] @ Kc -> [Dc]
                out_vec = torch.empty((Dc,), dtype=torch.float32, device=device)

                # 5) Triton matvec kernel: attn_flat[i*L:(i+1)*L], Kc_flat, out_vec
                attn_flat = attn[i, :].contiguous()  # [L_tokens]
                matvec_kernel[(1,)](attn_flat, Kc, out_vec, H=1, Dc=Dc, L=L_tokens, BLOCK_D=128, BLOCK_T=64)

                # Store to output[b, i] as bfloat16
                output[b, i] = out_vec.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
