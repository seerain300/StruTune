import math
import torch
import triton
import triton.language as tl


# Triton kernel: gather rows from a flattened cache into an output buffer.
# cache_ptr: [P * D] flattened (D=512 or 64).
# tok_idx: [L_tokens] int32 indices.
# out_ptr: [L_tokens * D] flattened.
@triton.jit
def gather_rows_kernel(cache_ptr, tok_idx_ptr, out_ptr,
                        num_tokens: tl.constexpr, D: tl.constexpr):
    pid = tl.program_id(0)
    if pid >= num_tokens:
        return
    idx = tl.load(tok_idx_ptr + pid).to(tl.int32)
    base_in = idx * D
    # Write the entire row into out_ptr[pid*D : (pid+1)*D]
    for k in range(0, D):
        val = tl.load(cache_ptr + base_in + k)
        tl.store(out_ptr + pid * D + k, val)


# Triton kernel: compute logits_scaled per (head, token).
# Inputs:
#   qn_flat: [H*Dc] flattened query-normalized per head.
#   qp_flat: [H*Dp] flattened query-permanent per head.
#   Kc_flat: [L*Dc] flattened cached keys.
#   Kp_flat: [L*Dp] flattened cached positions.
#   logits_scaled_ptr: [H*L] flattened output per row.
# Launch grid=(H, L). Each program computes one row element.
@triton.jit
def compute_logits_kernel(qn_flat_ptr, qp_flat_ptr, Kc_flat_ptr, Kp_flat_ptr,
                          logits_scaled_ptr, H: tl.constexpr, Dc: tl.constexpr, Dp: tl.constexpr,
                          L: tl.constexpr, sm_scale: tl.constexpr):
    i = tl.program_id(0)  # head index
    t = tl.program_id(1)  # token index
    if (i >= H) or (t >= L):
        return
    # Offsets into flattened arrays
    # qn_flat[i*Dc + k] but we need dot product: sum_k qn[k] * Kc[t*Dc + k]
    # For efficiency, we compute qn[i] and Kc[t] by loading slices.
    # However, Triton does not support slicing, so we compute dot via loop over k.
    # Note: qn_flat_ptr stride is 1, Kc_flat_ptr stride is 1.
    dot_qn = 0.0
    dot_qp = 0.0
    # Loop over Dc for qn and Kc
    for k in range(0, Dc):
        qn_k = tl.load(qn_flat_ptr + i * Dc + k)
        Kc_k = tl.load(Kc_flat_ptr + t * Dc + k)
        dot_qn += qn_k * Kc_k
    # Loop over Dp for qp and Kp
    for k in range(0, Dp):
        qp_k = tl.load(qp_flat_ptr + i * Dp + k)
        Kp_k = tl.load(Kp_flat_ptr + t * Dp + k)
        dot_qp += qp_k * Kp_k
    logits = dot_qn + dot_qp
    logits_scaled = logits * sm_scale
    tl.store(logits_scaled_ptr + i * L + t, logits_scaled)


# Triton kernel: compute per-row max over L tokens for each head i.
# logits_ptr: [H*L] flattened, row i starts at i*L.
# m_ptr: [H] float32, initialized to -inf, then set to max.
@triton.jit
def row_max_kernel(logits_ptr, m_ptr, H: tl.constexpr, L: tl.constexpr, BLOCK_L: tl.constexpr):
    i = tl.program_id(0)
    if i >= H:
        return
    m = -float("inf")
    for off in range(0, L, BLOCK_L):
        idx = off + tl.arange(0, BLOCK_L)
        mask = idx < L
        vals = tl.load(logits_ptr + i * L + idx, mask=mask, other=-float("inf"))
        chunk_max = tl.max(vals, axis=0)
        m = tl.maximum(m, chunk_max)
    tl.store(m_ptr + i, m)


# Triton kernel: compute per-row sum(exp(logits - m)) over L tokens for each head i.
# logits_ptr: [H*L]
# m_ptr: [H]
# sum_ptr: [H]
@triton.jit
def row_sumexp_kernel(logits_ptr, sum_ptr, m_ptr, H: tl.constexpr, L: tl.constexpr, BLOCK_L: tl.constexpr):
    i = tl.program_id(0)
    if i >= H:
        return
    m = tl.load(m_ptr + i)
    sum_exp = 0.0
    for off in range(0, L, BLOCK_L):
        idx = off + tl.arange(0, BLOCK_L)
        mask = idx < L
        vals = tl.load(logits_ptr + i * L + idx, mask=mask, other=-float("inf"))
        expv = tl.exp(vals - m)
        # masked elements are -inf => exp = 0 after subtract m; sum properly
        chunk_sum = tl.sum(expv, axis=0)
        sum_exp += chunk_sum
    tl.store(sum_ptr + i, sum_exp)


# Triton kernel: softmax per row (head i) over L tokens, write to attn_ptr.
# attn_ptr: [H*L], row i at i*L.
# logits_ptr: [H*L]
@triton.jit
def softmax_row_kernel(logits_ptr, attn_ptr, H: tl.constexpr, L: tl.constexpr, BLOCK_L: tl.constexpr):
    i = tl.program_id(0)
    if i >= H:
        return
    # First pass: max
    m = -float("inf")
    for off in range(0, L, BLOCK_L):
        idx = off + tl.arange(0, BLOCK_L)
        mask = idx < L
        vals = tl.load(logits_ptr + i * L + idx, mask=mask, other=-float("inf"))
        chunk_max = tl.max(vals, axis=0)
        m = tl.maximum(m, chunk_max)
    # Second pass: sum exp
    sum_exp = 0.0
    for off in range(0, L, BLOCK_L):
        idx = off + tl.arange(0, BLOCK_L)
        mask = idx < L
        vals = tl.load(logits_ptr + i * L + idx, mask=mask, other=-float("inf"))
        expv = tl.exp(vals - m)
        sum_exp += tl.sum(expv, axis=0)
    # Third pass: write normalized
    inv_sum = 1.0 / sum_exp
    for off in range(0, L, BLOCK_L):
        idx = off + tl.arange(0, BLOCK_L)
        mask = idx < L
        vals = tl.load(logits_ptr + i * L + idx, mask=mask, other=-float("inf"))
        attn = tl.exp(vals - m) * inv_sum
        tl.store(attn_ptr + i * L + idx, attn, mask=mask)


# Triton kernel: matvec per head. For each head i, compute out[i, :] = attn_row[i, :] @ Kc[:, :]
# attn_flat: [H*L] flattened (row-major). Each row is softmax over logits_scaled for that head.
# Kc_flat: [L*Dc] flattened.
# out_flat: [H*Dc] flattened.
# We implement this by looping over tokens: out[i, d] += attn[i, t] * Kc[t, d] for all t.
@triton.jit
def matvec_kernel_2d(attn_flat_ptr, Kc_flat_ptr, out_flat_ptr,
                     H: tl.constexpr, Dc: tl.constexpr, L: tl.constexpr,
                     BLOCK_D: tl.constexpr):
    i = tl.program_id(0)
    if i >= H:
        return
    # out[i, :] as contiguous [Dc]
    # We will compute in chunks of Dc
    for d0 in range(0, Dc, BLOCK_D):
        d_range = d0 + tl.arange(0, BLOCK_D)
        mask_d = d_range < Dc
        out_vec = tl.zeros((BLOCK_D,), dtype=tl.float32)
        # Accumulate over tokens
        for t in range(0, L):
            attn_val = tl.load(attn_flat_ptr + i * L + t)  # scalar
            Kc_chunk = tl.load(Kc_flat_ptr + t * Dc + d_range)  # [BLOCK_D]
            out_vec += attn_val * Kc_chunk
        tl.store(out_flat_ptr + i * Dc + d_range, out_vec, mask=mask_d)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Shapes
        B = q_nope.shape[0]
        H = q_nope.shape[1]
        Dc = q_nope.shape[2]
        Dp = q_pe.shape[2]
        assert H == 16, "num_qo_heads must be 16"
        assert Dc == 512, "head_dim_ckv must be 512"
        assert Dp == 64, "head_dim_kpe must be 64"
        # Device setup
        device = q_nope.device

        # Squeeze size-1 dim on cache to get [P, D]
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [P, Dc]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [P, Dp]

        # Allocate outputs
        output = torch.empty((B, H, Dc), dtype=torch.bfloat16, device=device)
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        inv_ln2 = 1.0 / math.log(2.0)

        for b in range(B):
            # Number of tokens for this batch element
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            if L_tokens <= 0:
                lse[b] = torch.zeros((H,), dtype=torch.float32, device=device)
                # output[b, :] zero
                for i in range(H):
                    output[b, i] = torch.zeros((Dc,), dtype=torch.bfloat16, device=device)
                continue

            # Gather token indices for this batch element
            tok_idx = kv_indices[int(kv_indptr[b].item()):int(kv_indptr[b + 1].item())].contiguous()  # [L_tokens]

            # 1) Gather rows from caches into contiguous Kc_flat and Kp_flat (float32)
            Kc_flat = torch.empty((L_tokens * Dc,), dtype=torch.float32, device=device)
            Kp_flat = torch.empty((L_tokens * Dp,), dtype=torch.float32, device=device)

            grid_gather = (L_tokens,)
            gather_rows_kernel[grid_gather](Kc_all, tok_idx, Kc_flat, L_tokens, Dc)
            Kc = Kc_flat.view(L_tokens, Dc)  # [L_tokens, Dc]

            gather_rows_kernel[grid_gather](Kp_all, tok_idx, Kp_flat, L_tokens, Dp)
            Kp = Kp_flat.view(L_tokens, Dp)  # [L_tokens, Dp]

            # 2) Compute logits_scaled per (head, token) in Triton
            qn_flat = q_nope[b].to(torch.float32).contiguous().view(-1)  # [H*Dc]
            qp_flat = q_pe[b].to(torch.float32).contiguous().view(-1)   # [H*Dp]
            logits_scaled = torch.empty((H * L_tokens,), dtype=torch.float32, device=device)

            grid_logit = (H, L_tokens)
            compute_logits_kernel[grid_logit](
                qn_flat, qp_flat, Kc.contiguous().view(-1), Kp.contiguous().view(-1),
                logits_scaled, H=H, Dc=Dc, Dp=Dp, L=L_tokens, sm_scale=float(sm_scale)
            )

            # 3) Triton reductions: row-wise max and sum-exp
            m = torch.empty((H,), dtype=torch.float32, device=device)
            sum_exp = torch.empty((H,), dtype=torch.float32, device=device)
            row_max_kernel[(H,)](
                logits_scaled, m, H=H, L=L_tokens, BLOCK_L=128
            )
            row_sumexp_kernel[(H,)](
                logits_scaled, sum_exp, m, H=H, L=L_tokens, BLOCK_L=128
            )

            # 4) lse in base-2
            lse[b] = m + torch.log(sum_exp) * inv_ln2  # [H]

            # 5) Triton softmax per row
            attn = torch.empty((H * L_tokens,), dtype=torch.float32, device=device)
            softmax_row_kernel[(H,)](
                logits_scaled, attn, H=H, L=L_tokens, BLOCK_L=128
            )
            # Reshape attn to [H, L]
            attn_2d = attn.view(H, L_tokens)

            # 6) Triton matvec: out[i, :] = attn_2d[i, :] @ Kc
            out_flat = torch.empty((H * Dc,), dtype=torch.float32, device=device)
            matvec_kernel_2d[(H,)](
                attn_2d.contiguous().view(-1), Kc.contiguous().view(-1), out_flat,
                H=H, Dc=Dc, L=L_tokens, BLOCK_D=128
            )
            out_vec = out_flat.view(H, Dc)  # [H, Dc]

            # Store output[b, i, :] as bfloat16
            for i in range(H):
                output[b, i] = out_vec[i].to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
