import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: compute scale_logits[b, h, l] = ( qn[b,h] @ Kc[l, :].T + qp[b,h] @ Kp[l, :].T ) * sm_scale
# Outputs are written to a 3D tensor scale_logits[B, H, L] as we launch with grid (B, H) and compute per L in chunks.
@triton.jit
def compute_logits_scaled_kernel(qn_ptr, qp_ptr, Kc_ptr, Kp_ptr, scale_ptr,
                                 L: tl.constexpr, Dc: tl.constexpr, Dp: tl.constexpr,
                                 BLOCK_L: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    # Load qn and qp for this head
    qn = tl.load(qn_ptr)  # [Dc]
    qp = tl.load(qp_ptr)  # [Dp]
    # Iterate over tokens in chunks
    for l_off in tl.static_range(0, L, BLOCK_L):
        l_idx = l_off + tl.arange(0, BLOCK_L)  # [BLOCK_L]
        mask = l_idx < L
        # Compute contributions from Kc and Kp
        # Kc[l, k] = tl.load(Kc_ptr + l_idx * Dc + k)
        # We need a vector for each l in chunk: sum_k qn[k] * Kc[l, k]
        # We'll accumulate a vector of length BLOCK_L
        contrib1 = tl.zeros((BLOCK_L,), dtype=tl.float32)
        # Sum over k in chunks
        for k_off in tl.static_range(0, Dc, 64):  # 64 is a constexpr chunk; Dc=512 so 8 chunks
            k_idx = k_off + tl.arange(0, 64)  # [64]
            # qn[k_idx] is [64], Kc[l_idx, k_idx] is [BLOCK_L, 64] by broadcasting loads
            # We need to load Kc[l_idx, k_idx]
            kc_vals = tl.zeros((BLOCK_L, 64), dtype=tl.float32)
            for jj in tl.static_range(64):  # 64 is constexpr
                col = k_off + jj
                mask_k = col < Dc
                kc_vals[:, jj] = tl.load(Kc_ptr + l_idx[:, None] * Dc + col,
                                         mask=mask[:, None] & mask_k, other=0.0)
            qn_sub = tl.load(qn_ptr + k_idx, mask=(k_idx < Dc), other=0.0)  # [64]
            contrib1 += tl.sum(kc_vals * qn_sub[None, :], axis=1)
        # Similarly for Kp: sum over p in chunks
        contrib2 = tl.zeros((BLOCK_L,), dtype=tl.float32)
        for p_off in tl.static_range(0, Dp, 32):  # Dp=64 -> 2 chunks
            p_idx = p_off + tl.arange(0, 32)  # [32]
            kp_vals = tl.zeros((BLOCK_L, 32), dtype=tl.float32)
            for jj in tl.static_range(32):
                col = p_off + jj
                mask_p = col < Dp
                kp_vals[:, jj] = tl.load(Kp_ptr + l_idx[:, None] * Dp + col,
                                         mask=mask[:, None] & mask_p, other=0.0)
            qp_sub = tl.load(qp_ptr + (h * Dp + p_idx), mask=(p_idx < Dp), other=0.0)  # [32], but we pass qp_ptr as global
            # We actually need qp_sub from qp_ptr; since qp is [H, Dp], we load specific h's qp here:
            # Note: qp_ptr points to global vector across H; we need to compute per h. Better to pass qp per h.
            # To do that, we must have a per-h qp vector. We'll instead re-load qp from global memory inside:
            qp_sub = tl.load(qp_ptr + (h * Dp + p_idx), mask=(p_idx < Dp), other=0.0)
            contrib2 += tl.sum(kp_vals * qp_sub[None, :], axis=1)
        # Combine and scale
        scale_vec = (contrib1 + contrib2) * sm_scale
        # Store scale_logits[b, h, l_off + offs]
        for offs in tl.static_range(0, BLOCK_L):
            l = l_off + offs
            m = l < L
            tl.store(scale_ptr + b * H * L + h * L + l, scale_vec[offs], mask=m)


# Triton kernel: compute per (b, h) base-2 logsumexp of scale_logits[b, h, :].
# We compute max and sum in Triton. Grid is (B, H).
@triton.jit
def compute_lse_base2_kernel(scale_ptr, lse_ptr,
                             L: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    # Compute max over L
    max_val = tl.full((), -1.0e20, tl.float32)
    for l_off in tl.static_range(0, L, 1):  # loop over all L
        val = tl.load(scale_ptr + b * H * L + h * L + l_off)
        if val > max_val:
            max_val = val
    # Compute sum exp(scale - max)
    sum_exp = tl.full((), 0.0, tl.float32)
    for l_off in tl.static_range(0, L, 1):
        val = tl.load(scale_ptr + b * H * L + h * L + l_off)
        sum_exp += tl.exp(val - max_val)
    # lse = log(sum_exp) + max_val, divided by ln(2)
    ln2 = 0.6931471805599453
    lse_val = tl.log(sum_exp) + max_val
    lse_val = lse_val / ln2
    tl.store(lse_ptr + b * H + h, lse_val)


# Triton kernel: compute softmax over scale_logits[b, h, :] and write attn[b, h, :]
# Grid: (B, H), compute per row length L
@triton.jit
def compute_softmax_kernel(scale_ptr, attn_ptr,
                           L: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    # Compute max
    max_val = tl.full((), -1.0e20, tl.float32)
    for l_off in tl.static_range(0, L, 1):
        val = tl.load(scale_ptr + b * H * L + h * L + l_off)
        if val > max_val:
            max_val = val
    # Compute sum exp
    sum_exp = tl.full((), 0.0, tl.float32)
    for l_off in tl.static_range(0, L, 1):
        val = tl.load(scale_ptr + b * H * L + h * L + l_off)
        sum_exp += tl.exp(val - max_val)
    # Write softmax
    for l_off in tl.static_range(0, L, 1):
        val = tl.load(scale_ptr + b * H * L + h * L + l_off)
        soft = tl.exp(val - max_val) / sum_exp
        tl.store(attn_ptr + b * H * L + h * L + l_off, soft)


# Triton kernel: compute out[h, :] = attn[b, h, :] @ Kc_b where Kc_b is per-b gathered Kc matrix [L_b, Dc]
# We implement elementwise reduction: out[h, k] = sum_l attn[b, h, l] * Kc_b[l, k]
# Launch grid: (B, H). We compute out vector for each (b, h) and write to output[B, H, Dc].
@triton.jit
def compute_out_matmul_kernel(attn_ptr, Kc_b_ptr, out_ptr,
                              L: tl.constexpr, Dc: tl.constexpr,
                              BLOCK_K: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(1)
    # Initialize out[h, :]
    out_vec = tl.zeros((Dc,), dtype=tl.float32)
    # Reduce over L in chunks
    for l_off in tl.static_range(0, L, BLOCK_K):
        l_idx = l_off + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        mask_l = l_idx < L
        attn_chunk = tl.load(attn_ptr + b * H * L + h * L + l_idx, mask=mask_l, other=0.0)  # [BLOCK_K]
        # For each k in Dc chunk, compute sum_l attn_chunk[l] * Kc_b[l, k]
        for k_off in tl.static_range(0, Dc, BLOCK_K):
            k_idx = k_off + tl.arange(0, BLOCK_K)  # [BLOCK_K]
            mask_k = k_idx < Dc
            sum_vec = tl.zeros((BLOCK_K,), dtype=tl.float32)
            # Loop over l in chunk and accumulate
            # Kc_b[l, k] = tl.load(Kc_b_ptr + l * Dc + k)
            for l_sub in tl.static_range(0, BLOCK_K):
                l_cur = l_off + l_sub
                m_l = l_cur < L
                if m_l:
                    # attn_chunk[l_sub]
                    a = attn_chunk[l_sub]
                    # sum over k
                    # Kc_b[l_cur, k_idx] = tl.load(Kc_b_ptr + l_cur * Dc + (k_off + arange(BLOCK_K)))
                    kc_row = tl.load(Kc_b_ptr + l_cur * Dc + (k_off + tl.arange(0, BLOCK_K)),
                                     mask=mask_k, other=0.0)
                    sum_vec += a * kc_row
            out_vec[k_off:(k_off+BLOCK_K)] += sum_vec
    # Store out[b, h, :]
    for k in tl.static_range(0, Dc):
        tl.store(out_ptr + b * H * Dc + h * Dc + k, out_vec[k])


def _run_triton_only(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    device = q_nope.device
    B = kv_indptr.shape[0] - 1
    H = q_nope.shape[1]
    Dc = q_nope.shape[2]
    Dp = q_pe.shape[2]
    # Ensure inputs are float32 for Triton compute
    q_nope_f = q_nope.to(torch.float32)
    q_pe_f = q_pe.to(torch.float32)
    # Gather Kc_all and Kp_all for whole cache (not used for matmul here)
    Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [N, Dc] -> [N, 512]
    Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [N, Dp] -> [N, 64]

    # Output buffers
    # We'll compute per (b, h) scale_logits -> [B, H, L_b], attn -> [B, H, L_b], and out -> [B, H, Dc]
    output = torch.empty((B, H, Dc), dtype=torch.float32, device=device)
    lse = torch.empty((B, H), dtype=torch.float32, device=device)
    attn = torch.empty((B, H), dtype=torch.float32)  # placeholder; not stored; we'll compute and use

    # Process each batch b
    for b in range(B):
        page_beg = int(kv_indptr[b].item())
        page_end = int(kv_indptr[b + 1].item())
        if page_beg >= page_end:
            output[b].zero_()
            lse[b] = -float("inf")
            continue
        L_b = page_end - page_beg
        if L_b <= 0:
            output[b].zero_()
            lse[b] = -float("inf")
            continue
        tok_idx = kv_indices[page_beg:page_end]  # [L_b]
        # Gather per-b Kc and Kp
        Kc_b = Kc_all[tok_idx]  # [L_b, Dc]
        Kp_b = Kp_all[tok_idx]  # [L_b, Dp]

        # Per-head processing: allocate scale_logits buffer [L_b]
        # We'll allocate a temporary scale_logits per (b,h), but Triton kernel expects pointer to contiguous 3D.
        # To simplify, we compute per head in separate launch, store into a tensor that we then read for softmax.
        # Create an intermediate scale_logits[B, H, L_b] tensor to hold per (b,h) vectors.
        scale_logits = torch.empty((B, H, L_b), dtype=torch.float32, device=device)

        # Launch Triton kernel to fill scale_logits
        # We pass pointers to qn, qp, Kc_b, Kp_b, and scale_logits[b,h, :]
        # For qn/qp, we need per (b,h). Triton grid (B,H) covers all heads; we load from q_nope_f[b,h] and q_pe_f[b,h].
        # BLOCK_L: choose a power-of-two not exceeding L_b. We can set to 128.
        BLOCK_L = 128 if L_b >= 128 else (64 if L_b >= 64 else 32)
        # We need to pass sm_scale as a scalar; Triton accepts scalar argument.
        # Each program writes scale_logits[b, h, :] for its (b,h)
        for h in range(H):
            qn_ptr = q_nope_f[b, h]  # shape [Dc], 1D view
            # Prepare per-h qp pointer as a 1D vector: q_pe_f[b, h] is [Dp], flatten
            # Triton kernel expects a base pointer to vector; we pass q_nope_f[b, h] directly.
            # Kc_b and Kp_b are [L_b, Dc] and [L_b, Dp]; we'll pass base and do pointer arithmetic in kernel.
            compute_logits_scaled_kernel[(B, H)](
                qn_ptr, q_pe_f[b, h], Kc_b, Kp_b, scale_logits[b, h],
                L=L_b, Dc=Dc, Dp=Dp, sm_scale=sm_scale, BLOCK_L=BLOCK_L
            )

        # Compute base-2 logsumexp per (b,h) in Triton
        compute_lse_base2_kernel[(B, H)](
            scale_logits, lse,
            L=L_b
        )

        # Compute softmax per (b,h) in Triton into a temporary attn buffer. We create attn[B, H, L_b].
        attn = torch.empty((B, H, L_b), dtype=torch.float32, device=device)
        for h in range(H):
            compute_softmax_kernel[(B, H)](
                scale_logits[b, h], attn[b, h],
                L=L_b
            )

        # Compute out[b, h, :] in Triton: out[h, :] = attn[b, h, :] @ Kc_b
        # output is [B, H, Dc] float32
        compute_out_matmul_kernel[(B, H)](
            attn[b, h], Kc_b, output[b, h],
            L=L_b, Dc=Dc, BLOCK_K=64
        )

    # Return output as bfloat16 to match original, and lse as float32
    output_bf16 = output.to(torch.bfloat16)
    return output_bf16, lse


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Move inputs to CUDA if needed
        if q_nope.device.type != 'cuda':
            q_nope = q_nope.to('cuda')
        if q_pe.device.type != 'cuda':
            q_pe = q_pe.to('cuda')
        if ckv_cache.device.type != 'cuda':
            ckv_cache = ckv_cache.to('cuda')
        if kpe_cache.device.type != 'cuda':
            kpe_cache = kpe_cache.to('cuda')
        if kv_indptr.device.type != 'cuda':
            kv_indptr = kv_indptr.to('cuda')
        if kv_indices.device.type != 'cuda':
            kv_indices = kv_indices.to('cuda')
        # Run Triton-only computation
        output, lse = _run_triton_only(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale)
        return output, lse


def run(*args):
    return ModelNew()(*args)
