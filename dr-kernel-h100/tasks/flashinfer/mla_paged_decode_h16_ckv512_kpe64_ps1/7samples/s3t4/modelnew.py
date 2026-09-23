import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Kernel: compute logits_scaled[h, :] for one batch b and one head h, and store in attn_ptr
@triton.jit
def compute_logits_scaled_kernel_full(qn_ptr, qh_ptr, Kc_ptr, Kp_ptr, attn_ptr,
                                      H: tl.constexpr, L: tl.constexpr, Dc: tl.constexpr, Dp: tl.constexpr,
                                      BLOCK_L: tl.constexpr, sm_scale: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(2)
    qn_vec = tl.load(qn_ptr + h * Dc + tl.arange(0, Dc), mask=True, other=0.0)  # [Dc]
    qh_vec = tl.load(qh_ptr + h * Dp + tl.arange(0, Dp), mask=True, other=0.0)  # [Dp]
    logits = tl.zeros((L,), dtype=tl.float32)
    for offs in range(0, L, BLOCK_L):
        l_offsets = offs + tl.arange(0, BLOCK_L)
        mask = l_offsets < L
        Kc_rows = tl.load(Kc_ptr + l_offsets[:, None] * Dc + tl.arange(0, Dc), mask=mask[:, None], other=0.0)  # [BLOCK_L, Dc]
        Kp_rows = tl.load(Kp_ptr + l_offsets[:, None] * Dp + tl.arange(0, Dp), mask=mask[:, None], other=0.0)  # [BLOCK_L, Dp]
        contrib_qn = tl.sum(qn_vec[None, :] * Kc_rows, axis=1)  # [BLOCK_L]
        contrib_qh = tl.sum(qh_vec[None, :] * Kp_rows, axis=1)  # [BLOCK_L]
        logits = logits + tl.where(mask, contrib_qn + contrib_qh, 0.0)
    logits_scaled = logits * sm_scale
    tl.store(attn_ptr + l_offsets, logits_scaled, mask=True)


# Kernel: compute lse per (b,h) = logsumexp(logits_scaled) / ln(2), store into lse_out_ptr[b*H + h]
@triton.jit
def compute_lse_kernel(attn_ptr, lse_out_ptr,
                       L: tl.constexpr, H: tl.constexpr, BLOCK_L: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(2)
    max_val = tl.full((), -float('inf'), tl.float32)
    for offs in range(0, L, BLOCK_L):
        l_offsets = offs + tl.arange(0, BLOCK_L)
        mask = l_offsets < L
        vals = tl.load(attn_ptr + l_offsets, mask=mask, other=-float('inf'))
        chunk_max = tl.max(vals, axis=0)
        max_val = tl.maximum(max_val, chunk_max)
    sum_exp = tl.zeros((), dtype=tl.float32)
    for offs in range(0, L, BLOCK_L):
        l_offsets = offs + tl.arange(0, BLOCK_L)
        mask = l_offsets < L
        vals = tl.load(attn_ptr + l_offsets, mask=mask, other=-float('inf'))
        sum_exp += tl.sum(tl.exp(vals - max_val), axis=0)
    lse_val = tl.log(sum_exp) + max_val
    ln2 = 0.6931471805599453
    lse_val = lse_val / ln2
    out_idx = b * H + h
    tl.store(lse_out_ptr + out_idx, lse_val)


# Kernel: compute softmax over attn_ptr (size L) into out_ptr (size L), numerically stable
@triton.jit
def compute_softmax_kernel(attn_ptr, out_ptr,
                           L: tl.constexpr, BLOCK_L: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(2)
    max_val = tl.full((), -float('inf'), tl.float32)
    for offs in range(0, L, BLOCK_L):
        l_offsets = offs + tl.arange(0, BLOCK_L)
        mask = l_offsets < L
        vals = tl.load(attn_ptr + l_offsets, mask=mask, other=-float('inf'))
        chunk_max = tl.max(vals, axis=0)
        max_val = tl.maximum(max_val, chunk_max)
    sum_exp = tl.zeros((), dtype=tl.float32)
    for offs in range(0, L, BLOCK_L):
        l_offsets = offs + tl.arange(0, BLOCK_L)
        mask = l_offsets < L
        vals = tl.load(attn_ptr + l_offsets, mask=mask, other=-float('inf'))
        sum_exp += tl.sum(tl.exp(vals - max_val), axis=0)
    for offs in range(0, L, BLOCK_L):
        l_offsets = offs + tl.arange(0, BLOCK_L)
        mask = l_offsets < L
        vals = tl.load(attn_ptr + l_offsets, mask=mask, other=-float('inf'))
        out_vals = tl.exp(vals - max_val) / sum_exp
        tl.store(out_ptr + l_offsets, out_vals, mask=mask)


# Kernel: compute out[b, h, :] = attn_vec @ Kc, where Kc is [L, Dc]
@triton.jit
def compute_out_kernel(attn_ptr, Kc_ptr, out_ptr,
                       L: tl.constexpr, Dc: tl.constexpr, BLOCK_L: tl.constexpr):
    b = tl.program_id(0)
    h = tl.program_id(2)
    acc = tl.zeros((Dc,), dtype=tl.float32)
    for offs in range(0, L, BLOCK_L):
        l_offsets = offs + tl.arange(0, BLOCK_L)
        mask = l_offsets < L
        attn_vals = tl.load(attn_ptr + l_offsets, mask=mask, other=0.0)  # [BLOCK_L]
        Kc_rows = tl.load(Kc_ptr + l_offsets[:, None] * Dc + tl.arange(0, Dc), mask=mask[:, None], other=0.0)  # [BLOCK_L, Dc]
        contrib = attn_vals[:, None] * Kc_rows  # [BLOCK_L, Dc]
        acc += tl.sum(contrib, axis=0)  # reduce over BLOCK_L
    tl.store(out_ptr, acc)


def _run_triton_only(B, H, Dc, Dp, num_pages, kv_indptr, kv_indices, sm_scale):
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")

    device = 'cuda'
    # Ensure tensors are on device and float32
    q_nope_b = [t.to(torch.float32, device=device) for t in q_nope]  # [B, H, Dc]
    q_pe_b = [t.to(torch.float32, device=device) for t in q_pe]     # [B, H, Dp]
    Kc_all = ckv_cache.squeeze(1).to(torch.float32, device=device)  # [num_pages, 512]
    Kp_all = kpe_cache.squeeze(1).to(torch.float32, device=device)  # [num_pages, 64]

    output = torch.empty((B, H, Dc), dtype=torch.float32, device=device)
    lse_out = torch.empty((B, H), dtype=torch.float32, device=device)

    for b in range(B):
        start = int(kv_indptr[b].item())
        end = int(kv_indptr[b + 1].item())
        if start >= end:
            output[b].zero_()
            lse_out[b, :] = 0.0
            continue

        tok_idx = kv_indices[start:end].to(device)  # int32
        L_tokens = end - start

        Kc = Kc_all[tok_idx]  # [L_tokens, 512]
        Kp = Kp_all[tok_idx]  # [L_tokens, 64]

        qn_b = q_nope_b[b]  # [H, 512]
        qh_b = q_pe_b[b]    # [H, 64]

        # Buffer for scaled logits
        attn_buf = torch.empty((L_tokens,), dtype=torch.float32, device=device)

        for h in range(H):
            # Compute scaled logits
            compute_logits_scaled_kernel_full[(B, H)](
                qn_b, qh_b, Kc, Kp, attn_buf,
                H=H, L=L_tokens, Dc=Dc, Dp=Dp,
                BLOCK_L=128, sm_scale=sm_scale
            )

            # Compute lse for this (b, h) and store
            compute_lse_kernel[(B, H)](
                attn_buf, lse_out,
                L=L_tokens, H=H, BLOCK_L=128
            )

            # Compute softmax and out
            softmax_vec = torch.empty((L_tokens,), dtype=torch.float32, device=device)
            compute_softmax_kernel[(B, H)](
                attn_buf, softmax_vec,
                L=L_tokens, BLOCK_L=128
            )
            out_vec = torch.empty((Dc,), dtype=torch.float32, device=device)
            compute_out_kernel[(B, 1)](
                softmax_vec, Kc, out_vec,
                L=L_tokens, Dc=Dc, BLOCK_L=128
            )
            output[b, h, :] = out_vec

    # Return output as bfloat16
    output_bf16 = output.to(torch.bfloat16)
    return output_bf16, lse_out


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        B = kv_indptr.shape[0] - 1
        H = q_nope.shape[1]
        Dc = q_nope.shape[2]
        Dp = q_pe.shape[2]
        num_pages = ckv_cache.shape[0]
        # Ensure inputs are on CUDA and float32 for compute
        q_nope = q_nope.to(device='cuda', dtype=torch.float32)
        q_pe = q_pe.to(device='cuda', dtype=torch.float32)
        ckv_cache = ckv_cache.to(device='cuda', dtype=torch.float32)
        kpe_cache = kpe_cache.to(device='cuda', dtype=torch.float32)
        kv_indptr = kv_indptr.to(device='cuda')
        kv_indices = kv_indices.to(device='cuda')
        output, lse = _run_triton_only(B, H, Dc, Dp, num_pages, kv_indptr, kv_indices, sm_scale)
        return output, lse