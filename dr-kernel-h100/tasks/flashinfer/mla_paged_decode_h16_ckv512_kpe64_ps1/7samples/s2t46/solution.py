import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: one program per (b, h). Computes output vector and lse for head h of batch b.
@triton.jit
def _compute_single_head_kernel(
    # Inputs
    q_nope_ptr,       # [B, H, Dc] fp32
    q_pe_ptr,         # [B, H, Dp] fp32
    Kc_all_ptr,       # [num_pages, Dc] fp32
    Kp_all_ptr,       # [num_pages, Dp] fp32
    tok_idx_ptr,      # [L_tokens] int32, token indices for this batch
    out_ptr,          # [B, H, Dc] fp32 (we'll store fp32 and cast later)
    lse_ptr,          # [B, H] fp32
    # Dimensions
    B: tl.constexpr, H: tl.constexpr,
    Dc: tl.constexpr, Dp: tl.constexpr,
    L_tokens: tl.constexpr,
    sm_scale: tl.float32,
):
    # We launch with grid=(B, H); pid is implicit per program
    # b is selected by the host when launching (pointer arithmetic uses b)
    # However, Triton doesn't provide b from here, so this kernel is intended to be called from ModelNew with explicit b,h.
    # To avoid confusion, we rely on host passing b via q_nope_ptr etc. by indexing correctly.

    # Step 1: Prepare qn and qp for this head (assume b is passed indirectly; this is a single program per (b,h)).
    # We need b. Triton kernel doesn't receive b directly, but since we launch with grid=(B, H), we can compute b from program_id(0).
    # Triton doesn't support pid in this snippet, so we rely on ModelNew to pass correct pointers. Here, we assume q_nope_ptr, q_pe_ptr point to [H, D] for fixed b,h.

    # Load qn and qp vectors (length Dc and Dp respectively)
    qn = tl.load(q_nope_ptr)       # shape [Dc], fp32
    qp = tl.load(q_pe_ptr)         # shape [Dp], fp32

    # Step 2: Build Kc_rows and Kp_rows: [L_tokens, Dc] and [L_tokens, Dp] by gathering rows indexed by tok_idx
    # Kc_rows[t, i] = Kc_all[tok_idx[t], i]
    # Kp_rows[t, j] = Kp_all[tok_idx[t], j]
    Kc_rows = tl.zeros((L_tokens, Dc), dtype=tl.float32)
    Kp_rows = tl.zeros((L_tokens, Dp), dtype=tl.float32)

    # Load tok_idx as a vector
    t_offsets = tl.arange(0, L_tokens)  # [L_tokens]
    tok_idx = tl.load(tok_idx_ptr + t_offsets)  # [L_tokens], int32

    # Build pointers for each row t; we iterate t via vectorized pointer arithmetic
    # For Kc_rows: row_ptr_t = Kc_all_ptr + tok_idx[t] * Dc + i
    # For Kp_rows: row_ptr_t = Kp_all_ptr + tok_idx[t] * Dp + j

    # Since Triton doesn't allow direct 2D loads from pointer array easily, we build each row:
    # But Triton supports scalar indexing; we can loop t explicitly (small L_tokens) to construct rows.
    # Loop over t scalarly; Triton supports scalar loop for small sizes.

    # We will fill Kc_rows and Kp_rows by looping t:
    for t in range(0, L_tokens):
        # i dimension
        i = tl.arange(0, Dc)
        Kc_row_ptr = Kc_all_ptr + tok_idx[t] * Dc + i
        Kc_rows[t, :] = tl.load(Kc_row_ptr)
        # j dimension
        j = tl.arange(0, Dp)
        Kp_row_ptr = Kp_all_ptr + tok_idx[t] * Dp + j
        Kp_rows[t, :] = tl.load(Kp_row_ptr)

    # Step 3: Compute logits vector [L_tokens]
    logits = tl.zeros((L_tokens,), dtype=tl.float32)

    # Dot product qn @ Kc_rows + qp @ Kp_rows per token t
    # qn[i], Kc_rows[t, i] with i in [0..Dc-1]
    for t in range(0, L_tokens):
        sum_qn = 0.0
        for i in range(0, Dc):
            sum_qn += qn[i] * Kc_rows[t, i]
        sum_qp = 0.0
        for j in range(0, Dp):
            sum_qp += qp[j] * Kp_rows[t, j]
        logits[t] = sum_qn + sum_qp

    # Scale and compute lse (stable)
    scaled = logits * sm_scale
    m = tl.max(scaled)
    sum_exp = 0.0
    for t in range(0, L_tokens):
        sum_exp += tl.exp(scaled[t] - m)
    lse_bh = (m + tl.log(sum_exp)) / tl.log(2.0)
    tl.store(lse_ptr, lse_bh)

    # Compute attention vector
    attn = tl.zeros((L_tokens,), dtype=tl.float32)
    for t in range(0, L_tokens):
        attn[t] = tl.exp(scaled[t] - lse_bh) / tl.log(2.0)

    # Compute output vector: out[h, :] = sum_t attn[t] * Kc_rows[t, :]
    out_vec = tl.zeros((Dc,), dtype=tl.float32)
    for t in range(0, L_tokens):
        # out_vec += attn[t] * Kc_rows[t, :]
        # Broadcast attn[t] to [Dc] by multiplying with vector of ones or just loop-wise add scalar times vector
        # Triton allows scalar times vector in expression; emulate via loop adding scalar to out_vec
        out_vec += attn[t] * Kc_rows[t, :]

    # Store out vector
    tl.store(out_ptr, out_vec)


def _run_triton_only(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    """
    Triton-only runner: computes output and lse using Triton kernels.
    q_nope: [B, H, Dc] bfloat16
    q_pe: [B, H, Dp] bfloat16
    ckv_cache: [num_pages, Dc] bfloat16
    kpe_cache: [num_pages, Dp] bfloat16
    kv_indptr: [len_indptr] int32
    kv_indices: [num_kv_indices] int32
    sm_scale: float
    Returns: (out [B, H, Dc] bfloat16, lse [B, H] float32)
    """
    B, H, Dc = q_nope.shape
    _, _, Dp = q_pe.shape
    num_pages = ckv_cache.shape[0]

    # Prepare outputs
    out = torch.empty((B, H, Dc), dtype=torch.float32, device=q_nope.device)  # fp32 compute, cast later
    lse = torch.empty((B, H), dtype=torch.float32, device=q_nope.device)

    # Ensure inputs contiguous and fp32
    q_nope_f32 = q_nope.to(torch.float32).contiguous()  # [B, H, Dc]
    q_pe_f32 = q_pe.to(torch.float32).contiguous()     # [B, H, Dp]
    Kc_all = ckv_cache.to(torch.float32).contiguous()  # [num_pages, Dc]
    Kp_all = kpe_cache.to(torch.float32).contiguous()  # [num_pages, Dp]
    kv_indptr = kv_indptr.to(torch.int32)
    kv_indices = kv_indices.to(torch.int32)

    # For each batch element, compute tok_idx = kv_indices[kv_indptr[b]:kv_indptr[b+1]]
    for b in range(B):
        page_beg = int(kv_indptr[b].item())
        page_end = int(kv_indptr[b + 1].item())
        L_tokens = page_end - page_beg
        if L_tokens <= 0:
            # No tokens for this batch element
            # Initialize outputs to zeros and set lse to -inf
            out[b].zero_()
            lse[b].fill_(-float("inf"))
            continue

        tok_idx = kv_indices[page_beg:page_end].to(torch.int32).contiguous()  # [L_tokens]

        # Launch Triton kernel for each head
        for h in range(H):
            # Prepare pointers: q_nope[b, h, :] and q_pe[b, h, :]
            qn_ptr = q_nope_f32[b, h, :]  # 1D vector
            qp_ptr = q_pe_f32[b, h, :]    # 1D vector
            # Output and lse pointers
            out_ptr = out[b, h, :]        # 1D vector
            lse_ptr = lse[b]              # scalar
            # Launch kernel: one program per (b,h)
            _compute_single_head_kernel[(1,)](
                qn_ptr, qp_ptr, Kc_all, Kp_all, tok_idx, out_ptr, lse_ptr,
                B=B, H=H, Dc=Dc, Dp=Dp, L_tokens=L_tokens, sm_scale=float(sm_scale),
                num_warps=4, num_stages=2
            )

    # Cast output to bfloat16 and return
    out_bf16 = out.to(torch.bfloat16)
    return out_bf16, lse


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        """
        Triton-only forward: computes output and lse using Triton kernels.
        """
        # Ensure tensors are on CUDA for Triton
        if (not TRITON_AVAILABLE) or (not q_nope.is_cuda):
            raise RuntimeError("Triton is not available or tensors are not on CUDA")
        out, lse = _run_triton_only(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale)
        return out, lse


def run(*args):
    return ModelNew()(*args)
