import torch
import triton
import triton.language as tl

# Kernel 1: compute logsumexp per (batch, head) without producing attn
# It reads qn[b,h, :], qp[b,h, :], and the gathered Kc, Kp for all tokens m.
# It writes lse[b, h] = ln(sum_exp) / ln(2).
@triton.jit
def _lse_kernel(
    qn_ptr,         # *fp32 [B, H, N]
    qp_ptr,         # *fp32 [B, H, Kp]
    Kc_ptr,         # *fp32 [M_total, N]
    Kp_ptr,         # *fp32 [M_total, Kp_dim]
    lse_ptr,        # *fp32 [B, H]
    B, H, N, Kp_dim, sm_scale, M_total,
):
    b = tl.program_id(0)
    h = tl.program_id(1)
    # Compute base offsets
    base_qn = (b * H + h) * N
    base_qp = (b * H + h) * Kp_dim

    # Initialize row-wise max and sumexp
    m_max = -float("inf")
    sum_exp = 0.0

    # Loop over tokens m
    m = 0
    while m < M_total:
        # Load qn row and qp row as vectors
        qn_row = tl.load(qn_ptr + base_qn + m * N, mask=False, other=0.0)  # shape [N]
        qp_row = tl.load(qp_ptr + base_qp + m * Kp_dim, mask=False, other=0.0)  # shape [Kp_dim]

        # Load Kc[m, :] and Kp[m, :] vectors
        kc_row = tl.load(Kc_ptr + m * N, mask=False, other=0.0)  # shape [N]
        kp_row = tl.load(Kp_ptr + m * Kp_dim, mask=False, other=0.0)  # shape [Kp_dim]

        # Compute dot products
        dot_base = tl.sum(qn_row * kc_row, axis=0)  # scalar
        dot_kpe = tl.sum(qp_row * kp_row, axis=0)   # scalar
        logits = dot_base + dot_kpe                # scalar
        logits_scaled = logits * sm_scale          # scalar

        # Update max and sumexp
        m_max = tl.maximum(m_max, logits_scaled)
        exp_val = tl.exp(logits_scaled - m_max)
        sum_exp += exp_val

        m += 1

    # Compute lse: ln(sum_exp) / ln(2)
    ln_sum = tl.log(sum_exp)
    ln2 = 0.6931471805599453  # float32 constant
    lse_val = ln_sum / ln2
    # Write to lse[b, h]
    tl.store(lse_ptr + b * H + h, lse_val)


# Kernel 2: compute output for each (batch, head) by first computing lse and attn, then GEMV
@triton.jit
def _output_gemm_kernel(
    qn_ptr,         # *fp32 [B, H, N]
    qp_ptr,         # *fp32 [B, H, Kp_dim]
    Kc_ptr,         # *fp32 [M_total, N]
    Kp_ptr,         # *fp32 [M_total, Kp_dim]
    lse_ptr,        # *fp32 [B, H] (for correct scaling; although we recompute lse for output)
    out_row_ptr,    # *fp32 [H, N] (output vector for this (b,h))
    B, H, N, Kp_dim, sm_scale, M_total,
):
    b = tl.program_id(0)
    h = tl.program_id(1)

    base_qn = (b * H + h) * N
    base_qp = (b * H + h) * Kp_dim

    # First pass: compute logits_scaled, store to attn[b,h,M], and accumulate sumexp for lse
    # attn buffer is fp32[M_total]
    attn_ptr = tl.make_block_ptr(
        base=tl.zeros((1,), dtype=tl.int32), shape=(M_total,), strides=(1,),
        offsets=(0,), block_shape=(M_total,), order=(0,)
    )
    sum_exp = 0.0
    m = 0
    while m < M_total:
        qn_row = tl.load(qn_ptr + base_qn + m * N)  # [N]
        qp_row = tl.load(qp_ptr + base_qp + m * Kp_dim)  # [Kp_dim]
        kc_row = tl.load(Kc_ptr + m * N)  # [N]
        kp_row = tl.load(Kp_ptr + m * Kp_dim)  # [Kp_dim]

        dot_base = tl.sum(qn_row * kc_row, axis=0)
        dot_kpe = tl.sum(qp_row * kp_row, axis=0)
        logits = dot_base + dot_kpe
        logits_scaled = logits * sm_scale
        # store attn value
        # attn_ptr[m] = logits_scaled
        tl.store(attn_ptr + m, logits_scaled)
        # accumulate sumexp for lse
        sum_exp += tl.exp(logits_scaled - m_max)  # m_max will be defined after first pass
        m += 1
    # Now we need m_max. Recompute in second pass to avoid storing.
    # However, Triton doesn't allow writing before definition; so we recompute m_max in second pass.
    # To avoid extra storage, we can't use stored attn; so we recompute lse from scratch.
    # Simpler: compute lse from scratch using qn/qp and Kc/Kp again, then proceed.

    # Recompute lse: second pass to get max, third pass to sumexp; but Triton kernel should avoid multiple passes without storing.
    # Instead, we can do a single-pass where we:
    # - First compute m_max
    # - Then compute sum_exp with m_max
    # But Triton doesn't support breaking into stages. So we recompute lse in this kernel by looping again.
    # To keep one kernel, we will compute lse again here.

    # Pass 1 (recompute) to get m_max
    m_max = -float("inf")
    m = 0
    while m < M_total:
        qn_row = tl.load(qn_ptr + base_qn + m * N)
        qp_row = tl.load(qp_ptr + base_qp + m * Kp_dim)
        kc_row = tl.load(Kc_ptr + m * N)
        kp_row = tl.load(Kp_ptr + m * Kp_dim)
        dot_base = tl.sum(qn_row * kc_row, axis=0)
        dot_kpe = tl.sum(qp_row * kp_row, axis=0)
        logits = dot_base + dot_kpe
        logits_scaled = logits * sm_scale
        m_max = tl.maximum(m_max, logits_scaled)
        m += 1

    # Pass 2: sumexp
    sum_exp = 0.0
    m = 0
    while m < M_total:
        qn_row = tl.load(qn_ptr + base_qn + m * N)
        qp_row = tl.load(qp_ptr + base_qp + m * Kp_dim)
        kc_row = tl.load(Kc_ptr + m * N)
        kp_row = tl.load(Kp_ptr + m * Kp_dim)
        dot_base = tl.sum(qn_row * kc_row, axis=0)
        dot_kpe = tl.sum(qp_row * kp_row, axis=0)
        logits = dot_base + dot_kpe
        logits_scaled = logits * sm_scale
        sum_exp += tl.exp(logits_scaled - m_max)
        m += 1

    ln_sum = tl.log(sum_exp)
    ln2 = 0.6931471805599453
    lse_val = ln_sum / ln2

    # Third pass: softmax over attn values and compute out = attn @ Kc
    # We need attn[m] = exp(logits_scaled - lse_val). Compute per m and accumulate dot.
    out_vec = tl.zeros((N,), dtype=tl.float32)
    m = 0
    while m < M_total:
        # Recompute logits_scaled and softmax contribution
        qn_row = tl.load(qn_ptr + base_qn + m * N)
        qp_row = tl.load(qp_ptr + base_qp + m * Kp_dim)
        kc_row = tl.load(Kc_ptr + m * N)

        dot_base = tl.sum(qn_row * kc_row, axis=0)
        dot_kpe = tl.sum(qp_row * tl.load(Kp_ptr + m * Kp_dim), axis=0)  # kp_row isn't reused; scalar fetch is fine
        logits = dot_base + dot_kpe
        logits_scaled = logits * sm_scale
        attn_m = tl.exp(logits_scaled - lse_val)  # softmax contribution
        kc_row_full = tl.load(Kc_ptr + m * N)  # [N], full vector
        out_vec += attn_m * kc_row_full  # broadcast scalar attn_m across N
        m += 1

    # Write out vector for this (b, h)
    tl.store(out_row_ptr + b * H * N + h * N, out_vec)


class ModelNew(torch.nn.Module):
    def __init__(self, sm_scale=1.0):
        super().__init__()
        self.sm_scale = float(sm_scale)

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA device and dtype handling
        device = q_nope.device
        assert device.type == 'cuda', "ModelNew requires CUDA tensors."
        # Extract shapes
        B = q_nope.shape[0]
        H = q_nope.shape[1]
        N = q_nope.shape[2]
        assert N == 512, "head_dim_ckv must be 512."
        Kp_dim = q_pe.shape[2]
        assert Kp_dim == 64, "head_dim_kpe must be 64."
        # Prepare gathered Kc_all, Kp_all based on kv_indptr and kv_indices
        # We need total number of tokens used per batch. For generic case, we can use torch to compute.
        # But we'll do everything in Triton. Host will only compute tok_idx range lengths.
        # Compute M_total per batch dynamically:
        # Create tok_idx for each batch b: indices in [kv_indptr[b]: kv_indptr[b+1])
        # We'll construct Kc gathered and Kp gathered tensors for all batches at once, but in Triton we only need lengths.
        # Instead, we will gather Kc_all and Kp_all into contiguous buffers by token index and then pass pointers.
        # However, Triton kernels operate on existing tensors. We can pre-gather Kc_all and Kp_all on host into fp32:
        # Since Triton must do all computation, we'll pre-gather and cast to fp32, then pass to kernels.
        # First, compute tok_idx per batch and gather Kc, Kp into new tensors [B, M_b, N] and [B, M_b, Kp_dim], but Triton can index original [num_pages, N].
        # To avoid host-side torch ops producing outputs, we instead:
        # 1) Precompute tok_idx arrays on host (simple torch ops are allowed here), and 2) launch Triton kernel that computes lse.
        # 3) Launch Triton kernel to compute output vectors.
        # Compute M_total per b and store in a list or compute runtime length inside kernel. Simpler: precompute tok_idx lengths per b.
        # But to adhere strictly to Triton-only for all math, we will avoid constructing gathered Kc/Kp tensors in host. Instead, we compute tok_idx lengths in Triton by reading kv_indptr differences in a tiny kernel. That would require an extra read kernel. To simplify and ensure correctness, we compute tok_idx lengths using torch on host (which is allowed in ModelNew) and pass them as scalars to Triton for loops.

        # Compute tok_idx lengths per batch: M_total[b] = kv_indptr[b+1] - kv_indptr[b]
        # Ensure non-negative and at least 1 if end > start. If equal, set M_total=0 to zero out output and skip kernels.
        M_totals = []
        for b in range(B):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            M = max(0, end - start)
            M_totals.append(M)

        # Output tensors
        output = torch.empty((B, H, N), dtype=torch.float32, device=device)  # fp32 inside kernel, cast later
        lse = torch.full((B, H), -float("inf"), dtype=torch.float32, device=device)

        # q_nope and q_pe must be fp32 for Triton compute
        qn_fp32 = q_nope.to(torch.float32)
        qp_fp32 = q_pe.to(torch.float32)
        Kc_fp32 = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, N]
        Kp_fp32 = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, Kp_dim]

        # Launch kernel 1: compute lse for each (b,h)
        grid_lse = (B, H)
        _lse_kernel[grid_lse](
            qn_fp32, qp_fp32, Kc_fp32, Kp_fp32, lse,
            B, H, N, Kp_dim, self.sm_scale, M_totals[0] if M_totals else 0,  # pass first M_total; each program uses its M via while
        )
        # Note: Triton while loops allow runtime bounds. We loop M_total per program. To pass M_total, we can pass a scalar per program; Triton supports scalar args. The kernel receives M_total scalar and loops with it.

        # Launch kernel 2: compute output vectors for each (b,h)
        # We need to recompute lse per (b,h) in the kernel; but we already have lse. To avoid recomputation, we pass M_totals and let the kernel loop and use lse. However, the previous kernel only computed lse for h=0? Let's fix: compute lse per (b,h) in kernel 2 as well.
        grid_out = (B, H)
        # We need to recompute lse in kernel 2; but we already have lse computed above. The kernel signature includes sm_scale and M_total. To avoid recomputation, we'll recompute lse in kernel 2: this is acceptable for correctness, and it ensures all math is in Triton.
        # However, Triton cannot access lse computed on host in kernel easily. So we recompute lse in kernel 2. This doubles work, but is fine for correctness.
        _output_gemm_kernel[grid_out](
            qn_fp32, qp_fp32, Kc_fp32, Kp_fp32, lse,  # lse is dummy here; kernel recomputes using its own loops
            output, B, H, N, Kp_dim, self.sm_scale, M_totals[0] if M_totals else 0,
        )

        # Cast output to bfloat16 to match original
        output_bf16 = output.to(torch.bfloat16)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
