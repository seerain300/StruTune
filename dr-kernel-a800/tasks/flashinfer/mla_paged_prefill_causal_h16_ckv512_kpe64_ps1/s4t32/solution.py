import math
import torch
import triton
import triton.language as tl


@triton.jit
def compute_dot_qn_Kc(
    qn_ptr,           # *f32, shape [H, Dc], contiguous
    Kc_ptr,           # *f32, shape [L, Dc], contiguous
    dot_ptr,          # *f32, shape [H, L], contiguous
    H: tl.constexpr,
    Dc: tl.constexpr,
    L: tl.constexpr,
    grid_dim0: tl.constexpr  # ignored but allows grid as (H, L)
):
    h = tl.program_id(0)
    t = tl.program_id(1)
    qn_base = qn_ptr + h * Dc
    Kc_row = Kc_ptr + t * Dc
    acc = 0.0
    for d in range(0, Dc, 64):
        offs = d + tl.arange(0, 64)
        mask = offs < Dc
        qn_vec = tl.load(qn_base + offs, mask=mask, other=0.0)
        Kc_vec = tl.load(Kc_row + offs, mask=mask, other=0.0)
        acc += tl.sum(qn_vec * Kc_vec, axis=0)
    tl.store(dot_ptr + h * L + t, acc)


@triton.jit
def compute_dot_qp_Kp(
    qp_ptr,           # *f32, shape [H, Dp], contiguous
    Kp_ptr,           # *f32, shape [L, Dp], contiguous
    dot_ptr,          # *f32, shape [H, L], contiguous
    H: tl.constexpr,
    Dp: tl.constexpr,
    L: tl.constexpr,
    grid_dim0: tl.constexpr
):
    h = tl.program_id(0)
    t = tl.program_id(1)
    qp_base = qp_ptr + h * Dp
    Kp_row = Kp_ptr + t * Dp
    acc = 0.0
    for d in range(0, Dp, 64):
        offs = d + tl.arange(0, 64)
        mask = offs < Dp
        qp_vec = tl.load(qp_base + offs, mask=mask, other=0.0)
        Kp_vec = tl.load(Kp_row + offs, mask=mask, other=0.0)
        acc += tl.sum(qp_vec * Kp_vec, axis=0)
    tl.store(dot_ptr + h * L + t, acc)


@triton.jit
def compute_logits_lse_attn(
    dot_qn_ptr,       # *f32, shape [H, L]
    dot_qp_ptr,       # *f32, shape [H, L]
    lse_out_ptr,      # *f32, shape [H]
    attn_out_ptr,     # *f32, shape [H, L] (flattened)
    H: tl.constexpr,
    L: tl.constexpr,
    sm_scale: tl.constexpr,
    i_abs: tl.constexpr  # absolute query position
):
    h = tl.program_id(0)  # one program per head
    logits = tl.zeros((L,), dtype=tl.float32)
    for t in range(0, L):
        logits[t] = tl.load(dot_qn_ptr + h * L + t) + tl.load(dot_qp_ptr + h * L + t)
    # causal mask
    for t in range(0, L):
        if t > i_abs:
            logits[t] = -float('inf')

    row_max = -float('inf')
    for t in range(0, L):
        row_max = tl.maximum(row_max, logits[t])
    sumexp = 0.0
    for t in range(0, L):
        sumexp += tl.exp(logits[t] - row_max)
    # base-2 logsumexp
    lse = row_max + math.log(sumexp) / math.log(2.0)
    tl.store(lse_out_ptr + h, lse)

    denom = 0.0
    for t in range(0, L):
        denom += tl.exp(logits[t] - lse)
    for t in range(0, L):
        attn_out_ptr[h * L + t] = tl.exp(logits[t] - lse) / denom


@triton.jit
def matmul_vec_by_mat(
    attn_ptr,         # *f32, shape [H, L], flattened
    Kc_ptr,           # *f32, shape [L, Dc], contiguous
    out_ptr,          # *f32, shape [Dc]
    H: tl.constexpr,
    L: tl.constexpr,
    Dc: tl.constexpr,
    h_idx: tl.constexpr
):
    h = h_idx
    out_vec = tl.zeros((Dc,), dtype=tl.float32)
    for t in range(0, L, 64):
        offs_t = t + tl.arange(0, 64)
        mask_t = offs_t < L
        attn_vec = tl.load(attn_ptr + h * L + offs_t, mask=mask_t, other=0.0)  # [64]
        acc_partial = tl.zeros((Dc,), dtype=tl.float32)
        for d in range(0, Dc, 64):
            offs_d = d + tl.arange(0, 64)
            mask_d = offs_d < Dc
            Kc_row = tl.load(Kc_ptr + offs_t * Dc + offs_d, mask=mask_d, other=0.0)  # [64]
            acc_partial += tl.sum(attn_vec[:, None] * Kc_row[None, :], axis=0)
        out_vec += acc_partial
    for d in range(0, Dc):
        tl.store(out_ptr + d, out_vec[d])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Move inputs to CUDA (assumed available)
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "Triton requires CUDA tensors"
        device = q_nope.device

        # Shapes
        total_q, H, Dc = q_nope.shape
        _, _, Dp = q_pe.shape
        len_indptr = qo_indptr.numel()
        batch_size = len_indptr - 1

        # Output buffer in float32 (we will cast to bfloat16 before returning to match original)
        output_fp32 = torch.empty((total_q, H, Dc), dtype=torch.float32, device=device)
        lse_out = torch.empty((total_q, H), dtype=torch.float32, device=device)  # placeholder, not used in return

        # For each batch element
        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            if q_start >= q_end:
                continue

            # Build tok_idx for this batch element: indices into kv_indptr window
            tok_idx = kv_indices[kv_indptr[b]: kv_indptr[b + 1]].to(torch.int32)  # [L]
            L = tok_idx.numel()

            # Build Kc and Kp for this batch segment: [L, Dc] and [L, Dp]
            Kc_batch = ckv_cache[tok_idx].to(torch.float32)  # [L, Dc]
            Kp_batch = kpe_cache[tok_idx].to(torch.float32)  # [L, Dp]

            # For each query i in this batch segment
            for i in range(q_start, q_end):
                # Prepare qn[h, :] and qp[h, :] flattened for Triton
                qn_flat = q_nope[i].reshape(H * Dc).to(torch.float32)  # [H*Dc]
                qp_flat = q_pe[i].reshape(H * Dp).to(torch.float32)   # [H*Dp]

                # Allocate intermediate buffers
                dot_qn = torch.empty((H * L,), dtype=torch.float32, device=device)
                dot_qp = torch.empty((H * L,), dtype=torch.float32, device=device)
                attn_out = torch.empty((H * L,), dtype=torch.float32, device=device)

                # Launch dot kernels: grid (H, L)
                compute_dot_qn_Kc[(H, L)](qn_flat, Kc_batch, dot_qn, H=H, Dc=Dc, L=L, grid_dim0=H)
                compute_dot_qp_Kp[(H, L)](qp_flat, Kp_batch, dot_qp, H=H, Dp=Dp, L=L, grid_dim0=H)

                # Compute logits, lse, attn per head: launch grid (H,)
                lse_h = torch.empty((H,), dtype=torch.float32, device=device)
                i_abs = i  # absolute query position
                compute_logits_lse_attn[(H,)](dot_qn, dot_qp, lse_h, attn_out, H=H, L=L, sm_scale=sm_scale, i_abs=i_abs)

                # Reshape attn_out to [H, L] for matmul
                attn_2d = attn_out.view(H, L)

                # Compute output[i, h, :] = sum_t attn[h, t] * Kc[t, :]
                for h_idx in range(H):
                    out_vec = torch.empty((Dc,), dtype=torch.float32, device=device)
                    matmul_vec_by_mat


def run(*args):
    return ModelNew()(*args)
