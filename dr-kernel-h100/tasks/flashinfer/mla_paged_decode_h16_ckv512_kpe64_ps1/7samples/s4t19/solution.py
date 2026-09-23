import math
import triton
import triton.language as tl


@triton.jit
def compute_logits_and_lse_kernel(
    qn_ptr,           # *float32, flattened [B*N*Dc]
    qp_ptr,           # *float32, flattened [B*N*Dp]
    Kc_ptr,           # *float32, flattened [P*Dc]
    Kp_ptr,           # *float32, flattened [P*Dp]
    tok_idx_ptr,      # *int32, flattened [M_b]
    attn_ptr,         # *float32, flattened [B*N*M_b]
    lse_ptr,          # *float32, flattened [B*N]
    B: tl.constexpr,      # batch size
    N: tl.constexpr,      # num heads
    Dc: tl.constexpr,     # head_dim_ckv
    Dp: tl.constexpr,     # head_dim_kpe
    M_b: tl.constexpr,    # tokens in this batch
    sm_scale: tl.constexpr,  # scaling factor
    BLOCK_T: tl.constexpr     # token chunk
):
    # program per (b,h)
    pid = tl.program_id(0)
    b = pid // N
    h = pid % N

    # base offsets
    qn_base = (b * N + h) * Dc
    qp_base = (b * N + h) * Dp

    # load q vectors
    qn_vec = tl.load(qn_ptr + qn_base + tl.arange(0, Dc))
    qp_vec = tl.load(qp_ptr + qp_base + tl.arange(0, Dp))

    # running max for logsumexp
    m = tl.full([1], -float("inf"), dtype=tl.float32)

    # loop over tokens in chunks
    for t_start in range(0, M_b, BLOCK_T):
        offs = t_start + tl.arange(0, BLOCK_T)
        mask = offs < M_b
        # load token indices
        tok = tl.load(tok_idx_ptr + offs, mask=mask, other=0)
        # compute Kc and Kp rows for this chunk
        Kc_rows = tl.load(Kc_ptr + tok * Dc + tl.arange(0, Dc), mask=mask, other=0.0)  # [BLOCK_T, Dc]
        Kp_rows = tl.load(Kp_ptr + tok * Dp + tl.arange(0, Dp), mask=mask, other=0.0)  # [BLOCK_T, Dp]

        # compute logits for each token in the chunk: [BLOCK_T]
        logits_chunk = tl.zeros([BLOCK_T], dtype=tl.float32)
        # qn · Kc.T + qp · Kp.T
        for i in range(Dc):
            logits_chunk += qn_vec[i] * Kc_rows[:, i]
        for i in range(Dp):
            logits_chunk += qp_vec[i] * Kp_rows[:, i]
        logits_scaled = logits_chunk * sm_scale

        # update running max
        chunk_max = tl.max(tl.where(mask, logits_scaled, -float("inf")))
        m_new = tl.maximum(m, chunk_max)
        # exp sum
        exp_sum = tl.sum(tl.exp(logits_scaled - m_new), mask=mask)
        m = m_new

        # store attention weights (scaled logits)
        tl.store(attn_ptr + (b * N + h) * M_b + offs, logits_scaled, mask=mask)

    # finalize base-2 LSE
    ln2 = 0.6931471805599453
    lse_val = m + tl.log(exp_sum / ln2)
    tl.store(lse_ptr + b * N + h, lse_val)


@triton.jit
def matvec_proj_kernel(
    attn_ptr,         # *float32, flattened [B*N*M_b]
    Kc_ptr,           # *float32, flattened [P*Dc]
    out_ptr,          # *float32, flattened [B*N*Dc]
    B: tl.constexpr,      # batch size
    N: tl.constexpr,      # num heads
    Dc: tl.constexpr,     # head_dim_ckv
    M_b: tl.constexpr,    # tokens
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr
):
    # program per (b,h)
    pid = tl.program_id(0)
    b = pid // N
    h = pid % N

    # base output offset for this (b,h)
    out_base = (b * N + h) * Dc

    # accumulate output vector [Dc]
    out_vec = tl.zeros([Dc], dtype=tl.float32)

    # loop over tokens in chunks and reduce
    for t_start in range(0, M_b, BLOCK_T):
        offs_t = t_start + tl.arange(0, BLOCK_T)
        mask_t = offs_t < M_b

        # load attention weights for this chunk
        attn_chunk = tl.load(attn_ptr + (b * N + h) * M_b + offs_t, mask=mask_t, other=0.0)  # [BLOCK_T]
        # loop over Dc in chunks and accumulate
        for d_start in range(0, Dc, BLOCK_D):
            offs_d = d_start + tl.arange(0, BLOCK_D)
            mask_d = offs_d < Dc

            # vector accumulator for this chunk of D
            acc = tl.zeros([BLOCK_D], dtype=tl.float32)

            # dot over tokens in chunk
            for t in range(BLOCK_T):
                tt = t_start + t
                mtt = tt < M_b
                # load alpha_t = attn[b,h,tt] if valid
                alpha_t = tl.load(attn_ptr + (b * N + h) * M_b + tt, mask=mtt, other=0.0)
                # load Kc rows for all d in chunk
                Kc_chunk = tl.load(Kc_ptr + tt * Dc + offs_d, mask=mask_d & mtt, other=0.0)  # [BLOCK_D]
                # accumulate
                acc += Kc_chunk * alpha_t

            # store acc into out_vec
            out_vec[offs_d] += acc  # masked store via offsets

    # write final out_vec
    tl.store(out_ptr + out_base + tl.arange(0, Dc), out_vec)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # you may add constants if needed
        self.block_t = 128  # token chunk
        self.block_d = 64   # head chunk (Dc=512, use 64 for good performance)

    def forward(self, *args):
        # Expect: q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale, _unused (ignore last if present)
        assert len(args) >= 7, "Not enough inputs"
        q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale, *_ = args

        # Ensure tensors are on CUDA device
        device = q_nope.device
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "All tensors must be CUDA"

        # Shapes
        B = q_nope.shape[0]
        N = q_nope.shape[1]
        P = ckv_cache.shape[0]  # total cached tokens
        Dc = q_nope.shape[2]
        Dp = q_pe.shape[2]

        # Prepare outputs
        output = torch.empty((B, N, Dc), dtype=torch.float32, device=device)
        lse = torch.empty((B, N), dtype=torch.float32, device=device)

        # Process each batch element
        for b in range(B):
            # compute M_b and tok_idx
            M_b = int(kv_indptr[b + 1].item() - kv_indptr[b].item())
            tok_idx = kv_indices[kv_indptr[b]: kv_indptr[b + 1]].to(torch.int32).to(device)

            # Kc_sub and Kp_sub as contiguous rows
            Kc_sub = ckv_cache[tok_idx].contiguous().to(torch.float32)  # [M_b, Dc]
            Kp_sub = kpe_cache[tok_idx].contiguous().to(torch.float32)  # [M_b, Dp]

            # Flatten q for kernel
            qn_flat = q_nope[b].contiguous().to(torch.float32).view(N * Dc)
            qp_flat = q_pe[b].contiguous().to(torch.float32).view(N * Dp)

            # Allocate attention buffer
            attn = torch.empty((B * N * M_b), dtype=torch.float32, device=device)

            # Launch compute_logits_and_lse_kernel for this batch
            grid = (B * N,)
            compute_logits_and_lse_kernel[grid](
                qn_flat,
                qp_flat,
                Kc_sub.contiguous().view(-1),
                Kp_sub.contiguous().view(-1),
                tok_idx,
                attn,
                lse[b],
                B=B, N=N, Dc=Dc, Dp=Dp, M_b=M_b, sm_scale=float(sm_scale),
                BLOCK_T=self.block_t
            )

            # Launch matvec projection kernel
            matvec_proj_kernel[grid](
                attn,
                Kc_sub.contiguous().view(-1),
                output.view(B * N, Dc),
                B=B, N=N, Dc=Dc, M_b=M_b,
                BLOCK_T=self.block_t, BLOCK_D=self.block_d
            )

        # Cast output to bfloat16 to match original
        output = output.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
