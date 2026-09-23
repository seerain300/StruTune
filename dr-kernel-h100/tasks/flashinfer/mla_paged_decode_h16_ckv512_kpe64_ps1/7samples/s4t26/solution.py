import torch
import math
import triton
import triton.language as tl


@triton.jit
def fused_logits_lse_kernel(
    qn_ptr,            # *float32, [B, N, Dc] flattened
    qp_ptr,            # *float32, [B, N, Dp] flattened
    Kc_ptr,            # *float32, [M_total, Dc] flattened, this is the per-batch subset Kc_sub (we set M_total=M_b and pass the right pointer)
    Kp_ptr,            # *float32, [M_total, Dp] flattened, this is the per-batch subset Kp_sub (we set M_total=M_b and pass the right pointer)
    tok_idx_ptr,       # *int32, [M_total], token indices for this batch (we set M_total=M_b and pass the right pointer)
    attn_ptr,          # *float32, [B, N, M_total] flattened, will store attention weights
    lse_ptr,           # *float32, [B, N] flattened, will store base-2 LSE
    B: tl.constexpr,   # batch size
    N: tl.constexpr,   # number of qo heads
    Dc: tl.constexpr,  # head_dim_ckv (512)
    Dp: tl.constexpr,  # head_dim_kpe (64)
    M_total: tl.constexpr,  # number of tokens for this batch (M_b)
    sm_scale: tl.constexpr, # scaling factor
    BLOCK_N: tl.constexpr,  # token tile size (e.g., 128)
    BLOCK_D: tl.constexpr   # vector tile size for Kc dimension (e.g., 32)
):
    # program ids: one per (b, h)
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    # Base offsets for this (b, h)
    # qn_ptr layout: [B, N, Dc] contiguous -> linear index = ((b*N + h)*Dc + d)
    qn_base = (pid_b * N + pid_h) * Dc
    # qp_ptr layout: [B, N, Dp] contiguous -> linear index = ((b*N + h)*Dp + d)
    qp_base = (pid_b * N + pid_h) * Dp

    # Load qn and qp vectors
    qn = tl.load(qn_ptr + qn_base + tl.arange(0, Dc))  # [Dc]
    qp = tl.load(qp_ptr + qp_base + tl.arange(0, Dp))  # [Dp]

    # Accumulators for logits_scaled
    logits_scaled = tl.zeros([BLOCK_N], dtype=tl.float32)  # will hold up to BLOCK_N tokens
    # m for logsumexp stability
    m = tl.full([1], -float("inf"), dtype=tl.float32)

    # Loop over tokens in chunks of BLOCK_N
    for start in range(0, M_total, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < M_total

        # Load token indices
        tok_idx = tl.load(tok_idx_ptr + offs, mask=mask, other=0)  # [BLOCK_N], int32
        # Convert to pointer offsets for Kc/Kp: Kc_ptr and Kp_ptr are [M_total, Dc/Dp], so linear index = tok_idx * Dc/Dp + d
        Kc_offsets = tok_idx * Dc + tl.arange(0, BLOCK_D)  # [BLOCK_N, BLOCK_D]
        Kp_offsets = tok_idx * Dp + tl.arange(0, BLOCK_D)  # [BLOCK_N, BLOCK_D]

        # Load Kc and Kp blocks
        Kc_block = tl.load(Kc_ptr + Kc_offsets, mask=mask[:, None], other=0.0)  # [BLOCK_N, BLOCK_D]
        Kp_block = tl.load(Kp_ptr + Kp_offsets, mask=mask[:, None], other=0.0)  # [BLOCK_N, BLOCK_D]

        # Compute logits for this chunk: sum over D tiles
        sum1 = tl.zeros([BLOCK_N], dtype=tl.float32)
        for d in range(0, Dc, BLOCK_D):
            kd = d + tl.arange(0, BLOCK_D)
            Kc_part = Kc_block[:, kd]  # [BLOCK_N, BLOCK_D]
            # matmul qn [Dc] x Kc_part [BLOCK_N, BLOCK_D] -> [BLOCK_N, BLOCK_D]
            # For each d block, sum along kd: but here we need to align qn [Dc] with Kc_part [BLOCK_N, BLOCK_D]
            # We'll do a pairwise dot per BLOCK_N by broadcasting qn across D block
            # Instead, compute pairwise for each d in BLOCK_D: sum_j qn[j] * Kc_part[i,j]
            sum_block = tl.zeros([BLOCK_N], dtype=tl.float32)
            for j in range(BLOCK_D):
                col = Kc_part[:, j]  # [BLOCK_N]
                sum_block += qn[kd[j]] * col
            sum1 += sum_block

        sum2 = tl.zeros([BLOCK_N], dtype=tl.float32)
        for d in range(0, Dp, BLOCK_D):
            kp = d + tl.arange(0, BLOCK_D)
            Kp_part = Kp_block[:, kp]  # [BLOCK_N, BLOCK_D]
            sum_block = tl.zeros([BLOCK_N], dtype=tl.float32)
            for j in range(BLOCK_D):
                col = Kp_part[:, j]  # [BLOCK_N]
                sum_block += qp[kp[j]] * col
            sum2 += sum_block

        logits = sum1 + sum2  # [BLOCK_N]
        logits_scaled = tl.where(mask, logits_scaled + logits, logits_scaled)

        # Update max for logsumexp
        local_max = tl.max(logits_scaled, axis=0)  # scalar
        m = tl.maximum(m, local_max)

        # Store attention weights for this chunk
        # attn layout: [B, N, M_total], linear index = ((b*N + h)*M_total + tok)
        attn_base = (pid_b * N + pid_h) * M_total
        attn_offsets = start + tl.arange(0, BLOCK_N)
        attn_mask = attn_offsets < M_total
        attn_vals = tl.exp((logits_scaled - m) * sm_scale)  # scaled exp
        tl.store(attn_ptr + attn_base + attn_offsets, attn_vals, mask=attn_mask)

    # Finalize LSE: sum exp(logits_scaled - m) / sm_scale, then log2
    sum_exp = tl.zeros([1], dtype=tl.float32)
    for start in range(0, M_total, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < M_total
        attn_offsets = (pid_b * N + pid_h) * M_total + offs
        attn_chunk = tl.load(attn_ptr + attn_offsets, mask=mask, other=0.0)  # [BLOCK_N]
        sum_exp += tl.sum(attn_chunk, axis=0)

    lse_val = tl.log(1.0) + tl.log(sum_exp[0]) * (1.0 / math.log(2.0))  # log2(sum_exp)
    lse_base = pid_b * N + pid_h
    tl.store(lse_ptr + lse_base, lse_val)


@triton.jit
def matvec_kernel(
    attn_ptr,      # *float32, [B, N, M_total] flattened
    Kc_ptr,        # *float32, [M_total, Dc] flattened (this is Kc_sub)
    out_ptr,       # *float32, [B, N, Dc] flattened (we'll store per (b,h) as [N, Dc])
    B: tl.constexpr,   # batch size
    N: tl.constexpr,   # number of qo heads
    Dc: tl.constexpr,  # head_dim_ckv
    M_total: tl.constexpr,  # number of tokens
    BLOCK_D: tl.constexpr   # Dc tile size
):
    # one program per (b,h)
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    # Prepare output vector [Dc]
    out_vec = tl.zeros([Dc], dtype=tl.float32)

    # Loop over tokens to compute out_vec = sum_i attn[b,h,i] * Kc[i, :]
    # attn layout: [B, N, M_total], linear index = ((b*N + h)*M_total + i)
    attn_base = (pid_b * N + pid_h) * M_total

    for i in range(0, M_total):
        attn_val = tl.load(attn_ptr + attn_base + i)  # scalar
        Kc_row_base = i * Dc
        # Load Kc[i, :] in BLOCK_D chunks
        for d in range(0, Dc, BLOCK_D):
            kd = d + tl.arange(0, BLOCK_D)
            Kc_chunk = tl.load(Kc_ptr + Kc_row_base + kd)  # [BLOCK_D]
            out_vec[kd] += attn_val * Kc_chunk

    # Store out[b, h, :] flattened as [N, Dc] contiguous
    out_base = (pid_b * N + pid_h) * Dc
    tl.store(out_ptr + out_base + tl.arange(0, Dc), out_vec)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # You can tune these block sizes if desired
        self.block_n = 128
        self.block_d = 32

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale, _unused=None):
        # Ensure device is CUDA
        device = q_nope.device
        assert device.type == "cuda", "ModelNew requires CUDA device for Triton kernels."

        # Dimensions
        B = q_nope.shape[0]
        N = q_nope.shape[1]  # number of qo heads
        Dc = q_nope.shape[2]  # head_dim_ckv, expected 512
        Dp = q_pe.shape[2]    # head_dim_kpe, expected 64

        # Prepare output and lse
        out = torch.empty((B, N, Dc), dtype=torch.float32, device=device)
        lse = torch.empty((B, N), dtype=torch.float32, device=device)

        # Prepare attn buffer [B, N, M_total]
        attn = torch.empty((B, N, 0), dtype=torch.float32, device=device)  # dummy, will be filled by kernel

        # For each batch b, compute M_b and subset Kc_sub, Kp_sub
        # tok_idx = kv_indices[kv_indptr[b]: kv_indptr[b+1]]
        for b in range(B):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            M_b = end - start
            if M_b <= 0:
                # No tokens for this batch, output zeros and lse = -inf
                out[b].zero_()
                lse[b].fill_(-float("inf"))
                continue

            tok_idx = kv_indices[start:end].to(torch.int32).to(device)

            # Build Kc_sub and Kp_sub by indexing the original caches (host does indexing; no torch.cat)
            # Kc_all is [P, Dc], Kp_all is [P, Dp]
            Kc_sub = ckv_cache[start:end]  # [M_b, Dc]
            Kp_sub = kpe_cache[start:end]  # [M_b, Dp]
            # Flatten to [M_total, Dc] and [M_total, Dp] for kernel
            Kc_flat = Kc_sub.reshape(-1).contiguous()  # [M_b*Dc]
            Kp_flat = Kp_sub.reshape(-1).contiguous()  # [M_b*Dp]
            # Note: We need [M_total, Dc] layout for kernels. Reshape back:
            Kc_ptr = Kc_sub.reshape(M_b, Dc).contiguous().view(-1)  # [M_b*Dc] -> but reshape back for pointer: use as is
            # Create a flat view for kernel: we pass 2D pointers as 1D via reshape(M_b, Dc).contiguous().view(-1) would be wrong; instead, pass Kc_sub directly to kernel and rely on pointer arithmetic with tok_idx.
            # To pass 2D pointers correctly in Triton, we need to ensure the kernel accesses [M_total, Dc] via tok_idx and base address. Therefore, we pass Kc_sub as 2D tensor, and in kernel compute offsets as tok_idx * Dc + d.
            # Here, we avoid torch.cat and reshape: we pass Kc_sub and Kp_sub as they are, and in kernel compute offsets using tok_idx.

            # Cast q_nope and q_pe to float32 for compute
            qn = q_nope[b].to(torch.float32).reshape(-1)  # [N*Dc]
            qp = q_pe[b].to(torch.float32).reshape(-1)    # [N*Dp]

            # Launch fused_logits_lse_kernel for each head h
            grid = (B, N)
            fused_logits_lse_kernel[grid](
                qn, qp,
                Kc_sub.reshape(M_b, Dc).contiguous(),  # [M_b, Dc]
                Kp_sub.reshape(M_b, Dp).contiguous(),  # [M_b, Dp]
                tok_idx,
                attn,  # attention buffer
                lse[b],
                B, N, Dc, Dp, M_b, float(sm_scale),
                BLOCK_N=self.block_n, BLOCK_D=self.block_d
            )

            # Launch matvec_kernel: compute out[b, h, :] for each h
            grid_mat = (B, N)
            matvec_kernel[grid_mat](
                attn,  # note: attn here is per-(b,h) attention over M_b tokens; each program uses pid_b, pid_h to access correct slice
                Kc_sub.reshape(M_b, Dc).contiguous(),  # [M_b, Dc]
                out[b],
                B, N, Dc, M_b,
                BLOCK_D=self.block_d
            )

        # Cast output to bfloat16 to match original run signature
        out = out.to(torch.bfloat16)

        return out, lse


def run(*args):
    return ModelNew()(*args)
