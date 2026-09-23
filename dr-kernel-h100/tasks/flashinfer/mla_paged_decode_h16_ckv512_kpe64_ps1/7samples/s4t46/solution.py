import torch
import math
import triton
import triton.language as tl


@triton.jit
def compute_logsumexp_and_attn_kernel(
    qn_ptr,            # *float32, flattened [B*N*Dc]
    qp_ptr,            # *float32, flattened [B*N*Dp]
    Kc_ptr,            # *float32, flattened [P*Dc] (ckv_cache squeezed)
    Kp_ptr,            # *float32, flattened [P*Dp] (kpe_cache squeezed)
    tok_idx_ptr,       # *int32, flattened [M_b]
    attn_ptr,          # *float32, flattened [B*N*M_b] (will store attention weights)
    lse_ptr,           # *float32, flattened [B*N] (will store base-2 logsumexp)
    B: tl.constexpr,   # int
    N: tl.constexpr,   # int (num_qo_heads)
    Dc: tl.constexpr,  # int (head_dim_ckv, e.g., 512)
    Dp: tl.constexpr,  # int (head_dim_kpe, e.g., 64)
    M_b: tl.constexpr, # int (tokens in this batch)
    Kc_size: tl.constexpr,   # int (P, total cached tokens)
    sm_scale: tl.constexpr,   # float scaling
    BLOCK_N: tl.constexpr      # token tile size
):
    # program per (b, h)
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    # Compute base offsets for this (b, h)
    base_qn = (pid_b * N + pid_h) * Dc
    base_qp = (pid_b * N + pid_h) * Dp

    # Load qn and qp as vectors
    qn_vec = tl.load(qn_ptr + base_qn + tl.arange(0, Dc))
    qp_vec = tl.load(qp_ptr + base_qp + tl.arange(0, Dp))

    # Accumulate max and sum for logsumexp
    m = tl.full([1], -float("inf"), dtype=tl.float32)
    sum_exp = tl.zeros([1], dtype=tl.float32)

    # Loop over tokens in chunks
    for t0 in range(0, M_b, BLOCK_N):
        offs = t0 + tl.arange(0, BLOCK_N)
        mask = offs < M_b
        tok = tl.load(tok_idx_ptr + offs, mask=mask, other=0)
        # Pointer to Kc row for this token
        Kc_row_ptrs = Kc_ptr + tok * Dc + tl.arange(0, Dc)
        Kp_row_ptrs = Kp_ptr + tok * Dp + tl.arange(0, Dp)
        Kc_row = tl.load(Kc_row_ptrs, mask=mask, other=0.0)  # [BLOCK_N, Dc]
        Kp_row = tl.load(Kp_row_ptrs, mask=mask, other=0.0)  # [BLOCK_N, Dp]
        # Compute logits for this chunk: (qn @ Kc) + (qp @ Kp)
        logits_qn = tl.dot(qn_vec, Kc_row.T)                  # [BLOCK_N]
        logits_qp = tl.dot(qp_vec, Kp_row.T)                 # [BLOCK_N]
        logits = logits_qn + logits_qp
        logits_scaled = logits * sm_scale
        # Numerically stable logsumexp in base-2 for this chunk
        # m = max(m, max(logits_scaled)), sum_exp += sum(exp(logits_scaled - m))
        chunk_max = tl.max(tl.where(mask, logits_scaled, -float("inf")), axis=0)
        new_m = tl.maximum(m, chunk_max)
        # sum over masked elements: masked ones contribute 0
        sum_chunk = tl.sum(tl.where(mask, tl.exp(logits_scaled - new_m), 0.0), axis=0)
        m = new_m
        sum_exp += sum_chunk

    # LSE in base-2
    lse = m + tl.log(sum_exp) / tl.log(2.0)
    # Store lse for (b, h)
    tl.store(lse_ptr + pid_b * N + pid_h, lse)

    # Store attention weights: attn[b,h,t] = exp(logits_scaled[t] - lse) / M_b
    # Recompute logits per token and write attn
    for t0 in range(0, M_b, BLOCK_N):
        offs = t0 + tl.arange(0, BLOCK_N)
        mask = offs < M_b
        tok = tl.load(tok_idx_ptr + offs, mask=mask, other=0)
        Kc_row_ptrs = Kc_ptr + tok * Dc + tl.arange(0, Dc)
        Kp_row_ptrs = Kp_ptr + tok * Dp + tl.arange(0, Dp)
        Kc_row = tl.load(Kc_row_ptrs, mask=mask, other=0.0)
        Kp_row = tl.load(Kp_row_ptrs, mask=mask, other=0.0)
        logits_qn = tl.dot(qn_vec, Kc_row.T)
        logits_qp = tl.dot(qp_vec, Kp_row.T)
        logits = logits_qn + logits_qp
        logits_scaled = logits * sm_scale
        attn_row = tl.exp(logits_scaled - lse) / M_b
        tl.store(attn_ptr + (pid_b * N + pid_h) * M_b + offs, attn_row, mask=mask)


@triton.jit
def matvec_proj_kernel(
    attn_ptr,          # *float32, [B*N*M_b]
    Kc_ptr,            # *float32, [P*Dc]
    out_ptr,           # *float32, [B*N*Dc]
    B: tl.constexpr,   # int
    N: tl.constexpr,   # int
    Dc: tl.constexpr,  # int
    M_b: tl.constexpr, # int
    BLOCK_D: tl.constexpr  # int (tile along Dc, e.g., 64 or 128)
):
    # One program per (b, h)
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    # out[h, :] = sum_t attn[b,h,t] * Kc[t,:]
    # We accumulate across tokens M_b using BLOCK_D tiling across Dc
    out = tl.zeros([Dc], dtype=tl.float32)

    for t0 in range(0, M_b):
        attn_val = tl.load(attn_ptr + (pid_b * N + pid_h) * M_b + t0)
        # For each chunk along Dc
        for d0 in range(0, Dc, BLOCK_D):
            d = d0 + tl.arange(0, BLOCK_D)
            mask = d < Dc
            Kc_row = tl.load(Kc_ptr + t0 * Dc + d, mask=mask, other=0.0)
            # out[d] += attn_val * Kc_row
            out += tl.where(mask, attn_val * Kc_row, 0.0)

    tl.store(out_ptr + (pid_b * N + pid_h) * Dc + tl.arange(0, Dc), out)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Tunable kernel params
        self.block_n = 128  # for token chunks
        self.block_d = 64   # for Dc reduction

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale, *unused):
        # Ensure device and dtype
        device = q_nope.device
        B = q_nope.shape[0]
        N = q_nope.shape[1]
        Dc = q_nope.shape[2]
        assert q_pe.shape[0] == B and q_pe.shape[1] == N and q_pe.shape[2] == 64, "q_pe shape must be [B, N, 64]"
        # Prepare flattened qn and qp
        qn_flat = q_nope.contiguous().view(-1).to(torch.float32)  # [B*N*Dc]
        qp_flat = q_pe.contiguous().view(-1).to(torch.float32)    # [B*N*Dp]

        # Squeeze cached caches (as in original)
        Kc_all = ckv_cache.squeeze(1).contiguous().view(-1).to(torch.float32)  # [P*Dc]
        Kp_all = kpe_cache.squeeze(1).contiguous().view(-1).to(torch.float32)  # [P*Dp]

        # Prepare tok_idx per batch
        # len_indptr should be [B+1], kv_indices [M]
        # M_b per batch
        M_b_list = []
        for b in range(B):
            begin = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            M_b = end - begin
            if M_b <= 0:
                M_b = 1  # minimal guard, though original asserts end > begin
            M_b_list.append(M_b)
        # We need to run per-batch. Triton grid requires fixed shape; we launch per batch in a loop.

        # Output buffers
        out = torch.empty((B, N, Dc), dtype=torch.float32, device=device)  # we will cast to bfloat16 at the end
        lse = torch.empty((B, N), dtype=torch.float32, device=device)

        for b in range(B):
            M_b = M_b_list[b]
            tok_idx = kv_indices[int(kv_indptr[b].item()): int(kv_indptr[b + 1].item())].to(torch.int32).contiguous()

            # Launch fused kernel: grid = (1, N), compute lse and attn for this batch
            grid = (1, N)
            compute_logsumexp_and_attn_kernel[grid](
                qn_flat, qp_flat, Kc_all, Kp_all, tok_idx, out[b].contiguous(), lse[b].contiguous(),
                B, N, Dc, 64, M_b, Kc_all.numel() // Dc, sm_scale, self.block_n
            )

            # Now compute projection per head: grid = (1, N)
            grid_proj = (1, N)
            matvec_proj_kernel[grid_proj](
                out[b].view(N, M_b), Kc_all, out[b].view(N, Dc),
                1, N, Dc, M_b, self.block_d
            )

        # Cast to bfloat16 to match original
        out = out.to(torch.bfloat16)

        return out, lse


def run(*args):
    return ModelNew()(*args)
