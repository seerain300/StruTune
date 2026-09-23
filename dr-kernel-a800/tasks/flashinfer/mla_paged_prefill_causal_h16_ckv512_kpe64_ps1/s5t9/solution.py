import torch
import math
import triton
import triton.language as tl


# Matmul: out[H, L] = qn[H, D] @ kc[L, D]^T
@triton.jit
def matmul_qn_kc(qn_ptr, kc_ptr, out_ptr,
                 H: tl.constexpr, D: tl.constexpr, L: tl.constexpr,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    m = H
    n = L
    k = D

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, k, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        q = tl.load(
            qn_ptr + (offs_m[:, None] * D + offs_k[None, :]),
            mask=(offs_m[:, None] < m) & (offs_k[None, :] < k),
            other=0.0
        )  # [BLOCK_M, BLOCK_K]
        # kc[n, k] -> tile [offs_k, offs_n]
        k_tile = tl.load(
            kc_ptr + (offs_n[None, :] * D + offs_k[:, None]),
            mask=(offs_k[:, None] < k) & (offs_n[None, :] < n),
            other=0.0
        )  # [BLOCK_K, BLOCK_N]
        acc += tl.dot(q, k_tile)

    out_offsets = offs_m[:, None] * L + offs_n[None, :]
    tl.store(
        out_ptr + out_offsets,
        acc,
        mask=(offs_m[:, None] < m) & (offs_n[None, :] < n)
    )


# Matmul: out[H, L] = qp[H, P] @ kp[L, P]^T
@triton.jit
def matmul_qp_kp(qp_ptr, kp_ptr, out_ptr,
                 H: tl.constexpr, P: tl.constexpr, L: tl.constexpr,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    m = H
    n = L
    k = P

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, k, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        q = tl.load(
            qp_ptr + (offs_m[:, None] * P + offs_k[None, :]),
            mask=(offs_m[:, None] < m) & (offs_k[None, :] < k),
            other=0.0
        )  # [BLOCK_M, BLOCK_K]
        k_tile = tl.load(
            kp_ptr + (offs_n[None, :] * P + offs_k[:, None]),
            mask=(offs_k[:, None] < k) & (offs_n[None, :] < n),
            other=0.0
        )  # [BLOCK_K, BLOCK_N]
        acc += tl.dot(q, k_tile)

    out_offsets = offs_m[:, None] * L + offs_n[None, :]
    tl.store(
        out_ptr + out_offsets,
        acc,
        mask=(offs_m[:, None] < m) & (offs_n[None, :] < n)
    )


# Add: C = A + B
@triton.jit
def add_logits(A_ptr, B_ptr, C_ptr,
               H: tl.constexpr, L: tl.constexpr,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    m = H
    n = L

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    A = tl.load(
        A_ptr + (offs_m[:, None] * L + offs_n[None, :]),
        mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
        other=0.0
    )
    B = tl.load(
        B_ptr + (offs_m[:, None] * L + offs_n[None, :]),
        mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
        other=0.0
    )
    C = A + B

    out_offsets = offs_m[:, None] * L + offs_n[None, :]
    tl.store(
        C_ptr + out_offsets,
        C,
        mask=(offs_m[:, None] < m) & (offs_n[None, :] < n)
    )


# Scale: C = A * scale
@triton.jit
def scale_logits(A_ptr, C_ptr, scale, H: tl.constexpr, L: tl.constexpr,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    m = H
    n = L

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    A = tl.load(
        A_ptr + (offs_m[:, None] * L + offs_n[None, :]),
        mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
        other=0.0
    )
    C = A * scale

    out_offsets = offs_m[:, None] * L + offs_n[None, :]
    tl.store(
        C_ptr + out_offsets,
        C,
        mask=(offs_m[:, None] < m) & (offs_n[None, :] < n)
    )


# Apply mask: D_masked[h, j] = D[h, j] if j > query_abs_pos else -inf
# We process per head h in blocks across j.
@triton.jit
def apply_mask(D_ptr, out_ptr, L: tl.constexpr, query_abs_pos: tl.int32,
               BLOCK: tl.constexpr):
    h = tl.program_id(0)  # one program per head
    for k in range(0, L, BLOCK):
        offs = k + tl.arange(0, BLOCK)
        D = tl.load(D_ptr + h * L + offs, mask=offs < L, other=0.0)
        keep = offs > query_abs_pos
        D_masked = tl.where(keep, D, -float("inf"))
        tl.store(out_ptr + h * L + offs, D_masked, mask=offs < L)


# Row-wise logsumexp: lse[h] = log(sum_j exp(D[h, j] - m[h])) + m[h], per head h
@triton.jit
def row_logsumexp_masked(D_ptr, lse_ptr, L: tl.constexpr, BLOCK: tl.constexpr):
    h = tl.program_id(0)  # one program per head
    # First pass: max
    m = -float("inf")
    for k in range(0, L, BLOCK):
        offs = k + tl.arange(0, BLOCK)
        D = tl.load(D_ptr + h * L + offs, mask=offs < L, other=-float("inf"))
        # Clamp invalid to -inf
        D = tl.where(offs < L, D, -float("inf"))
        m = tl.maximum(m, tl.max(D, axis=0))
    # Second pass: sumexp
    sumexp = 0.0
    for k in range(0, L, BLOCK):
        offs = k + tl.arange(0, BLOCK)
        D = tl.load(D_ptr + h * L + offs, mask=offs < L, other=-float("inf"))
        # Shift by max
        D = tl.where(offs < L, D - m, -float("inf"))
        # exp(-inf) = 0
        sumexp += tl.sum(tl.exp(D), axis=0)
    lse = m + tl.log(sumexp)  # log2(sumexp) is not needed; original uses natural log
    tl.store(lse_ptr + h, lse)


# Row-wise softmax using precomputed lse[h]
# Softmax[h, j] = exp(D[h, j] - lse[h]) / sum_k exp(D[h, k] - lse[h])
@triton.jit
def softmax_row_masked(D_ptr, lse_ptr, out_ptr, L: tl.constexpr, BLOCK: tl.constexpr):
    h = tl.program_id(0)  # one program per head
    # Compute denominator
    denom = 0.0
    for k in range(0, L, BLOCK):
        offs = k + tl.arange(0, BLOCK)
        D = tl.load(D_ptr + h * L + offs, mask=offs < L, other=0.0)
        D = D - tl.load(lse_ptr + h)
        e = tl.exp(D)
        denom += tl.sum(e, axis=0)
    # Write normalized outputs
    for k in range(0, L, BLOCK):
        offs = k + tl.arange(0, BLOCK)
        D = tl.load(D_ptr + h * L + offs, mask=offs < L, other=0.0)
        D = D - tl.load(lse_ptr + h)
        e = tl.exp(D)
        out = e / denom
        tl.store(out_ptr + h * L + offs, out, mask=offs < L)


# Matmul: out[H, D] = attn[H, L] @ kc[L, D]
@triton.jit
def matmul_attn_kc(attn_ptr, kc_ptr, out_ptr,
                   H: tl.constexpr, L: tl.constexpr, D: tl.constexpr,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    m = H
    n = D
    k = L

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, k, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        attn = tl.load(
            attn_ptr + (offs_m[:, None] * L + offs_k[None, :]),
            mask=(offs_m[:, None] < m) & (offs_k[None, :] < k),
            other=0.0
        )  # [BLOCK_M, BLOCK_K]
        k_tile = tl.load(
            kc_ptr + (offs_n[None, :] * D + offs_k[:, None]),
            mask=(offs_k[:, None] < k) & (offs_n[None, :] < n),
            other=0.0
        )  # [BLOCK_K, BLOCK_N]
        acc += tl.dot(attn, k_tile)

    out_offsets = offs_m[:, None] * D + offs_n[None, :]
    tl.store(
        out_ptr + out_offsets,
        acc,
        mask=(offs_m[:, None] < m) & (offs_n[None, :] < n)
    )


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Prepare constants and tensors
        device = q_nope.device
        H = 16
        D = 512
        P = 64

        # Prepare Kc_all and Kp_all: squeeze batch dim
        Kc_all = ckv_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, D]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, P]

        total_q = int(qo_indptr[1].item()) - int(qo_indptr[0].item())
        batch_size = int(kv_indptr[1].item()) - int(kv_indptr[0].item())
        num_qo_heads = H
        head_dim_ckv = D
        head_dim_kpe = P

        # Output buffers
        output = torch.empty((total_q, H, D), dtype=torch.bfloat16, device=device)
        lse = torch.full((total_q, H), -float("inf"), dtype=torch.float32, device=device)

        # Loop over batches
        for b in range(batch_size):
            # Compute q_start, q_end using indptr[b] and indptr[b+1]
            qo_start = int(qo_indptr[b].item())
            qo_end = int(qo_indptr[b + 1].item())

            # Compute tok_idx range and Kc, Kp for this batch
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())
            num_keys = kv_end - kv_start
            if num_keys <= 0 or qo_end - qo_start <= 0:
                continue

            tok_idx = kv_indices[kv_start:kv_end].to(torch.int64)  # [num_keys]
            Kc = Kc_all[tok_idx].to(torch.float32)  # [num_keys, D]
            Kp = Kp_all[tok_idx].to(torch.float32)  # [num_keys, P]

            # Slice q_nope and q_pe for this batch
            q_nope_b = q_nope[qo_start:qo_end].to(torch.float32)  # [Q, H, D]
            q_pe_b = q_pe[qo_start:qo_end].to(torch.float32)      # [Q, H, P]
            Q = q_nope_b.shape[0]

            # Loop over queries in the batch
            for i in range(Q):
                # Compute prefix_len = number of previously processed tokens
                # prefix_len = num_keys - Q  # NOT CORRECT: see mask logic per query
                # However for lse, we only need max and sumexp of logits[h, :].
                # We'll handle mask based on per-query query_abs_pos.

                # Prepare qn and qp for this query
                qn = q_nope_b[i].contiguous()  # [H, D]
                qp = q_pe_b[i].contiguous()    # [H, P]

                # Allocate intermediates
                A = torch.empty((H, num_keys), dtype=torch.float32, device=device)  # qn @ Kc.T
                B = torch.empty((H, num_keys), dtype=torch.float32, device=device)  # qp @ Kp.T
                logits = torch.empty((H, num_keys), dtype=torch.float32, device=device)  # A + B
                logits_scaled = torch.empty((H, num_keys), dtype=torch.float32, device=device)  # scaled
                D_masked = torch.empty((H, num_keys), dtype=torch.float32, device=device)  # masked
                lse_vec = torch.empty((H,), dtype=torch.float32, device=device)            # lse per head

                # Launch matmul qn @ Kc.T
                grid_qn = (triton.cdiv(H, 16), triton.cdiv(num_keys, 64))
                matmul_qn_kc[grid_qn](
                    qn, Kc, A,
                    H, D, num_keys,
                    16, 64, 64
                )

                # Launch matmul qp @ Kp.T
                grid_qp = (triton.cdiv(H, 16), triton.cdiv(num_keys, 64))
                matmul_qp_kp[grid_qp](
                    qp, Kp, B,
                    H, P, num_keys,
                    16, 64, 64
                )

                # Add
                grid_add = (triton.cdiv(H, 16), triton.cdiv(num_keys, 64))
                add_logits[grid_add](
                    A, B, logits,
                    H, num_keys,
                    16, 64
                )

                # Scale
                grid_scale = (triton.cdiv(H, 16), triton.cdiv(num_keys, 64))
                scale_logits[grid_scale](
                    logits, logits_scaled,
                    sm_scale,
                    H, num_keys,
                    16, 64
                )

                # Compute query_abs_pos: j > (num_keys - Q + i)
                # We want to keep j where j > (num_keys - Q + i). If num_keys - Q + i < 0, keep all.
                prefix_len = num_keys - Q
                query_abs_pos = prefix_len + i
                if query_abs_pos < 0:
                    # Keep all positions
                    D_masked.copy_(logits_scaled)
                else:
                    # Apply mask
                    # Note: BLOCK_N can be set to 128 or 64; here set to 128 for efficiency
                    grid_mask = (H,)
                    apply_mask[grid_mask](
                        logits_scaled, D_masked,
                        num_keys, query_abs_pos,
                        128
                    )

                # Row-wise logsumexp over D_masked: lse = logsumexp(D_masked) per head
                grid_lse = (H,)
                row_logsumexp_masked[grid_lse](
                    D_masked, lse_vec,
                    num_keys, 64
                )

                # Compute softmax per row using lse_vec
                attn = torch.empty((H, num_keys), dtype=torch.float32, device=device)
                grid_softmax = (H,)
                softmax_row_masked[grid_softmax](
                    D_masked, lse_vec, attn,
                    num_keys, 128
                )

                # Final output: out[h] = attn[h] @ Kc -> [H, D]
                out_q = torch.empty((H, D), dtype=torch.float32, device=device)
                grid_matmul_out = (triton.cdiv(H, 16), triton.cdiv(D, 64))
                matmul_attn_kc[grid_matmul_out](
                    attn, Kc, out_q,
                    H, num_keys, D,
                    16, 64, 64
                )

                # Store results
                out_row = out_q.to(torch.bfloat16)
                output[qo_start + i] = out_row
                lse[qo_start + i] = lse_vec

        return output, lse


def run(*args):
    return ModelNew()(*args)
