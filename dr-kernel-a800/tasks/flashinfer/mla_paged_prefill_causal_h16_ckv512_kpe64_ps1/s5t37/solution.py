import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernels
if TRITON_AVAILABLE:
    @triton.jit
    def matmul_qn_kc(qn_ptr, kc_ptr, out_ptr,
                     H: tl.constexpr, D: tl.constexpr, L: tl.constexpr,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        # out[H, L] = qn[H, D] @ kc[L, D]^T
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
            # qn tile: [BLOCK_M, BLOCK_K]
            q = tl.load(
                qn_ptr + (offs_m[:, None] * D + offs_k[None, :]),
                mask=(offs_m[:, None] < m) & (offs_k[None, :] < k),
                other=0.0
            )
            # kc tile: kc[n, k] -> [offs_k, offs_n]
            k_tile = tl.load(
                kc_ptr + (offs_n[None, :] * D + offs_k[:, None]),
                mask=(offs_k[:, None] < k) & (offs_n[None, :] < n),
                other=0.0
            )
            acc += tl.dot(q, k_tile)

        out_offsets = offs_m[:, None] * L + offs_n[None, :]
        tl.store(
            out_ptr + out_offsets,
            acc,
            mask=(offs_m[:, None] < m) & (offs_n[None, :] < n)
        )


    @triton.jit
    def matmul_qp_kp(qp_ptr, kp_ptr, out_ptr,
                     H: tl.constexpr, P: tl.constexpr, L: tl.constexpr,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
        # out[H, L] = qp[H, P] @ kp[L, P]^T
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
            )
            k_tile = tl.load(
                kp_ptr + (offs_n[None, :] * P + offs_k[:, None]),
                mask=(offs_k[:, None] < k) & (offs_n[None, :] < n),
                other=0.0
            )
            acc += tl.dot(q, k_tile)

        out_offsets = offs_m[:, None] * L + offs_n[None, :]
        tl.store(
            out_ptr + out_offsets,
            acc,
            mask=(offs_m[:, None] < m) & (offs_n[None, :] < n)
        )


    @triton.jit
    def softmax_row(inp_ptr, out_ptr, scale_logsumexp: tl.constexpr, H: tl.constexpr, L: tl.constexpr):
        # Softmax across L for each row h, optionally dividing by log(2).
        m = H
        n = L
        pid_m = tl.program_id(0)
        offs_m = pid_m * m + tl.arange(0, m)

        # Compute per-row max
        max_val = -float("inf")
        for j in range(0, n):
            val = tl.load(inp_ptr + offs_m * n + j)
            if val > max_val:
                max_val = val

        # Compute denominator
        denom = 0.0
        for j in range(0, n):
            val = tl.load(inp_ptr + offs_m * n + j)
            denom += tl.exp(val - max_val)

        inv_denom = 1.0 / denom

        # Write softmax
        for j in range(0, n):
            val = tl.load(inp_ptr + offs_m * n + j)
            soft = tl.exp(val - max_val) * inv_denom
            tl.store(out_ptr + offs_m * n + j, soft / (math.log(2.0) if scale_logsumexp == 1 else 1.0))


    @triton.jit
    def matmul_attn_kc(soft_ptr, kc_ptr, out_ptr, H: tl.constexpr, L: tl.constexpr, D: tl.constexpr):
        # out[H, D] = soft[H, L] @ kc[L, D]
        m = H
        n = D
        k = L

        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * m + tl.arange(0, m)
        offs_n = pid_n * n + tl.arange(0, n)
        acc = tl.zeros((m, n), dtype=tl.float32)

        for k0 in range(0, k, 128):
            offs_k = k0 + tl.arange(0, 128)
            a = tl.load(soft_ptr + (offs_m[:, None] * k + offs_k[None, :]), mask=(offs_m[:, None] < m) & (offs_k[None, :] < k), other=0.0)
            b = tl.load(kc_ptr + (offs_k[:, None] * D + offs_n[None, :]), mask=(offs_k[:, None] < k) & (offs_n[None, :] < n), other=0.0)
            acc += tl.dot(a, b)

        out_offsets = offs_m[:, None] * D + offs_n[None, :]
        tl.store(out_ptr + out_offsets, acc, mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure device consistency
        device = q_nope.device
        H = 16
        D = 512
        P = 64

        total_q = int(qo_indptr[-1].item())
        len_indptr = qo_indptr.numel()

        # Allocate outputs
        output = torch.zeros((total_q, H, D), dtype=torch.bfloat16, device=device)
        lse = torch.full((total_q, H), -float("inf"), dtype=torch.float32, device=device)

        # Iterate over batches
        for b in range(len_indptr - 1):
            # Compute q range and L
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            Q = q_end - q_start  # number of queries in this batch

            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())
            L = kv_end - kv_start

            # Skip if no queries or no kv
            if Q <= 0 or L <= 0:
                continue

            # Prepare Kc and Kp (CPU-side metadata; no heavy compute here)
            Kc = ckv_cache.squeeze(1).to(torch.float32)[kv_start:kv_end]  # [L, D]
            Kp = kpe_cache.squeeze(1).to(torch.float32)[kv_start:kv_end]  # [L, P]

            # Process each query in this batch
            for i in range(Q):
                qn = q_nope[q_start + i].to(torch.float32).contiguous()  # [H, D]
                qp = q_pe[q_start + i].to(torch.float32).contiguous()   # [H, P]

                # Matmul: logits_qn [H, L] = qn @ Kc.T
                logits_qn = torch.empty((H, L), dtype=torch.float32, device=device)
                BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64
                grid = (triton.cdiv(H, BLOCK_M), triton.cdiv(L, BLOCK_N))
                matmul_qn_kc[grid](qn, Kc.transpose(0, 1).contiguous(), logits_qn, H, D, L, BLOCK_M, BLOCK_N, BLOCK_K)

                # Matmul: logits_qp [H, L] = qp @ Kp.T
                logits_qp = torch.empty((H, L), dtype=torch.float32, device=device)
                matmul_qp_kp[grid](qp, Kp.transpose(0, 1).contiguous(), logits_qp, H, P, L, BLOCK_M, BLOCK_N, BLOCK_K)

                # Add and scale
                logits = logits_qn + logits_qp
                logits_scaled = logits * sm_scale

                # Apply causal mask: j <= (L - Q + i) -> -inf
                # Construct mask in torch for robustness
                query_abs_pos = (L - Q) + i
                idxs = torch.arange(L, device=device)
                mask = (idxs > query_abs_pos)
                masked = torch.where(mask, logits_scaled, torch.tensor(-float("inf"), device=device))

                # Compute lse per row using torch: lse[h] = logsumexp(masked[h]) (natural log)
                row_max = torch.amax(masked, dim=1)                      # [H]
                sumexp = torch.sum(torch.exp(masked - row_max.view(-1, 1)), dim=1)  # [H]
                lse[q_start + i] = torch.log(sumexp) + row_max          # [H]

                # Softmax over L per row using lse
                softmax = torch.exp(masked - lse[q_start + i].view(-1, 1))  # [H, L]

                # Final output: softmax @ Kc -> [H, D]
                out_row = torch.empty((H, D), dtype=torch.float32, device=device)
                BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 128
                grid = (triton.cdiv(H, BLOCK_M), triton.cdiv(D, BLOCK_N))
                matmul_attn_kc[grid](softmax, Kc, out_row, H, L, D, BLOCK_M, BLOCK_N, BLOCK_K)

                # Store output in bfloat16
                output[q_start + i] = out_row.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
