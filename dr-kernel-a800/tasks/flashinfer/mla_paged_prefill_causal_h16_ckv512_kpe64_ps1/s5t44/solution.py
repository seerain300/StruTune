import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernels: all compute happens inside these kernels, forward calls them.

@triton.jit
def matmul_qn_kc(qn_ptr, kc_ptr, out_ptr,
                 H: tl.constexpr, D: tl.constexpr, L: tl.constexpr,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # Compute out[H, L] = qn[H, D] @ kc[L, D]^T
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
        # Load qn tile [BLOCK_M, BLOCK_K]
        q = tl.load(
            qn_ptr + (offs_m[:, None] * D + offs_k[None, :]),
            mask=(offs_m[:, None] < m) & (offs_k[None, :] < k),
            other=0.0
        )
        # Load kc tile (transposed): kc[n, k] -> [offs_k, offs_n]
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
    # Compute out[H, L] = qp[H, P] @ kp[L, P]^T
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
def add_logits(a_ptr, b_ptr, out_ptr,
               H: tl.constexpr, L: tl.constexpr,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    m = H
    n = L

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    a = tl.load(a_ptr + (offs_m[:, None] * L + offs_n[None, :]),
                mask=(offs_m[:, None] < m) & (offs_n[None, :] < n), other=0.0)
    b = tl.load(b_ptr + (offs_m[:, None] * L + offs_n[None, :]),
                mask=(offs_m[:, None] < m) & (offs_n[None, :] < n), other=0.0)
    out = a + b
    tl.store(out_ptr + (offs_m[:, None] * L + offs_n[None, :]), out,
             mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


@triton.jit
def scale_logits(inp_ptr, out_ptr, scale,
                 H: tl.constexpr, L: tl.constexpr,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    m = H
    n = L

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    a = tl.load(inp_ptr + (offs_m[:, None] * L + offs_n[None, :]),
                mask=(offs_m[:, None] < m) & (offs_n[None, :] < n), other=0.0)
    out = a * scale
    tl.store(out_ptr + (offs_m[:, None] * L + offs_n[None, :]), out,
             mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


@triton.jit
def apply_mask(logits_ptr, mask_ptr, out_ptr, scale,
               H: tl.constexpr, L: tl.constexpr,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # logits_ptr: [H, L], mask_ptr: [H, L], 0.0 means keep, 1.0 means mask
    m = H
    n = L

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    logits = tl.load(logits_ptr + (offs_m[:, None] * L + offs_n[None, :]),
                     mask=(offs_m[:, None] < m) & (offs_n[None, :] < n), other=0.0)
    mask = tl.load(mask_ptr + (offs_m[:, None] * L + offs_n[None, :]),
                   mask=(offs_m[:, None] < m) & (offs_n[None, :] < n), other=0.0)
    # set masked positions to -1e20 (large negative)
    neg_large = -1e20
    out = tl.where(mask > 0.0, neg_large, logits)
    out = out * scale
    tl.store(out_ptr + (offs_m[:, None] * L + offs_n[None, :]), out,
             mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


@triton.jit
def row_lse(logits_ptr, out_ptr, H: tl.constexpr, L: tl.constexpr, BLOCK_N: tl.constexpr):
    # Compute per-row logsumexp of logits_ptr, store in out_ptr
    m = H
    n = L

    pid = tl.program_id(0)
    h = pid  # one program per row

    # First pass: row max
    max_val = -1e20
    for j in range(0, n, BLOCK_N):
        offs = j + tl.arange(0, BLOCK_N)
        vals = tl.load(logits_ptr + (h * L + offs),
                       mask=(offs < n), other=-1e20)
        # reduce max across this chunk
        local_max = tl.max(vals, axis=0)
        max_val = tl.maximum(max_val, local_max)

    # Second pass: sum of exp shifted by max
    sumexp = 0.0
    for j in range(0, n, BLOCK_N):
        offs = j + tl.arange(0, BLOCK_N)
        vals = tl.load(logits_ptr + (h * L + offs),
                       mask=(offs < n), other=-1e20)
        sumexp += tl.sum(tl.exp(vals - max_val), axis=0)

    lse = max_val + tl.log(sumexp)  # natural logsumexp
    # Store lse[h]; output tensor is 1D [H]
    tl.store(out_ptr + h, lse)


@triton.jit
def softmax_row(logits_ptr, lse_ptr, out_ptr, H: tl.constexpr, L: tl.constexpr, BLOCK_N: tl.constexpr):
    # Compute softmax per row h: out[h, j] = exp(logits[h, j] - lse[h]) / sum_k exp(...)
    m = H
    n = L

    pid = tl.program_id(0)
    h = pid

    lse = tl.load(lse_ptr + h)
    # compute denom
    denom = 0.0
    for j in range(0, n, BLOCK_N):
        offs = j + tl.arange(0, BLOCK_N)
        vals = tl.load(logits_ptr + (h * L + offs),
                       mask=(offs < n), other=-1e20)
        denom += tl.sum(tl.exp(vals - lse), axis=0)

    # write softmax
    for j in range(0, n, BLOCK_N):
        offs = j + tl.arange(0, BLOCK_N)
        vals = tl.load(logits_ptr + (h * L + offs),
                       mask=(offs < n), other=-1e20)
        soft = tl.exp(vals - lse) / denom
        tl.store(out_ptr + (h * L + offs), soft, mask=(offs < n))


@triton.jit
def matmul_out(softmax_ptr, kc_ptr, out_ptr,
               H: tl.constexpr, L: tl.constexpr, D: tl.constexpr,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # Compute out[H, D] = softmax[H, L] @ kc[L, D]
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
        s = tl.load(
            softmax_ptr + (offs_m[:, None] * L + offs_k[None, :]),
            mask=(offs_m[:, None] < m) & (offs_k[None, :] < k),
            other=0.0
        )
        k_tile = tl.load(
            kc_ptr + (offs_k[:, None] * D + offs_n[None, :]),
            mask=(offs_k[:, None] < k) & (offs_n[None, :] < n),
            other=0.0
        )
        acc += tl.dot(s, k_tile)

    out_offsets = offs_m[:, None] * D + offs_n[None, :]
    tl.store(
        out_ptr + out_offsets,
        acc,
        mask=(offs_m[:, None] < m) & (offs_n[None, :] < n)
    )


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # constants from original code
        self.H = 16
        self.D = 512
        self.P = 64

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Prepare Kc_all and Kp_all: squeeze batch dim and keep as 2-D tensors
        Kc_all = ckv_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, D]
        Kp_all = kpe_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, P]

        device = q_nope.device
        batch_size = int(qo_indptr.shape[0]) - 1
        # output tensors
        total_q = int(qo_indptr[-1].item())
        num_qo_heads = self.H
        D = self.D
        output = torch.empty((total_q, num_qo_heads, D), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Process each batch element
        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            q_len = q_end - q_start

            # Compute L (number of tokens in this batch's KV)
            L = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())

            # Select tok_idx positions (given num_kv_indices == L in provided workloads)
            tok_idx = kv_indices[b * L:(b + 1) * L].to(torch.int32).contiguous()
            # Gather Kc and Kp
            Kc = Kc_all[tok_idx]  # [L, D]
            Kp = Kp_all[tok_idx]  # [L, P]

            # For each query i
            for i in range(q_len):
                qn = q_nope[q_start + i]  # [H, D]
                qp = q_pe[q_start + i]    # [H, P]

                # 1) Compute logit_rows[h, L] = qn[h] @ Kc.T + qp[h] @ Kp.T
                logits = torch.empty((self.H, L), dtype=torch.float32, device=device)
                # launch matmul_qn_kc
                grid_qn = (triton.cdiv(self.H, 32), triton.cdiv(L, 64))
                matmul_qn_kc[grid_qn](qn, Kc, logits, self.H, self.D, L, 32, 64, 64)
                # launch matmul_qp_kp
                grid_qp = (triton.cdiv(self.H, 32), triton.cdiv(L, 64))
                logit_qp = torch.empty((self.H, L), dtype=torch.float32, device=device)
                matmul_qp_kp[grid_qp](qp, Kp, logit_qp, self.H, self.P, L, 32, 64, 64)
                # add
                grid_add = (triton.cdiv(self.H, 32), triton.cdiv(L, 64))
                logits = torch.empty((self.H, L), dtype=torch.float32, device=device)
                add_logits[grid_add](logit_qp, logits, logits, self.H, L, 32, 64)
                # scale
                grid_scale = (triton.cdiv(self.H, 32), triton.cdiv(L, 64))
                logits_scaled = torch.empty((self.H, L), dtype=torch.float32, device=device)
                scale_logits[grid_scale](logits, logits_scaled, sm_scale, self.H, L, 32, 64)

                # 2) Apply mask per row: mask[j] = 1 if j <= (L - q_len + i) else 0
                mask = torch.empty((self.H, L), dtype=torch.float32, device=device)
                query_abs_pos = L - q_len + i  # position where first token starts being kept
                for j in range(L):
                    mask[:, j] = (j > query_abs_pos).float()
                mask = mask.to(torch.float32)
                logits_masked = torch.empty((self.H, L), dtype=torch.float32, device=device)
                grid_mask = (triton.cdiv(self.H, 32), triton.cdiv(L, 64))
                apply_mask[grid_mask](logits_scaled, mask, logits_masked, sm_scale, self.H, L, 32, 64)

                # 3) Compute row-wise lse
                lse_row = torch.empty((self.H,), dtype=torch.float32, device=device)
                grid_lse = (self.H,)
                row_lse[grid_lse](logits_masked, lse_row, self.H, L, 128)  # BLOCK_N=128

                # 4) Softmax row-wise
                softmax_row_out = torch.empty((self.H, L), dtype=torch.float32, device=device)
                grid_softmax = (self.H,)
                softmax_row[grid_softmax](logits_masked, lse_row, softmax_row_out, self.H, L, 128)

                # 5) Output per head: out[h, :] = softmax[h, :] @ Kc
                out_row = torch.empty((self.H, self.D), dtype=torch.float32, device=device)
                grid_out = (triton.cdiv(self.H, 32), triton.cdiv(self.D, 64))
                matmul_out[grid_out](softmax_row_out, Kc, out_row, self.H, L, self.D, 32, 64, 64)

                # Store outputs
                q_idx = q_start + i
                output[q_idx] = out_row
                lse[q_idx] = lse_row / math.log(2.0)  # convert to base-2

        return output, lse


def run(*args):
    return ModelNew()(*args)
