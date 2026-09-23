import torch
import triton
import triton.language as tl

# Triton kernels
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
        q = tl.load(qn_ptr + (offs_m[:, None] * D + offs_k[None, :]),
                    mask=(offs_m[:, None] < m) & (offs_k[None, :] < k),
                    other=0.0)
        k_tile = tl.load(kc_ptr + (offs_n[None, :] * D + offs_k[:, None]),
                         mask=(offs_k[:, None] < k) & (offs_n[None, :] < n),
                         other=0.0)
        acc += tl.dot(q, k_tile)
    out_offsets = offs_m[:, None] * L + offs_n[None, :]
    tl.store(out_ptr + out_offsets, acc, mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


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
        q = tl.load(qp_ptr + (offs_m[:, None] * P + offs_k[None, :]),
                    mask=(offs_m[:, None] < m) & (offs_k[None, :] < k),
                    other=0.0)
        k_tile = tl.load(kp_ptr + (offs_n[None, :] * P + offs_k[:, None]),
                         mask=(offs_k[:, None] < k) & (offs_n[None, :] < n),
                         other=0.0)
        acc += tl.dot(q, k_tile)
    out_offsets = offs_m[:, None] * L + offs_n[None, :]
    tl.store(out_ptr + out_offsets, acc, mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


@triton.jit
def add_logits(a_ptr, b_ptr, out_ptr, H: tl.constexpr, L: tl.constexpr):
    m = H
    n = L
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * m + tl.arange(0, m)
    offs_n = pid_n * n + tl.arange(0, n)
    a = tl.load(a_ptr + (offs_m[:, None] * L + offs_n[None, :]),
                mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
                other=0.0)
    b = tl.load(b_ptr + (offs_m[:, None] * L + offs_n[None, :]),
                mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
                other=0.0)
    tl.store(out_ptr + (offs_m[:, None] * L + offs_n[None, :]), a + b,
             mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


@triton.jit
def scale_logits(in_ptr, out_ptr, scale, H: tl.constexpr, L: tl.constexpr):
    m = H
    n = L
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * m + tl.arange(0, m)
    offs_n = pid_n * n + tl.arange(0, n)
    inp = tl.load(in_ptr + (offs_m[:, None] * L + offs_n[None, :]),
                  mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
                  other=0.0)
    tl.store(out_ptr + (offs_m[:, None] * L + offs_n[None, :]), inp * scale,
             mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


@triton.jit
def apply_mask_logits(logits_ptr, mask_ptr, out_ptr, H: tl.constexpr, L: tl.constexpr):
    # out[h, j] = logits[h, j] if mask[h, j] else -inf
    m = H
    n = L
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * m + tl.arange(0, m)
    offs_n = pid_n * n + tl.arange(0, n)
    logits = tl.load(logits_ptr + (offs_m[:, None] * L + offs_n[None, :]),
                     mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
                     other=0.0)
    mask = tl.load(mask_ptr + (offs_m[:, None] * L + offs_n[None, :]),
                   mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
                   other=0.0)
    out = tl.where(mask > 0, logits, -float('inf'))
    tl.store(out_ptr + (offs_m[:, None] * L + offs_n[None, :]), out,
             mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


@triton.jit
def row_lse(logits_ptr, lse_ptr, H: tl.constexpr, L: tl.constexpr):
    # lse[h] = log(sum_j exp(logits[h, j] - max)) + max
    m = H
    n = L
    pid_m = tl.program_id(0)
    h = pid_m
    row_max = -float('inf')
    for j0 in range(0, n, 128):
        offs_j = j0 + tl.arange(0, 128)
        logit_vec = tl.load(logits_ptr + (h * n + offs_j),
                            mask=(offs_j < n),
                            other=-float('inf'))
        row_max = tl.maximum(row_max, tl.max(logit_vec, axis=0))
    sumexp = 0.0
    for j0 in range(0, n, 128):
        offs_j = j0 + tl.arange(0, 128)
        logit_vec = tl.load(logits_ptr + (h * n + offs_j),
                            mask=(offs_j < n),
                            other=-float('inf'))
        sumexp += tl.sum(tl.exp(logit_vec - row_max), axis=0)
    lse_val = tl.log(sumexp) + row_max
    tl.store(lse_ptr + h, lse_val)


@triton.jit
def softmax_row_masked(logits_ptr, lse_ptr, out_ptr, H: tl.constexpr, L: tl.constexpr):
    # out[h, :] = softmax(logits[h, :] - lse[h])
    m = H
    n = L
    pid_m = tl.program_id(0)
    h = pid_m
    lse_val = tl.load(lse_ptr + h)
    for j0 in range(0, n, 128):
        offs_j = j0 + tl.arange(0, 128)
        logits_vec = tl.load(logits_ptr + (h * n + offs_j), mask=(offs_j < n), other=-float('inf'))
        numerator = tl.exp(logits_vec - lse_val)
        denom = tl.sum(numerator, axis=0)
        out_vec = numerator / denom
        tl.store(out_ptr + (h * n + offs_j), out_vec, mask=(offs_j < n))


@triton.jit
def matmul_attn_kc(attention_ptr, kc_ptr, out_ptr,
                   H: tl.constexpr, L: tl.constexpr, D: tl.constexpr,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # out[H, D] = attention[H, L] @ kc[L, D]
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
        a = tl.load(attention_ptr + (offs_m[:, None] * L + offs_k[None, :]),
                    mask=(offs_m[:, None] < m) & (offs_k[None, :] < k),
                    other=0.0)
        k_tile = tl.load(kc_ptr + (offs_k[:, None] * L + offs_n[None, :]),
                         mask=(offs_k[:, None] < k) & (offs_n[None, :] < n),
                         other=0.0)
        acc += tl.dot(a, k_tile)
    out_offsets = offs_m[:, None] * D + offs_n[None, :]
    tl.store(out_ptr + out_offsets, acc, mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_qo_heads = 16
        self.head_dim_ckv = 512
        self.head_dim_kpe = 64
        self.page_size = 1

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA tensors
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda
        device = q_nope.device

        num_pages = ckv_cache.shape[0]
        # Prepare Kc_all and Kp_all (squeeze size-1 dim)
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, D]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, P]

        total_q, H, D = q_nope.shape
        P = q_pe.shape[-1]
        assert H == self.num_qo_heads and D == self.head_dim_ckv and P == self.head_dim_kpe

        output = torch.empty((total_q, H, D), dtype=torch.float32, device=device)  # store as float32
        lse = torch.full((total_q, H), -float('inf'), dtype=torch.float32, device=device)

        # batch_size = number of batches in qo_indptr
        batch_size = int(qo_indptr.numel()) - 1
        if batch_size < 0:
            return output, lse

        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            q_len = q_end - q_start

            if q_len <= 0:
                continue

            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            L = page_end - page_beg

            tok_idx = kv_indices[page_beg:page_end]  # 1-D int64
            Kc = Kc_all[tok_idx]  # [L, D]
            Kp = Kp_all[tok_idx]  # [L, P]

            # Select current batch queries
            qn_batch = q_nope[q_start:q_end].to(torch.float32).contiguous()  # [q_len, H, D]
            qp_batch = q_pe[q_start:q_end].to(torch.float32).contiguous()   # [q_len, H, P]

            for i in range(q_len):
                qn = qn_batch[i]  # [H, D]
                qp = qp_batch[i]  # [H, P]

                # Intermediate buffers
                logits_qn = torch.empty((H, L), dtype=torch.float32, device=device)
                logits_qp = torch.empty((H, L), dtype=torch.float32, device=device)
                logits_sum = torch.empty((H, L), dtype=torch.float32, device=device)

                # GEMMs


def run(*args):
    return ModelNew()(*args)
