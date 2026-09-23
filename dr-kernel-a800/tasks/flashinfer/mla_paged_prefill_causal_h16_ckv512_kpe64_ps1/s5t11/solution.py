import math
import triton
import triton.language as tl

# 1) Matmul: out[H, L] = qn[H, D] @ Kc[L, D]^T
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
        # qn: [H, D]
        q = tl.load(qn_ptr + (offs_m[:, None] * D + offs_k[None, :]),
                    mask=(offs_m[:, None] < m) & (offs_k[None, :] < k),
                    other=0.0)
        # kc: [L, D], we want kc[n, k] -> shape [offs_k, offs_n]
        kc_tile = tl.load(kc_ptr + (offs_n[None, :] * D + offs_k[:, None]),
                          mask=(offs_k[:, None] < k) & (offs_n[None, :] < n),
                          other=0.0)
        acc += tl.dot(q, kc_tile)
    out_offsets = offs_m[:, None] * L + offs_n[None, :]
    tl.store(out_ptr + out_offsets,
             acc,
             mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))

# 2) Matmul: out[H, L] = qp[H, P] @ Kp[L, P]^T
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
        q = tl.load(qp_ptr + (offs_m[:, None] * P + offs_k[None, :]),
                    mask=(offs_m[:, None] < m) & (offs_k[None, :] < k),
                    other=0.0)
        kp_tile = tl.load(kp_ptr + (offs_n[None, :] * P + offs_k[:, None]),
                          mask=(offs_k[:, None] < k) & (offs_n[None, :] < n),
                          other=0.0)
        acc += tl.dot(q, kp_tile)
    out_offsets = offs_m[:, None] * L + offs_n[None, :]
    tl.store(out_ptr + out_offsets,
             acc,
             mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))

# 3) Elementwise add
@triton.jit
def add_logits(a_ptr, b_ptr, out_ptr, H: tl.constexpr, L: tl.constexpr,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    m = H
    n = L
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    a = tl.load(a_ptr + (offs_m[:, None] * L + offs_n[None, :]),
                mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
                other=0.0)
    b = tl.load(b_ptr + (offs_m[:, None] * L + offs_n[None, :]),
                mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
                other=0.0)
    c = a + b
    tl.store(out_ptr + (offs_m[:, None] * L + offs_n[None, :]),
             c,
             mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))

# 4) Scale elementwise
@triton.jit
def scale_logits(inp_ptr, out_ptr, scale: tl.float32, H: tl.constexpr, L: tl.constexpr,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    m = H
    n = L
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    x = tl.load(inp_ptr + (offs_m[:, None] * L + offs_n[None, :]),
                mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
                other=0.0)
    y = x * scale
    tl.store(out_ptr + (offs_m[:, None] * L + offs_n[None, :]),
             y,
             mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))

# 5) Apply mask per row: set -inf where j <= query_abs_pos
@triton.jit
def apply_mask(logits_ptr, mask_ptr, H: tl.constexpr, L: tl.constexpr, pos: tl.int32,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    m = H
    n = L
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    vals = tl.load(logits_ptr + (offs_m[:, None] * L + offs_n[None, :]),
                   mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
                   other=0.0)
    # pos is scalar, offs_n is vector; create row-wise vector of pos
    keep = offs_n[None, :] > pos
    # mask: True means keep; False set to -inf
    vals = tl.where(keep, vals, -float("inf"))
    tl.store(mask_ptr + (offs_m[:, None] * L + offs_n[None, :]),
             vals,
             mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))

# 6) Row-wise logsumexp: lse[h] = log(sum_j exp(D[h, j] - m[h])) + m[h]
@triton.jit
def row_logsumexp(inp_ptr, lse_ptr, H: tl.constexpr, L: tl.constexpr,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    m = H
    n = L
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # Compute per-row max
    max_val = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
    for k0 in range(0, n, BLOCK_N):
        offs_n = k0 + tl.arange(0, BLOCK_N)
        x = tl.load(inp_ptr + (offs_m[:, None] * n + offs_n[None, :]),
                    mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
                    other=-float("inf"))
        # Reduce max across N
        block_max = tl.max(x, axis=1)  # shape [BLOCK_M]
        max_val = tl.maximum(max_val, block_max)
    # Compute sumexp with shifted values
    sumexp = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k0 in range(0, n, BLOCK_N):
        offs_n = k0 + tl.arange(0, BLOCK_N)
        x = tl.load(inp_ptr + (offs_m[:, None] * n + offs_n[None, :]),
                    mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
                    other=-float("inf"))
        shifted = x - max_val[:, None]
        expv = tl.exp(shifted)
        sumexp += tl.sum(expv, axis=1)
    lse_vals = tl.log(sumexp) + max_val  # logsumexp per row
    tl.store(lse_ptr + offs_m, lse_vals, mask=(offs_m < m))

# 7) Softmax per row over L, using lse: out[h, j] = exp(logits[h, j] - lse[h]) / sum_k exp(logits[h, k] - lse[h])
@triton.jit
def softmax_row_masked(logits_ptr, lse_ptr, out_ptr, H: tl.constexpr, L: tl.constexpr,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    m = H
    n = L
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # Load lse for this row
    lse_val = tl.load(lse_ptr + offs_m, mask=(offs_m < m), other=0.0)  # shape [BLOCK_M]
    # Compute numerator and denominator
    num = tl.zeros((BLOCK_M, n), dtype=tl.float32)
    den = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for k0 in range(0, n, BLOCK_N):
        offs_n = k0 + tl.arange(0, BLOCK_N)
        x = tl.load(logits_ptr + (offs_m[:, None] * n + offs_n[None, :]),
                    mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
                    other=0.0)
        num_block = tl.exp(x - lse_val[:, None])  # numerator
        # accumulate denominator
        den += tl.sum(num_block, axis=1)
        # store num for later normalization
        tl.store(out_ptr + (offs_m[:, None] * n + offs_n[None, :]),
                 num_block,
                 mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))
    # normalize num by den
    for k0 in range(0, n, BLOCK_N):
        offs_n = k0 + tl.arange(0, BLOCK_N)
        num_block = tl.load(out_ptr + (offs_m[:, None] * n + offs_n[None, :]),
                            mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
                            other=0.0)
        num_block = num_block / den[:, None]
        tl.store(out_ptr + (offs_m[:, None] * n + offs_n[None, :]),
                 num_block,
                 mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))

# 8) Matmul: out[H, D] = softmax[H, L] @ Kc[L, D]
@triton.jit
def matmul_attn_kc(softmax_ptr, kc_ptr, out_ptr,
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
        soft = tl.load(softmax_ptr + (offs_m[:, None] * L + offs_k[None, :]),
                       mask=(offs_m[:, None] < m) & (offs_k[None, :] < k),
                       other=0.0)
        kc_tile = tl.load(kc_ptr + (offs_k[:, None] * D + offs_n[None, :]),
                          mask=(offs_k[:, None] < k) & (offs_n[None, :] < n),
                          other=0.0)
        acc += tl.dot(soft, kc_tile)
    out_offsets = offs_m[:, None] * D + offs_n[None, :]
    tl.store(out_ptr + out_offsets,
             acc,
             mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Constants as in original
        self.num_qo_heads = 16
        self.head_dim_ckv = 512
        self.head_dim_kpe = 64
        self.sm_scale = 1.0  # float32 scalar

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        device = q_nope.device
        # Prepare Kc_all and Kp_all: squeeze (remove '1' dim) and cast to float32 for compute
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, head_dim_ckv]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, head_dim_kpe]

        total_q = int(qo_indptr[-1].item())  # number of queries across all batches
        num_pages = Kc_all.shape[0]
        len_indptr = qo_indptr.shape[0]
        batch_size = len_indptr - 1  # original logic

        # Output buffers
        output = torch.zeros(
            (total_q, self.num_qo_heads, self.head_dim_ckv), dtype=torch.bfloat16, device=device
        )
        lse = torch.full(
            (total_q, self.num_qo_heads), -float("inf"), dtype=torch.float32, device=device
        )

        # Iterate over batches
        for b in range(len(qo_indptr) - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            # If empty batch, skip
            if q_start >= q_end:
                continue

            # Extract tok_idx for this batch (ensure 1-D indices)
            tok_idx = kv_indices[kv_indptr[b]:kv_indptr[b + 1]]
            L = int((kv_indptr[b + 1] - kv_indptr[b]).item())

            # Slice Kc and Kp for this batch
            Kc = Kc_all[tok_idx]  # [L, D]
            Kp = Kp_all[tok_idx]  # [L, P]

            # For each query in the batch
            for i in range(q_start + 1, q_end + 1):
                # Compute query vectors
                qn = q_nope[i].to(torch.float32).contiguous()  # [H, D]
                qp = q_pe[i].to(torch.float32).contiguous()    # [H, P]
                # If i exceeds total_q, skip (defensive)
                if i >= total_q:
                    break

                # Preallocate intermediate buffers
                logits = torch.empty((self.num_qo_heads, L), dtype=torch.float32, device=device)
                logits_scaled = torch.empty_like(logits)
                masked_logits = torch.empty_like(logits)

                # Matmul: qn @ Kc.T -> [H, L]
                BLOCK_M = 16; BLOCK_N = 64; BLOCK_K = 64
                grid_qn = (triton.cdiv(self.num_qo_heads, BLOCK_M), triton.cdiv(L, BLOCK_N))
                matmul_qn_kc[grid_qn](
                    qn, Kc, logits,
                    H=self.num_qo_heads, D=self.head_dim_ckv, L=L,
                    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
                )

                # Matmul: qp @ Kp.T -> [H, L]
                logits_qp = torch.empty((self.num_qo_heads, L), dtype=torch.float32, device=device)
                grid_qp = (triton.cdiv(self.num_qo_heads, BLOCK_M), triton.cdiv(L, BLOCK_N))
                matmul_qp_kp[grid_qp](
                    qp, Kp, logits_qp,
                    H=self.num_qo_heads, P=self.head_dim_kpe, L=L,
                    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
                )

                # Add: logits = logits_qn + logits_qp
                grid_add = (triton.cdiv(self.num_qo_heads, BLOCK_M), triton.cdiv(L, BLOCK_N))
                add_logits[grid_add](logits, logits_qp, logits,
                                     H=self.num_qo_heads, L=L,
                                     BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)

                # Scale by sm_scale
                grid_scale = (triton.cdiv(self.num_qo_heads, BLOCK_M), triton.cdiv(L, BLOCK_N))
                scale_logits[grid_scale](
                    logits, logits_scaled,
                    scale=sm_scale,
                    H=self.num_qo_heads, L=L,
                    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
                )

                # Apply causal-like mask: keep j where j > (L - (q_end - q_start) + i)
                # Here (q_end - q_start) is number of queries in this batch (q_len), but we can compute it from qo_indptr
                # Compute q_len robustly: number of queries in this batch equals q_end - q_start
                q_len = q_end - q_start
                query_abs_pos = q_len - L + (i - q_start)
                # Launch mask kernel
                grid_mask = (triton.cdiv(self.num_qo_heads, BLOCK_M), triton.cdiv(L, BLOCK_N))
                apply_mask[grid_mask](
                    logits_scaled, masked_logits,
                    H=self.num_qo_heads, L=L,
                    pos=int(query_abs_pos),
                    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
                )

                # Compute lse per row: logsumexp(masked_logits) / ln(2)
                # lse tensor is [total_q, H]; we need to write into lse[i, :]
                lse_vec = torch.empty((self.num_qo_heads,), dtype=torch.float32, device=device)
                grid_lse = (triton.cdiv(self.num_qo_heads, BLOCK_M),)
                row_logsumexp[grid_lse](
                    masked_logits, lse_vec,
                    H=self.num_qo_heads, L=L,
                    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
                )
                lse[i] = lse_vec / math.log(2.0)

                # Softmax per row over L
                softmax_out = torch.empty((self.num_qo_heads, L), dtype=torch.float32, device=device)
                grid_softmax = (triton.cdiv(self.num_qo_heads, BLOCK_M), triton.cdiv(L, BLOCK_N))
                softmax_row_masked[grid_softmax](
                    masked_logits, lse[i], softmax_out,
                    H=self.num_qo_heads, L=L,
                    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N
                )

                # Final output: softmax @ Kc -> [H, D]
                out_batch = torch.empty((self.num_qo_heads, self.head_dim_ckv), dtype=torch.float32, device=device)
                grid_final = (triton.cdiv(self.num_qo_heads, BLOCK_M), triton.cdiv(self.head_dim_ckv, BLOCK_N))
                matmul_attn_kc[grid_final](
                    softmax_out, Kc, out_batch,
                    H=self.num_qo_heads, L=L, D=self.head_dim_ckv,
                    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_N
                )

                # Store output[i] in the correct position
                if i < output.shape[0]:
                    output[i] = out_batch.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
