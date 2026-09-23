import math
import torch
import triton
import triton.language as tl

# Matmul: A[M, K] @ B[N, K]^T -> C[M, N]
@triton.jit
def matmul_qn_kc(qn_ptr, kc_ptr, out_ptr,
                 M: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        q = tl.load(qn_ptr + (offs_m[:, None] * K + offs_k[None, :]),
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        k_tile = tl.load(kc_ptr + (offs_n[None, :] * K + offs_k[:, None]),
                         mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(q, k_tile)
    out_offsets = offs_m[:, None] * N + offs_n[None, :]
    tl.store(out_ptr + out_offsets,
             acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

@triton.jit
def matmul_qp_kp(qp_ptr, kp_ptr, out_ptr,
                 M: tl.constexpr, K: tl.constexpr, N: tl.constexpr,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # qp[M, K] @ kp[N, K]^T -> [M, N]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        q = tl.load(qp_ptr + (offs_m[:, None] * K + offs_k[None, :]),
                    mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        k_tile = tl.load(kp_ptr + (offs_n[None, :] * K + offs_k[:, None]),
                         mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(q, k_tile)
    out_offsets = offs_m[:, None] * N + offs_n[None, :]
    tl.store(out_ptr + out_offsets,
             acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

@triton.jit
def add_logits(a_ptr, b_ptr, out_ptr,
               M: tl.constexpr, N: tl.constexpr,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    a = tl.load(a_ptr + (offs_m[:, None] * N + offs_n[None, :]),
                mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0)
    b = tl.load(b_ptr + (offs_m[:, None] * N + offs_n[None, :]),
                mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0)
    out = a + b
    tl.store(out_ptr + (offs_m[:, None] * N + offs_n[None, :]),
             out,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

@triton.jit
def scale_logits(in_ptr, out_ptr, scale: tl.float32,
                 M: tl.constexpr, N: tl.constexpr,
                 BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    x = tl.load(in_ptr + (offs_m[:, None] * N + offs_n[None, :]),
                mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0)
    y = x * scale
    tl.store(out_ptr + (offs_m[:, None] * N + offs_n[None, :]),
             y,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

@triton.jit
def apply_mask_logits(in_ptr, out_ptr, mask_ptr, M: tl.constexpr, N: tl.constexpr,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Apply per-row mask: load mask[M, N] and set positions with mask[j] == 0 to -inf in out_ptr
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    logits = tl.load(in_ptr + (offs_m[:, None] * N + offs_n[None, :]),
                     mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0)
    mask = tl.load(mask_ptr + (offs_m[:, None] * N + offs_n[None, :]),
                   mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0)
    # Convert mask to boolean: True where keep, else False
    keep = mask > 0  # mask values: 1 where keep, 0 where mask out
    inf_val = -float('inf')
    masked_logits = tl.where(keep, logits, inf_val)
    tl.store(out_ptr + (offs_m[:, None] * N + offs_n[None, :]),
             masked_logits,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

@triton.jit
def row_lse(in_ptr, lse_ptr,
            M: tl.constexpr, N: tl.constexpr,
            scale_log2: tl.float32,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Compute per-row logsumexp(in_ptr[M, N]) and store in lse_ptr[M]
    pid_m = tl.program_id(0)
    # Single tile per row
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # BLOCK_M == 16 here
    offs_n = tl.arange(0, BLOCK_N)  # BLOCK_N will cover N
    x = tl.load(in_ptr + (offs_m[:, None] * N + offs_n[None, :]),
                mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=-float('inf'))
    row_max = tl.max(x, axis=1)  # [BLOCK_M]
    x_shift = x - row_max[:, None]
    exp_x = tl.exp(x_shift)
    exp_x = tl.where(exp_x == 0, 0.0, exp_x)  # safety, though not needed
    sumexp = tl.sum(exp_x, axis=1)  # [BLOCK_M]
    lse_row = tl.log(sumexp) + row_max
    lse_row = lse_row / scale_log2  # divide by ln(2)
    tl.store(lse_ptr + offs_m, lse_row, mask=(offs_m < M))

@triton.jit
def softmax_row_masked(in_ptr, lse_ptr, out_ptr,
                       M: tl.constexpr, N: tl.constexpr,
                       scale_log2: tl.float32,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    x = tl.load(in_ptr + (offs_m[:, None] * N + offs_n[None, :]),
                mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0)
    lse_row = tl.load(lse_ptr + offs_m, mask=(offs_m < M), other=-float('inf'))
    x_shift = x - (lse_row * scale_log2)  # convert lse back from scaled to unscaled
    exp_x = tl.exp(x_shift)
    denom = tl.sum(exp_x, axis=1)  # [BLOCK_M]
    soft = exp_x / denom[:, None]
    tl.store(out_ptr + (offs_m[:, None] * N + offs_n[None, :]),
             soft,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

@triton.jit
def matmul_attn_kc(attn_ptr, kc_ptr, out_ptr,
                   M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    # attn[M, N] @ kc[N, K] -> [M, K]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        attn_tile = tl.load(attn_ptr + (offs_m[:, None] * N + offs_n[None, :]),
                            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0)
        kc_tile = tl.load(kc_ptr + (offs_n[None, :] * K + offs_k[:, None]),
                          mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(attn_tile, kc_tile)
    out_offsets = offs_m[:, None] * K + offs_n[None, :]
    tl.store(out_ptr + out_offsets,
             acc,
             mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


class ModelNew(torch.nn.Module):
    def __init__(self, num_qo_heads=16, head_dim_ckv=512, head_dim_kpe=64):
        super().__init__()
        self.num_qo_heads = num_qo_heads
        self.head_dim_ckv = head_dim_ckv
        self.head_dim_kpe = head_dim_kpe

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Assert shapes (not strictly required for performance, but keep for clarity)
        assert q_nope.shape[1:] == (self.num_qo_heads, self.head_dim_ckv)
        assert q_pe.shape[-1] == self.head_dim_kpe
        # Ensure CUDA and contiguity
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda
        device = q_nope.device
        total_q, H, D = q_nope.shape
        P = q_pe.shape[-1]
        assert H == self.num_qo_heads and D == self.head_dim_ckv and P == self.head_dim_kpe

        # Prepare Kc_all and Kp_all
        num_pages = ckv_cache.shape[0]
        Kc_all = ckv_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, D]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, P]

        # Outputs (compute in float32, cast at end)
        output = torch.empty((total_q, H, D), dtype=torch.float32, device=device)
        lse_out = torch.full((total_q, H), -float('inf'), dtype=torch.float32, device=device)

        # batch_size = len(qo_indptr) - 1
        batch_size = int(qo_indptr.numel()) - 1
        if batch_size < 0:
            return output, lse_out

        # Constants for Triton grids
        BLOCK_M = H  # 16
        BLOCK_N = 64  # tuning; fits typical L
        BLOCK_K_D = 64  # for D=512
        BLOCK_K_P = 64  # for P=64
        scale_log2 = 1.0 / math.log(2.0)  # because original divides by log(2)

        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            q_len = q_end - q_start

            if q_len <= 0:
                continue

            # Compute number of tokens in this batch's kv
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            L = page_end - page_beg

            # Collect tok indices and corresponding Kc, Kp
            tok_idx = kv_indices[page_beg:page_end]  # 1-D tensor of indices
            Kc = Kc_all[tok_idx].to(torch.float32).contiguous()  # [L, D]
            Kp = Kp_all[tok_idx].to(torch.float32).contiguous()  # [L, P]

            # Loop over queries in the batch
            for i in range(q_len):
                cur_q_start = q_start + i

                # Select qn and qp
                qn = q_nope[cur_q_start].to(torch.float32).contiguous()  # [H, D]
                qp = q_pe[cur_q_start].to(torch.float32).contiguous()   # [H, P]

                # Prepare intermediate buffers
                logits_qn = torch.empty((H, L), dtype=torch.float32, device=device)
                logits_qp = torch.empty((H, L), dtype=torch.float32, device=device)
                logits_sum = torch.empty((H, L), dtype=torch.float32, device=device)
                logits_scaled = torch.empty((H, L), dtype=torch.float32, device=device)
                masked_logits = torch.empty((H, L), dtype=torch.float32, device=device)
                lse_row = torch.empty((H,), dtype=torch.float32, device=device)
                softmax = torch.empty((H, L), dtype=torch.float32, device=device)
                out_row = torch.empty((H, D), dtype=torch.float32, device=device)

                # 1) qn @ Kc.T -> [H, L]
                grid_qn = (H, L)
                matmul_qn_kc[grid_qn](qn, Kc, logits_qn,
                                      M=H, K=D, N=L,
                                      BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K_D)

                # 2) qp @ Kp.T -> [H, L]
                grid_qp = (H, L)
                matmul_qp_kp[grid_qp](qp, Kp, logits_qp,
                                      M=H, K=P, N=L,
                                      BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K_P)

                # 3) Add
                grid_add = (H, L)
                add_logits[grid_add](logits_qn, logits_qp, logits_sum,
                                     M=H, N=L,
                                     BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)

                # 4) Scale
                grid_scale = (H, L)
                scale_logits[grid_scale](logits_sum, logits_scaled,
                                         scale=sm_scale,
                                         M=H, N=L,
                                         BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)

                # 5) Build mask: per row h, keep j where j > (L - q_len + i)
                #    query_abs_pos = L - q_len + i
                query_abs_pos = L - q_len + i
                mask_mat = torch.ones((H, L), dtype=torch.float32, device=device)
                # Set invalid positions to 0: j <= query_abs_pos
                for j in range(L):
                    if j <= query_abs_pos:
                        mask_mat[:, j].fill_(0.0)
                grid_mask = (H, L)
                apply_mask_logits[grid_mask](logits_scaled, masked_logits, mask_mat,
                                             M=H, N=L,
                                             BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)

                # 6) Row-wise LSE: compute per row logsumexp of masked_logits
                grid_lse = (H,)
                row_lse[grid_lse](masked_logits, lse_row,
                                  M=H, N=L,
                                  scale_log2=scale_log2,
                                  BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)

                # 7) Softmax per row on masked logits
                grid_softmax = (H, L)
                softmax_row_masked[grid_softmax](masked_logits, lse_row, softmax,
                                                 M=H, N=L,
                                                 scale_log2=scale_log2,
                                                 BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)

                # 8) Output = softmax @ Kc -> [H, D]
                grid_out = (H, D)
                matmul_attn_kc[grid_out](softmax, Kc, out_row,
                                         M=H, N=L, K=D,
                                         BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K_D)

                # Store to output and lse_out
                # output has shape [total_q, H, D], lse_out has shape [total_q, H]
                output[cur_q_start] = out_row  # this is fine since Triton writes float32
                lse_out[cur_q_start] = lse_row

        # Cast output to bfloat16 and lse_out to float32 to match original signatures
        output = output.to(torch.bfloat16)
        lse_out = lse_out  # already float32
        return output, lse_out


def run(*args):
    return ModelNew()(*args)
