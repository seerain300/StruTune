import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: Compute logits for a single head h:
# logits = qn[h] @ Kc.T + qp[h] @ Kp.T, for all tokens (length L_tokens).
# Inputs:
#   qn_ptr: [Hc] float32
#   qp_ptr: [Hp] float32
#   Kc_ptr: [L_tokens, Hc] float32
#   Kp_ptr: [L_tokens, Hp] float32
#   out_ptr: [L_tokens] float32 (accumulated logits)
#   sm_scale: float32
# Launch: one program per (b, h). Call inside forward with grid=(1,).
@triton.jit
def matmul_add_row_kernel(
    qn_ptr, qp_ptr, Kc_ptr, Kp_ptr, out_ptr,
    sm_scale: tl.constexpr,
    L_tokens: tl.constexpr, Hc: tl.constexpr, Hp: tl.constexpr,
    qn_stride, qp_stride,
    Kc_stride0, Kc_stride1,
    Kp_stride0, Kp_stride1,
    out_stride,
    BLOCK_K: tl.constexpr
):
    # We run a single program per head. For the given shape, we assume only one head per call as per forward.
    # Reduce across K dimension in chunks of BLOCK_K.
    offs_k = tl.arange(0, BLOCK_K)
    # Accumulators for logits vector of length L_tokens
    logits = tl.zeros((L_tokens,), dtype=tl.float32)

    # Contribution from qn @ Kc.T
    acc_qn = tl.zeros((), dtype=tl.float32)  # scalar accumulator
    for k in range(0, L_tokens, BLOCK_K):
        k_idx = k + offs_k
        mask = k_idx < L_tokens
        # Kc[k_idx, :] for all dims Hc
        kc = tl.load(Kc_ptr + k_idx * Kc_stride0 + tl.arange(0, Hc) * Kc_stride1, mask=mask, other=0.0)  # shape (Hc,)
        # qn[k_idx] scalar dot kc
        qn_val = tl.load(qn_ptr + k_idx * qn_stride, mask=mask, other=0.0)  # shape (BLOCK_K,)
        acc_qn += tl.sum(qn_val * kc, axis=0)  # sum over Hc for each k in chunk
    logits += acc_qn

    # Contribution from qp @ Kp.T
    acc_qp = tl.zeros((), dtype=tl.float32)
    for k in range(0, L_tokens, BLOCK_K):
        k_idx = k + offs_k
        mask = k_idx < L_tokens
        kp = tl.load(Kp_ptr + k_idx * Kp_stride0 + tl.arange(0, Hp) * Kp_stride1, mask=mask, other=0.0)  # shape (Hp,)
        qp_val = tl.load(qp_ptr + k_idx * qp_stride, mask=mask, other=0.0)  # shape (BLOCK_K,)
        acc_qp += tl.sum(qp_val * kp, axis=0)  # sum over Hp for each k in chunk
    logits += acc_qp

    logits = logits * sm_scale
    # Store logits
    for k in range(0, L_tokens):
        tl.store(out_ptr + k * out_stride, logits[k])


# Triton kernel: Compute row-wise softmax and logsumexp for a single (b, head) row over tokens (length L_tokens).
# Inputs:
#   x_ptr: [L_tokens] float32, logits
#   lse_ptr: scalar pointer to output lse per row
# Launch: one program per (b, h). Call inside forward with grid=(1,).
@triton.jit
def softmax_logsumexp_row_kernel(x_ptr, lse_ptr, L_tokens: tl.constexpr):
    # Pass 1: compute max
    max_val = tl.full((), -float('inf'), tl.float32)
    for i in range(0, L_tokens):
        val = tl.load(x_ptr + i)
        max_val = tl.maximum(max_val, val)
    # Pass 2: compute sum(exp(x - max))
    sum_exp = tl.zeros((), dtype=tl.float32)
    for i in range(0, L_tokens):
        val = tl.load(x_ptr + i)
        sum_exp += tl.exp(val - max_val)
    lse = tl.log(sum_exp) / tl.log(2.0)
    tl.store(lse_ptr, lse)


# Triton kernel: Compute out_row = attn_row @ Kc, where attn_row is per-head softmax over tokens (length L_tokens),
# and Kc is [L_tokens, Hc]. We produce a chunk of output dims (BLOCK_N) and accumulate across tokens in chunks (BLOCK_M).
# Inputs:
#   attn_ptr: [L_tokens] float32 (softmaxed logits)
#   Kc_ptr: [L_tokens, Hc] float32
#   out_ptr: [Hc] float32 (accumulated output vector)
# Launch: grid over output column chunks. In forward, we call this kernel with grid=(triton.cdiv(Hc, BLOCK_N),).
@triton.jit
def matvec_row_kernel(attn_ptr, Kc_ptr, out_ptr,
                      L_tokens: tl.constexpr, Hc: tl.constexpr,
                      Kc_stride0, Kc_stride1,
                      out_stride,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Each program handles a chunk of output columns [offs_n, offs_n+BLOCK_N)
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < Hc
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Loop over tokens in chunks of BLOCK_M
    for m in range(0, L_tokens, BLOCK_M):
        offs_m = m + tl.arange(0, BLOCK_M)
        mask_m = offs_m < L_tokens

        # Load attn chunk: attn[offs_m]
        attn_chunk = tl.load(attn_ptr + offs_m, mask=mask_m, other=0.0)  # shape (BLOCK_M,)

        # Load Kc chunk: Kc[offs_m, offs_n]
        kc = tl.load(Kc_ptr + offs_m[:, None] * Kc_stride0 + offs_n[None, :] * Kc_stride1,
                     mask=mask_m[:, None] & mask_n[None, :], other=0.0)  # shape (BLOCK_M, BLOCK_N)

        # Accumulate: acc += sum_m attn_chunk[m] * kc[m, :]
        acc += tl.sum(kc * attn_chunk[:, None], axis=0)

    # Store accumulated output chunk
    tl.store(out_ptr + offs_n * out_stride, acc, mask=mask_n)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Extract shapes and ensure CUDA / contiguous
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "All tensors must be on CUDA"
        device = q_nope.device

        batch_size = q_nope.shape[0]
        num_qo_heads = q_nope.shape[1]
        head_dim_ckv = q_nope.shape[2]  # Hc
        head_dim_kpe = q_pe.shape[2]    # Hp

        # Kc_all and Kp_all: squeeze the (1) dimension from cache
        Kc_all = ckv_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, Hc]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, Hp]

        # Prepare output tensors
        output = torch.empty((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        # Precompute L_tokens and tok_idx for each batch b
        # kv_indptr is [len_indptr], kv_indices is [num_kv_indices]
        for b in range(batch_size):
            # Read tok_idx range for this batch
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            L_tokens = end - start
            if L_tokens <= 0:
                # No tokens for this batch element, zero output and lse
                output[b].zero_()
                lse[b] = -float('inf')
                continue

            tok_idx = kv_indices[start:start + L_tokens].to(torch.int32).contiguous()  # [L_tokens]

            # Gather Kc and Kp for these tokens
            Kc = Kc_all[tok_idx]  # [L_tokens, Hc], float32
            Kp = Kp_all[tok_idx]  # [L_tokens, Hp], float32

            # Initialize per-head outputs and lse vectors
            # We will compute per head h in a loop to keep code simple.
            for h in range(num_qo_heads):
                # Prepare pointers
                # qn: [Hc], qp: [Hp]
                qn = q_nope[b, h, :].to(torch.float32).contiguous()  # [Hc]
                qp = q_pe[b, h, :].to(torch.float32).contiguous()   # [Hp]

                # Buffer for logits: [L_tokens]
                logits = torch.empty((L_tokens,), dtype=torch.float32, device=device)

                # Kernel 1: compute logits for head h
                BLOCK_K = 128  # tuneable
                matmul_add_row_kernel[(1,)](
                    qn, qp, Kc, Kp, logits,
                    sm_scale=float(sm_scale),
                    L_tokens=L_tokens, Hc=head_dim_ckv, Hp=head_dim_kpe,
                    qn_stride=qn.stride(0), qp_stride=qp.stride(0),
                    Kc_stride0=Kc.stride(0), Kc_stride1=Kc.stride(1),
                    Kp_stride0=Kp.stride(0), Kp_stride1=Kp.stride(1),
                    out_stride=logits.stride(0),
                    BLOCK_K=BLOCK_K,
                    num_warps=4, num_stages=2
                )

                # Kernel 2: compute softmax and logsumexp for this row
                lse[b, h] = torch.empty((), dtype=torch.float32, device=device)  # placeholder, will be overwritten
                # Triton kernel for lse
                softmax_logsumexp_row_kernel[(1,)](
                    logits, lse[b, h],
                    L_tokens=L_tokens
                )

                # Kernel 3: compute output vector for head h: out_row = softmax(logits) @ Kc
                out_row = torch.empty((head_dim_ckv,), dtype=torch.float32, device=device)
                BLOCK_N = 128  # output chunk
                BLOCK_M = 128  # token chunk
                # We launch over chunks of output dims; grid is over columns
                grid = (triton.cdiv(head_dim_ckv, BLOCK_N),)
                matvec_row_kernel[grid](
                    logits, Kc, out_row,
                    L_tokens=L_tokens, Hc=head_dim_ckv,
                    Kc_stride0=Kc.stride(0), Kc_stride1=Kc.stride(1),
                    out_stride=out_row.stride(0),
                    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                    num_warps=4, num_stages=2
                )

                # Store to output[b, h, :]
                output[b, h, :] = out_row.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
