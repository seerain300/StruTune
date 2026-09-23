import torch
import triton
import triton.language as tl
import math

# Constants derived from original asserts
HEAD_DIM_CKV = 512      # head_dim_ckv
HEAD_DIM_KPE = 64       # head_dim_kpe
NUM_QO_HEADS = 16       # num_qo_heads
LN2 = math.log(2.0)


@triton.jit
def matvec_row_kernel(
    q_ptr,            # *float32, q vector, length K
    B_ptr,            # *float32, matrix B, shape [M_CONST, K], contiguous
    C_ptr,            # *float32, output vector, shape [M_CONST], contiguous
    K: tl.constexpr,  # compile-time K
    M_CONST: tl.constexpr,  # compile-time number of rows
    BLOCK_K: tl.constexpr = 64
):
    # One program computes a single output element: C[i] = q @ B[i, :]
    i = tl.program_id(0)  # 0..M_CONST-1
    acc = 0.0
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K
        a = tl.load(q_ptr + offs_k, mask=mask_k, other=0.0)
        b = tl.load(B_ptr + i * K + offs_k, mask=mask_k, other=0.0)
        acc += tl.sum(a * b, axis=0)
    tl.store(C_ptr + i, acc)


@triton.jit
def compute_lse_and_attn_kernel(
    logits_ptr,       # *float32, logits vector, length M_CONST
    attn_ptr,         # *float32, attn vector, length M_CONST
    lse_ptr,          # *float32, lse per head, length 1 (single head)
    M_CONST: tl.constexpr
):
    # Compute max for numerical stability
    max_val = -float("inf")
    for i in range(0, M_CONST):
        val = tl.load(logits_ptr + i)
        if val > max_val:
            max_val = val
    # Compute sum of exp(logits - max)
    sum_exp = 0.0
    for i in range(0, M_CONST):
        val = tl.load(logits_ptr + i) - max_val
        sum_exp += tl.exp(val)
    # lse = log(sum_exp) / ln(2)
    lse = tl.log(sum_exp) / LN2
    tl.store(lse_ptr, lse)
    # Compute attn and store
    inv_sum = 1.0 / sum_exp
    for i in range(0, M_CONST):
        val = tl.load(logits_ptr + i) - max_val
        attn_i = tl.exp(val) * inv_sum
        tl.store(attn_ptr + i, attn_i)


@triton.jit
def matvec_out_kernel(
    attn_ptr,         # *float32, attn vector, length M_CONST
    Kc_ptr,           # *float32, Kc rows, shape [M_CONST, K], contiguous
    out_ptr,          # *float32, output vector, shape [K], contiguous
    K: tl.constexpr,  # compile-time K
    M_CONST: tl.constexpr,
    BLOCK_M: tl.constexpr = 128
):
    # One program computes one output element of 'out'
    o = tl.program_id(0)  # 0..K-1
    acc = 0.0
    for m0 in range(0, M_CONST, BLOCK_M):
        offs_m = m0 + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M_CONST
        attn_chunk = tl.load(attn_ptr + offs_m, mask=mask_m, other=0.0)
        Kc_chunk = tl.load(Kc_ptr + offs_m * K + o, mask=mask_m, other=0.0)
        acc += tl.sum(attn_chunk * Kc_chunk, axis=0)
    tl.store(out_ptr + o, acc)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        """
        Triton-only implementation of the original run function.
        - q_nope: [B, 16, 512] bfloat16 (but we convert to float32 for compute)
        - q_pe: [B, 16, 64] bfloat16 (convert to float32)
        - ckv_cache: [num_pages, 1, 512] bfloat16 (we use tokens to index rows)
        - kpe_cache: [num_pages, 1, 64] bfloat16
        - kv_indptr: [B+1] int32
        - kv_indices: [M] int32 (tokens per batch)
        - sm_scale: float32 scalar
        Returns:
        - output: [B, 16, 512] bfloat16
        - lse: [B, 16] float32
        """
        B = q_nope.shape[0]
        device = q_nope.device
        Kc_dim = HEAD_DIM_CKV
        Kp_dim = HEAD_DIM_KPE
        assert q_nope.shape[1] == NUM_QO_HEADS, "num_qo_heads must be 16"
        assert q_nope.shape[2] == Kc_dim, "head_dim_ckv must be 512"
        assert q_pe.shape[2] == Kp_dim, "head_dim_kpe must be 64"

        # Prepare output and lse tensors
        output = torch.empty((B, NUM_QO_HEADS, Kc_dim), dtype=torch.float32, device=device)
        lse = torch.empty((B, NUM_QO_HEADS), dtype=torch.float32, device=device)

        # Process each batch element b
        for b in range(B):
            # Compute M (number of tokens for this batch element)
            if kv_indptr.numel() <= b + 1:
                # Degenerate case, no tokens
                lse[b] = -float("inf")
                output[b] = 0.0
                continue

            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            M = max(0, page_end - page_beg)
            if M == 0:
                lse[b] = -float("inf")
                output[b] = 0.0
                continue

            # Gather tokens
            tok_idx = kv_indices[page_beg:page_end].to(torch.int32).contiguous()
            M_CONST = M  # compile-time constant per launch

            # Gather Kc and Kp rows for these tokens
            Kc_all = ckv_cache.to(torch.float32)  # [num_pages, 512]
            Kp_all = kpe_cache.to(torch.float32)  # [num_pages, 64]
            Kc = Kc_all[tok_idx]  # [M, 512], contiguous
            Kp = Kp_all[tok_idx]  # [M, 64], contiguous

            # Prepare q vectors for each head: qn, qp
            for h in range(NUM_QO_HEADS):
                qn = q_nope[b, h].to(torch.float32)        # [512]
                qp = q_pe[b, h].to(torch.float32)         # [64]

                # Compute logits[i] = qn @ Kc[i, :]  (no second term for now; will add in host)
                logits = torch.empty(M_CONST, dtype=torch.float32, device=device)
                grid = (M_CONST,)
                matvec_row_kernel[grid](
                    qn, Kc, logits,
                    K=Kc_dim,
                    M_CONST=M_CONST,
                    BLOCK_K=64,
                    num_warps=4
                )

                # Compute second term for logits: logits += qp @ Kp[:, ]
                logits2 = torch.empty(M_CONST, dtype=torch.float32, device=device)
                matvec_row_kernel[grid](
                    qp, Kp, logits2,
                    K=Kp_dim,
                    M_CONST=M_CONST,
                    BLOCK_K=64,
                    num_warps=4
                )
                logits += logits2

                # Scale logits
                logits = logits * sm_scale

                # Compute lse and attn in Triton
                attn = torch.empty(M_CONST, dtype=torch.float32, device=device)
                compute_lse_and_attn_kernel[(1,)](
                    logits, attn, lse[b, h],
                    M_CONST=M_CONST
                )
                # Compute final output vector: out = attn @ Kc
                out_vec = torch.empty(Kc_dim, dtype=torch.float32, device=device)
                matvec_out_kernel[(Kc_dim,)](
                    attn, Kc, out_vec,
                    K=Kc_dim,
                    M_CONST=M_CONST,
                    BLOCK_M=128,
                    num_warps=4
                )

                # Store output for head h
                output[b, h] = out_vec

        # Cast output to bfloat16 to match original return type
        output = output.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
