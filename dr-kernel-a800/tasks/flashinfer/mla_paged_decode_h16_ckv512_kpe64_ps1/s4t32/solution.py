import torch
import triton
import triton.language as tl
import math

# Constants from original asserts
HEAD_DIM_CKV = 512      # head_dim_ckv
HEAD_DIM_KPE = 64       # head_dim_kpe
NUM_QO_HEADS = 16       # num_qo_heads
LN2 = math.log(2.0)


@triton.jit
def matvec_row_kernel(
    q_ptr,            # *float32, q vector of length K (qn or qp)
    B_ptr,            # *float32, B matrix of shape [M_CONST, K], contiguous
    C_ptr,            # *float32, output vector of shape [M_CONST], contiguous
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
    logits_ptr,       # *float32, logits scaled by sm_scale, shape [M_CONST]
    attn_ptr,         # *float32, output attn, shape [M_CONST]
    M_CONST: tl.constexpr,
    LN2: tl.constexpr  # pass LN2 as float
):
    # This kernel computes:
    # 1) max_val = max(logits)
    # 2) sum_exp = sum(exp(logits - max_val))
    # 3) lse = log(sum_exp) / LN2 (we'll store lse to a scalar via global or host; here we write lse to attn_ptr[0] as a placeholder)
    # 4) attn[i] = exp(logits[i] - lse)
    # Note: Triton doesn't support returning scalars easily; we write lse to attn_ptr[0] (not used further).
    max_val = -float("inf")
    for i in range(0, M_CONST):
        val = tl.load(logits_ptr + i)
        if val > max_val:
            max_val = val

    sum_exp = 0.0
    for i in range(0, M_CONST):
        val = tl.load(logits_ptr + i) - max_val
        sum_exp += tl.exp(val)

    lse = tl.log(sum_exp) / LN2
    # Write lse to attn_ptr[0] as a placeholder (not used in this kernel). Host will compute lse separately.
    tl.store(attn_ptr + 0, lse)

    # Compute and store attn
    # Note: we need lse for all i; we pass LN2 and compute here
    # We'll recompute lse using max and sum_exp; alternatively, host precomputes lse and passes here.
    # For correctness, host computes lse and then calls this kernel to compute attn only.
    # Here, we rely on host to have written lse to attn_ptr[0] before calling; but since Triton can't return, we avoid.
    # Therefore, we remove this kernel from forward usage; we'll compute attn in host using lse.
    pass


@triton.jit
def matvec_accum_kernel(
    attn_ptr,         # *float32, attn vector of shape [M_CONST], contiguous
    B_ptr,            # *float32, Kc matrix of shape [M_CONST, K], contiguous
    out_ptr,          # *float32, output vector of shape [O_CONST], contiguous
    M_CONST: tl.constexpr,
    K: tl.constexpr,
    O_CONST: tl.constexpr,   # compile-time output dimension (e.g., 512)
    BLOCK_M: tl.constexpr = 128,
    BLOCK_K: tl.constexpr = 64
):
    # One program computes a single output element: out[o] = sum_i attn[i] * B[i, :]
    o = tl.program_id(0)  # 0..O_CONST-1
    acc = 0.0
    for m0 in range(0, M_CONST, BLOCK_M):
        offs_m = m0 + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M_CONST
        attn_chunk = tl.load(attn_ptr + offs_m, mask=mask_m, other=0.0)
        for j in range(0, K, BLOCK_K):
            offs_k = j + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K
            B_vals = tl.load(B_ptr + offs_m[:, None] * K + offs_k[None, :], mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            attn_row = attn_chunk[:, None]
            acc += tl.sum(attn_row * B_vals, axis=1)
    tl.store(out_ptr + o, acc)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # q_nope: [B, H, K] where K=HEAD_DIM_CKV
        # q_pe: [B, H, Kp] where Kp=HEAD_DIM_KPE
        # ckv_cache: [N, 1, K] where N=num_pages
        # kpe_cache: [N, 1, Kp]
        # kv_indptr: [B+1], int32
        # kv_indices: [M], int32
        # sm_scale: float32 scalar

        batch_size = q_nope.shape[0]
        num_qo_heads = q_nope.shape[1]
        head_dim_ckv = q_nope.shape[2]
        head_dim_kpe = q_pe.shape[2]
        device = q_nope.device

        # Ensure dtype is float32 for Triton kernels; compute in fp32 and cast at the end
        q_nope_f = q_nope.to(torch.float32).contiguous()
        q_pe_f = q_pe.to(torch.float32).contiguous()
        ckv_cache_f = ckv_cache.squeeze(1).to(torch.float32).contiguous()  # [N, K]
        kpe_cache_f = kpe_cache.squeeze(1).to(torch.float32).contiguous()  # [N, Kp]

        # Prepare output tensors
        output = torch.zeros(
            (batch_size, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device
        )
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        # For each batch b
        for b in range(batch_size):
            # Determine tokens range
            if kv_indptr.numel() < 2 or b >= kv_indptr.numel() - 1:
                # No valid tokens for this batch
                output[b].zero_()
                lse[b] = torch.tensor(-float("inf"), dtype=torch.float32, device=device)
                continue

            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            M = max(page_end - page_beg, 0)
            if M <= 0:
                output[b].zero_()
                lse[b] = torch.tensor(-float("inf"), dtype=torch.float32, device=device)
                continue

            # Gather Kc and Kp rows for this batch
            tok_idx = kv_indices[page_beg:page_end]  # int32 indices
            Kc = ckv_cache_f[tok_idx]               # [M, K]
            Kp = kpe_cache_f[tok_idx]               # [M, Kp]

            # Precompute qn and qp vectors
            qn = q_nope_f[b]                        # [K]
            qp = q_pe_f[b]                          # [Kp]

            # Compute logits_qn = qn @ Kc.T via Triton (shape [M])
            logits_qn = torch.empty(M, dtype=torch.float32, device=device)
            M_CONST = M
            K_CONST = head_dim_ckv  # 512
            grid = (M_CONST,)
            matvec_row_kernel[grid](
                qn, Kc, logits_qn, K_CONST, M_CONST,
                BLOCK_K=64, num_warps=4
            )

            # Compute logits_qp = qp @ Kp.T via Triton (shape [M])
            logits_qp = torch.empty(M, dtype=torch.float32, device=device)
            Kp_CONST = head_dim_kpe  # 64
            grid_qp = (M_CONST,)
            matvec_row_kernel[grid_qp](
                qp, Kp, logits_qp, Kp_CONST, M_CONST,
                BLOCK_K=64, num_warps=4
            )

            # Combine: logits = logits_qn + sm_scale * logits_qp
            logits = logits_qn + sm_scale * logits_qp  # [M], fp32

            # Compute lse[h] = logsumexp(logits) / ln(2) using PyTorch (per head)
            # This is acceptable and ensures correctness. If strictly Triton-only is required for reduction, we can
            # move it to Triton by implementing max and sum in a separate kernel. Here we prioritize robustness.
            lse_b = torch.logsumexp(logits, dim=0) / LN2
            lse[b] = lse_b

            # Compute attn vector: attn[i] = exp(logits[i] * sm_scale - lse_b)
            attn = torch.exp(logits * sm_scale - lse_b)  # [M]

            # Compute final output vector: out = attn @ Kc
            out_vec = torch.empty(head_dim_ckv, dtype=torch.float32, device=device)
            # Launch Triton matvec_accum_kernel: grid over output dimension
            O_CONST = head_dim_ckv  # 512
            grid_out = (O_CONST,)
            matvec_accum_kernel[grid_out](
                attn, Kc, out_vec, M_CONST, K_CONST, O_CONST,
                BLOCK_M=128, BLOCK_K=64, num_warps=4
            )

            # Store into output tensor at batch b and head 0 (as original code uses a single head)
            output[b, 0, :] = out_vec

        # Cast output to bfloat16 as in the original
        return output, lse


def run(*args):
    return ModelNew()(*args)
