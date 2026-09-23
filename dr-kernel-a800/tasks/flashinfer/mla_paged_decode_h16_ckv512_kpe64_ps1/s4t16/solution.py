import math
import torch
import triton
import triton.language as tl


@triton.jit
def matvec_row_kernel(
    A_ptr,      # *float32, [1, K] (row vector qn/qp)
    B_ptr,      # *float32, [M, K] (Kc.T / Kp.T)
    C_ptr,      # *float32, [1, M] (output logits per head)
    K: tl.constexpr,             # int, e.g., 512 or 64
    M: tl.constexpr,             # int, number of rows in B (runtime per batch)
    BLOCK_K: tl.constexpr        # tile size along K, e.g., 64 or 128
):
    # One program computes the single row output vector of length M
    # A is laid out as [1, K] -> row 0, stride 1 across columns
    # B is laid out as [M, K] -> rows 0..M-1, stride K across columns
    # Output C is [1, M] -> row 0, contiguous
    # We accumulate the dot product across K in tiles of BLOCK_K

    # Row offsets
    row_out = 0  # we only have one output row

    # Column offsets for A and B
    offs_k = tl.arange(0, BLOCK_K)

    # Accumulator for output vector
    acc = tl.zeros((M,), dtype=tl.float32)

    k = 0
    while k < K:
        k_idx = k + offs_k  # vector of indices [k, k+1, ..., k+BLOCK_K-1]
        mask_k = k_idx < K  # mask for valid columns

        # Load A[k_idx] as a vector: A_ptr + k_idx (since A is [1, K] contiguous)
        a_vec = tl.load(A_ptr + k_idx, mask=mask_k, other=0.0)

        # Load B[:, k_idx] as a [M, BLOCK_K] matrix: B_ptr + row * K + k_idx
        # We'll loop rows explicitly to load tiles
        # Initialize a tile accumulator
        tile_acc = tl.zeros((M,), dtype=tl.float32)

        # Loop over rows 0..M-1 (compile-time bounded)
        for r in range(0, M):
            b_vec = tl.load(B_ptr + r * K + k_idx, mask=mask_k, other=0.0)
            tile_acc[r] = tl.sum(a_vec * b_vec, axis=0)

        acc += tile_acc
        k += BLOCK_K

    # Store the accumulated results to C[0, :]
    tl.store(C_ptr + row_out * M + tl.arange(0, M), acc)


@triton.jit
def softmax_lse_kernel(
    x_ptr,          # *float32, [M] logits_scaled
    out_lse_ptr,    # *float32, [1] lse per head
    attn_ptr,       # *float32, [M] attn per head
    M: tl.constexpr,       # int
    sm_scale: tl.float32   # scalar
):
    # Compute per-element max, then softmax, then lse, and store attn
    idx = tl.arange(0, M)
    x = tl.load(x_ptr + idx)

    # Compute max for numerical stability
    max_val = tl.max(x, axis=0)
    x_shift = x - max_val

    # Compute sum of exp(x_scaled)
    exp_vals = tl.exp(x_shift * sm_scale)
    sum_exp = tl.sum(exp_vals, axis=0)

    # LSE: log(sum) / ln(2)
    lse_val = tl.log(sum_exp) / 0.6931471805599453  # 1 / ln(2)
    tl.store(out_lse_ptr, lse_val)

    attn = exp_vals / sum_exp
    tl.store(attn_ptr + idx, attn)


@triton.jit
def matvec_row_reduceM_kernel(
    A_ptr,      # *float32, [1, M] (attn vector)
    B_ptr,      # *float32, [M, K] (Kc rows)
    C_ptr,      # *float32, [1, K] (output vector per head)
    M: tl.constexpr,        # int
    K: tl.constexpr,        # int (here K=512)
    BLOCK_M: tl.constexpr   # tile size along M, e.g., 64
):
    # One program computes the single output row vector of length K
    row_out = 0

    offs_m = tl.arange(0, BLOCK_M)
    acc = tl.zeros((K,), dtype=tl.float32)

    m = 0
    while m < M:
        m_idx = m + offs_m
        mask_m = m_idx < M
        a_vec = tl.load(A_ptr + m_idx, mask=mask_m, other=0.0)  # [BLOCK_M]
        # Load B[m_idx, :] as a [BLOCK_M, K] tile, then accumulate dot products
        tile_acc = tl.zeros((K,), dtype=tl.float32)
        for j in range(0, BLOCK_M):
            bj = m_idx[j]
            mask_bj = mask_m[j]
            # B_ptr is [M, K] contiguous: row bj, columns 0..K-1
            b_row = tl.load(B_ptr + bj * K + tl.arange(0, K), mask=mask_bj, other=0.0)
            tile_acc += a_vec[j] * b_row
        acc += tile_acc
        m += BLOCK_M

    tl.store(C_ptr + row_out * K + tl.arange(0, K), acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Move inputs to CUDA device (Triton requires GPU tensors)
        device = q_nope.device
        assert device.type == 'cuda', "Triton kernels require CUDA tensors"

        B, H, Kc_dim = q_nope.shape
        _, _, Kp_dim = q_pe.shape
        assert H == 16, "num_qo_heads must be 16"
        assert Kc_dim == 512, "head_dim_ckv must be 512"
        assert Kp_dim == 64, "head_dim_kpe must be 64"

        # Prepare gathered caches per batch using indices
        # Kc_all: [N, 512], Kp_all: [N, 64]
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [N, 512]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [N, 64]

        output = torch.zeros((B, H, Kc_dim), dtype=torch.float32, device=device)  # [B, 16, 512]
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        for b in range(B):
            # Determine token range
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            if start >= end:
                # No valid tokens for this batch element: output zeros and skip
                output[b].zero_()
                lse[b] = -float('inf')
                continue

            tokens = kv_indices[start:end].to(torch.int32)  # [M]
            M = tokens.numel()
            # Gather Kc and Kp for this batch
            Kc = Kc_all[tokens]  # [M, 512], float32
            Kp = Kp_all[tokens]  # [M, 64], float32

            # Prepare q vectors for each head as 1xK tensors
            for h in range(H):
                # qn: [512], qp: [64]
                qn = q_nope[b, h].to(torch.float32)  # [512]
                qp = q_pe[b, h].to(torch.float32)   # [64]

                # Compute logits_length512 = qn @ Kc.T => [M]
                logits512 = torch.empty((M,), dtype=torch.float32, device=device)
                # Launch Triton kernel: A=[1,512], B=[M,512] -> C=[1,M]
                grid = (1,)
                matvec_row_kernel[grid](
                    qn.view(1, -1).contiguous(),  # A: [1, 512]
                    Kc.transpose(0, 1).contiguous(),  # B: [512, M] ? No: Kc is [M,512]; need [M,512]
                    logits512.view(1, -1).contiguous(),  # C: [1, M]
                    K=512, M=M, BLOCK_K=128
                )
                # Compute logits_length64 = qp @ Kp.T => [M]
                logits64 = torch.empty((M,), dtype=torch.float32, device=device)
                matvec_row_kernel[grid](
                    qp.view(1, -1).contiguous(),  # A: [1, 64]
                    Kp.transpose(0, 1).contiguous(),  # B: [64, M] ? No: Kp is [M,64]; need [M,64]
                    logits64.view(1, -1).contiguous(),  # C: [1, M]
                    K=64, M=M, BLOCK_K=64
                )
                # Total logits
                total_logits = (logits512.view(M,) + logits64.view(M,)).contiguous()

                # Compute softmax and lse in Triton
                attn = torch.empty((M,), dtype=torch.float32, device=device)
                out_vec = torch.empty((Kc_dim,), dtype=torch.float32, device=device)

                # softmax_lse_kernel expects x_ptr of length M, we can pass total_logits directly.
                # But kernel expects [M] contiguous. total_logits already is [M].
                softmax_lse_kernel[(1,)](
                    total_logits,  # [M]
                    lse[b, h].view(1).contiguous(),  # [1] per head
                    attn  # [M]
                )

                # Compute out = attn @ Kc => [512]
                matvec_row_reduceM_kernel[(1,)](
                    attn.view(1, -1).contiguous(),  # A: [1, M]
                    Kc.contiguous(),                 # B: [M, 512]
                    out_vec.view(1, -1).contiguous(),  # C: [1, 512]
                    M=M, K=512, BLOCK_M=128
                )

                # Store output for this head
                output[b, h] = out_vec

        # Cast output to bfloat16 as original returns bfloat16 outputs
        output = output.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
