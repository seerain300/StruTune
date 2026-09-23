import math
import torch
import triton
import triton.language as tl


@triton.jit
def matvec_row_kernel(A_ptr, B_ptr, C_ptr,
                      K: tl.constexpr,  # reduction dimension (512 or 64)
                      M,                # number of rows in B (runtime)
                      BLOCK_K: tl.constexpr):
    """
    Compute a single output row: A: [1, K], B: [M, K] -> C: [1, M]
    Grid: (1,) one program computes the whole row.
    """
    # Column offsets for reduction tiles
    offs_k = tl.arange(0, BLOCK_K)

    # Accumulator for output vector C[0, :]
    acc = tl.zeros((M,), dtype=tl.float32)

    k0 = 0
    while k0 < K:
        k_idx = k0 + offs_k  # [BLOCK_K]
        # Load A[k_idx] as a vector: A_ptr + k_idx, shape [BLOCK_K]
        a_vec = tl.load(A_ptr + k_idx, mask=k_idx < K, other=0.0)

        # Load B[:, k_idx] as a [M, BLOCK_K] matrix: B_ptr + row * K + k_idx
        # We build the 2D pointer for the tile: shape [M, BLOCK_K]
        # Note: Triton requires pointer to be computed with tl.arange or static_range.
        row_idx = tl.arange(0, M)  # M must be tl.constexpr; Triton specializes per launch.
        b_ptrs = B_ptr + row_idx * K + k_idx
        b_tile = tl.load(b_ptrs, mask=row_idx < M, other=0.0)  # [M, BLOCK_K]

        # Accumulate: sum over K tile, broadcasting a_vec over rows
        acc += tl.sum(b_tile * a_vec[None, :], axis=1)

        k0 += BLOCK_K

    # Write result to C[0, :]
    out_idx = tl.arange(0, M)
    tl.store(C_ptr + out_idx, acc)


@triton.jit
def softmax_lse_kernel(x_ptr, attn_ptr, out_lse_ptr,
                       M: tl.constexpr,  # length of logits (compile-time for specialization)
                       sm_scale: tl.float32):
    """
    Given logits_scaled vector x_ptr[M], compute:
      - attn = softmax(x_ptr)
      - lse = logsumexp(x_ptr) / ln(2)
    Writes attn to attn_ptr[M], lse to out_lse_ptr[0].
    """
    # Load logits_scaled into vector
    idx = tl.arange(0, M)
    logits_scaled = tl.load(x_ptr + idx)  # [M], float32

    # Compute max and sum(exp(...))
    max_val = tl.max(logits_scaled, axis=0)
    sum_exp = tl.sum(tl.exp(logits_scaled - max_val), axis=0)
    lse = (max_val + tl.log(sum_exp)) / 0.6931471805599453  # 1 / ln(2)

    # Compute attention
    attn = tl.exp(logits_scaled - max_val) / sum_exp  # [M]

    # Store attn and lse
    tl.store(attn_ptr + idx, attn)
    tl.store(out_lse_ptr, lse)


@triton.jit
def matvec_row_kernel_out(A_ptr, B_ptr, C_ptr,
                          Kc_dim: tl.constexpr,  # output dimension (e.g., 512)
                          M: tl.constexpr,       # reduction dimension (number of tokens)
                          BLOCK_M: tl.constexpr):
    """
    Compute a single output row vector: A: [M], B: [M, Kc_dim] -> C: [Kc_dim]
    Grid: (1,) one program computes the whole output vector.
    """
    # Output accumulator
    acc = tl.zeros((Kc_dim,), dtype=tl.float32)

    m0 = 0
    while m0 < M:
        offs_m = m0 + tl.arange(0, BLOCK_M)  # [BLOCK_M]
        # Load A[offs_m] as a vector
        a_vec = tl.load(A_ptr + offs_m, mask=offs_m < M, other=0.0)  # [BLOCK_M]

        # Load B[offs_m, :] as [BLOCK_M, Kc_dim]
        b_ptrs = B_ptr + offs_m[:, None] * Kc_dim + tl.arange(0, Kc_dim)[None, :]
        b_tile = tl.load(b_ptrs, mask=offs_m[:, None] < M, other=0.0)  # [BLOCK_M, Kc_dim]

        # Accumulate
        acc += tl.sum(b_tile * a_vec[:, None], axis=0)

        m0 += BLOCK_M

    # Store result
    out_d = tl.arange(0, Kc_dim)
    tl.store(C_ptr + out_d, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        """
        Triton-only implementation of the original run function.
        q_nope: [B, 16, 512] bfloat16
        q_pe: [B, 16, 64] bfloat16
        ckv_cache: [N, 1, 512] bfloat16
        kpe_cache: [N, 1, 64] bfloat16
        kv_indptr: [B+1] int32
        kv_indices: [L] int32
        sm_scale: float32 scalar
        Returns:
        - output: [B, 16, 512] bfloat16
        - lse: [B, 16] float32
        """
        device = q_nope.device
        B = q_nope.shape[0]
        H = q_nope.shape[1]
        Kc_dim = q_nope.shape[2]  # 512
        Kp_dim = q_pe.shape[2]    # 64

        # Prepare cached matrices
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [N, 512]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [N, 64]

        # Output and lse initialization
        output = torch.empty((B, H, Kc_dim), dtype=torch.float32, device=device)  # we'll cast later to bfloat16
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        for b in range(B):
            # Compute tokens for this batch
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            if start >= end:
                # No valid tokens for this batch
                output[b].zero_()
                lse[b].zero_()
                continue

            tokens = kv_indices[start:end]  # [M]
            M = tokens.numel()

            # Gather Kc and Kp for these tokens
            Kc = Kc_all[tokens]  # [M, 512]
            Kp = Kp_all[tokens]  # [M, 64]

            # Convert q_nope and q_pe to float32 and shape [H, K]
            qn = q_nope[b].to(torch.float32).transpose(0, 1).reshape(H, Kc_dim)  # [H, 512]
            qp = q_pe[b].to(torch.float32).transpose(0, 1).reshape(H, Kp_dim)   # [H, 64]

            # Compute logits_length512[h, m] = qn[h, :] @ Kc[m, :] for all h
            # We will compute per head using matvec_row_kernel
            logits512 = torch.empty((H, M), dtype=torch.float32, device=device)
            # Launch kernel per head: grid=(1,)
            for h in range(H):
                # A = qn[h, :] -> [1, 512], B = Kc.T -> [M, 512]
                A = qn[h].view(1, Kc_dim).contiguous()
                B = Kc.transpose(0, 1).contiguous()  # [M, 512]
                C = torch.empty((1, M), dtype=torch.float32, device=device)
                # Choose BLOCK_K based on Kc_dim
                BLOCK_K = 128 if Kc_dim >= 128 else 64
                matvec_row_kernel[(1,)](
                    A, B, C,
                    K=Kc_dim, M=M, BLOCK_K=BLOCK_K,
                    num_warps=4, num_stages=2
                )
                logits512[h] = C[0]  # [M]

            # Compute logits_length64[h, m] = qp[h, :] @ Kp[m, :] for all h
            logits64 = torch.empty((H, M), dtype=torch.float32, device=device)
            for h in range(H):
                A = qp[h].view(1, Kp_dim).contiguous()          # [1, 64]
                B = Kp.transpose(0, 1).contiguous()             # [M, 64]
                C = torch.empty((1, M), dtype=torch.float32, device=device)
                BLOCK_K = 64
                matvec_row_kernel[(1,)](
                    A, B, C,
                    K=Kp_dim, M=M, BLOCK_K=BLOCK_K,
                    num_warps=2, num_stages=2
                )
                logits64[h] = C[0]  # [M]

            # Total logits
            logits = logits512 + logits64  # [H, M]

            # Compute lse and attn per head using softmax_lse_kernel
            for h in range(H):
                # logits_scaled[h, :] = logits[h, :] * sm_scale
                x = logits[h] * sm_scale  # [M]
                attn = torch.empty((M,), dtype=torch.float32, device=device)
                lse_scalar = torch.empty((1,), dtype=torch.float32, device=device)
                # Make x a 1D tensor for kernel
                x_vec = x.contiguous()
                # Launch with M as constexpr specialization
                softmax_lse_kernel[(1,)](
                    x_vec, attn, lse_scalar,
                    M=M, sm_scale=sm_scale,
                    num_warps=4, num_stages=2
                )
                lse[b, h] = lse_scalar[0]
                # Now compute out[h, :] = attn @ Kc -> [512]
                A = attn.contiguous()  # [M]
                B = Kc  # [M, 512]
                C = torch.empty((Kc_dim,), dtype=torch.float32, device=device)
                BLOCK_M = 128
                matvec_row_kernel_out[(1,)](
                    A, B, C,
                    Kc_dim=Kc_dim, M=M, BLOCK_M=BLOCK_M,
                    num_warps=4, num_stages=2
                )
                output[b, h] = C  # [512]

        # Cast output to bfloat16 as original returns
        output = output.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
