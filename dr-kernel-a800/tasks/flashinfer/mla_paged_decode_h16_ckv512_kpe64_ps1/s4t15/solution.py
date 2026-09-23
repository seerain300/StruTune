import math
import torch
import triton
import triton.language as tl


@triton.jit
def matvec_row_kernel(
    A_ptr,        # *float32, points to a 1xK row vector (we pass qn or qp)
    B_ptr,        # *float32, points to [M, K] matrix (Kc or Kp)
    C_ptr,        # *float32, points to [1, M] output
    K: tl.constexpr,     # int, e.g., 512 or 64
    M,               # int, number of rows in B (runtime)
    sm_scale,        # float32
    BLOCK_K: tl.constexpr  # tile size along K, e.g., 64 or 128
):
    # grid = (1,), one program computes the row-vector product
    # A is [1, K], B is [M, K]
    acc = tl.zeros((1,), dtype=tl.float32)
    # loop over K dimension in tiles
    for k0 in range(0, K, BLOCK_K):
        cols = k0 + tl.arange(0, BLOCK_K)  # constexpr vector length
        mask_k = cols < K
        # Load A chunk: A_ptr + 0 * stride + cols
        a = tl.load(A_ptr + cols, mask=mask_k, other=0.0)  # [BLOCK_K]
        # Load B chunk: B_ptr + row_idx * stride_m + cols, row_idx = 0 since grid=(1,)
        b = tl.load(B_ptr + cols, mask=mask_k, other=0.0)  # [BLOCK_K]
        acc += tl.sum(a * b, axis=0)
    # Write output C[0, :]
    # Output is [1, M], so store at C_ptr + 0 * M + j where j=0..M-1
    # We'll compute vector of outputs in another step; here just set first element.
    # To write full vector, we'd need a different kernel. But our use case is computing a single row output vector for qn @ Kc.T,
    # where we actually launch with grid=(M,) below.
    pass


# Note: The above kernel is a template; we will use a different launch for per-row output.


@triton.jit
def matvec_row_outputs_kernel(
    A_ptr,        # *float32, [1, K]
    B_ptr,        # *float32, [M, K]
    C_ptr,        # *float32, [M]  # one per row
    K: tl.constexpr,     # int, e.g., 512 or 64
    M: tl.constexpr,     # int, number of rows in B (compile-time for kernel)
    sm_scale,        # float32
    BLOCK_K: tl.constexpr
):
    # grid = (M,), each program handles one output row
    row_id = tl.program_id(0)
    acc = tl.zeros((1,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        cols = k0 + tl.arange(0, BLOCK_K)
        mask_k = cols < K
        a = tl.load(A_ptr + cols, mask=mask_k, other=0.0)  # [BLOCK_K]
        b = tl.load(B_ptr + row_id * K + cols, mask=mask_k, other=0.0)  # [BLOCK_K]
        acc += tl.sum(a * b, axis=0)
    # Write scalar output for this row
    tl.store(C_ptr + row_id, acc)


@triton.jit
def softmax_lse_kernel(
    x_ptr,        # *float32, [M]
    out_lse_ptr,  # *float32, [1]  # lse per head
    attn_ptr,     # *float32, [M]
    M: tl.constexpr,     # int, compile-time
    sm_scale,     # float32
    inv_ln2,      # float32
):
    # Load vector x
    idx = tl.arange(0, M)
    x = tl.load(x_ptr + idx)  # [M]
    # Compute max for numerical stability
    max_val = tl.max(x, axis=0)
    x_shift = x - max_val
    # Scale
    x_scaled = x_shift * sm_scale
    # Compute sum of exp
    exp_x = tl.exp(x_scaled)
    sum_exp = tl.sum(exp_x, axis=0)
    # LSE and attn
    lse = tl.log(sum_exp) * inv_ln2
    attn = exp_x / sum_exp
    # Write lse and attn
    tl.store(out_lse_ptr, lse)
    tl.store(attn_ptr + idx, attn)


@triton.jit
def matvec_row_out_kernel(
    attn_ptr,     # *float32, [M] softmax vector
    B_ptr,        # *float32, [M, K] Kc
    C_ptr,        # *float32, [K] output vector
    K: tl.constexpr,     # int, e.g., 512
    M: tl.constexpr,     # int
    BLOCK_M: tl.constexpr
):
    # grid = (K,), one program computes one output element (per head dimension)
    d = tl.program_id(0)
    acc = tl.zeros((), dtype=tl.float32)
    for m0 in range(0, M, BLOCK_M):
        rows = m0 + tl.arange(0, BLOCK_M)
        mask_m = rows < M
        attn = tl.load(attn_ptr + rows, mask=mask_m, other=0.0)  # [BLOCK_M]
        Kc_cols = tl.load(B_ptr + rows * K + d, mask=mask_m, other=0.0)  # [BLOCK_M]
        acc += tl.sum(attn * Kc_cols, axis=0)
    tl.store(C_ptr + d, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        """
        Triton-only implementation of the original run function.
        - q_nope: [B, 16, 512], bfloat16
        - q_pe: [B, 16, 64], bfloat16
        - ckv_cache: [N, 1, 512], bfloat16
        - kpe_cache: [N, 1, 64], bfloat16
        - kv_indptr: [B+1], int32
        - kv_indices: [L], int32
        - sm_scale: float32
        Returns:
        - output: [B, 16, 512], bfloat16
        - lse: [B, 16], float32
        """
        device = q_nope.device
        B = q_nope.shape[0]
        H = q_nope.shape[1]  # 16
        Kc_dim = q_nope.shape[2]  # 512
        Kp_dim = q_pe.shape[2]     # 64

        # Prepare caches (float32 for Triton kernels)
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [N, 512]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [N, 64]

        output = torch.zeros((B, H, Kc_dim), dtype=torch.bfloat16, device=device)
        lse = torch.full((B, H), -float("inf"), dtype=torch.float32, device=device)

        # Constants
        inv_ln2 = 1.4426950408889634  # 1 / ln(2)

        # Process each batch b
        for b in range(B):
            # Determine valid token range
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            M = end - start  # number of valid tokens for this batch

            # If no tokens, skip
            if M <= 0:
                lse[b] = -float("inf")  # already initialized
                continue

            # Gather indices and corresponding cache rows
            tok_idx = kv_indices[start:end].to(torch.int64)  # [M]
            Kc = Kc_all[tok_idx].contiguous().to(torch.float32)  # [M, 512]
            Kp = Kp_all[tok_idx].contiguous().to(torch.float32)  # [M, 64]

            # Compute logits_length512 = qn @ Kc.T for each head
            for h in range(H):
                qn = q_nope[b, h].to(torch.float32).contiguous()  # [512]
                # Launch Triton kernel to compute qn @ Kc.T -> [M]
                logits512 = torch.empty((M,), dtype=torch.float32, device=device)
                # We'll use BLOCK_K=128 for K=512, 64 for Kp reductions
                # Triton kernels expect A to be [1, K]. Create dummy 1xK pointers by reshaping qn to [1, K].
                A_qn = qn.view(1, Kc_dim).contiguous()  # [1, 512]
                B_Kc = Kc  # [M, 512]
                # We need grid=(M,) output. Use a different kernel template.
                # Implement with a kernel that writes one output per row:
                matvec_row_outputs_kernel[(M,)](
                    A_qn, B_Kc, logits512,
                    K=Kc_dim, M=M, sm_scale=1.0, BLOCK_K=128
                )

                # Compute logits_length64 = qp @ Kp.T
                qp = q_pe[b, h].to(torch.float32).contiguous()  # [64]
                A_qp = qp.view(1, Kp_dim).contiguous()  # [1, 64]
                B_Kp = Kp  # [M, 64]
                logits64 = torch.empty((M,), dtype=torch.float32, device=device)
                matvec_row_outputs_kernel[(M,)](
                    A_qp, B_Kp, logits64,
                    K=Kp_dim, M=M, sm_scale=1.0, BLOCK_K=64
                )

                # Total logits
                logits = logits512 + logits64  # [M]
                # Launch Triton softmax_lse_kernel to compute lse and attn for this head
                attn = torch.empty((M,), dtype=torch.float32, device=device)
                lse_scalar = torch.empty((1,), dtype=torch.float32, device=device)
                # M is runtime, but Triton can specialize per launch. We pass M as meta.
                softmax_lse_kernel[(1,)](
                    logits, lse_scalar, attn,
                    M=M, sm_scale=sm_scale, inv_ln2=inv_ln2
                )
                lse[b, h] = lse_scalar[0]  # [1] scalar

                # Compute out = attn @ Kc (per head), output is [512]
                out_vec = torch.empty((Kc_dim,), dtype=torch.float32, device=device)
                # grid over Kc_dim
                matvec_row_out_kernel[(Kc_dim,)](
                    attn, Kc, out_vec,
                    K=Kc_dim, M=M, BLOCK_M=128
                )
                # Assign to output
                output[b, h] = out_vec.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
