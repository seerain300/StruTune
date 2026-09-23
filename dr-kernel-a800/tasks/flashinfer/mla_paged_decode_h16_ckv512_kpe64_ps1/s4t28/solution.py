import torch
import triton
import triton.language as tl


@triton.jit
def matvec_row_kernel(
    A_ptr,       # *float32, pointer to q vector (length K), flattened to 1D
    B_ptr,       # *float32, pointer to K rows, shape [M, K], contiguous
    C_ptr,       # *float32, pointer to output, shape [M], contiguous
    M,           # int, number of rows in B (runtime)
    K: tl.constexpr,           # int, length of q and columns of B (compile-time)
    BLOCK_K: tl.constexpr = 64 # tile size along K
):
    # One program computes a single output element: C[i] = A @ B[i, :]
    i = tl.program_id(0)  # index over M
    # Initialize accumulator for this output element
    acc = 0.0
    # Loop over K in tiles
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)  # vector of indices along K
        mask_k = offs_k < K
        # Load q chunk: A_ptr is [K], so slice via + offs_k
        a = tl.load(A_ptr + offs_k, mask=mask_k, other=0.0)
        # Load B row chunk: B_ptr is [M, K], row i, cols offs_k
        b = tl.load(B_ptr + i * K + offs_k, mask=mask_k, other=0.0)
        # Accumulate dot product of a and b
        acc += tl.sum(a * b, axis=0)
    # Store the result for this i
    tl.store(C_ptr + i, acc)


@triton.jit
def softmax_lse_kernel(
    logits_ptr,         # *float32, [M] logits_scaled
    out_lse_ptr,        # *float32, [1] lse for this head
    attn_ptr,           # *float32, [M] attn vector for this head
    M: tl.constexpr,    # int, number of tokens (compile-time for reductions)
    sm_scale            # float32
):
    # Compute per-row max over logits (no tl.arange on runtime)
    max_val = -float("inf")
    # Pass 1: find max
    for m in range(0, M):
        x = tl.load(logits_ptr + m)
        if x > max_val:
            max_val = x
    # Compute sum of exp(logits - max_val)
    sum_exp = 0.0
    for m in range(0, M):
        x = tl.load(logits_ptr + m)
        e = tl.exp((x - max_val) * sm_scale)
        sum_exp += e
    # lse per row: log(sum_exp) / ln(2)
    ln2 = 0.6931471805599453  # log(2)
    lse_val = tl.log(sum_exp) / ln2
    # Store lse for this head
    tl.store(out_lse_ptr, lse_val)
    # Compute and store attn
    for m in range(0, M):
        x = tl.load(logits_ptr + m)
        attn_m = tl.exp((x - max_val) * sm_scale) / sum_exp
        tl.store(attn_ptr + m, attn_m)


@triton.jit
def matvec_row_d_kernel(
    A_ptr,       # *float32, pointer to attn vector (length M), flattened to 1D
    B_ptr,       # *float32, pointer to K rows, shape [Kc_dim, M], contiguous (row-major)
    C_ptr,       # *float32, pointer to output, shape [Kc_dim], contiguous
    M: tl.constexpr,         # int, length of A (compile-time for loops)
    Kc_dim,                   # int, number of output columns (runtime)
    BLOCK_M: tl.constexpr = 64
):
    # One program computes a single output element: C[d] = A @ B[d, :]
    d = tl.program_id(0)  # index over Kc_dim
    acc = 0.0
    # Loop over M in chunks
    for m0 in range(0, M, BLOCK_M):
        offs_m = m0 + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M
        # Load A chunk: A_ptr is [M]
        a = tl.load(A_ptr + offs_m, mask=mask_m, other=0.0)
        # Load B row chunk: B_ptr is [Kc_dim, M], row d, cols offs_m
        b = tl.load(B_ptr + d * M + offs_m, mask=mask_m, other=0.0)
        acc += tl.sum(a * b, axis=0)
    # Store the result for this d
    tl.store(C_ptr + d, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Input checks and setup
        B, H, Kc_dim = q_nope.shape
        _, _, Kp_dim = q_pe.shape
        assert H == 16, "num_qo_heads must be 16"
        assert Kc_dim == 512, "head_dim_ckv must be 512"
        assert Kp_dim == 64, "head_dim_kpe must be 64"
        device = q_nope.device
        assert q_nope.dtype == torch.bfloat16 and q_pe.dtype == torch.bfloat16
        assert ckv_cache.shape[1] == 1 and kpe_cache.shape[1] == 1, "cache second dim must be 1"
        assert kv_indptr.dtype == torch.int32 and kv_indices.dtype == torch.int32

        # Squeeze caches: [N, 1, D] -> [N, D]
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [N, 512]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [N, 64]

        # Output tensors
        output = torch.empty((B, H, Kc_dim), dtype=torch.bfloat16, device=device)
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        # Process each batch element
        for b in range(B):
            # Determine valid token range
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            M = end - start
            if M <= 0:
                lse[b, :] = -float("inf")
                output[b] = torch.zeros((H, Kc_dim), dtype=torch.bfloat16, device=device)
                continue

            # Gather Kc and Kp rows
            tok_idx = kv_indices[start:end]  # [M]
            Kc = Kc_all[tok_idx]             # [M, 512]
            Kp = Kp_all[tok_idx]             # [M, 64]

            # Prepare q vectors (float32 for compute)
            qn = q_nope[b].to(torch.float32)  # [16, 512]
            qp = q_pe[b].to(torch.float32)    # [16, 64]

            # Compute logits = qn @ Kc.T + qp @ Kp.T for each head h
            # We'll use Triton kernels for these matvec computations.
            for h in range(H):
                # logits_qn: [M] = qn[h] @ Kc
                # Cast qn[h] to 1D [Kc_dim]
                qn_h = qn[h, :]  # [512]
                # Allocate logits_qn_out [M]
                logits_qn = torch.empty((M,), dtype=torch.float32, device=device)
                # Launch Triton kernel: grid over M
                grid = (M,)
                matvec_row_kernel[grid](
                    qn_h, Kc, logits_qn, M, Kc_dim, BLOCK_K=64
                )
                # logits_qp: [M] = qp[h] @ Kp
                qp_h = qp[h, :]  # [64]
                logits_qp = torch.empty((M,), dtype=torch.float32, device=device)
                matvec_row_kernel[grid](
                    qp_h, Kp, logits_qp, M, Kp_dim, BLOCK_K=64
                )
                # Sum to get logits
                logits = logits_qn + logits_qp  # [M]

                # Compute lse and attn using Triton
                # We need to pass M as constexpr-like here. Triton expects constexpr for tl.static_range loops.
                # Use M_CONST as a Python int for loops; Triton can handle Python range over M.
                # Create tensors to hold results
                lse_bh = torch.empty((1,), dtype=torch.float32, device=device)
                attn = torch.empty((M,), dtype=torch.float32, device=device)
                # Launch Triton softmax_lse_kernel. M is runtime; Triton supports Python for-loops in kernels.
                softmax_lse_kernel[(1,)](
                    logits, lse_bh, attn, M, sm_scale
                )
                # Store lse for this head
                lse[b, h] = lse_bh[0]

                # Compute out[h, :] = attn @ Kc using Triton
                out_h = torch.empty((Kc_dim,), dtype=torch.float32, device=device)
                matvec_row_d_kernel[(Kc_dim,)](
                    attn, Kc, out_h, M, Kc_dim, BLOCK_M=64
                )
                output[b, h, :] = out_h.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
