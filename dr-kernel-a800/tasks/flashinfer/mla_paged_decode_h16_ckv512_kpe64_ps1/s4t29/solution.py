import torch
import triton
import triton.language as tl


@triton.jit
def matvec_row_kernel(
    A_ptr,       # *float32, pointer to q vector (length K), flattened to 1D
    B_ptr,       # *float32, pointer to rows, shape [M, K], contiguous
    C_ptr,       # *float32, pointer to output, shape [M], contiguous
    M,           # int, number of rows in B (runtime)
    K: tl.constexpr,           # int, length of q and columns of B (compile-time)
    BLOCK_K: tl.constexpr = 64 # tile size along K
):
    # One program computes a single output element: C[i] = A @ B[i, :]
    i = tl.program_id(0)  # index over M, grid size must be >= M
    acc = 0.0
    # Loop over K in tiles; K is compile-time, so this loop unrolls
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)  # not used here since we loop, but kept for clarity
        # Accumulate dot product of A and row i of B
        for kk in range(k0, k0 + BLOCK_K):
            mask_k = kk < K
            a = tl.load(A_ptr + kk, mask=mask_k, other=0.0)  # scalar load
            b = tl.load(B_ptr + i * K + kk, mask=mask_k, other=0.0)  # scalar load
            acc += a * b
    tl.store(C_ptr + i, acc)


@triton.jit
def softmax_lse_row_kernel(
    logits_ptr,        # *float32, pointer to logits_scaled vector, length M_CONST
    out_lse_ptr,       # *float32, pointer to lse output scalar for this head
    M: tl.constexpr,   # number of logits (compile-time for kernel)
    sm_scale,          # float32 scalar
):
    # Compute per-row softmax and lse for this head. We'll do scalar accumulation over M.
    # First, compute max over logits to improve numerical stability.
    max_val = -float("inf")
    # Loop over M to find max
    for m in range(0, M):
        x = tl.load(logits_ptr + m)
        if x > max_val:
            max_val = x
    # Compute sum of exp(logits - max) * sm_scale
    sum_exp = 0.0
    for m in range(0, M):
        x = tl.load(logits_ptr + m)
        e = tl.exp((x - max_val) * sm_scale)
        sum_exp += e
    # lse = log(sum_exp) / ln(2) = log(sum_exp) * 1.4426950408889634 (log2_e)
    lse = tl.log(sum_exp) * 1.4426950408889634
    tl.store(out_lse_ptr, lse)


@triton.jit
def matvec_attn_kernel(
    attn_ptr,          # *float32, pointer to attn vector, length M
    B_ptr,             # *float32, pointer to K rows, shape [M, K], contiguous
    C_ptr,             # *float32, pointer to output, shape [K], contiguous
    M,                 # int, number of rows in B (runtime)
    K: tl.constexpr,           # int, length of columns (compile-time)
    BLOCK_M: tl.constexpr = 64  # tile size along M for accumulation
):
    # One program computes a single output element: C[d] = sum_i attn[i] * B[i, d]
    d = tl.program_id(0)  # index over K (grid size must be >= K)
    acc = 0.0
    # Loop over M in tiles; M is runtime but we use scalar loads in loop
    for m0 in range(0, M, BLOCK_M):
        for mm in range(m0, m0 + BLOCK_M):
            mask_m = mm < M
            attn_i = tl.load(attn_ptr + mm, mask=mask_m, other=0.0)  # scalar
            b_vec = tl.load(B_ptr + mm * K + tl.arange(0, K), mask=tl.arange(0, K) < K, other=0.0)  # vector for all K at mm
            # We need b[d] only, not the whole vector. Fix: scalar load b[mm, d]
            b_elem = tl.load(B_ptr + mm * K + d, mask=mm < M, other=0.0)
            acc += attn_i * b_elem
    tl.store(C_ptr + d, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Input types and shapes
        assert q_nope.dim() == 3 and q_pe.dim() == 3
        assert ckv_cache.dim() == 3 and kpe_cache.dim() == 3
        assert kv_indptr.dim() == 1 and kv_indices.dim() == 1
        B = q_nope.shape[0]
        H = q_nope.shape[1]
        Kc_dim = q_nope.shape[2]
        Kp_dim = q_pe.shape[2]
        N = ckv_cache.shape[0]
        assert ckv_cache.shape[1] == 1 and kpe_cache.shape[1] == 1
        assert kv_indptr.shape[0] == B + 1
        # Prepare device and dtypes
        device = q_nope.device
        # Ensure inputs are on same device and contiguous
        # q_nope, q_pe already tensors, ckv_cache, kpe_cache, kv_indptr, kv_indices may be on CPU; move if needed
        # However, Triton kernels require CUDA tensors. If inputs are on CPU, move to CUDA.
        if not q_nope.is_cuda:
            q_nope = q_nope.to(device='cuda')
            q_pe = q_pe.to(device='cuda')
            ckv_cache = ckv_cache.to(device='cuda')
            kpe_cache = kpe_cache.to(device='cuda')
            kv_indptr = kv_indptr.to(device='cuda')
            kv_indices = kv_indices.to(device='cuda')

        output = torch.empty((B, H, Kc_dim), dtype=torch.bfloat16, device=device)
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        # Precompute Kc_all and Kp_all for all tokens (squeeze dim=1)
        # But in original, Kc and Kp are gathered per batch using kv_indptr and kv_indices.
        # We will gather per batch below.

        for b in range(B):
            # Determine valid token range
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            M = end - start
            if M <= 0:
                # No valid tokens for this batch, output zeros
                output[b].zero_()
                lse[b].zero_()
                continue

            # Gather tokens for this batch
            tok_idx = kv_indices[start:start + M].to(torch.long)

            # Gather cache rows
            Kc = ckv_cache[tok_idx, 0, :].to(torch.float32)  # [M, 512]
            Kp = kpe_cache[tok_idx, 0, :].to(torch.float32)  # [M, 64]

            # Prepare q vectors for all heads
            # q_nope[b, :, :] shape [H, Kc_dim], q_pe[b, :, :] shape [H, Kp_dim]
            # We compute logits per head
            for h in range(H):
                qn = q_nope[b, h, :].to(torch.float32).contiguous()  # [Kc_dim]
                qp = q_pe[b, h, :].to(torch.float32).contiguous()    # [Kp_dim]

                # Compute logits = qn @ Kc.T -> [M]
                logits_Kc = torch.empty((M,), dtype=torch.float32, device=device)
                # Launch Triton kernel: grid=(M,), K=Kc_dim (compile-time loop)
                matvec_row_kernel[(M,)](
                    qn, Kc, logits_Kc,
                    M, Kc_dim,
                    BLOCK_K=128
                )

                # Compute logits = qp @ Kp.T -> [M]
                logits_Kp = torch.empty((M,), dtype=torch.float32, device=device)
                matvec_row_kernel[(M,)](
                    qp, Kp, logits_Kp,
                    M, Kp_dim,
                    BLOCK_K=128
                )

                # Elementwise add
                logits = logits_Kc + logits_Kp  # [M]

                # Compute lse for this head in Triton: lse[b, h] = logsumexp(logits * sm_scale) / ln(2)
                # We'll pass M as constexpr to kernel (set M_CONST = M). Triton allows passing scalars as constexpr if used consistently.
                # However, Triton kernels typically expect tl.constexpr as meta-parameters. We'll emulate by passing a scalar with constexpr binding.
                # Here, M is runtime, but we'll ensure Triton kernel uses only scalar loops.
                logits_scaled = logits * sm_scale
                out_lse = torch.empty((), dtype=torch.float32, device=device)
                softmax_lse_row_kernel[(1,)](
                    logits_scaled, out_lse,
                    M=M, sm_scale=sm_scale
                )
                lse[b, h] = out_lse

                # Compute attn = softmax(logits_scaled, dim=0)
                attn = torch.empty((M,), dtype=torch.float32, device=device)
                # Implement softmax in Triton: need vector or scalar; use scalar accumulation here for robustness.
                # Compute max
                max_val = -float("inf")
                for m in range(0, M):
                    x = logits_scaled[m]
                    if x > max_val:
                        max_val = x
                # Compute exp and normalize
                sum_exp = 0.0
                for m in range(0, M):
                    e = torch.exp((logits_scaled[m] - max_val) * sm_scale)
                    attn[m] = e
                    sum_exp += e
                attn = attn / sum_exp

                # Compute out = attn @ Kc -> [Kc_dim]
                out = torch.empty((Kc_dim,), dtype=torch.float32, device=device)
                matvec_attn_kernel[(Kc_dim,)](
                    attn, Kc, out,
                    M, Kc_dim,
                    BLOCK_M=128
                )
                output[b, h, :] = out.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
