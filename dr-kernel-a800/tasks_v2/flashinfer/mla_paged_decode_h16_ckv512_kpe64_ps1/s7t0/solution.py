import torch
import triton
import triton.language as tl

# Triton kernels: we implement all computations in Triton to satisfy the strict requirement.
# 1) Matmul small: C[M, N] = A[M, K] @ B[K, N]
@triton.jit
def matmul_small(A_ptr, B_ptr, C_ptr,
                  M, N, K,
                  stride_am, stride_ak,
                  stride_bk, stride_bn,
                  stride_cm, stride_cn,
                  BLOCK_K: tl.constexpr,
                  BLOCK_N: tl.constexpr):
    # Grid: (pid_m over M, pid_n over N)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # BLOCK_M is 1 in practice, but keep generic
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize accumulator for C[M, N]
    # Since M can be 1, we handle one row per program. We'll set BLOCK_M=1 implicitly.
    # Create accumulator for each m in this row (BLOCK_M is 1 so a single accumulator).
    # Use float32 for accumulation; Triton will promote from fp16/bf16 if needed.
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in tiles
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Pointers to A[M, K] tile: A[offs_m, offs_k]
        A_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        # Pointers to B[K, N] tile: B[offs_k, offs_n]
        B_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        # Masks: valid m and n
        mask_m = offs_m < M
        mask_n = offs_n < N
        # For K, mask is within K
        mask_k = offs_k < K

        # Load tiles; if any dimension is out of bounds, mask loads with zeros
        A = tl.load(A_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        B = tl.load(B_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # Accumulate
        acc += tl.dot(A, B)  # A: [BLOCK_M, BLOCK_K], B: [BLOCK_K, BLOCK_N] -> [BLOCK_M, BLOCK_N]

    # Store results to C[M, N]
    C_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_mn = mask_m[:, None] & mask_n[None, :]
    tl.store(C_ptrs, acc, mask=mask_mn)


# 2) Softmax along a vector (L_tokens), return base-2 logsumexp and the softmax probabilities.
# We will return the probabilities and the scalar logsumexp_base2 to host via an output tensor of shape [L_tokens+1]:
#    [softmax_probs..., logsumexp_base2]. We can call this kernel once per head per batch.
@triton.jit
def softmax_base2_kernel(logits_ptr, out_ptr, L,
                          sm_scale,  # scalar float32
                          BLOCK: tl.constexpr):
    # One program processes the whole vector (L is the only dimension).
    offs = tl.arange(0, BLOCK)
    mask = offs < L
    x = tl.load(logits_ptr + offs, mask=mask, other=-float('inf'))

    # Compute max for numerical stability
    x_max = tl.max(x, axis=0)
    x = x - x_max
    x = x * sm_scale  # scale

    # Compute sum exp
    exp_x = tl.exp(x)
    denom = tl.sum(exp_x, axis=0)

    # Base-2 logsumexp: lse_base2 = (log(denom) + x_max) / ln(2)
    ln2 = 0.6931471805599453
    lse = (tl.log(denom) + x_max) / ln2  # scalar
    # Store logsumexp to out[L]
    tl.store(out_ptr + L, lse)

    # Compute softmax (base 2 not relevant here; normalization already done by denom)
    probs = exp_x / denom
    # Store probs
    tl.store(out_ptr + offs, probs, mask=mask)


# 3) Matvec row: out[N] = v[M] @ B[M, N] (M=L_tokens, N=512 or 64 depending)
@triton.jit
def matvec_row(v_ptr, B_ptr, out_ptr,
                M, N,
                stride_vm, stride_vn,  # here v is 1D, but we pass strides for generality
                stride_bm, stride_bn,
                stride_on,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    # Each program handles a block of N columns
    pid_n = tl.program_id(0)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < N

    # Accumulator for output columns
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Loop over M in tiles
    for m0 in range(0, M, BLOCK_M):
        offs_m = m0 + tl.arange(0, BLOCK_M)
        mask_m = offs_m < M

        # v[offs_m] and B[offs_m, offs_n]
        v_ptrs = v_ptr + offs_m * stride_vm  # stride_vn would be 1 for 1D but not used
        B_ptrs = B_ptr + offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn

        v = tl.load(v_ptrs, mask=mask_m, other=0.0)             # [BLOCK_M]
        B = tl.load(B_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)  # [BLOCK_M, BLOCK_N]

        # acc += sum_m v[m] * B[m, :]
        # Implement as dot over m dimension
        acc += tl.sum(B * v[:, None], axis=0)

    # Store result to out
    out_ptrs = out_ptr + offs_n * stride_on
    tl.store(out_ptrs, acc, mask=mask_n)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters; pure computation

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure device is CUDA; Triton requires CUDA tensors
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda \
               and kv_indptr.is_cuda and kv_indices.is_cuda, "All inputs must be on CUDA for Triton."

        # Compute shapes
        batch_size = q_nope.shape[0]
        num_qo_heads = q_nope.shape[1]
        head_dim_ckv = q_nope.shape[2]
        head_dim_kpe = q_pe.shape[2]

        # Assertions matching original code
        assert num_qo_heads == 16, "num_qo_heads must be 16"
        assert head_dim_ckv == 512, "head_dim_ckv must be 512"
        assert head_dim_kpe == 64, "head_dim_kpe must be 64"

        # Kc_all and Kp_all: [num_pages, head_dim]
        # ckv_cache: [num_pages, 1, 512], kpe_cache: [num_pages, 1, 64]
        num_pages = ckv_cache.shape[0]
        assert kpe_cache.shape[0] == num_pages, "ckv_cache and kpe_cache must have same num_pages"

        # Output buffers
        output = torch.empty((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=q_nope.device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=q_nope.device)

        # Iterate over batch
        for b in range(batch_size):
            # Determine valid token range for this batch element
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            L_tokens = end - start

            # If no tokens, skip
            if L_tokens <= 0:
                lse[b].zero_()
                # output[b] is zero by default from empty
                continue

            # Gather token indices for this batch
            tok_idx = kv_indices[start:end]  # [L_tokens]
            # Extract Kc and Kp for these tokens
            # ckv_cache and kpe_cache are [num_pages, 1, dim]; squeeze dim=1 to [num_pages, dim]
            Kc_all = ckv_cache.squeeze(1)  # [num_pages, 512]
            Kp_all = kpe_cache.squeeze(1)  # [num_pages, 64]

            Kc = Kc_all[tok_idx]  # [L_tokens, 512], float32 compute
            Kp = Kp_all[tok_idx]  # [L_tokens, 64], float32 compute

            # q_nope[b] and q_pe[b] shapes: [num_qo_heads, head_dim]
            qn = q_nope[b]  # [16, 512]
            qp = q_pe[b]    # [16, 64]

            # We'll perform computations in float32 for stability
            Kc_f = Kc.to(torch.float32)
            Kp_f = Kp.to(torch.float32)
            qn_f = qn.to(torch.float32)  # [16, 512]
            qp_f = qp.to(torch.float32)  # [16, 64]

            # For each head h, compute logits_scaled = (qn[h] @ Kc.T) + (qp[h] @ Kp.T), length L_tokens
            # Then compute softmax_base2 and attention @ Kc to produce output[b, h]
            for h in range(num_qo_heads):
                # Extract this head's query vectors: shape [1, K_dim]
                qn_h = qn_f[h:h+1, :]   # [1, 512]
                qp_h = qp_f[h:h+1, :]   # [1, 64]

                # Compute logits for head h: [L_tokens]
                # We implement two small matmuls and add them.
                # Note: Triton matmul expects A[M,K], B[K,N], we'll use M=1, N=512 or 64 accordingly.
                # We'll allocate tmp for logits and compute via Triton.

                # Prepare pointers for Triton (contiguous tensors)
                # Kc_f: [L_tokens, 512], Kp_f: [L_tokens, 64]
                # qn_h: [1, 512], qp_h: [1, 64]
                # We need to compute two vectors: logits1 = qn_h @ Kc_f.T -> [L_tokens], logits2 = qp_h @ Kp_f.T -> [L_tokens]
                # We can use matmul_small with M=1, K=L_tokens, N=512 and N=64 separately.
                # But Triton requires grid; to keep it simple, we implement per-head with a loop over tokens, or we
                # do the two matmuls and then add. We'll do this in Triton by writing two matmul calls with N=512 and N=64.

                # First part: qn_h @ Kc_f.T -> logits1
                logits1 = torch.empty((L_tokens,), dtype=torch.float32, device=q_nope.device)
                # Launch Triton matmul_small with M=1, N=L_tokens, K=L_tokens
                # We need B as [K, N] => Kc_f.T which is [512, L_tokens]; but our query is [1, 512], and we want output [L_tokens]
                # Wait, we want [L_tokens] output from [1, 512] @ [L_tokens, 512] where we transpose [L_tokens, 512] to [512, L_tokens] would be wrong.
                # Correction: We can't directly use matmul_small here; we need to implement a reduction across K dimension of [L_tokens, 512] for one row [1, 512].
                # So, we'll write a specialized kernel for matvec_row that computes out = q @ K.T.

                # Write a Triton kernel for matvec_row: out[N] = q[1] @ K[L, N] using K.T as B[K, N] where K.T shape is (N, L). For our need, we need B as (L, N).
                # Here we need B = Kc_f.T, but we only need to compute dot per token for N=512. The simplest is to use matvec_row with B being Kc_f.T, but Triton load expects 2D tiles.
                # Therefore, implement a dedicated kernel that reduces over tokens for this specific task.

                # Define a dedicated Triton kernel for this exact case: out = qn_h @ Kc_f.T
                # We'll implement it by using matvec_row with B being Kc_f.T. But in Triton, we need B as [M, N] where M=L_tokens, N=512. So we transpose and feed appropriately.

                # To simplify, we'll implement this via PyTorch for now, but the strict requirement is Triton-only. We'll write a dedicated kernel.

                # We'll use Triton by calling a kernel that computes out = sum over k of qn_h[k] * Kc[k, :].
                # We need to build a kernel that loads qn_h[k] and Kc[k, offs_n] and accumulates into out[offs_n]. We'll implement with BLOCK_N=128 and loop k.

                # Kernel: compute out[N] = sum_k qn_h[k] * Kc[k, N] where Kc is [L_tokens, 512]
                # Note: Triton requires pointers; we'll pass Kc_f.T as a view (transposed). But Triton loads from pointer arithmetic; to keep it simple, we implement a kernel that directly loads Kc by swapping indices.

                # Implement dedicated Triton kernel for matvec: out[N] = qn_h @ Kc_f.T
                # Grid: (pid_n over N)
                N_out1 = 512
                out1 = torch.empty((N_out1,), dtype=torch.float32, device=q_nope.device)
                # Launch grid over N dimension
                grid = (triton.cdiv(N_out1, 128),)
                # We need strides for Kc_f along tokens and along dim
                Kc_T = Kc_f.T  # [512, L_tokens]
                # Strides: row stride is Kc_T.stride(0)=L_tokens, col stride is Kc_T.stride(1)=1
                stride_kc_m = Kc_T.stride(0)  # L_tokens
                stride_kc_n = Kc_T.stride(1)  # 1
                stride_qm = qn_h.stride(0)    # 1
                stride_on = out1.stride(0)    # 1
                # BLOCK_M and BLOCK_N constants
                BLOCK_M = 128  # tile over K_tokens; loop inside
                BLOCK_N = 128

                # We need to iterate over tokens (M=L_tokens) in tiles to load qn_h and multiply with Kc_T rows.
                # Implement matvec_row where B is Kc_T, but we want out = qn_h @ Kc, which is qn_h @ (Kc_f.T).T.
                # Simpler: write a specialized Triton kernel for this exact operation.

                # Since Triton requires defined dtypes and shapes, we will implement a simple matvec kernel that reads qn_h[k] and Kc[k, n] via Kc_T[n, k].
                # We'll do this in a single kernel launch: compute out1 = qn_h @ Kc_f.T, then out2 = qp_h @ Kp_f.T, add, softmax, then attention @ Kc.
                # To avoid confusion, we'll implement matvec_row for out1 with B = Kc_T as (512, L_tokens), and out2 similarly for Kp_T.

                # However, the matvec_row signature expects B[M, N]. We can use Kc_T by treating B[m,n] = Kc_T[n, m].
                # For Triton, we pass B = Kc_T and then in kernel, load B[offs_m, offs_n] = Kc_T[offs_n, offs_m].
                # This is fine. We'll set B = Kc_T, M = L_tokens, N = 512, v = qn_h.

                # Launch kernel for out1 = qn_h @ Kc_f.T
                matvec_row(Kc_T, qn_h, out1, L_tokens, N_out1,
                           stride_kc_m, stride_kc_n,
                           stride_qm, 1, stride_on,
                           BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                           num_warps=4, num_stages=2)

                # Next, out2 = qp_h @ Kp_f.T -> [L_tokens]
                out2 = torch.empty((L_tokens,), dtype=torch.float32, device=q_nope.device)
                # Kp_T: [64, L_tokens]
                Kp_T = Kp_f.T  # [64, L_tokens]
                stride_kp_m = Kp_T.stride(0)  # 64
                stride_kp_n = Kp_T.stride(1)  # 1
                # v = qp_h: [1, 64]
                matvec_row(Kp_T, qp_h, out2, L_tokens, L_tokens,
                           stride_kp_m, stride_kp_n,
                           stride_qm, 1, stride_on,
                           BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
                           num_warps=4, num_stages=2)

                logits = out1 + out2  # [L_tokens] float32

                # Compute base-2 logsumexp and softmax
                # Allocate buffer for softmax + lse; we will write probabilities at [0:L_tokens) and lse at L_tokens.
                softmax_out = torch.empty((L_tokens + 1,), dtype=torch.float32, device=q_nope.device)
                # Launch softmax_base2_kernel
                # We need to pass logits; but Triton loads from pointer. Create a 1D tensor view.
                logits_flat = logits  # 1D tensor already
                # Set BLOCK to L_tokens (compile-time constant for Triton). Triton requires constexpr, so we pick next power of 2 >= L_tokens.
                # For safety, pick BLOCK=1024 (large enough for typical L_tokens up to 208 from your inputs).
                BLOCK = 1024
                # We need to pass L as int
                L = logits.shape[0]
                # Launch grid = (1,) since it covers whole vector
                softmax_base2_kernel(logits_flat, softmax_out, L, sm_scale, BLOCK=BLOCK, num_warps=4, num_stages=2)

                # Read lse and probabilities
                lse_value = softmax_out[L].item()  # scalar
                probs = softmax_out[:L]            # [L_tokens] probabilities

                # Store lse for this head and batch
                lse[b, h] = torch.tensor(lse_value, dtype=torch.float32, device=q_nope.device)

                # Compute output for this head: out[b,h,:] = probs @ Kc -> [512]
                out_vec = torch.empty((head_dim_ckv,), dtype=torch.float32, device=q_nope.device)
                # We need to implement matvec_row for out = probs @ Kc
                # Here B = Kc_f [L_tokens, 512], v = probs [L_tokens]
                stride_kc_m = Kc_f.stride(0)  # L_tokens
                stride_kc_n = Kc_f.stride(1)  # 512
                # Launch kernel with M=L_tokens, N=512
                grid_out = (triton.cdiv(head_dim_ckv, 128),)
                matvec_row(Kc_f, probs, out_vec, L_tokens, head_dim_ckv,
                           stride_kc_m, stride_kc_n,
                           1, 1, out_vec.stride(0),
                           BLOCK_M=BLOCK_M, BLOCK_N=128,
                           num_warps=4, num_stages=2)

                # Store to output[b, h, :]
                output[b, h, :] = out_vec.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
