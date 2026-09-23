import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Kernel 1: Compute logits = (qn @ Kc.T) + (qp @ Kp.T) for a single head h
# Inputs:
#   qn_ptr: [Hc] float32 (per-head slice of q_nope[b])
#   qp_ptr: [Hp] float32 (per-head slice of q_pe[b])
#   Kc_ptr: [L, Hc] float32 (gathered tokens from ckv_cache)
#   Kp_ptr: [L, Hp] float32 (gathered tokens from kpe_cache)
#   out_ptr: [L] float32 (logits buffer)
#   sm_scale: float32
#   L: int (number of tokens for this batch)
# Launch: one program per (b, h) iteration handled on host; we set grid=(1,) and pass pointers accordingly.
@triton.jit
def matmul_add_row_kernel(
    qn_ptr, qp_ptr, Kc_ptr, Kp_ptr, out_ptr,
    sm_scale: tl.constexpr,
    L: tl.constexpr,
    Kc_stride0, Kc_stride1,
    Kp_stride0, Kp_stride1,
    out_stride,
    BLOCK_K: tl.constexpr,
):
    # One program computes the entire logits vector for this head
    # We loop over tokens in chunks of BLOCK_K and accumulate
    offs = tl.arange(0, BLOCK_K)
    acc = tl.zeros([BLOCK_K], dtype=tl.float32)

    # Loop over tokens
    for k in range(0, L, BLOCK_K):
        k_idx = k + offs
        mask = k_idx < L
        # Load qn, qp (scalars for this head)
        # qn_ptr and qp_ptr are 1D vectors of length Hc/Hp, we index with 0
        qn_val = tl.load(qn_ptr + 0)
        qp_val = tl.load(qp_ptr + 0)
        # Load Kc and Kp rows corresponding to token indices
        Kc_row = tl.load(Kc_ptr + k_idx * Kc_stride0 + 0 * Kc_stride1, mask=mask, other=0.0)  # other dimension is Hc; we just reduce over tokens
        Kp_row = tl.load(Kp_ptr + k_idx * Kp_stride0 + 0 * Kp_stride1, mask=mask, other=0.0)  # other dimension is Hp

        # Accumulate qn @ Kc.T and qp @ Kp.T into acc
        # qn_val is scalar; Kc_row is [BLOCK_K] (masked). We treat this as outer product accumulation for a single head.
        acc += qn_val * Kc_row
        acc += qp_val * Kp_row

    # Apply scaling
    acc *= sm_scale
    # Store logits
    tl.store(out_ptr + k_idx * out_stride, acc, mask=k_idx < L)


# Kernel 2: Compute row-wise logsumexp and softmax probabilities (one program per row)
# Inputs:
#   x_ptr: [L] float32 (logits_scaled)
#   L: int (length of row)
#   lse_ptr: [1] float32 (per-row logsumexp / log(2))
#   attn_ptr: [L] float32 (per-row softmax probabilities)
# Launch: grid = (batch_size * num_qo_heads,)
@triton.jit
def softmax_logsumexp_row_kernel(x_ptr, lse_ptr, attn_ptr, L: tl.constexpr):
    row_id = tl.program_id(0)  # not used here; 1 program handles entire row
    # Compute max for numerical stability
    m = -float("inf")
    for i in range(0, L):
        m = tl.maximum(m, tl.load(x_ptr + i))
    # Compute sum of exp(x - m)
    sum_exp = 0.0
    for i in range(0, L):
        sum_exp += tl.exp(tl.load(x_ptr + i) - m)
    # lse = log(sum_exp) / log(2)
    lse = tl.log(sum_exp) / 1.4426950408889634  # 1.0 / ln(2)
    tl.store(lse_ptr, lse)
    # Write attn (softmax probabilities)
    # Note: We don't actually need attn on host; we keep this for completeness if ever used.
    for i in range(0, L):
        p = tl.exp(tl.load(x_ptr + i) - m) / sum_exp
        tl.store(attn_ptr + i, p)


# Kernel 3: Compute out_row = softmax(logits_scaled) @ Kc for a single head (GEMV-like, Triton)
# Inputs:
#   logits_ptr: [L] float32 (scaled logits for this head)
#   Kc_ptr: [L, Hc] float32
#   out_row_ptr: [Hc] float32
# Launch: grid over output columns in chunks of BLOCK_N
@triton.jit
def matvec_row_kernel(
    logits_ptr, Kc_ptr, out_row_ptr,
    Hc: tl.constexpr,  # output dimension
    L: tl.constexpr,   # number of tokens
    Kc_stride0, Kc_stride1,
    out_stride,
    BLOCK_N: tl.constexpr,
):
    col_block = tl.program_id(0)
    offs = col_block * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < Hc
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Accumulate over tokens
    for i in range(0, L):
        attn_i = tl.exp(tl.load(logits_ptr + i))  # softmax probability for token i
        Kc_col = tl.load(Kc_ptr + i * Kc_stride0 + offs * Kc_stride1, mask=mask, other=0.0)
        acc += attn_i * Kc_col

    tl.store(out_row_ptr + offs * out_stride, acc, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA and contiguous
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All tensors must be CUDA for Triton."
        device = q_nope.device
        # q_nope: [B, 16, 512], q_pe: [B, 16, 64], ckv_cache: [P, 1, 512], kpe_cache: [P, 1, 64]
        B = q_nope.shape[0]
        num_qo_heads = q_nope.shape[1]
        Hc = q_nope.shape[2]
        Hp = q_pe.shape[2]
        # Derive tok_idx per batch b: L_tokens = kv_indptr[b+1] - kv_indptr[b]
        len_indptr = kv_indptr.shape[0]
        assert len_indptr == B + 1, "kv_indptr length must be batch_size + 1."
        # Compute tok_idx and gather Kc/Kp
        Kc_all = ckv_cache.squeeze(1).contiguous().to(torch.float32)  # [P, Hc]
        Kp_all = kpe_cache.squeeze(1).contiguous().to(torch.float32)  # [P, Hp]

        output = torch.empty((B, num_qo_heads, Hc), dtype=torch.float32, device=device)  # we will return bfloat16
        lse = torch.empty((B, num_qo_heads), dtype=torch.float32, device=device)

        # Loop over batch and heads
        for b in range(B):
            # Determine token range for this batch element
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            if start >= end:
                # No tokens for this batch element
                # output[b] should be zeros; lse[b] can be zeros
                output[b] = torch.zeros((num_qo_heads, Hc), dtype=torch.float32, device=device)
                for h in range(num_qo_heads):
                    lse[b, h] = 0.0
                continue

            tok_idx = kv_indices[start:end].to(torch.int32)  # [L]
            L_tokens = tok_idx.numel()

            # Gather Kc and Kp for these tokens
            Kc = Kc_all[tok_idx]  # [L, Hc], float32
            Kp = Kp_all[tok_idx]  # [L, Hp], float32

            # For each head h, compute logits, lse, and output
            for h in range(num_qo_heads):
                # Extract qn and qp for this head (they are contiguous along last dim)
                qn = q_nope[b, h, :].contiguous().to(torch.float32)  # [Hc]
                qp = q_pe[b, h, :].contiguous().to(torch.float32)   # [Hp]

                # Buffer for logits (length L_tokens)
                logits = torch.empty((L_tokens,), dtype=torch.float32, device=device)

                # Kernel 1: compute logits = (qn @ Kc.T) + (qp @ Kp.T)
                # Choose a reasonable BLOCK_K (e.g., 64 or 128). We'll loop over L_tokens in chunks of 128.
                BLOCK_K = 128
                matmul_add_row_kernel[(1,)](
                    qn, qp, Kc, Kp, logits,
                    sm_scale,
                    L_tokens,
                    Kc.stride(0), Kc.stride(1),
                    Kp.stride(0), Kp.stride(1),
                    1,  # out_stride is 1 since logits is 1D
                    BLOCK_K=BLOCK_K,
                    num_warps=2, num_stages=2
                )

                # Scale logits
                logits_scaled = logits * sm_scale

                # Kernel 2: compute row-wise logsumexp and softmax (lse and attn)
                lse_row = torch.empty((1,), dtype=torch.float32, device=device)
                attn = torch.empty((L_tokens,), dtype=torch.float32, device=device)
                softmax_logsumexp_row_kernel[(1,)](
                    logits_scaled, lse_row, attn, L_tokens
                )
                lse[b, h] = lse_row[0]

                # Kernel 3: compute out_row = softmax(logits_scaled) @ Kc for this head
                out_row = torch.empty((Hc,), dtype=torch.float32, device=device)
                BLOCK_N = 128
                grid = (triton.cdiv(Hc, BLOCK_N),)
                matvec_row_kernel[grid](
                    logits_scaled, Kc, out_row,
                    Hc, L_tokens,
                    Kc.stride(0), Kc.stride(1),
                    1,
                    BLOCK_N=BLOCK_N,
                    num_warps=2, num_stages=2
                )
                # Store to output
                output[b, h, :] = out_row

        # Return output in bfloat16 to match original q_nope dtype, lse in float32
        return output.to(torch.bfloat16), lse


# Optional: get_inputs to generate CUDA tensors (kept simple; Triton kernels require CUDA)
def get_inputs():
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16, device='cuda')
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16, device='cuda')
    ckv_cache = torch.randn([989669, 1, 512], dtype=torch.bfloat16, device='cuda')
    kpe_cache = torch.randn([989669, 1, 64], dtype=torch.bfloat16, device='cuda')
    _n = 1; _t = 8
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32).to(device='cuda')
    kv_indices = torch.randint(0, 989669, [8], dtype=torch.int32).to(device='cuda')
    sm_scale = 1.0
    return [q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale]


def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6):
    return ModelNew().forward(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6)


def run(*args):
    return ModelNew()(*args)
