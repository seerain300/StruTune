import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: matvec C = A dot B, where
# - A is a 1D vector of length K (we pass a [K] pointer)
# - B is a [M, K] matrix (rows = M, cols = K)
# - C is a 1D vector of length M
@triton.jit
def matvec_row_kernel(A_ptr, B_ptr, C_ptr, M, K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    """
    Compute C[i] = sum_k A[k] * B[i, k] for i in 0..M-1, with A of length K and B of shape [M, K].
    We process M in chunks of BLOCK_M and K in chunks of BLOCK_K using nested loops and masks.
    """
    pid_m = tl.program_id(0)
    m0 = pid_m * BLOCK_M
    # Accumulator for BLOCK_M outputs
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Reduce over K in chunks
    for k0 in tl.static_range(0, K, BLOCK_K):
        # Loop over rows within this program's chunk
        for dm in tl.static_range(0, BLOCK_M):
            i = m0 + dm
            # Load A chunk
            offs_k = k0 + tl.arange(0, BLOCK_K)
            mask_k = (offs_k < K)
            A_chunk = tl.load(A_ptr + offs_k, mask=mask_k, other=0.0)  # [BLOCK_K]
            # Load B[i, :] chunk: pointer arithmetic B[i, k] = B_ptr + i*K + k
            B_row_ptrs = B_ptr + (i * K) + (k0 + tl.arange(0, BLOCK_K))
            mask_k2 = (i < M) & mask_k
            B_chunk = tl.load(B_row_ptrs, mask=mask_k2, other=0.0)
            # Accumulate: acc[dm] += sum over k of A_chunk * B_chunk
            acc[dm] += tl.sum(B_chunk * A_chunk, axis=0)

    # Store results for this chunk
    offs_m = m0 + tl.arange(0, BLOCK_M)
    mask_m = offs_m < M
    tl.store(C_ptr + offs_m, acc, mask=mask_m)


# Triton kernel: softmax + logsumexp over a vector (scaled logits), output attn and lse
@triton.jit
def softmax_lse_kernel(x_ptr, attn_ptr, lse_ptr, M: tl.constexpr, sm_scale: tl.float32):
    """
    x_ptr: *float32, vector of length M (logits_scaled)
    attn_ptr: *float32, vector of length M (attention probs)
    lse_ptr: *float32, scalar (lse) written at position 0
    We compute lse = logsumexp(x) / ln(2) and attn = softmax(x) using tiled loops (no tl.arange with runtime M).
    """
    # Compute max
    max_val = -float("inf")
    for dm in tl.static_range(0, M):
        val = tl.load(x_ptr + dm)
        max_val = tl.maximum(max_val, val)
    # Compute sum of exp
    sum_exp = 0.0
    for dm in tl.static_range(0, M):
        val = tl.load(x_ptr + dm)
        sum_exp += tl.exp(val - max_val)
    # lse
    lse_val = max_val + tl.log(sum_exp)  # logsumexp
    lse_val = lse_val / 3.46573590309  # 1 / ln(2)
    tl.store(lse_ptr, lse_val)

    # Compute attn
    for dm in tl.static_range(0, M):
        val = tl.load(x_ptr + dm)
        attn_val = tl.exp(val - max_val) / sum_exp
        tl.store(attn_ptr + dm, attn_val)


# Triton kernel: out = attn dot Kc, i.e., C[h, :] where C[h, d] = sum_i attn[i] * Kc[i, d]
@triton.jit
def matvec_out_kernel(attn_ptr, Kc_ptr, out_ptr, M, Kc_dim: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr):
    """
    Compute out[d] = sum_i attn[i] * Kc[i, d] for d in 0..Kc_dim-1.
    attn is [M], Kc is [M, Kc_dim], out is [Kc_dim].
    We process Kc_dim in chunks of BLOCK_K and reduce over M in chunks of BLOCK_M.
    """
    pid_d = tl.program_id(0)
    d0 = pid_d * BLOCK_K
    acc = tl.zeros([BLOCK_K], dtype=tl.float32)

    for m0 in tl.static_range(0, M, BLOCK_M):
        for dm in tl.static_range(0, BLOCK_M):
            i = m0 + dm
            # Load attn[i]
            attn_i = tl.load(attn_ptr + i)
            # Load Kc[i, d:d+BLOCK_K]
            offs_d = d0 + tl.arange(0, BLOCK_K)
            mask_d = (offs_d < Kc_dim)
            Kc_ptrs = Kc_ptr + (i * Kc_dim) + offs_d
            Kc_chunk = tl.load(Kc_ptrs, mask=mask_d, other=0.0)
            # Accumulate
            acc += Kc_chunk * attn_i

    offs_d = d0 + tl.arange(0, BLOCK_K)
    mask_d = offs_d < Kc_dim
    tl.store(out_ptr + offs_d, acc, mask=mask_d)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # constants expected by original code
        self.head_dim_ckv = 512
        self.num_qo_heads = 16
        self.head_dim_kpe = 64
        self.page_size = 1

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Device and dtype handling
        device = q_nope.device
        # Ensure inputs are contiguous and in float32 for kernels
        q_nope_f32 = q_nope.to(torch.float32).contiguous()
        q_pe_f32 = q_pe.to(torch.float32).contiguous()
        # K-all are [N, 1, ...], squeeze dim=1 for gather
        Kc_all = ckv_cache.squeeze(1).contiguous().to(torch.float32)
        Kp_all = kpe_cache.squeeze(1).contiguous().to(torch.float32)
        kv_indptr = kv_indptr.contiguous()
        kv_indices = kv_indices.contiguous()

        batch_size = q_nope_f32.shape[0]
        H = self.num_qo_heads

        # Output buffers (float32 for compute, cast to bfloat16 at the end)
        output = torch.empty((batch_size, H, self.head_dim_ckv), dtype=torch.float32, device=device)
        lse = torch.empty((batch_size, H), dtype=torch.float32, device=device)

        # Constants for kernels
        Kc_dim = self.head_dim_ckv  # 512
        Kp_dim = self.head_dim_kpe  # 64

        # Tile sizes (can be tuned)
        BLOCK_M = 64
        BLOCK_K = 128

        # Compute per batch
        for b in range(batch_size):
            # Determine token range
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            if start >= end:
                # No valid tokens for this batch, output zeros
                output[b].zero_()
                lse[b].zero_()
                continue

            tokens = kv_indices[start:end]  # [M]
            M = tokens.numel()

            # Gather Kc and Kp rows: [M, Kc_dim] and [M, Kp_dim]
            Kc = Kc_all[tokens]  # [M, 512]
            Kp = Kp_all[tokens]  # [M, 64]

            # For each head h
            for h in range(H):
                # Load qn and qp
                qn = q_nope_f32[b, h, :]  # [512]
                qp = q_pe_f32[b, h, :]    # [64]

                # Compute logits_qn = qn @ Kc.T -> [M]
                logits_qn = torch.empty((M,), dtype=torch.float32, device=device)
                grid_m = (triton.cdiv(M, BLOCK_M),)
                matvec_row_kernel[grid_m](qn, Kc, logits_qn, M, Kc_dim, BLOCK_M, BLOCK_K)

                # Compute logits_qp = qp @ Kp.T -> [M]
                logits_qp = torch.empty((M,), dtype=torch.float32, device=device)
                grid_m2 = (triton.cdiv(M, BLOCK_M),)
                matvec_row_kernel[grid_m2](qp, Kp, logits_qp, M, Kp_dim, BLOCK_M, BLOCK_K)

                # Combine and scale
                logits_scaled = (logits_qn + logits_qp) * sm_scale  # [M], contiguous

                # Compute lse and attn in Triton (vector ops via kernels)
                attn = torch.empty((M,), dtype=torch.float32, device=device)
                lse_vec = torch.empty((1,), dtype=torch.float32, device=device)  # scalar buffer
                grid_lse = (1,)
                softmax_lse_kernel[grid_lse](logits_scaled, attn, lse_vec, M, sm_scale)
                lse_val = lse_vec[0]  # scalar float32
                lse[b, h] = lse_val  # store per head

                # Compute output for this head using attn @ Kc (matvec)
                out_vec = torch.empty((Kc_dim,), dtype=torch.float32, device=device)
                grid_out = (triton.cdiv(Kc_dim, BLOCK_K),)
                matvec_out_kernel[grid_out](attn, Kc, out_vec, M, Kc_dim, BLOCK_M, BLOCK_K)

                # Store results
                output[b, h, :] = out_vec

        # Cast output to bfloat16 as original code returns bfloat16 output
        output = output.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
