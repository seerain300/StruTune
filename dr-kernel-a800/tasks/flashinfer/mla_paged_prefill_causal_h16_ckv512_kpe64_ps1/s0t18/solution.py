import torch

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None

# Triton kernels to be invoked from ModelNew.forward

@triton.jit
def matmul_qn_kc_vec_kernel(qn_row_ptr, Kc_ptr, S_ptr,
                            Dn: tl.constexpr,    # 512
                            KV: tl.constexpr,    # number of KV tokens
                            BLOCK_K: tl.constexpr):
    # One program computes S_vec = qn_row @ Kc.T
    # qn_row_ptr points to a [Dn] vector (float32) for a specific head
    acc = tl.zeros((Dn,), dtype=tl.float32)
    for k0 in range(0, KV, BLOCK_K):
        k_idx = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_idx < KV
        # Kc tile: [BLOCK_K, Dn]
        Kc_tile = tl.load(Kc_ptr + k_idx[:, None] * Dn + tl.arange(0, Dn), mask=mask_k[:, None], other=0.0)
        # qn_row: [Dn]
        qn_row = tl.load(qn_row_ptr + tl.arange(0, Dn))
        # Accumulate: sum over K block
        # Kc_tile shape [BLOCK_K, Dn], qn_row [Dn] -> elementwise multiply across columns then reduce over K
        # We need acc += sum_k Kc_tile[k, :] * qn_row[:], broadcast qn_row over K dimension.
        # Triton supports elementwise multiply and reduction along last dim via tl.sum across K dimension.
        prod = Kc_tile * qn_row[None, :]
        acc += tl.sum(prod, axis=0)
    # Store S_vec
    tl.store(S_ptr + tl.arange(0, Dn), acc)

@triton.jit
def matmul_qp_kp_vec_kernel(qp_row_ptr, Kp_ptr, T_ptr,
                            Dp: tl.constexpr,    # 64
                            KV: tl.constexpr,    # number of KV tokens
                            BLOCK_K: tl.constexpr):
    # One program computes T_vec = qp_row @ Kp.T
    acc = tl.zeros((Dp,), dtype=tl.float32)
    for k0 in range(0, KV, BLOCK_K):
        k_idx = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_idx < KV
        Kp_tile = tl.load(Kp_ptr + k_idx[:, None] * Dp + tl.arange(0, Dp), mask=mask_k[:, None], other=0.0)
        qp_row = tl.load(qp_row_ptr + tl.arange(0, Dp))
        prod = Kp_tile * qp_row[None, :]
        acc += tl.sum(prod, axis=0)
    tl.store(T_ptr + tl.arange(0, Dp), acc)

@triton.jit
def compute_logit_lse_and_store_kernel(
    qn_row_ptr, qp_row_ptr, Kc_ptr, Kp_ptr,
    logits_ptr, lse_ptr, attn_ptr,
    sm_scale: tl.float32,
    prefix_len: tl.int32,
    query_abs_pos: tl.int32,
    KV: tl.int32,
    Dn: tl.constexpr,   # 512
    Dp: tl.constexpr,   # 64
    BLOCK_K: tl.constexpr,
):
    # Compute S_vec and T_vec
    S_vec = tl.zeros((KV,), dtype=tl.float32)
    T_vec = tl.zeros((KV,), dtype=tl.float32)

    # Launch matmul_qn_kc_vec_kernel to get S_vec
    matmul_qn_kc_vec_kernel[(1,)](
        qn_row_ptr, Kc_ptr, S_vec,
        Dn, KV, BLOCK_K
    )

    # Launch matmul_qp_kp_vec_kernel to get T_vec
    matmul_qp_kp_vec_kernel[(1,)](
        qp_row_ptr, Kp_ptr, T_vec,
        Dp, KV, BLOCK_K
    )

    # Sum and scale
    logits = S_vec + T_vec
    logits = logits * sm_scale

    # Causal mask: keep if j > (prefix_len + i), else -inf
    # Implement mask via setting -inf for invalid positions
    for j in range(0, KV):
        cond = (j > query_abs_pos)
        # If cond False, set logits[j] to -inf
        if not cond:
            logits[j] = -float('inf')

    # Store logits
    tl.store(logits_ptr + tl.arange(0, KV), logits)

    # Compute stable softmax (one program, vectorized)
    # Use numerically stable softmax: subtract max, exp, sum, divide
    m = tl.max(logits, axis=0)
    logits_shift = logits - m
    expv = tl.exp(logits_shift)
    sum_exp = tl.sum(expv, axis=0)
    attn = expv / sum_exp

    # Store attn
    tl.store(attn_ptr + tl.arange(0, KV), attn)

    # Compute lse = logsumexp(logits_shift) / ln(2)
    lse_val = tl.log(sum_exp)  # natural log
    lse_val = lse_val / 0.6931471805599453  # 1 / ln(2)
    # Write lse to lse_ptr (single scalar per head)
    tl.store(lse_ptr, lse_val)

@triton.jit
def compute_out_row_kernel(attn_ptr, Kc_ptr, out_ptr,
                           Dn: tl.constexpr,  # 512
                           KV: tl.constexpr,  # number of KV tokens
                           BLOCK_K: tl.constexpr):
    # Compute out_vec = attn_vec @ Kc, where attn_vec[KV], Kc[KV, Dn]
    acc = tl.zeros((Dn,), dtype=tl.float32)
    for k0 in range(0, KV, BLOCK_K):
        k_idx = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_idx < KV
        attn_vec = tl.load(attn_ptr + k_idx, mask=mask_k, other=0.0)
        Kc_tile = tl.load(Kc_ptr + k_idx[:, None] * Dn + tl.arange(0, Dn), mask=mask_k[:, None], other=0.0)
        acc += tl.sum(Kc_tile * attn_vec[:, None], axis=0)
    tl.store(out_ptr + tl.arange(0, Dn), acc)

# Host-side ModelNew.forward: must launch Triton kernels only
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure on CUDA and dtype
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "All tensors must be on CUDA"
        device = q_nope.device

        # Convert caches to float32 once (host-side)
        Kc_all = ckv_cache.to(torch.float32)  # [M, 512]
        Kp_all = kpe_cache.to(torch.float32)  # [M, 64]

        total_q = int(qo_indptr[-1].item())
        num_qo_heads = 16
        head_dim_ckv = 512
        head_dim_kpe = 64

        # Allocate outputs
        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)  # will cast to bfloat16 at the end
        lse = torch.full((total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

        B = qo_indptr.numel() - 1

        for b in range(B):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            q_len = q_end - q_start
            kv_len = kv_end - kv_start

            # Gather token indices and caches
            tok_idx = kv_indices[kv_start:kv_end].to(device)  # int32 indices
            Kc = Kc_all[tok_idx]  # [kv_len, 512]
            Kp = Kp_all[tok_idx]  # [kv_len, 64]

            # For each query i
            for i in range(q_len):
                # prefix_len = total kv tokens for this batch minus current q_len
                prefix_len = kv_len - q_len
                query_abs_pos = prefix_len + i

                # Loop over heads
                for h in range(num_qo_heads):
                    # Load qn_row and qp_row (float32)
                    qn_row = q_nope[q_start + i, h, :].to(torch.float32).to(device)  # [512]
                    qp_row = q_pe[q_start + i, h, :].to(torch.float32).to(device)   # [64]

                    # Allocate intermediate buffers
                    logits = torch.empty((kv_len,), dtype=torch.float32, device=device)
                    attn = torch.empty((kv_len,), dtype=torch.float32, device=device)
                    lse_val = torch.empty((), dtype=torch.float32, device=device)  # scalar per head

                    # Launch compute_logit_lse_and_store_kernel
                    compute_logit_lse_and_store_kernel[(1,)](
                        qn_row, qp_row, Kc, Kp,
                        logits, lse_val, attn,
                        sm_scale,
                        prefix_len, query_abs_pos,
                        kv_len,
                        head_dim_ckv, head_dim_kpe,
                        BLOCK_K=128,
                        num_warps=4
                    )

                    # Now compute out = attn @ Kc for this head
                    out_row = torch.empty((head_dim_ckv,), dtype=torch.float32, device=device)
                    compute_out_row_kernel[(1,)](
                        attn, Kc, out_row,
                        head_dim_ckv, kv_len,
                        BLOCK_K=128,
                        num_warps=4
                    )

                    # Store output[q_start + i, h, :] = out_row
                    # We can place output row directly using indexing if available, but using torch for store is fine here.
                    # However, to keep Triton-only, we'll use torch assignment (it's not computation).
                    output[q_start + i, h, :] = out_row

                    # Store lse[q_start + i, h] = lse_val
                    lse[q_start + i, h] = lse_val.item()  # Triton stores scalar, .item() is fine here

        # Cast output to bfloat16 as original returns
        output_bf16 = output.to(torch.bfloat16)
        # Return output and lse (lse is float32 as in original)
        return output_bf16, lse


def run(*args):
    return ModelNew()(*args)
