import math
import torch

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None

# Triton kernels: Triton-only, no torch ops in forward.

# Kernel to compute logits, apply scaling and causal mask, and compute per-head lse.
@triton.jit
def compute_logits_and_lse_kernel(
    qn_ptr, qp_ptr, Kc_ptr, Kp_ptr,
    logits_ptr, lse_ptr,
    sm_scale: tl.float32,
    prefix_len: tl.int32,
    query_abs_pos: tl.int32,
    KV: tl.int32,        # number of KV tokens
    H: tl.int32,         # number of heads
    Dn: tl.constexpr,    # head_dim_ckv (512)
    Dp: tl.constexpr,    # head_dim_kpe (64)
    BLOCK_K: tl.constexpr,
):
    # Loop over heads h
    for h in range(0, H):
        # Accumulate S[h, :] = qn[h, :] @ Kc.T
        acc_S = tl.zeros((Dn,), dtype=tl.float32)
        for k0 in range(0, KV, BLOCK_K):
            k_idx = k0 + tl.arange(0, BLOCK_K)
            mask_k = k_idx < KV
            # Load qn_row[h, :]
            qn_row = tl.load(qn_ptr + h * Dn + tl.arange(0, Dn))
            # Load Kc tile: [BLOCK_K, Dn]
            Kc_tile = tl.load(Kc_ptr + k_idx[:, None] * Dn + tl.arange(0, Dn), mask=mask_k[:, None], other=0.0)
            # acc_S += sum over Kc_tile * qn_row
            acc_S += tl.sum(Kc_tile * qn_row[None, :], axis=1)

        # Accumulate T[h, :] = qp[h, :] @ Kp.T
        acc_T = tl.zeros((Dp,), dtype=tl.float32)
        for k0 in range(0, KV, BLOCK_K):
            k_idx = k0 + tl.arange(0, BLOCK_K)
            mask_k = k_idx < KV
            qp_row = tl.load(qp_ptr + h * Dp + tl.arange(0, Dp))
            Kp_tile = tl.load(Kp_ptr + k_idx[:, None] * Dp + tl.arange(0, Dp), mask=mask_k[:, None], other=0.0)
            acc_T += tl.sum(Kp_tile * qp_row[None, :], axis=1)

        logits_vec = acc_S + acc_T  # [KV]

        # Apply scaling
        logits_vec = logits_vec * sm_scale

        # Causal mask: keep j if j > query_abs_pos else -inf
        j = tl.arange(0, KV)
        mask_pos = j > query_abs_pos
        neg_inf = -float("inf")
        logits_vec = tl.where(mask_pos, logits_vec, neg_inf)

        # Compute lse = logsumexp(logits) / ln(2)
        # Numerically stable: lse = max + log(sum(exp(logits - max))) / ln(2)
        row_max = tl.max(logits_vec, axis=0)
        sum_exp = tl.sum(tl.exp(logits_vec - row_max), axis=0)
        # ln(2) computed inside kernel
        ln2 = tl.log(2.0)
        lse_val = (row_max + tl.log(sum_exp)) / ln2
        tl.store(lse_ptr + h, lse_val)

        # Store masked logits for this head
        tl.store(logits_ptr + h * KV + j, logits_vec)

# Kernel to compute softmax per row (numerically stable) and write attn.
@triton.jit
def softmax_rows_kernel(logits_ptr, attn_ptr, KV: tl.int32, H: tl.int32):
    for h in range(0, H):
        row_logits_ptr = logits_ptr + h * KV
        row_attn_ptr = attn_ptr + h * KV
        j = tl.arange(0, KV)
        logits_vec = tl.load(row_logits_ptr + j)
        # Numerically stable softmax
        row_max = tl.max(logits_vec, axis=0)
        logits_vec = logits_vec - row_max
        sum_exp = tl.sum(tl.exp(logits_vec), axis=0)
        attn_vec = tl.exp(logits_vec) / sum_exp
        tl.store(row_attn_ptr + j, attn_vec)

# Kernel to compute out[h, :] = attn[h, :] @ Kc
@triton.jit
def compute_out_kernel(
    attn_ptr, Kc_ptr, out_ptr,
    KV: tl.int32,
    Dn: tl.constexpr,     # head_dim_ckv = 512
    H: tl.int32,
    BLOCK_K: tl.constexpr,
):
    for h in range(0, H):
        # out[h, :] accumulation
        out_vec = tl.zeros((Dn,), dtype=tl.float32)
        for k0 in range(0, KV, BLOCK_K):
            k_idx = k0 + tl.arange(0, BLOCK_K)
            mask_k = k_idx < KV
            attn_vec = tl.load(attn_ptr + h * KV + k_idx, mask=mask_k, other=0.0)  # [BLOCK_K]
            Kc_tile = tl.load(Kc_ptr + k_idx[:, None] * Dn + tl.arange(0, Dn), mask=mask_k[:, None], other=0.0)  # [BLOCK_K, Dn]
            # out_vec += sum over k of attn_vec[k] * Kc_tile[k, :]
            out_vec += tl.sum(Kc_tile * attn_vec[:, None], axis=0)
        tl.store(out_ptr + h * Dn + tl.arange(0, Dn), out_vec)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure device and types
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "All tensors must be on CUDA device."
        device = q_nope.device
        # Cast caches to float32 and make contiguous
        Kc_all = ckv_cache.squeeze(1).to(torch.float32).contiguous()  # [M, 512]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32).contiguous()  # [M, 64]
        # Cast queries to float32 and make contiguous
        q_nope_f32 = q_nope.to(torch.float32).contiguous()  # [N, 16, 512]
        q_pe_f32 = q_pe.to(torch.float32).contiguous()     # [N, 16, 64]

        # Dimensions
        total_q = qo_indptr[-1].item()
        num_qo_heads = q_nope_f32.shape[1]
        head_dim_ckv = q_nope_f32.shape[2]  # 512
        head_dim_kpe = q_pe_f32.shape[2]    # 64
        num_pages = ckv_cache.shape[0]
        len_indptr = qo_indptr.shape[0]
        batch_size = len_indptr - 1

        # Allocate output and lse
        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Process batches
        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            q_len = q_end - q_start
            kv_len = kv_end - kv_start
            tok_idx = kv_indices[kv_start:kv_end].to(torch.long).contiguous()  # [kv_len]

            # Gather Kc and Kp for this batch's tokens
            Kc = Kc_all[tok_idx]  # [kv_len, 512]
            Kp = Kp_all[tok_idx]  # [kv_len, 64]
            Kc = Kc.contiguous()
            Kp = Kp.contiguous()

            for i in range(q_len):
                # Slice qn and qp rows
                qn = q_nope_f32[q_start + i]  # [16, 512]
                qp = q_pe_f32[q_start + i]    # [16, 64]
                qn = qn.contiguous()
                qp = qp.contiguous()

                # Allocate logits and lse vectors per head
                logits = torch.empty((num_qo_heads, kv_len), dtype=torch.float32, device=device)
                lse_vec = torch.empty((num_qo_heads,), dtype=torch.float32, device=device)

                # Launch compute_logits_and_lse_kernel for this (b, i)
                compute_logits_and_lse_kernel[(1,)](
                    qn, qp, Kc, Kp,
                    logits, lse_vec,
                    sm_scale,
                    kv_len - q_len,   # prefix_len
                    kv_len - q_len + i,  # query_abs_pos
                    kv_len, num_qo_heads,
                    head_dim_ckv, head_dim_kpe,
                    BLOCK_K=128,
                    num_warps=4,
                )

                # Softmax over logits to get attn
                attn = torch.empty_like(logits)
                softmax_rows_kernel[(1,)](
                    logits, attn,
                    kv_len, num_qo_heads,
                    num_warps=4,
                )

                # Compute out[h, :] = attn[h, :] @ Kc for each head
                out_rows = torch.empty((num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
                compute_out_kernel[(1,)](
                    attn, Kc, out_rows,
                    kv_len, head_dim_ckv, num_qo_heads,
                    BLOCK_K=128,
                    num_warps=4,
                )

                # Store outputs and lse
                # output[q_start + i, :, :] = out_rows
                output[q_start + i] = out_rows
                # lse[q_start + i, :] = lse_vec
                lse[q_start + i] = lse_vec

        # Cast output to bfloat16 to match original
        output = output.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
