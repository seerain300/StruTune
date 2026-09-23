import math
import torch
import triton
import triton.language as tl


@triton.jit
def compute_logits_kernel(
    qn_vec_ptr, qp_vec_ptr, Kc_ptr, Kp_ptr, logits_scaled_ptr,
    H: tl.constexpr, K: tl.constexpr, Kp: tl.constexpr, L: tl.constexpr,
    tok_idx_ptr,
    sm_scale,
):
    # This kernel computes logits_scaled for a single head h. It is launched with h as a grid dimension.
    h = tl.program_id(0)  # head index, constexpr-like for loop range

    # Compute base strides for q vectors
    # qn_vec_ptr: length H*K
    # qp_vec_ptr: length H*Kp
    # We will index qn_vec_ptr by h*K + k, and qp_vec_ptr by h*Kp + kp

    # Prepare accumulators for logits_scaled for this head
    # We will compute one element at a time for simplicity and store to logits_scaled_ptr[h*L + l]
    for l in tl.static_range(0, L):
        # Read token index
        tok = tl.load(tok_idx_ptr + l)
        # Accumulate over K and Kp features
        acc = 0.0
        for k in tl.static_range(0, K):
            qn_k = tl.load(qn_vec_ptr + h * K + k)
            Kc_val = tl.load(Kc_ptr + tok * K + k)
            acc += qn_k * Kc_val
        for kp in tl.static_range(0, Kp):
            qp_kp = tl.load(qp_vec_ptr + h * Kp + kp)
            Kp_val = tl.load(Kp_ptr + tok * Kp + kp)
            acc += qp_kp * Kp_val
        acc = acc * sm_scale
        tl.store(logits_scaled_ptr + h * L + l, acc)


@triton.jit
def compute_lse_kernel(
    logits_scaled_ptr, lse_ptr,
    L: tl.constexpr,
):
    h = tl.program_id(0)  # head index
    # Compute max over logits_scaled[h, :]
    max_val = -float('inf')
    for l in tl.static_range(0, L):
        val = tl.load(logits_scaled_ptr + h * L + l)
        max_val = tl.maximum(max_val, val)
    # Compute sum of exp(logits - max)
    sum_exp = 0.0
    for l in tl.static_range(0, L):
        sum_exp += tl.exp(tl.load(logits_scaled_ptr + h * L + l) - max_val)
    lse = tl.log(sum_exp) + max_val
    ln2 = 0.6931471805599453  # math.log(2.0)
    tl.store(lse_ptr + h, lse / ln2)


@triton.jit
def gemv_out_kernel(
    attn_ptr, Kc_ptr, out_ptr,
    H: tl.constexpr, L: tl.constexpr, K: tl.constexpr, tok_idx_ptr,
):
    # For a given head h, compute out[h, :] = sum_l attn[h, l] * Kc[tok_idx[l], :]
    h = tl.program_id(0)
    out_vec = tl.zeros((K,), dtype=tl.float32)
    for l in tl.static_range(0, L):
        attn_l = tl.load(attn_ptr + h * L + l)
        tok = tl.load(tok_idx_ptr + l)
        # Dot product over K: out_vec += attn_l * Kc[tok, :]
        for k in tl.static_range(0, K):
            kval = tl.load(Kc_ptr + tok * K + k)
            out_vec[k] += attn_l * kval
    # Store out_vec to out_ptr
    for k in tl.static_range(0, K):
        tl.store(out_ptr + h * K + k, out_vec[k])


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure all inputs are on the same device and dtype for computation
        device = q_nope.device

        # Squeeze caches to [P, K] and [P, Kp]
        Kc_all = ckv_cache.squeeze(1).to(device=device, dtype=torch.float32)  # [P, 512]
        Kp_all = kpe_cache.squeeze(1).to(device=device, dtype=torch.float32)  # [P, 64]

        total_q, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]
        assert num_qo_heads == 16, "num_qo_heads must be 16"
        assert head_dim_ckv == 512, "head_dim_ckv must be 512"
        assert head_dim_kpe == 64, "head_dim_kpe must be 64"

        # Compute output and lse
        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Process each batch element
        for b in range(qo_indptr.shape[0] - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            # If no queries or KV for this batch, skip
            if q_start >= q_end:
                continue

            # Token indices for this batch
            tok_idx = kv_indices[int(kv_indptr[b].item()): int(kv_indptr[b + 1].item())].to(torch.int32).to(device)
            L = tok_idx.numel()

            # Iterate over queries in this batch
            for i in range(q_end - q_start):
                q_abs = q_start + i
                # Prepare flattened q vectors
                qn_vec = q_nope[q_abs].contiguous().view(-1).to(torch.float32).to(device)  # H*K = 16*512 = 8192
                qp_vec = q_pe[q_abs].contiguous().view(-1).to(torch.float32).to(device)    # H*Kp = 16*64 = 1024

                # Allocate logits_scaled per head
                logits_scaled = torch.empty((num_qo_heads, L), dtype=torch.float32, device=device)
                # Launch compute_logits_kernel for each head h
                for h in range(num_qo_heads):
                    grid = (1,)
                    compute_logits_kernel[grid](
                        qn_vec, qp_vec, Kc_all, Kp_all, logits_scaled[h],
                        H=num_qo_heads, K=head_dim_ckv, Kp=head_dim_kpe, L=L,
                        tok_idx_ptr=tok_idx,
                        sm_scale=float(sm_scale),
                    )

                # Compute lse per head
                for h in range(num_qo_heads):
                    lse[b, h] = compute_lse_kernel[(1,)](
                        logits_scaled[h], lse[b, h],
                        L=L,
                    )

                # Compute attn per head (softmax of logits_scaled - lse). For robustness, we reconstruct attn here.
                # Note: Triton kernel does not return attn; we compute it on host from logits_scaled. We still can use Triton for GEMV:
                out_vec = torch.empty((head_dim_ckv,), dtype=torch.float32, device=device)

                # We need attn[h, :]. Since Triton kernel doesn't produce it, reconstruct:
                for h in range(num_qo_heads):
                    # attn[h, l] = exp((logits_scaled[h, l] - lse[b, h]) / ln2)
                    # Here, we don't scale by ln2 inside kernel, so we compute directly on host:
                    attn = torch.exp(logits_scaled[h] - lse[b, h])  # logits_scaled already scaled by sm_scale
                    # Now perform GEMV: output[q_abs, h, :] += attn @ Kc[tok_idx, :]
                    # We can do this GEMV in Triton to avoid torch @ on host:
                    # Prepare attn_ptr
                    attn_ptr = attn.to(torch.float32).contiguous().view(-1)  # [L]
                    # Launch gemv_out_kernel
                    gemv_out_kernel[(1,)](
                        attn_ptr, Kc_all, out_vec,
                        H=num_qo_heads, L=L, K=head_dim_ckv, tok_idx_ptr=tok_idx,
                    )
                    # Store result to output at position (q_abs, h, :)
                    output[q_abs, h] = out_vec.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
