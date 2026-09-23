import torch
import triton
import triton.language as tl


# Kernel 1: Compute logits_scaled[h, :] for a given query and head
# Args:
#   qn_vec_ptr: *fp32, flattened qn for this query, length H*K
#   qp_vec_ptr: *fp32, flattened qp for this query, length H*Kp
#   Kc_ptr: *fp32, base pointer to Kc_all flattened (P*K)
#   Kp_ptr: *fp32, base pointer to Kp_all flattened (P*Kp)
#   tok_idx_ptr: *int32, token indices for this batch, length L
#   L: number of tokens in this batch segment
#   H, K, Kp: dimensions
#   sm_scale: fp32 scalar
#   head: constexpr, which head we compute for
# Output:
#   logits_scaled_ptr: *fp32, vector of length L, logits_scaled[h, :]
@triton.jit
def compute_logits_kernel(
    qn_vec_ptr, qp_vec_ptr, Kc_ptr, Kp_ptr, logits_scaled_ptr,
    H, K, Kp, L,
    tok_idx_ptr,
    sm_scale,
    stride_qn_h, stride_qn_k,
    stride_qp_h, stride_qp_kp,
    stride_log_l,
    head: tl.constexpr,
):
    # Precompute strides and launch per-query program
    for l in range(0, L):
        # Compute dot products for this token l
        acc = 0.0
        for k in range(0, K):
            idx = head * stride_qn_h + k * stride_qn_k
            qn_k = tl.load(qn_vec_ptr + idx)
            tok = tl.load(tok_idx_ptr + l)
            Kc_val = tl.load(Kc_ptr + tok * K + k)
            acc += qn_k * Kc_val

        acc2 = 0.0
        for kp in range(0, Kp):
            idx = head * stride_qp_h + kp * stride_qp_kp
            qp_kp = tl.load(qp_vec_ptr + idx)
            tok = tl.load(tok_idx_ptr + l)
            Kp_val = tl.load(Kp_ptr + tok * Kp + kp)
            acc2 += qp_kp * Kp_val

        val = acc + acc2
        tl.store(logits_scaled_ptr + l, val * sm_scale)


# Kernel 2: Compute lse[h] = logsumexp(logits_scaled[h, :]) / ln(2), zeroing invalid positions
@triton.jit
def compute_lse_kernel(logits_scaled_ptr, lse_ptr, L):
    # Single program handles one row (here we assume one head row)
    # We load the entire row, apply mask, compute max, sum exp, then lse.
    # Since Triton kernels typically operate on blocks, we implement a simple loop here.
    # We assume logits_scaled_ptr points to a vector of length L.
    max_val = -1.0e30
    for j in range(0, L):
        val = tl.load(logits_scaled_ptr + j)
        # No mask applied here; assume logits_scaled already zeroed invalid positions in compute_logits_kernel
        if val > max_val:
            max_val = val

    sum_exp = 0.0
    for j in range(0, L):
        val = tl.load(logits_scaled_ptr + j)
        sum_exp += tl.exp(val - max_val)

    lse = tl.log(sum_exp) + max_val  # logsumexp after subtracting max
    lse_div = lse / 0.6931471805599453  # 1 / ln(2)
    tl.store(lse_ptr, lse_div)


# Kernel 3: Compute attn[h, :] = softmax(logits_scaled[h, :]) relative to lse[h], zero invalid positions
@triton.jit
def compute_softmax_kernel(logits_scaled_ptr, lse_ptr, attn_ptr, L):
    lse_val = tl.load(lse_ptr)
    for j in range(0, L):
        val = tl.load(logits_scaled_ptr + j)
        # invalid positions should be zero in logits_scaled after mask
        attn_ptr[j] = tl.exp(val - lse_val)


# Kernel 4: GEMV per head: out[h, :] = attn[h, :] @ Kc_all[tok_idx[:], :] -> vector of length K
# We implement a simple loop-based GEMV here, accumulating across L tokens.
@triton.jit
def gemv_out_kernel(attn_ptr, Kc_ptr, tok_idx_ptr, out_vec_ptr, L, K):
    acc = 0.0
    for l in range(0, L):
        attn_l = tl.load(attn_ptr + l)
        tok = tl.load(tok_idx_ptr + l)
        # Reduce over K
        row_sum = 0.0
        for k in range(0, K):
            Kc_val = tl.load(Kc_ptr + tok * K + k)
            row_sum += attn_l * Kc_val
        acc += attn_l * row_sum  # Note: attn_l is scalar; this line incorrect. Fix below.
        # Correct accumulation should be: for each k, accumulate attn_l * Kc_val, but we need per-output k accumulation.
        # To correctly implement GEMV, we should have out_vec_ptr indexed by k. However, Triton does not support
        # vectorized writes with per-iteration indexing easily; instead, we implement as:
        # out_vec_ptr += attn_l * row_sum for each k. This requires vectorization; we'll use a separate
        # approach: compute attn_ptr vector and then launch a second kernel to do GEMV, but here we simplify by
        # computing row_sum and writing a scalar? We need a vector output. We'll instead write a small wrapper in
        # forward that calls this kernel and handles output vector via torch for simplicity. However, to keep Triton-only
        # forward, we should compute the full output vector inside the kernel. We'll re-implement gemv to produce
        # a vector of length K by iterating over k and accumulating attn contributions for each k:
    # Recompute proper GEMV: initialize out_vec to zeros then accumulate for each k.
    out_vec = tl.zeros((K,), dtype=tl.float32)
    for k in range(0, K):
        col_sum = 0.0
        for l in range(0, L):
            attn_l = tl.load(attn_ptr + l)
            tok = tl.load(tok_idx_ptr + l)
            Kc_val = tl.load(Kc_ptr + tok * K + k)
            col_sum += attn_l * Kc_val
        out_vec[k] = col_sum
    # Store out_vec; pointer is out_vec_ptr which is a vector of length K
    # Triton does not allow returning vectors; we write them elementwise in forward. Here, we store scalar acc.
    # We need a way to return out_vec to PyTorch. Triton kernel returns scalars, so we cannot return vector directly.
    # Instead, forward should allocate out vector and pass pointers appropriately. For correctness, we implement
    # GEMV in Triton to produce vector output by using a second kernel that writes out_vec_ptr[k] = computed value.
    # Since Triton kernels cannot return vectors, we will use a scalar output placeholder and rely on Triton to
    # write to out_vec_ptr via a loop. Triton will execute the loop and write values, but we must ensure out_vec_ptr
    # points to a valid tensor. We will implement the full vector write here:
    for k in range(0, K):
        col_sum = 0.0
        for l in range(0, L):
            attn_l = tl.load(attn_ptr + l)
            tok = tl.load(tok_idx_ptr + l)
            Kc_val = tl.load(Kc_ptr + tok * K + k)
            col_sum += attn_l * Kc_val
        tl.store(out_vec_ptr + k, col_sum)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA device
        if q_nope.device.type != 'cuda':
            raise RuntimeError("ModelNew requires tensors on CUDA device.")
        device = q_nope.device

        # Cast caches to fp32 for compute
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [P, 512]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [P, 64]

        total_q = q_nope.shape[0]
        batch_size = qo_indptr.shape[0] - 1

        # Output tensors
        output = torch.empty((total_q, NUM_QO_HEADS, HEAD_DIM_CKV), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, NUM_QO_HEADS), dtype=torch.float32, device=device)

        H = NUM_QO_HEADS
        K = HEAD_DIM_CKV  # 512
        Kp = 64

        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            q_len = q_end - q_start
            if q_len <= 0:
                continue

            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            L = end - start
            tok_idx = kv_indices[start:end].to(torch.int32).to(device).contiguous()  # [L]

            # For each query i
            for i in range(q_len):
                q_abs = q_start + i

                # Flatten qn and qp for this query
                qn_base = q_nope[q_abs].to(torch.float32)  # [H, K]
                qp_base = q_pe[q_abs].to(torch.float32)   # [H, Kp]
                # Flattened pointers (views)
                qn_vec = qn_base.view(-1).to(torch.float32).contiguous()
                qp_vec = qp_base.view(-1).to(torch.float32).contiguous()

                # Compute logits_scaled per head (we'll compute logits for all heads and apply softmax)
                logits_scaled = torch.empty((L,), dtype=torch.float32, device=device)

                # Launch compute_logits_kernel for each head
                for h in range(H):
                    # Launch kernel: one program, iterate L (simple loop). Triton requires args as tensors; here we pass
                    # scalars for H, K, Kp, L, sm_scale; and pointers for qn_vec, qp_vec, Kc_all, Kp_all, tok_idx, logits_scaled.
                    # To pass strides, we can use K and Kp as strides in compute: stride_qn_h = K, stride_qn_k = 1, etc.
                    compute_logits_kernel[(1,)](
                        qn_vec, qp_vec, Kc_all, Kp_all, logits_scaled,
                        H=H, K=K, Kp=Kp, L=L,
                        tok_idx=tok_idx,
                        sm_scale=float(sm_scale),
                        stride_qn_h=K, stride_qn_k=1,
                        stride_qp_h=Kp, stride_qp_kp=1,
                        stride_log_l=1,
                        head=h,
                    )

                    # Compute lse for this head
                    lse_row = torch.empty((), dtype=torch.float32, device=device)  # scalar tensor for lse
                    compute_lse_kernel[(1,)](
                        logits_scaled, lse_row, L=L
                    )

                    # Compute attn for this head
                    attn = torch.empty((L,), dtype=torch.float32, device=device)
                    compute_softmax_kernel[(1,)](
                        logits_scaled, lse_row, attn, L=L
                    )

                    # GEMV per head: out[h, :] = attn[h, :] @ Kc_all[tok_idx[:], :]
                    # We need a vector output of length K. Triton kernel writes elementwise to out_vec_ptr[k].
                    out_vec = torch.empty((K,), dtype=torch.float32, device=device)
                    gemv_out_kernel[(1,)](
                        attn, Kc_all, tok_idx, out_vec, L=L, K=K
                    )

                    # Store output as bfloat16
                    output[q_abs, h, :] = out_vec.to(torch.bfloat16)

                    # Update lse for this query
                    lse[q_abs, h] = lse_row.item()  # not ideal to call .item(); however, Triton cannot return tensors.
                    # Since Triton kernels cannot return tensors, we store computed lse via PyTorch assignment above.

        return output, lse


def run(*args):
    return ModelNew()(*args)
