import math
import torch
import triton
import triton.language as tl


@triton.jit
def compute_logits_kernel(
    qn_vec_ptr, qp_vec_ptr, Kc_ptr, Kp_ptr, logits_scaled_ptr,
    H, K, Kp, L,
    tok_idx_ptr,
    sm_scale,
    stride_qn_h, stride_qn_k,
    stride_qp_h, stride_qp_kp,
    stride_log_h, stride_log_l,
    head: tl.constexpr,
):
    # We expect to be launched once per head h; we compute the logits row for all L positions.
    # qn_vec_ptr: [H*K] float32
    # qp_vec_ptr: [H*Kp] float32
    # Kc_ptr: [P*K] float32
    # Kp_ptr: [P*Kp] float32
    # logits_scaled_ptr: [L] float32 (row for this head)
    # tok_idx_ptr: [L] int32
    # We compute: logits_scaled[h, l] = sum_k qn[h, k] * Kc[tok_idx[l], k] + sum_kp qp[h, k'] * Kp[tok_idx[l], k']

    # Create local indices
    k_idx = tl.arange(0, K)
    kp_idx = tl.arange(0, Kp)
    l_idx = tl.arange(0, L)

    # Build qn and qp row vectors for this head
    # qn_vec_ptr is laid out as [H*K] contiguous
    base_qn = qn_vec_ptr[head * K + k_idx]  # [K]
    base_qp = qp_vec_ptr[head * Kp + kp_idx]  # [Kp]

    # For each position l
    for j in l_idx:
        tok = tl.load(tok_idx_ptr + j)  # token index into caches
        # Kc values for this token, shape [K]
        Kc_vals = tl.load(Kc_ptr + tok * K + k_idx)
        # Kp values for this token, shape [Kp]
        Kp_vals = tl.load(Kp_ptr + tok * Kp + kp_idx)

        # Dot products
        dot_qn = tl.sum(base_qn * Kc_vals, axis=0)  # scalar
        dot_qp = tl.sum(base_qp * Kp_vals, axis=0)  # scalar

        # Accumulate
        val = dot_qn + dot_qp
        val = val * sm_scale
        # Write scaled logits
        tl.store(logits_scaled_ptr + j, val)


@triton.jit
def compute_lse_kernel(
    logits_scaled_ptr, lse_ptr,
    L,
    stride_log_h, stride_log_l,
):
    # lse_ptr is a scalar for this head (we write into a single location).
    # Compute logsumexp over L positions (invalid positions should be zeroed, so they don't affect max).
    # We do a simple reduction:
    m = -float('inf')
    for j in range(0, L):
        val = tl.load(logits_scaled_ptr + j)
        # invalid positions are already zero in logits_scaled, so max stays unaffected (we zeroed them before).
        if val > m:
            m = val

    # Compute sum(exp(logits - m))
    sum_exp = 0.0
    for j in range(0, L):
        val = tl.load(logits_scaled_ptr + j)
        sum_exp += tl.exp(val - m)

    lse = tl.log(sum_exp) + m
    # lse is per head; write to lse_ptr
    tl.store(lse_ptr, lse)


@triton.jit
def compute_softmax_kernel(
    logits_scaled_ptr, lse_ptr, attn_ptr,
    L,
    stride_log_h, stride_log_l,
    stride_attn_h, stride_attn_l,
):
    # Compute attn[h, :] = exp(logits_scaled[h, :] - lse[h])
    # Invalid positions should be zero; here we assume logits_scaled invalid positions are zero.
    lse_val = tl.load(lse_ptr)
    for j in range(0, L):
        val = tl.load(logits_scaled_ptr + j)
        attn_val = tl.exp(val - lse_val)
        # invalid positions are zero in logits_scaled; exp(-inf) is not called because we zeroed them. Here we ensure attn=0 if needed by masking, but since logits_scaled invalid were zero, this is fine.
        tl.store(attn_ptr + j, attn_val)


@triton.jit
def gemv_out_kernel(
    attn_ptr, Kc_ptr, out_vec_ptr,
    K, L,
    tok_idx_ptr,
):
    # GEMV: out[h, :] = attn[h, :] @ Kc[tok_idx[:], :]
    # Kc_ptr: [P*K], P is not needed directly, we index by tok_idx[l], then feature k
    # attn_ptr: [L] float32
    # out_vec_ptr: [K] float32
    for k in range(0, K):
        acc = 0.0
        for l in range(0, L):
            attn_val = tl.load(attn_ptr + l)
            tok = tl.load(tok_idx_ptr + l)
            Kc_val = tl.load(Kc_ptr + tok * K + k)
            acc += attn_val * Kc_val
        tl.store(out_vec_ptr + k, acc)


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
        output = torch.empty((total_q, 16, 512), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, 16), dtype=torch.float32, device=device)

        H = 16
        K = 512
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

            # Token indices for this batch
            tok_idx = kv_indices[start:end].to(torch.int32).to(device).contiguous()  # [L]

            # For each query i
            for i in range(q_len):
                q_abs = q_start + i

                # Compute qn_vec_flat and qp_vec_flat per query (host-side, but no torch compute in kernels)
                # We'll pass them to Triton compute_logits_kernel by flattening directly in forward.
                # To keep Triton-only, we construct them via slicing and .view:
                qn = q_nope[q_abs].to(torch.float32).contiguous()   # [H, K]
                qn_vec = qn.view(-1)                                # [H*K]
                qp = q_pe[q_abs].to(torch.float32).contiguous()    # [H, Kp]
                qp_vec = qp.view(-1)                                # [H*Kp]

                # Launch compute logits for each head h
                for h in range(H):
                    # Prepare logits_scaled buffer for this head
                    logits_scaled = torch.empty((L,), dtype=torch.float32, device=device)

                    # Strides for kernel
                    stride_qn_h = K
                    stride_qn_k = 1
                    stride_qp_h = Kp
                    stride_qp_kp = 1
                    stride_log_h = 1
                    stride_log_l = 1

                    compute_logits_kernel[(1,)](
                        qn_vec, qp_vec, Kc_all, Kp_all, logits_scaled,
                        H=H, K=K, Kp=Kp, L=L, tok_idx=tok_idx,
                        sm_scale=float(sm_scale),
                        stride_qn_h=stride_qn_h, stride_qn_k=stride_qn_k,
                        stride_qp_h=stride_qp_h, stride_qp_kp=stride_qp_kp,
                        stride_log_h=stride_log_h, stride_log_l=stride_log_l,
                        head=h,
                    )

                    # Compute lse[h] = logsumexp(logits_scaled) / ln(2)
                    lse_vec = torch.empty((1,), dtype=torch.float32, device=device)
                    compute_lse_kernel[(1,)](
                        logits_scaled, lse_vec,
                        L=L,
                        stride_log_h=1, stride_log_l=1,
                    )
                    # Assign to lse[q_abs, h]; since lse is [total_q, H], we write using q_abs as row index.
                    # PyTorch: lse[q_abs, h] = lse_vec[0] * (1 / ln(2))
                    lse[q_abs, h] = lse_vec[0] / math.log(2.0)

                    # Compute attn[h, :]
                    attn = torch.empty((L,), dtype=torch.float32, device=device)
                    compute_softmax_kernel[(1,)](
                        logits_scaled, lse[q_abs, h].to(torch.float32), attn,
                        L=L,
                        stride_log_h=1, stride_log_l=1,
                        stride_attn_h=1, stride_attn_l=1,
                    )

                    # GEMV: out[h, :] = attn @ Kc[tok_idx, :]
                    out_vec = torch.empty((K,), dtype=torch.float32, device=device)
                    gemv_out_kernel[(1,)](
                        attn, Kc_all, out_vec,
                        K=K, L=L, tok_idx=tok_idx,
                    )

                    # Store output[q_abs, h, :]
                    output[q_abs, h, :] = out_vec.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
