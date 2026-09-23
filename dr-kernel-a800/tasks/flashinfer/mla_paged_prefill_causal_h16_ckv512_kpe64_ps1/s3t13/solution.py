import torch
import triton
import triton.language as tl


@triton.jit
def compute_logits_kernel(
    qn_vec_ptr, qp_vec_ptr, Kc_ptr, Kp_ptr, logits_scaled_ptr,
    K: tl.constexpr, Kp: tl.constexpr, L: tl.constexpr,
    tok_idx_ptr,
    sm_scale,
    stride_qn_h, stride_qn_k,
    stride_qp_h, stride_qp_kp,
    stride_log_h, stride_log_l,
    head: tl.constexpr,
):
    # Compute per-token logits for this head
    l = 0
    while l < L:
        idx_l = tl.load(tok_idx_ptr + l)  # int32
        acc = tl.zeros((), dtype=tl.float32)
        # Sum over K features from qn @ Kc
        k = 0
        while k < K:
            qn_val = tl.load(qn_vec_ptr + head * K + k)
            kc_val = tl.load(Kc_ptr + idx_l * K + k)
            acc += qn_val * kc_val
            k += 1
        # Sum over Kp features from qp @ Kp
        kp = 0
        while kp < Kp:
            qp_val = tl.load(qp_vec_ptr + head * Kp + kp)
            kp_val = tl.load(Kp_ptr + idx_l * Kp + kp)
            acc += qp_val * kp_val
            kp += 1
        # Scale
        acc = acc * sm_scale
        # Store to logits_scaled[h, l] at address head * L + l
        tl.store(logits_scaled_ptr + head * L + l, acc)
        l += 1


@triton.jit
def compute_lse_kernel(
    logits_scaled_ptr, lse_ptr,
    L: tl.constexpr,
    stride_log_h, stride_log_l,
    prefix_len,  # int32 scalar
):
    head = 0
    # Initialize max
    max_val = tl.full((), -1e30, dtype=tl.float32)
    j = 0
    while j < L:
        val = tl.load(logits_scaled_ptr + head * L + j)
        valid = (j <= prefix_len)
        if valid:
            max_val = tl.maximum(max_val, val)
        j += 1

    sum_exp = tl.zeros((), dtype=tl.float32)
    j = 0
    while j < L:
        val = tl.load(logits_scaled_ptr + head * L + j)
        valid = (j <= prefix_len)
        if valid:
            sum_exp += tl.exp(val - max_val)
        j += 1

    lse_val = max_val + tl.log(sum_exp)
    ln2 = 0.6931471805599453
    tl.store(lse_ptr + head, lse_val / ln2)


@triton.jit
def compute_softmax_kernel(
    logits_scaled_ptr, lse_ptr, attn_ptr,
    L: tl.constexpr,
    stride_log_h, stride_log_l,
    stride_attn_h, stride_attn_l,
    prefix_len,  # int32 scalar
):
    head = 0
    lse_val = tl.load(lse_ptr + head)
    j = 0
    while j < L:
        val = tl.load(logits_scaled_ptr + head * L + j)
        valid = (j <= prefix_len)
        soft = 0.0
        if valid:
            soft = tl.exp(val - lse_val)
        tl.store(attn_ptr + head * L + j, soft)
        j += 1


@triton.jit
def gemv_out_kernel(
    attn_ptr, Kc_ptr, out_ptr,
    K: tl.constexpr, L: tl.constexpr,
    stride_attn_h, stride_attn_l,
    stride_Kc_p, stride_Kc_k,
    stride_out_h, stride_out_k,
    tok_idx_ptr,
    head: tl.constexpr,
):
    # For each output feature k, accumulate sum over l of attn[h, l] * Kc[tok_idx[l], k]
    k = 0
    acc = tl.zeros((), dtype=tl.float32)
    while k < K:
        l = 0
        while l < L:
            attn_val = tl.load(attn_ptr + head * L + l)
            idx_l = tl.load(tok_idx_ptr + l)  # int32
            kc_val = tl.load(Kc_ptr + idx_l * K + k)
            acc += attn_val * kc_val
            l += 1
        tl.store(out_ptr + head * K + k, acc)
        k += 1


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure Triton/CUDA
        if q_nope.device.type != "cuda":
            raise RuntimeError("ModelNew requires CUDA tensors for Triton kernels.")

        # Squeeze caches: ckv_cache and kpe_cache are [P, 1, ...], so squeeze dim=1 → [P, K] and [P, Kp]
        # Note: We do not use any torch .to(...) or .contiguous(...) on Triton tensors (to keep Triton-only).
        Kc_all = ckv_cache.squeeze(1)  # [P, K]
        Kp_all = kpe_cache.squeeze(1)  # [P, Kp]

        # Output buffers
        H = 16
        K = 512
        total_q = int(qo_indptr[-1].item())
        output = torch.empty((total_q, H, K), dtype=torch.bfloat16, device=q_nope.device)
        lse = torch.empty((total_q, H), dtype=torch.float32, device=q_nope.device)

        # Process each batch element b
        for b in range(kv_indptr.numel() - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            q_len = q_end - q_start
            if q_len == 0:
                continue

            tok_start = int(kv_indptr[b].item())
            tok_end = int(kv_indptr[b + 1].item())
            L = tok_end - tok_start
            if L == 0:
                continue

            # tok_idx as int32 tensor on device (no torch .to inside Triton kernels)
            tok_idx = kv_indices[tok_start:tok_end].to(torch.int32)  # [L], int32 (PyTorch tensor)

            # Loop over queries in this batch
            for i in range(q_len):
                q_abs = q_start + i

                # Prepare flattened vectors for qn and qp (these are inputs; we pass pointers to Triton)
                # We must ensure q_nope[q_abs] and q_pe[q_abs] are contiguous when used, but Triton pointers accept any layout—
                # better to create flattened views as torch tensors for simplicity and avoid Triton-side .contiguous().
                # However, Triton kernels cannot call .view()/.contiguous(); we instead pass the full [H, K] or [H, Kp] and use strides.
                # Here, we will compute qn and qp directly as torch tensors and pass their pointers to Triton kernels which read
                # elements using strides. This avoids any Triton tensor .to/.contiguous.

                # Load qn and qp from PyTorch tensors and pass pointers; Triton kernels will read them (no torch ops inside).
                qn = q_nope[q_abs]  # [H, K] tensor (PyTorch), already on CUDA
                qp = q_pe[q_abs]    # [H, Kp] tensor (PyTorch), already on CUDA

                # Allocate temporary per-head buffers
                logits_scaled = torch.empty((H, L), dtype=torch.float32, device=q_nope.device)
                attn = torch.empty((H, L), dtype=torch.float32, device=q_nope.device)

                # Launch Triton kernels per head
                for h in range(H):
                    # Compute logits_scaled[h, :]
                    stride_qn_h, stride_qn_k = K, 1
                    stride_qp_h, stride_qp_kp = Kp_all.shape[1], 1
                    stride_log_h, stride_log_l = L, 1

                    compute_logits_kernel[(1,)](
                        qn_ptr=qn, qp_ptr=qp, Kc_ptr=Kc_all, Kp_ptr=Kp_all,
                        logits_scaled_ptr=logits_scaled[h],
                        K=K, Kp=Kp_all.shape[1], L=L,
                        tok_idx_ptr=tok_idx,
                        sm_scale=float(sm_scale),
                        stride_qn_h=stride_qn_h, stride_qn_k=stride_qn_k,
                        stride_qp_h=stride_qp_h, stride_qp_kp=stride_qp_kp,
                        stride_log_h=stride_log_h, stride_log_l=stride_log_l,
                        head=h,
                    )

                    # Compute lse[h] with prefix_len = L - q_len (per batch element)
                    prefix_len = L - q_len
                    compute_lse_kernel[(1,)](
                        logits_scaled[h], lse[q_abs],
                        L=L,
                        stride_log_h=stride_log_h, stride_log_l=1,
                        prefix_len=prefix_len,
                    )

                    # Compute attn[h, :]
                    compute_softmax_kernel[(1,)](
                        logits_scaled[h], lse[q_abs], attn[h],
                        L=L,
                        stride_log_h=stride_log_h, stride_log_l=1,
                        stride_attn_h=L, stride_attn_l=1,
                        prefix_len=prefix_len,
                    )

                    # Compute output[h, :] via GEMV with Kc rows indexed by tok_idx
                    out_vec = torch.empty((K,), dtype=torch.float32, device=q_nope.device)
                    stride_Kc_p, stride_Kc_k = K, 1
                    stride_out_h, stride_out_k = K, 1

                    gemv_out_kernel[(1,)](
                        attn[h], Kc_all, out_vec,
                        K=K, L=L,
                        stride_attn_h=L, stride_attn_l=1,
                        stride_Kc_p=stride_Kc_p, stride_Kc_k=stride_Kc_k,
                        stride_out_h=stride_out_h, stride_out_k=1,
                        tok_idx_ptr=tok_idx,
                        head=h,
                    )

                    # Store output as bfloat16
                    output[q_abs, h] = out_vec.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
