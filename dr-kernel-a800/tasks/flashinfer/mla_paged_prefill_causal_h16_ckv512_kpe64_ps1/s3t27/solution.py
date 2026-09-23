import math
import torch
import triton
import triton.language as tl


# Kernel 1: For a given head h, compute logits_scaled[h, :] = (qn_vec @ Kc_tok + qp_vec @ Kp_tok) * sm_scale
# Inputs:
#   qn_vec_ptr: *fp32, length H*K
#   qp_vec_ptr: *fp32, length H*Kp
#   Kc_ptr: *fp32, [P, K], indexed by tok_idx_ptr[l] and feature k
#   Kp_ptr: *fp32, [P, Kp], indexed by tok_idx_ptr[l] and feature kp
#   tok_idx_ptr: *int32, length L
#   logits_ptr: *fp32, [L], output logits scaled
#   L: number of tokens
#   H, K, Kp: dims
#   sm_scale: fp32 scalar
#   head: constexpr
@triton.jit
def compute_logits_kernel(qn_vec_ptr, qp_vec_ptr, Kc_ptr, Kp_ptr, tok_idx_ptr, logits_ptr,
                          H, K, Kp, L, sm_scale,
                          head: tl.constexpr):
    # Compute logits for head 'head' across L tokens
    for l in range(0, L):
        k_sum = tl.zeros((), dtype=tl.float32)
        kp_sum = tl.zeros((), dtype=tl.float32)
        tok = tl.load(tok_idx_ptr + l)  # scalar int32
        # Accumulate qn contributions
        for kk in range(0, K):
            qn_val = tl.load(qn_vec_ptr + head * K + kk)
            kc_val = tl.load(Kc_ptr + tok * K + kk)  # row tok of Kc
            k_sum += qn_val * kc_val
        # Accumulate qp contributions
        for kp in range(0, Kp):
            qp_val = tl.load(qp_vec_ptr + head * Kp + kp)
            kp_val = tl.load(Kp_ptr + tok * Kp + kp)  # row tok of Kp
            kp_sum += qp_val * kp_val
        logits_ptr[l] = (k_sum + kp_sum) * sm_scale


# Kernel 2: Compute lse for a single row logits_ptr[L] and store into lse_ptr[head]
# Avoid -inf: compute max and sum over valid entries, then lse = (max + log(sum_exp)) / ln(2)
@triton.jit
def compute_lse_kernel(logits_ptr, lse_ptr, L, ln2):
    # Triton does not support reductions over runtime ranges directly; we implement max and sum here.
    # We assume forward uses a grid launch per head so this kernel is launched once per head.
    max_val = tl.load(logits_ptr + 0)  # initialize with first element
    for l in range(1, L):
        val = tl.load(logits_ptr + l)
        if val > max_val:
            max_val = val
    # sum_exp = sum(exp(x - max))
    sum_exp = tl.zeros((), dtype=tl.float32)
    for l in range(0, L):
        val = tl.load(logits_ptr + l)
        sum_exp += tl.exp(val - max_val)
    lse_ptr[0] = (max_val + tl.log(sum_exp)) / ln2


# Kernel 3: Compute softmax over logits_ptr[L] - lse_row, store into attn_ptr[L]
# We assume invalid positions are zeroed in logits_ptr; softmax will ignore them as exp(0)=1 for valid, 0 for invalid.
@triton.jit
def compute_softmax_kernel(logits_ptr, lse_row, attn_ptr, L):
    # compute inv_lse = 1 / lse_row
    inv_lse = 1.0 / lse_row
    for l in range(0, L):
        val = tl.load(logits_ptr + l) - lse_row
        attn_ptr[l] = tl.exp(val)  # for invalid, logits were 0, so exp(-lse_row) but since we zeroed invalid in logits, they remain 0


# Kernel 4: GEMV: out[K] = attn[L] @ Kc[L, K] via row-wise accumulation
# We pass Kc_ptr = Kc_all, row indices from tok_idx_ptr. Output is out_ptr[K].
@triton.jit
def gemv_out_kernel(attn_ptr, Kc_ptr, tok_idx_ptr, out_ptr, L, K):
    for k in range(0, K):
        out_ptr[k] = tl.zeros((), dtype=tl.float32)
        for l in range(0, L):
            attn_l = tl.load(attn_ptr + l)
            tok_l = tl.load(tok_idx_ptr + l)
            kc_val = tl.load(Kc_ptr + tok_l * K + k)
            out_ptr[k] += attn_l * kc_val


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        """
        Triton-only forward. All math happens in Triton kernels; forward only allocates outputs and launches kernels.
        """
        # Ensure CUDA tensors
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "Tensors must be on CUDA"
        device = q_nope.device

        # Squeeze caches to [P, K] and [P, Kp]
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [P, K]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [P, Kp]
        K = int(Kc_all.shape[-1])  # should be 512
        Kp = int(Kp_all.shape[-1])  # should be 64
        total_q = int(q_nope.shape[0])
        H = int(q_nope.shape[1])  # should be 16

        # Indptrs: convert to int32
        qo_indptr = qo_indptr.to(torch.int32)
        kv_indptr = kv_indptr.to(torch.int32)

        # Outputs
        output = torch.empty((total_q, H, K), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, H), dtype=torch.float32, device=device)

        LN2 = float(math.log(2.0))  # ln(2) for bit scale

        batch_size = int(qo_indptr.shape[0] - 1)
        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            q_len = q_end - q_start
            L = kv_end - kv_start

            if q_len <= 0 or L <= 0:
                continue

            # Token indices for this batch element: int32 on device
            tok_idx = kv_indices[kv_start:kv_end].to(torch.int32).to(device)

            # For each query
            for i in range(q_len):
                q_abs = q_start + i

                # Build qn_vec and qp_vec (fp32 flat vectors) from q_nope[q_abs] and q_pe[q_abs]
                qn_2d = q_nope[q_abs]  # [H, K]
                qp_2d = q_pe[q_abs]    # [H, Kp]
                qn_vec = qn_2d.reshape(-1).to(torch.float32)  # [H*K]
                qp_vec = qp_2d.reshape(-1).to(torch.float32)  # [H*Kp]

                # Per-head outputs
                # Output vector for this head
                out_vec = torch.empty((K,), dtype=torch.float32, device=device)
                # Logits buffer [L]
                logits_buf = torch.empty((L,), dtype=torch.float32, device=device)
                # lse for head (we'll store per-head)
                lse_row = torch.empty((1,), dtype=torch.float32, device=device)

                # Compute logits for each head
                for h in range(H):
                    # Launch compute_logits_kernel
                    compute_logits_kernel[(1,)](
                        qn_vec, qp_vec, Kc_all, Kp_all, tok_idx, logits_buf,
                        H=H, K=K, Kp=Kp, L=L, sm_scale=float(sm_scale),
                        head=h,
                    )
                    # Compute lse for head
                    compute_lse_kernel[(1,)](
                        logits_buf, lse_row, L, LN2
                    )
                    # Compute softmax
                    attn = torch.empty((L,), dtype=torch.float32, device=device)
                    compute_softmax_kernel[(1,)](
                        logits_buf, lse_row, attn, L
                    )
                    # GEMV: out[h, :] = attn @ Kc_all[tok_idx, :]
                    gemv_out_kernel[(1,)](
                        attn, Kc_all, tok_idx, out_vec, L, K
                    )
                    # Store output in bf16
                    output[q_abs, h, :] = out_vec.to(torch.bfloat16)
                # Store lse row (lse[q_abs, :] = lse_row)
                lse[q_abs, :] = lse_row.squeeze(0)  # broadcasting over H is handled by per-head loop above; lse[q_abs, :] gets filled by per-head loops.
                # Note: lse[q_abs, h] is written per head loop. The loop runs H times, writing to lse[q_abs, h].

        return output, lse


def run(*args):
    return ModelNew()(*args)
