import torch
import triton
import triton.language as tl


# Kernel 1: Compute logits[h, l] = sum_k qn[h, k] * Kc_all[tok_idx[l], k] + sum_kp qp[h, k'] * Kp_all[tok_idx[l], k']
# Launch: one program per head h
@triton.jit
def compute_logits_kernel(
    q_nope_ptr, q_pe_ptr, Kc_all_ptr, Kp_all_ptr,
    logits_ptr,
    q_start,  # int32 scalar q_start
    H, K, Kp, L,  # int32 scalars (compile-time for loop bounds)
    sm_scale,
    stride_qn_h, stride_qn_k,
    stride_qp_h, stride_qp_kp,
    stride_Kc_p, stride_Kc_k,
    stride_Kp_p, stride_Kp_kp,
    stride_log_h, stride_log_l,
    tok_idx_ptr,  # int32, length L
    head: tl.constexpr,  # head index for this program
):
    # Compute qn_vec and qp_vec
    qn_vec = tl.zeros((K,), dtype=tl.float32)
    qp_vec = tl.zeros((Kp,), dtype=tl.float32)
    k = 0
    while k < K:
        qn_vec[k] = tl.load(q_nope_ptr + (head * stride_qn_h) + k * stride_qn_k)
        k += 1
    kp = 0
    while kp < Kp:
        qp_vec[kp] = tl.load(q_pe_ptr + (head * stride_qp_h) + kp * stride_qp_kp)
        kp += 1

    # Loop over L (compile-time)
    l = 0
    while l < L:
        idx_l = tl.load(tok_idx_ptr + l)
        # Load Kc_row and Kp_row
        Kc_row = tl.zeros((K,), dtype=tl.float32)
        Kp_row = tl.zeros((Kp,), dtype=tl.float32)
        k = 0
        while k < K:
            Kc_row[k] = tl.load(Kc_all_ptr + idx_l * stride_Kc_p + k * stride_Kc_k)
            k += 1
        kp = 0
        while kp < Kp:
            Kp_row[kp] = tl.load(Kp_all_ptr + idx_l * stride_Kp_p + kp * stride_Kp_kp)
            kp += 1

        dot1 = 0.0
        dot2 = 0.0
        for k in range(K):
            dot1 += qn_vec[k] * Kc_row[k]
        for kp in range(Kp):
            dot2 += qp_vec[kp] * Kp_row[kp]

        logit = (dot1 + dot2) * sm_scale
        tl.store(logits_ptr + head * stride_log_h + l * stride_log_l, logit)
        l += 1


# Kernel 2: Compute logsumexp per head with causal mask and base-2 scaling
# Inputs: logits[H, L], write lse[H]
@triton.jit
def compute_lse_kernel(
    logits_ptr, lse_ptr,
    H, L,
    stride_log_h, stride_log_l,
    prefix_len,  # int scalar = L - q_len
    head: tl.constexpr,
):
    max_val = -float("inf")
    # Only valid positions j <= prefix_len
    j = 0
    while j < L:
        if j <= prefix_len:
            val = tl.load(logits_ptr + head * stride_log_h + j * stride_log_l)
            max_val = tl.maximum(max_val, val)
        j += 1

    sum_exp = 0.0
    j = 0
    while j < L:
        val = tl.load(logits_ptr + head * stride_log_h + j * stride_log_l)
        if j <= prefix_len:
            sum_exp += tl.exp(val - max_val)
        j += 1

    lse_val = max_val + tl.log(sum_exp)
    ln2 = 0.6931471805599453
    tl.store(lse_ptr + head, lse_val / ln2)


# Kernel 3: Compute softmax per head with causal mask; write attn[H, L]
@triton.jit
def compute_softmax_kernel(
    logits_ptr, lse_ptr, attn_ptr,
    H, L,
    stride_log_h, stride_log_l,
    stride_attn_h, stride_attn_l,
    prefix_len,  # int scalar = L - q_len
    head: tl.constexpr,
):
    lse_val = tl.load(lse_ptr + head)
    l = 0
    while l < L:
        val = tl.load(logits_ptr + head * stride_log_h + l * stride_log_l)
        if l <= prefix_len:
            soft = tl.exp(val - lse_val)
        else:
            soft = 0.0
        tl.store(attn_ptr + head * stride_attn_h + l * stride_attn_l, soft)
        l += 1


# Kernel 4: GEMV out[h, k] = sum_l attn[h, l] * Kc_all[tok_idx[l], k]
@triton.jit
def gemv_out_kernel(
    attn_ptr, Kc_all_ptr, out_ptr,
    H, L, K,
    stride_attn_h, stride_attn_l,
    stride_Kc_p, stride_Kc_k,
    stride_out_h, stride_out_k,
    tok_idx_ptr,  # int32, length L
    head: tl.constexpr,
):
    k = 0
    acc = tl.zeros((), dtype=tl.float32)
    while k < K:
        l = 0
        while l < L:
            idx_l = tl.load(tok_idx_ptr + l)
            attn_val = tl.load(attn_ptr + head * stride_attn_h + l * stride_attn_l)
            kc_val = tl.load(Kc_all_ptr + idx_l * stride_Kc_p + k * stride_Kc_k)
            acc += attn_val * kc_val
            l += 1
        tl.store(out_ptr + head * stride_out_h + k * stride_out_k, acc)
        k += 1


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure Triton/CUDA availability
        if not triton.runtime.driver.active or not torch.cuda.is_available():
            raise RuntimeError("Triton/CUDA required for ModelNew.")

        # Move everything to CUDA
        device = torch.device("cuda")
        q


def run(*args):
    return ModelNew()(*args)
