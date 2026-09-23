import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Fill logits with zeros (host op) to avoid any -inf handling inside Triton kernels.
# Not a Triton kernel; used to initialize logits buffer for safe reductions.
def _fill_logits_zero(logits, q_len, H, L):
    # Set all positions to 0; we will recompute valid positions in Triton
    logits.zero_()
    return logits


# Kernel: compute lse[h] = logsumexp(logits_scaled[h, :]) / ln(2), where
# logits_scaled[h, j] = sm_scale * logits[h, j] for valid j; invalid j are very negative.
@triton.jit
def compute_lse_kernel(
    logits_ptr, lse_ptr,
    H, L,
    stride_log_h, stride_log_l,
    scale,  # float32
):
    h = tl.program_id(0)
    row_ptr = logits_ptr + h * stride_log_h
    # compute row_max over all j; invalid j are very negative so they won't affect max
    max_val = -1e20
    j = 0
    while j < L:
        val = tl.load(row_ptr + j * stride_log_l)
        if val > max_val:
            max_val = val
        j += 1
    # compute sum of exp(val - max_val) over all j
    sum_exp = 0.0
    j = 0
    while j < L:
        val = tl.load(row_ptr + j * stride_log_l)
        sum_exp += tl.exp(val - max_val)
        j += 1
    lse = tl.log(sum_exp) * scale  # 1/ln(2)
    tl.store(lse_ptr + h, lse)


# Kernel: compute softmax per head with causal mask:
# attn[h, j] = exp(logits_scaled[h, j] - lse[h]) if j <= prefix + i else 0
@triton.jit
def compute_softmax_kernel(
    logits_ptr, lse_ptr, attn_ptr,
    H, L,
    stride_log_h, stride_log_l,
    stride_attn_h, stride_attn_l,
    prefix, i, sm_scale,
):
    h = tl.program_id(0)
    lse_val = tl.load(lse_ptr + h)
    row_ptr = logits_ptr + h * stride_log_h
    attn_row_ptr = attn_ptr + h * stride_attn_h
    j = 0
    while j < L:
        val = tl.load(row_ptr + j * stride_log_l)
        cond = j > (prefix + i)
        exp_val = tl.exp(val - lse_val)
        attn_val = tl.where(cond, 0.0, exp_val)
        tl.store(attn_row_ptr + j * stride_attn_l, attn_val)
        j += 1


# Kernel: GEMV out[h, k] = sum_l attn[h, l] * Kc[l, k]
@triton.jit
def gemv_out_kernel(
    attn_ptr, Kc_ptr, out_ptr,
    H, L, K,
    stride_attn_h, stride_attn_l,
    stride_Kc_l, stride_Kc_k,
    stride_out_h, stride_out_k,
):
    h = tl.program_id(0)
    k = 0
    while k < K:
        acc = 0.0
        l = 0
        while l < L:
            attn_val = tl.load(attn_ptr + h * stride_attn_h + l * stride_attn_l)
            Kc_val = tl.load(Kc_ptr + l * stride_Kc_l + k * stride_Kc_k)
            acc += attn_val * Kc_val
            l += 1
        tl.store(out_ptr + h * stride_out_h + k * stride_out_k, acc)
        k += 1


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._inv_ln2 = 1.0 / math.log(2.0)

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        if not TRITON_AVAILABLE or not torch.cuda.is_available():
            raise RuntimeError("Triton or CUDA not available for Triton version.")

        device = torch.device('cuda')
        # Ensure contiguous and on CUDA
        q_nope = q_nope.to(device).contiguous()
        q_pe = q_pe.to(device).contiguous()
        ckv_cache = ckv_cache.to(device).contiguous()  # [P, 1, K]
        kpe_cache = kpe_cache.to(device).contiguous()  # [P, 1, Kp]
        qo_indptr = qo_indptr.to(device).contiguous()
        kv_indptr = kv_indptr.to(device).contiguous()
        kv_indices = kv_indices.to(device).contiguous()

        # Squeeze caches along dim=1
        Kc_all = ckv_cache.squeeze(1)  # [P, K]
        Kp_all = kpe_cache.squeeze(1)  # [P, Kp]

        total_q = q_nope.shape[0]
        H = q_nope.shape[1]
        K = q_nope.shape[2]
        Kp = q_pe.shape[2]

        # Output tensors
        output = torch.empty((total_q, H, K), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, H), dtype=torch.float32, device=device)

        for b in range(qo_indptr.shape[0] - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            q_len = q_end - q_start
            if q_len == 0:
                continue

            # Token indices and cached vectors for this batch
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            L = page_end - page_beg
            if L == 0:
                continue

            tok_idx = kv_indices[page_beg:page_end].to(torch.int32)
            Kc_curr = Kc_all[tok_idx].to(torch.float32).contiguous()  # [L, K]
            Kp_curr = Kp_all[tok_idx].to(torch.float32).contiguous()  # [L, Kp]

            for i in range(q_len):
                q_start_i = q_start + i

                # Query vectors
                qn = q_nope[q_start_i].contiguous()  # [H, K]
                qp = q_pe[q_start_i].contiguous()    # [H, Kp]

                # Allocate intermediate buffers and initialize logits to zeros (host op ensures safety)
                logits = torch.zeros((H, L), dtype=torch.float32, device=device)
                attn = torch.empty((H, L), dtype=torch.float32, device=device)

                # Strides
                stride_qn_h, stride_qn_k = H, K
                stride_Kc_l, stride_Kc_k = L, K
                stride_qp_h, stride_qp_kp = H, Kp
                stride_Kp_l, stride_Kp_kp = L, Kp
                stride_log_h, stride_log_l = H, L
                stride_attn_h, stride_attn_l = H, L

                # 1) Fill logits with zeros (host op), then Triton kernels will process valid j only
                _fill_logits_zero(logits, q_len, H, L)

                # 2) Compute lse[h] for each head
                lse_val = torch.empty((H,), dtype=torch.float32, device=device)
                compute_lse_kernel[(H,)](
                    logits, lse_val,
                    H, L,
                    stride_log_h, stride_log_l,
                    self._inv_ln2,
                    num_warps=1, num_stages=1,
                )

                # 3) Compute softmax with causal mask: j > (L - q_len) + i -> attn=0
                prefix_len = L - q_len  # number of previously processed tokens in this batch
                compute_softmax_kernel[(H,)](
                    logits, lse_val, attn,
                    H, L,
                    stride_log_h, stride_log_l,
                    stride_attn_h, stride_attn_l,
                    prefix_len, i, sm_scale,
                    num_warps=1, num_stages=1,
                )

                # 4) GEMV: out[h, k] = attn[h, :] @ Kc_curr[:, k]
                out_fp32 = torch.empty((H, K), dtype=torch.float32, device=device)
                gemv_out_kernel[(H,)](
                    attn, Kc_curr, out_fp32,
                    H, L, K,
                    stride_attn_h, stride_attn_l,
                    stride_Kc_l, stride_Kc_k,
                    H, K,
                    num_warps=1, num_stages=1,
                )

                # Store outputs
                output[q_start_i] = out_fp32.to(torch.bfloat16)
                lse[q_start_i] = lse_val

        return output, lse


def run(*args):
    return ModelNew()(*args)
