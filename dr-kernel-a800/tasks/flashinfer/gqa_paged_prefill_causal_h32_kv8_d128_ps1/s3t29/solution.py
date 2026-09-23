import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Kernel A: compute logits = q_vec @ k_rows.T, q_vec: [HEAD_DIM], k_rows: [NUM_KV, HEAD_DIM] -> logits: [NUM_KV]
@triton.jit
def dot_logits_kernel(q_vec, k_rows, logits, HEAD_DIM: tl.constexpr, NUM_KV: tl.constexpr):
    for j in tl.static_range(NUM_KV):
        k_row = k_rows[j]  # [HEAD_DIM]
        acc = 0.0
        for d in tl.static_range(HEAD_DIM):
            acc += q_vec[d] * k_row[d]
        logits[j] = acc


# Kernel B: compute lse_scaled = logsumexp(x) / ln(2), where x = logits * sm_scale
# Stable two-pass: max, then sum(exp(. - max)), lse = log(sumexp) + max
@triton.jit
def logsumexp_two_pass_kernel(vec, sm_scale, lse_out, NUM: tl.constexpr, VEC_SIZE: tl.constexpr):
    max_val = -float("inf")
    for i in tl.static_range(NUM):
        max_val = tl.maximum(max_val, vec[i] * sm_scale)
    sumexp = 0.0
    for i in tl.static_range(NUM):
        sumexp += tl.exp((vec[i] - max_val) * sm_scale)
    ln2 = 0.6931471805599453
    lse_val = tl.log(sumexp) + max_val
    lse_val = lse_val / ln2
    lse_out[0] = lse_val


# Fused kernel C: compute softmax(logits_scaled) and output_vec = softmax @ v_rows in one pass
# Inputs:
# - logits: [NUM_KV] (unnormalized), sm_scale
# - v_rows: [NUM_KV, HEAD_DIM]
# Outputs:
# - softmax: [NUM_KV] (will be stored into a 1-element tensor per kernel invocation)
# - out_vec: [HEAD_DIM]
@triton.jit
def softmax_matvec_fused_kernel(logits, sm_scale, v_rows, out_vec, softmax, HEAD_DIM: tl.constexpr, NUM: tl.constexpr, stride_v: tl.constexpr):
    # Compute sumexp = sum(exp((logits[i] * sm_scale)))
    sumexp = 0.0
    for i in tl.static_range(NUM):
        sumexp += tl.exp(logits[i] * sm_scale)

    # Compute softmax values and write to softmax[0..NUM-1]
    for i in tl.static_range(NUM):
        soft_i = tl.exp(logits[i] * sm_scale) / sumexp
        softmax[i] = soft_i

    # Compute out_vec = softmax @ v_rows (elementwise multiply and reduce along NUM)
    for d in tl.static_range(HEAD_DIM):
        acc = 0.0
        for i in tl.static_range(NUM):
            val = v_rows[i, d]
            acc += softmax[i] * val
        out_vec[d] = acc


class ModelNew(torch.nn.Module):
    def __init__(self, head_dim=128, num_qo_heads=32, num_kv_heads=8, sm_scale=None):
        super().__init__()
        self.HEAD_DIM = head_dim
        self.num_qo_heads = num_qo_heads
        self.num_kv_heads = num_kv_heads
        self.gqa_ratio = num_qo_heads // num_kv_heads
        if sm_scale is None:
            self.sm_scale = 1.0 / math.sqrt(head_dim)
        else:
            self.sm_scale = sm_scale

    def forward(self, q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale=None):
        device = q.device
        if sm_scale is None:
            sm_scale = self.sm_scale

        # Shapes and checks
        total_q, num_qo_heads, head_dim = q.shape
        num_pages, p, num_kv_heads, _ = k_cache.shape
        assert p == 1, "k_cache second dim must be 1"
        assert num_qo_heads == 32
        assert num_kv_heads == 8
        assert head_dim == 128
        assert qo_indptr.dim() == 1 and kv_indptr.dim() == 1
        assert kv_indices.dim() == 1
        assert total_q == qo_indptr[-1].item(), "qo_indptr[-1] must equal total_q"

        # Output buffers
        output = torch.empty((total_q, num_qo_heads, head_dim), dtype=torch.float32, device=device)
        lse = torch.full((total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

        # Flatten k_cache, v_cache along the middle dimension
        k_cache_flat = k_cache.squeeze(1)  # [num_pages, 8, 128]
        v_cache_flat = v_cache.squeeze(1)  # [num_pages, 8, 128]

        len_indptr = qo_indptr.shape[0]
        for b in range(len_indptr - 1):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end:
                continue
            if kv_start >= kv_end:
                continue

            kv_ids = kv_indices[kv_start:kv_end].contiguous()  # [num_segments_b]
            num_q_tokens = q_end - q_start
            num_segments_b = kv_ids.shape[0]

            # Process each query token
            q_batch = q[q_start:q_end]  # [num_q_tokens, 32, 128]
            for q_idx in range(num_q_tokens):
                global_q_idx = q_start + q_idx

                # causal-like bound
                delta = num_segments_b - num_q_tokens
                max_kv_idx = min(q_idx + 1 + delta, num_segments_b)
                if max_kv_idx <= 0:
                    continue

                # Iterate query heads
                for h in range(self.num_qo_heads):
                    kv_head = h // self.gqa_ratio

                    # Load q_vec [128] as fp32
                    q_vec = q_batch[q_idx, h, :].to(torch.float32)  # [128]

                    # Load K/V rows for kv_head for this segment
                    k_rows = k_cache_flat[kv_ids[:max_kv_idx], kv_head]  # [max_kv_idx, 128], fp32
                    v_rows = v_cache_flat[kv_ids[:max_kv_idx], kv_head]  # [max_kv_idx, 128], fp32

                    # Compute logits: [max_kv_idx]
                    logits = torch.empty((max_kv_idx,), dtype=torch.float32, device=device)
                    dot_logits_kernel[(1,)](
                        q_vec, k_rows, logits,
                        HEAD_DIM=self.HEAD_DIM, NUM_KV=max_kv_idx,
                    )

                    # Compute logsumexp of scaled logits
                    vec_pad = torch.empty((self.HEAD_DIM,), dtype=torch.float32, device=device)
                    vec_pad[:max_kv_idx] = logits
                    lse_pos = torch.empty((1,), dtype=torch.float32, device=device)
                    logsumexp_two_pass_kernel[(1,)](
                        vec_pad, sm_scale, lse_pos,
                        NUM=max_kv_idx, VEC_SIZE=self.HEAD_DIM,
                    )
                    lse[global_q_idx, h] = lse_pos[0]

                    # Fused softmax and matvec: out_vec, and softmax vector for possible future use
                    out_vec = torch.empty((self.HEAD_DIM,), dtype=torch.float32, device=device)
                    softmax = torch.empty((max_kv_idx,), dtype=torch.float32, device=device)
                    softmax_matvec_fused_kernel[(1,)](
                        logits, sm_scale, v_rows, out_vec, softmax,
                        HEAD_DIM=self.HEAD_DIM, NUM=max_kv_idx, stride_v=self.HEAD_DIM,
                    )

                    # Store output
                    output[global_q_idx, h] = out_vec

        return output, lse


def run(*args):
    return ModelNew()(*args)
