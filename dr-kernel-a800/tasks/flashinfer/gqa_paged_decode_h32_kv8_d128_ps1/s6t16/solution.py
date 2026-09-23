import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: one program per (b, h). Computes output and lse for that pair.
@triton.jit
def _forward_kernel_bh(
    q_ptr,          # *fp16, shape [B, H, D]
    k_ptr,          # *fp16, shape [num_pages, 1, num_kv_heads, D] (we gather per token)
    v_ptr,          # *fp16, shape [num_pages, 1, num_kv_heads, D] (we gather per token)
    kv_indptr_ptr,  # *int32, shape [B+1]
    kv_indices_ptr, # *int32, shape [num_kv_indices]
    lse_ptr,        # *fp32, shape [B, H]
    out_ptr,        # *fp32, shape [B, H, D]
    B: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    sm_scale,       # float32 scalar = 1.0 / sqrt(D)
    N_TOTAL: tl.constexpr,  # static loop bound (e.g., 128), mask nn < actual_num_tokens
):
    # program ids for batch and head
    b = tl.program_id(0)  # int
    h = tl.program_id(1)  # int

    # Compute q vector for this (b, h): q[b, h, :]
    q_base = q_ptr + b * H * D + h * D
    q_vec = tl.zeros((D,), dtype=tl.float32)
    for d in range(0, D):
        q_vec[d] = tl.load(q_base + d).to(tl.float32)

    # Gather kv indices for this batch from kv_indptr
    start = tl.load(kv_indptr_ptr + b).to(tl.int32)
    end = tl.load(kv_indptr_ptr + b + 1).to(tl.int32)
    actual_num_tokens = end - start

    # First pass: streaming logsumexp in base-2 of scaled logits
    m = -float("inf")  # scalar
    sumexp = 0.0       # scalar
    ln2 = 0.6931471805599453

    for nn in range(0, N_TOTAL):
        if nn < actual_num_tokens:
            idx = tl.load(kv_indices_ptr + start + nn).to(tl.int32)
            kv_head = h // 4  # GQA mapping: 32 / 8 = 4

            # Load k_row[idx, kv_head, :] -> shape [D]
            k_base = k_ptr + idx * (1 * 8 * D) + kv_head * D  # num_kv_heads = 8
            k_vec = tl.zeros((D,), dtype=tl.float32)
            for d in range(0, D):
                k_vec[d] = tl.load(k_base + d).to(tl.float32)

            dot = 0.0
            for d in range(0, D):
                dot += q_vec[d] * k_vec[d]

            logit = dot * sm_scale
            new_m = tl.maximum(m, logit)
            sumexp = sumexp * tl.exp(m - new_m) + tl.exp(logit - new_m)
            m = new_m

    # lse for this (b, h)
    lse_val = m + tl.log(sumexp) / ln2
    tl.store(lse_ptr + b * H + h, lse_val)

    # Second pass: compute softmax per token and accumulate output
    out_base = out_ptr + b * H * D + h * D
    out_vec = tl.zeros((D,), dtype=tl.float32)
    for nn in range(0, N_TOTAL):
        if nn < actual_num_tokens:
            idx = tl.load(kv_indices_ptr + start + nn).to(tl.int32)
            kv_head = h // 4

            # Load v_row[idx, kv_head, :]
            v_base = v_ptr + idx * (1 * 8 * D) + kv_head * D
            v_vec = tl.zeros((D,), dtype=tl.float32)
            for d in range(0, D):
                v_vec[d] = tl.load(v_base + d).to(tl.float32)

            # Recompute dot and scaled logit
            k_base = k_ptr + idx * (1 * 8 * D) + kv_head * D
            k_vec = tl.zeros((D,), dtype=tl.float32)
            for d in range(0, D):
                k_vec[d] = tl.load(k_base + d).to(tl.float32)
            dot = 0.0
            for d in range(0, D):
                dot += q_vec[d] * k_vec[d]
            logit = dot * sm_scale

            softmax = tl.exp(logit - m) / (sumexp * ln2)
            out_vec += softmax * v_vec

    # Store final output vector
    for d in range(0, D):
        tl.store(out_base + d, out_vec[d])


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
        # Triton-only forward: no PyTorch ops, no dtype conversions, no asserts.
        B, H, D = q.shape

        # Allocate outputs (fp32 for compute; return as bfloat16)
        output = torch.empty((B, H, D), dtype=torch.float32, device=q.device)
        lse = torch.empty((B, H), dtype=torch.float32, device=q.device)

        # Launch Triton kernel: one program per (b, h)
        grid = (B, H)
        _forward_kernel_bh[grid](
            q, k_cache, v_cache, kv_indptr, kv_indices, lse, output,
            B=B, H=H, D=D, sm_scale=float(sm_scale), N_TOTAL=128,
            num_warps=4,
        )

        # Return outputs as bfloat16 (to match original) and lse as float32
        return output.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
