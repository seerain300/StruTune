import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Kernel 1: Compute logits = (qn @ Kc.T) + (qp @ Kp.T) for a single (b, h)
# qn_ptr: [Hc] float32, qp_ptr: [Hp] float32
# Kc_ptr: [L, Hc] float32, Kp_ptr: [L, Hp] float32
# out_ptr: [L] float32
# We iterate over tokens l in a 1D range and accumulate scalar dot-products.
@triton.jit
def matmul_add_row_kernel(
    qn_ptr, qp_ptr, Kc_ptr, Kp_ptr, out_ptr,
    L: tl.int32, Hc: tl.int32, Hp: tl.int32, sm_scale: tl.float32
):
    # Process a single head's logits vector over L tokens. This kernel is launched once per (b, h).
    # We will iterate over tokens l and accumulate scalar dot-products.
    # Note: Triton supports loops with runtime bounds, but the simplest and robust way is to
    # use a for-loop over a runtime L; the compilation environment has supported this in prior runs.
    for l in range(0, L):
        # Accumulate (qn @ Kc_col) + (qp @ Kp_col)
        acc = 0.0
        # Reduction over Hc dimension
        for j in range(0, Hc):
            kcj = tl.load(Kc_ptr + l * Hc + j)  # [l, j] element
            acc += qn_ptr[j] * kcj
        # Reduction over Hp dimension
        for j in range(0, Hp):
            kpj = tl.load(Kp_ptr + l * Hp + j)  # [l, j] element
            acc += qp_ptr[j] * kpj
        # Scale
        acc = acc * sm_scale
        # Store
        tl.store(out_ptr + l, acc)


# Kernel 2: Compute lse = logsumexp(logits) / log(2.0) for a single (b, h) row
# logits_ptr: [L] float32
# lse_ptr: [1] float32 (we pass a 1-element tensor to store the result)
@triton.jit
def softmax_logsumexp_row_kernel(
    logits_ptr, lse_ptr, L: tl.int32
):
    # Pass 1: compute max
    max_val = -1e20
    for l in range(0, L):
        v = tl.load(logits_ptr + l)
        max_val = tl.maximum(max_val, v)
    # Pass 2: compute sum of exp(v - max)
    sum_val = 0.0
    for l in range(0, L):
        v = tl.load(logits_ptr + l)
        sum_val += tl.exp(v - max_val)
    lse_val = tl.log(sum_val) / tl.log(2.0)
    # Store lse as a scalar
    tl.store(lse_ptr, lse_val)


# Kernel 3: Compute output_row = softmax(logits_scaled) @ Kc for a single (b, h)
# logits_ptr: [L] float32
# Kc_ptr: [L, Hc] float32
# out_ptr: [Hc] float32
@triton.jit
def matvec_row_kernel(
    logits_ptr, Kc_ptr, out_ptr,
    L: tl.int32, Hc: tl.int32
):
    # Compute softmax(logits) in a numerically stable way and then dot with Kc
    # First pass: compute max
    max_val = -1e20
    for l in range(0, L):
        v = tl.load(logits_ptr + l)
        max_val = tl.maximum(max_val, v)
    # Second pass: compute sum and accumulate output
    sum_val = 0.0
    for l in range(0, L):
        v = tl.load(logits_ptr + l)
        p = tl.exp(v - max_val)
        sum_val += p
    inv_sum = 1.0 / sum_val
    # Third pass: compute out_row
    # We write the entire out vector by iterating over Hc and accumulating over L
    # Note: We cannot directly use tl.dot for a 1D vector, so we loop over L chunks and accumulate into out
    # Create a zero vector
    # Triton does not support initializing tensors like this; we will implement accumulation per column.
    for j in range(0, Hc):
        out_val = 0.0
        for l in range(0, L):
            v = tl.load(logits_ptr + l)
            p = tl.exp(v - max_val) * inv_sum
            kcj = tl.load(Kc_ptr + l * Hc + j)  # [l, j]
            out_val += p * kcj
        tl.store(out_ptr + j, out_val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Extract shapes and device
        device = q_nope.device
        batch_size = q_nope.shape[0]
        num_qo_heads = q_nope.shape[1]
        head_dim_ckv = q_nope.shape[2]
        head_dim_kpe = q_pe.shape[2]

        # Prepare Kc_all and Kp_all by squeezing the segment dimension and casting to float32
        # The caches are shaped [num_pages, 1, H] and we only need the tokens selected by kv_indptr/kv_indices.
        # We will create Kc_all and Kp_all as float32 for numerical stability.
        # Note: ckv_cache.squeeze(1) is a view; make it contiguous.
        Kc_all = ckv_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, Hc]
        Kp_all = kpe_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, Hp]

        # Output buffers
        output = torch.empty((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        # Precompute num_tokens per batch element using kv_indptr
        # kv_indptr has shape [batch_size + 1]
        for b in range(batch_size):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            L_tokens = end - start

            if L_tokens <= 0:
                output[b].zero_()
                lse[b] = -float("inf")
                continue

            # Gather tokens Kc and Kp for this batch element
            tok_idx = kv_indices[start:end]  # 1D indices for tokens in this batch
            # Ensure tok_idx is 1D int64 and gather from Kc_all/Kp_all
            # Kc_all shape: [num_pages, Hc], gather along dim=0 using tok_idx
            Kc = Kc_all[tok_idx].contiguous().to(torch.float32)  # [L_tokens, Hc]
            Kp = Kp_all[tok_idx].contiguous().to(torch.float32)  # [L_tokens, Hp]

            # Prepare qn and qp for this head; we’ll iterate over all heads and launch kernels per (b, h)
            for h in range(num_qo_heads):
                qn = q_nope[b, h].to(torch.float32).contiguous()  # [Hc]
                qp = q_pe[b, h].to(torch.float32).contiguous()   # [Hp]

                # Allocate buffers
                logits = torch.empty(L_tokens, dtype=torch.float32, device=device)

                # Kernel 1: compute logits for this (b, h)
                matmul_add_row_kernel[(1,)](
                    qn, qp, Kc, Kp, logits,
                    L_tokens, head_dim_ckv, head_dim_kpe, float(sm_scale),
                    num_warps=1, num_stages=1
                )

                # Kernel 2: compute lse for this (b, h)
                # lse[b, h] is a 1-element tensor on device
                lse_b_h_ptr = lse[b]  # same tensor, index h not needed because we store scalar
                softmax_logsumexp_row_kernel[(1,)](
                    logits, lse_b_h_ptr,
                    L_tokens,
                    num_warps=1, num_stages=1
                )

                # Kernel 3: compute output[b, h, :]
                out_row = torch.empty(head_dim_ckv, dtype=torch.float32, device=device)
                matvec_row_kernel[(1,)](
                    logits, Kc, out_row,
                    L_tokens, head_dim_ckv,
                    num_warps=1, num_stages=1
                )
                output[b, h, :] = out_row

        # Return output in bfloat16, lse in float32 (matching original)
        return output.to(torch.bfloat16), lse


# Helpers to match the original interface
def get_inputs():
    # Create random inputs on CUDA
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16, device='cuda')
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16, device='cuda')
    ckv_cache = torch.randn([989669, 1, 512], dtype=torch.bfloat16, device='cuda')
    kpe_cache = torch.randn([989669, 1, 64], dtype=torch.bfloat16, device='cuda')
    _n = 1; _t = 8
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32).to(device='cuda')
    kv_indices = torch.randint(0, 989669, [8], dtype=torch.int32).to(device='cuda')
    sm_scale = 1.0  # float32 scalar
    return [q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale]


def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6):
    return ModelNew().forward(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6)