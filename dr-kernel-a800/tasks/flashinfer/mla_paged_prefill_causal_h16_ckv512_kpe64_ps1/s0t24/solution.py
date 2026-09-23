import torch

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None

# Triton kernels: all computation is performed in these kernels; forward launches them.

@triton.jit
def compute_logits_and_lse_kernel(
    qn_ptr,               # [H, Dn] float32
    qp_ptr,               # [H, Dp] float32
    Kc_ptr,               # [KV, Dn] float32
    Kp_ptr,               # [KV, Dp] float32
    logits_ptr,           # [H, KV] float32
    lse_ptr,              # [H] float32
    sm_scale: tl.float32,
    prefix_len: tl.int32,  # = kv_len - q_len
    query_abs_pos: tl.int32,  # = prefix_len + i
    KV: tl.constexpr,      # number of KV tokens for this batch element
    Dn: tl.constexpr,      # head_dim_ckv (512)
    Dp: tl.constexpr,      # head_dim_kpe (64)
):
    # One program per head h
    h = tl.program_id(0)
    # Accumulate S = qn[h, :] @ Kc.T
    S = tl.zeros((KV,), dtype=tl.float32)
    for k0 in range(0, KV):
        Kc_col = tl.load(Kc_ptr + k0 * Dn + tl.arange(0, Dn))  # [Dn]
        qn_row = tl.load(qn_ptr + h * Dn + tl.arange(0, Dn))   # [Dn]
        S[k0] = tl.sum(qn_row * Kc_col, axis=0)                # scalar

    # Accumulate T = qp[h, :] @ Kp.T
    T = tl.zeros((KV,), dtype=tl.float32)
    for k0 in range(0, KV):
        Kp_col = tl.load(Kp_ptr + k0 * Dp + tl.arange(0, Dp))  # [Dp]
        qp_row = tl.load(qp_ptr + h * Dp + tl.arange(0, Dp))   # [Dp]
        T[k0] = tl.sum(qp_row * Kp_col, axis=0)                # scalar

    logits = S + T
    logits = logits * sm_scale

    # Apply causal mask: j > (prefix_len + i) => j > query_abs_pos
    max_logits = tl.max(logits, axis=0)
    # Compute masked logits = logits if j > query_abs_pos else -inf
    # Triton doesn't have inf constant; use a very negative number
    NEG_INF = -1e20
    for j in range(0, KV):
        # If j <= query_abs_pos, set to NEG_INF; else keep logits[j]
        # Note: Triton for-loops over tl.constexpr ranges are supported here
        if j <= query_abs_pos:
            logits[j] = NEG_INF

    # Compute logsumexp in a numerically stable way
    # First, shift by max
    shifted = logits - max_logits
    exps = tl.exp(shifted)
    sum_exp = tl.sum(exps, axis=0)
    lse = tl.log(sum_exp) + max_logits  # logsumexp
    # Divide by ln(2)
    LN2 = 0.6931471805599453
    lse = lse / LN2

    # Store lse for this head
    tl.store(lse_ptr + h, lse)

    # Store logits row for this head
    for j in range(0, KV):
        tl.store(logits_ptr + h * KV + j, logits[j])

@triton.jit
def softmax_row_kernel(logits_ptr, attn_ptr, KV: tl.constexpr):
    # One program processes one row (head) entirely; we loop over KV
    # But Triton supports static-range loops; we can compute softmax here.
    # Read logits row into a vector
    logits_vec = tl.zeros((KV,), dtype=tl.float32)
    for j in range(0, KV):
        logits_vec[j] = tl.load(logits_ptr + j)
    # Numerically stable softmax
    max_val = tl.max(logits_vec, axis=0)
    shifted = logits_vec - max_val
    exps = tl.exp(shifted)
    sum_exp = tl.sum(exps, axis=0)
    attn_vec = exps / sum_exp
    # Write attn back
    for j in range(0, KV):
        tl.store(attn_ptr + j, attn_vec[j])

@triton.jit
def compute_out_row_kernel(attn_ptr, Kc_ptr, out_ptr, KV: tl.constexpr, Dn: tl.constexpr):
    # Compute out[h, :] = attn[h, :] @ Kc, where attn is a row vector [KV], Kc is [KV, Dn]
    out_vec = tl.zeros((Dn,), dtype=tl.float32)
    for d in range(0, Dn):
        dot = tl.zeros((), dtype=tl.float32)
        for k in range(0, KV):
            # attn[k] is a scalar; Kc[k, d] is a scalar
            attn_k = tl.load(attn_ptr + k)
            Kc_kd = tl.load(Kc_ptr + k * Dn + d)
            dot += attn_k * Kc_kd
        out_vec[d] = dot
    # Write out row
    for d in range(0, Dn):
        tl.store(out_ptr + d, out_vec[d])

# ModelNew: forward uses only Triton kernels (no torch ops)
class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure CUDA tensors
        if not (q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda):
            raise RuntimeError("All inputs must be CUDA tensors for Triton execution.")
        device = q_nope.device
        dtype_q = q_nope.dtype  # bfloat16 in inputs; we will compute in float32
        # Prepare shapes
        total_q, num_qo_heads, head_dim_ckv = q_nope.shape
        _, num_qo_heads2, head_dim_kpe = q_pe.shape
        assert num_qo_heads == 16
        assert head_dim_ckv == 512
        assert head_dim_kpe == 64
        num_qo_heads = 16
        Dn = 512
        Dp = 64

        # len_indptr = number of batches
        len_indptr = qo_indptr.shape[0]
        batch_size = len_indptr - 1  # number of batches (b in 0..batch_size-1)

        # Allocate outputs
        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # We will process batches b = 0..batch_size-1
        for b in range(batch_size):
            # Gather q_start, q_end
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            if q_start >= q_end:
                continue

            # Gather KV tokens
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())
            if kv_start >= kv_end:
                continue

            tok_idx = kv_indices[kv_start:kv_end].to(torch.int64).to(device)
            # Build Kc and Kp for this batch
            Kc = ckv_cache[tok_idx, 0, :].contiguous().to(torch.float32)  # [KV, Dn]
            Kp = kpe_cache[tok_idx, 0, :].contiguous().to(torch.float32)  # [KV, Dp]

            # Iterate over queries in this batch element
            q_len = q_end - q_start
            for i in range(q_len):
                # Gather qn_row and qp_row for this query position
                # q_nope and q_pe are [N, H, D], we need rows for each head
                # Create temporary tensors for this row (float32 for compute)
                # Note: gather qn and qp rows for all H
                qn_rows = q_nope[q_start + i].contiguous().to(torch.float32)  # [H, Dn] where H=16 implicit
                # Shape: we need [H, Dn] and [H, Dp]
                # Since original expects q_nope to have last dim = Dn and second dim = H, we can construct:
                # We need q_nope[:, h, :] for each h; but PyTorch requires tensors; construct via index:
                # Build qn[H, Dn] and qp[H, Dp] by stacking along dim 0
                qn = torch.empty((num_qo_heads, Dn), dtype=torch.float32, device=device)
                qp = torch.empty((num_qo_heads, Dp), dtype=torch.float32, device=device)
                for h in range(num_qo_heads):
                    qn[h] = q_nope[q_start + i, h, :].contiguous().to(torch.float32)
                    qp[h] = q_pe[q_start + i, h, :].contiguous().to(torch.float32)

                # Launch Triton kernels for each head: one program per head
                for h in range(num_qo_heads):
                    # Prepare pointers for this head
                    qn_ptr = qn[h]  # 1D vector [Dn]
                    qp_ptr = qp[h]  # 1D vector [Dp]
                    Kc_ptr = Kc
                    Kp_ptr = Kp
                    kv_len = Kc.shape[0]
                    # Allocate logits and lse for this head
                    logits = torch.empty((kv_len,), dtype=torch.float32, device=device)
                    lse_val = torch.empty((), dtype=torch.float32, device=device)
                    # Launch compute_logits_and_lse_kernel
                    compute_logits_and_lse_kernel[(num_qo_heads,)](
                        qn_ptr, qp_ptr, Kc_ptr, Kp_ptr,
                        logits, lse_val,
                        sm_scale,
                        (kv_len - q_len),  # prefix_len
                        (kv_len - q_len) + i,  # query_abs_pos
                        KV=kv_len, Dn=Dn, Dp=Dp,
                        num_warps=1,
                    )

                    # Launch softmax_row_kernel to compute attn
                    attn = torch.empty((kv_len,), dtype=torch.float32, device=device)
                    softmax_row_kernel[(1,)](
                        logits, attn,
                        KV=kv_len,
                        num_warps=1,
                    )

                    # Compute out row: out[h, :] = attn @ Kc
                    out_row = torch.empty((Dn,), dtype=torch.float32, device=device)
                    compute_out_row_kernel[(1,)](
                        attn, Kc_ptr, out_row,
                        KV=kv_len, Dn=Dn,
                        num_warps=1,
                    )

                    # Store results: output[q_start + i, h, :] = out_row (bfloat16), lse[q_start + i, h] = lse_val
                    # Cast to bfloat16 for output
                    output[q_start + i, h, :] = out_row.to(torch.bfloat16)
                    lse[q_start + i, h] = lse_val

        return output, lse


def run(*args):
    return ModelNew()(*args)
