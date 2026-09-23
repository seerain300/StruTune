import math
import torch
import triton
import triton.language as tl


# -----------------------
# Triton Kernels
# -----------------------

# Kernel to compute logits[i, h, j] for a given segment b.
# Grid is (B, Q, H, K). Each program handles one (i, h, j) within segment b.
# Masks ensure we only compute/store for (i < qo_len[b]) and (j < kv_len[b]).
@triton.jit
def compute_logits_segments_kernel(
    q_ptr, k_ptr, logits_ptr,
    Q, K,
    sm_scale,  # scalar float
    qo_len_ptr, kv_len_ptr,
    B: tl.constexpr,  # number of segments
    head_dim: tl.constexpr
):
    b = tl.program_id(0)
    i = tl.program_id(1)
    h = tl.program_id(2)
    j = tl.program_id(3)

    # Load segment lengths (assume qo_len_ptr and kv_len_ptr length == B)
    qo_len_b = tl.load(qo_len_ptr + b)
    kv_len_b = tl.load(kv_len_ptr + b)

    # Validity masks for this segment
    valid_i = i < qo_len_b
    valid_j = j < kv_len_b
    if not (valid_i and valid_j):
        return

    # Compute dot product over head_dim and store
    dot = 0.0
    for d in tl.static_range(0, head_dim):
        # q[i, h, d] address: i * (H * head_dim) + h * head_dim + d
        qd = tl.load(q_ptr + i * (32 * head_dim) + h * head_dim + d)
        # k[j, h, d] address: j * (8 * head_dim) + h * head_dim + d
        kd = tl.load(k_ptr + j * (8 * head_dim) + h * head_dim + d)
        dot += qd * kd
    logits_val = dot * sm_scale

    # Store to logits at [i, h, j] for segment b
    # We store into a single [Q_total, H, K] tensor, but we rely on host to
    # precompute segment offsets or to write only within valid segment ranges.
    # Here, we compute base offset assuming linear layout: (i*H+ h)*K + j
    # Note: i here is the global index within the whole q; the above validity masks ensure
    # we only write where i belongs to segment b.
    out_index = (i * H + h) * K + j
    tl.store(logits_ptr + out_index, logits_val)


# Kernel to compute lse[i, h] = logsumexp(logits[i, h, :]) / ln(2) for all segments.
# Grid is (B, Q, H). We perform two passes: max, then sum-exp. Per (b, i, h), iterate over j.
@triton.jit
def lse_segments_kernel(
    logits_ptr, lse_ptr,
    Q, K,
    qo_len_ptr,  # length B
    B: tl.constexpr
):
    b = tl.program_id(0)
    i = tl.program_id(1)
    h = tl.program_id(2)

    # Load segment length
    qo_len_b = tl.load(qo_len_ptr + b)
    # i must be within segment, but lse is computed per i,h regardless of segment since
    # logits_ptr stores per i,h across all j; here we assume i < total_q (host ensures that).
    # If we need per-segment lse, we only care when i < qo_len_b; however, we compute for all i.
    # So we proceed with i in [0, Q).

    max_val = -float('inf')
    for j in tl.static_range(0, K):
        val = tl.load(logits_ptr + (i * H + h) * K + j)
        if val > max_val:
            max_val = val

    sum_exp = 0.0
    for j in tl.static_range(0, K):
        val = tl.load(logits_ptr + (i * H + h) * K + j)
        sum_exp += tl.exp(val - max_val)

    lse_ih = tl.log(sum_exp) + max_val
    # Divide by ln(2)
    ln2 = 0.6931471805599453  # math.log(2.0)
    lse_ih = lse_ih / ln2
    tl.store(lse_ptr + i * H + h, lse_ih)


# Kernel to compute final output[i, h, :] for each segment b.
# Grid is (B, Q, H). We loop over K and accumulate dot products over head_dim.
# output_ptr is [Q_total, H, head_dim] and we write only for valid i in each segment.
@triton.jit
def output_segments_kernel(
    logits_ptr, v_ptr, lse_ptr, output_ptr,
    Q, K, head_dim: tl.constexpr,
    qo_len_ptr,  # length B
    B: tl.constexpr
):
    b = tl.program_id(0)
    i = tl.program_id(1)
    h = tl.program_id(2)

    qo_len_b = tl.load(qo_len_ptr + b)

    if i >= qo_len_b:
        return

    # Load lse[i, h]
    lse_ih = tl.load(lse_ptr + i * H + h)
    acc_vec = tl.zeros((head_dim,), dtype=tl.float32)

    # Accumulate over j
    for j in tl.static_range(0, K):
        val = tl.load(logits_ptr + (i * H + h) * K + j)
        w = tl.exp(val - lse_ih) / 0.6931471805599453  # 1/ln(2)

        # v_ptr layout: [K, 8, head_dim] contiguous
        # For v_expanded, k_idx = j, head_exp = h % 8
        h_exp = h % 8
        jkv = j  # since we did not repeat keys (keys length is K), this assumes K == original kv count.
        v_base = jkv * (8 * head_dim) + h_exp * head_dim
        # Sum over head_dim
        sum_v = 0.0
        for d in tl.static_range(0, head_dim):
            vd = tl.load(v_ptr + v_base + d)
            sum_v += w * vd

        # Store into output[i, h, :]
        out_base = i * H * head_dim + h * head_dim
        for d in tl.static_range(0, head_dim):
            tl.store(output_ptr + out_base + d, sum_v)

    # Note: This kernel writes the entire output[i, h, :] vector for all i within segment b.
    # In practice, we should avoid overwriting by masking, but since output_ptr is pre-zeroed,
    # writing only for valid i ensures correctness. However, better approach is to compute
    # per i inside the kernel and store only for valid i. To be precise, we'll return early if i >= qo_len_b.
    # The above 'return' already handles that. Now we store sum_v for all head_dim at the given i,h.
    # Since we wrote for all d in head_dim, we ensure vectorized store above is correct.

    # Replacing the above with a vectorized store: precompute sum_v across head_dim and write once:
    # We'll instead compute per d:
    # Implementing vectorized store above ensures correctness. The code above writes sum_v to all d,
    # which is intended behavior for output[i,h,d] across d.

# NOTE: The above kernel currently writes the same sum_v to all d, which is incorrect.
# We need to compute and store per d. We'll fix that below using a simplified version:
# Replace the last block with a per-d loop to ensure correctness.

# Updated output kernel: write per d
@triton.jit
def output_segments_kernel_v2(
    logits_ptr, v_ptr, lse_ptr, output_ptr,
    Q, K, head_dim: tl.constexpr,
    qo_len_ptr,  # length B
    B: tl.constexpr
):
    b = tl.program_id(0)
    i = tl.program_id(1)
    h = tl.program_id(2)

    qo_len_b = tl.load(qo_len_ptr + b)

    if i >= qo_len_b:
        return

    lse_ih = tl.load(lse_ptr + i * H + h)
    # For each d, compute the same sum (this is not ideal; see below for a better approach).
    # Better approach: compute per d by looping j and accumulating v contributions.
    # To match original, we need output[i,h,:] = sum_j exp(logits[i,h,j] - lse)*v_expanded[j,h,:]/ln(2).
    # We'll implement j-loop and d-loop explicitly.

    ln2_inv = 1.0 / 0.6931471805599453
    for d in tl.static_range(0, head_dim):
        sum_d = 0.0
        for j in tl.static_range(0, K):
            val = tl.load(logits_ptr + (i * H + h) * K + j)
            w = tl.exp(val - lse_ih) * ln2_inv
            # v_expanded[j,h,d] = v[j, h%8, d]
            h_exp = h % 8
            v_base = j * (8 * head_dim) + h_exp * head_dim
            vd = tl.load(v_ptr + v_base + d)
            sum_d += w * vd
        out_base = i * H * head_dim + h * head_dim
        tl.store(output_ptr + out_base + d, sum_d)


# -----------------------
# ModelNew: Triton forward
# -----------------------

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        """
        q: [total_q, 32, 128], k: [total_kv, 8, 128], v: [total_kv, 8, 128]
        qo_indptr: [B+1], kv_indptr: [B+1]
        Returns:
        - output: [total_q, 32, 128], dtype=bfloat16
        - lse: [total_q, 32], dtype=float32
        """

        device = q.device
        total_q = int(qo_indptr[-1].item())
        total_kv = int(kv_indptr[-1].item())
        B = qo_indptr.shape[0] - 1
        assert kv_indptr.shape[0] - 1 == B, "qo_indptr and kv_indptr length mismatch"

        # Precompute segment lengths (global indices)
        qo_len = (qo_indptr[1:] - qo_indptr[:-1]).to(device)  # [B]
        kv_len = (kv_indptr[1:] - kv_indptr[:-1]).to(device)  # [B]

        # Prepare q, k, v slices: we will compute per segment. For Triton, we keep full tensors and
        # use masks based on qo_len and kv_len.
        # Cast to float32 for compute
        q_f32 = q.to(torch.float32).contiguous()  # [total_q, 32, 128]
        k_f32 = k.to(torch.float32).contiguous()  # [total_kv, 8, 128]
        v_f32 = v.to(torch.float32).contiguous()  # [total_kv, 8, 128]

        # Allocate outputs (float32 compute, then cast)
        logits = torch.empty((total_q, 32, total_kv), dtype=torch.float32, device=device)  # [Q_total, 32, K_total]
        lse = torch.empty((total_q, 32), dtype=torch.float32, device=device)               # [Q_total, 32]
        output = torch.empty((total_q, 32, 128), dtype=torch.float32, device=device)       # [Q_total, 32, 128]

        # Launch Triton kernels per segment
        # Note: Triton does not support dynamic loops; grid must cover all segments.
        # We use grid (B, Q, 32, total_kv) for logits; (B, Q, 32) for lse; (B, Q, 32) for output.
        head_dim = 128  # compile-time constant

        # 1) Compute logits per segment
        grid_logits = (B, total_q, 32, total_kv)
        compute_logits_segments_kernel[grid_logits](
            q_f32, k_f32, logits,
            total_q, total_kv,
            sm_scale, qo_len, kv_len,
            B=B, head_dim=head_dim,
            num_warps=4, num_stages=2
        )

        # 2) Compute lse per (i, h). We iterate over i in


def run(*args):
    return ModelNew()(*args)
