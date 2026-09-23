import math
import torch
import triton
import triton.language as tl


@triton.jit
def _compute_logits_scaled_kernel(
    qn_ptr,        # *f32, [H, CK] flattened
    qp_ptr,        # *f32, [H, KP] flattened
    Kc_ptr,        # *f32, [L_tokens, CK] flattened
    Kp_ptr,        # *f32, [L_tokens, KP] flattened
    tok_idx_ptr,   # *i32, [L_tokens] (used to gather tokens if needed)
    logits_ptr,    # *f32, [H, L_tokens] flattened
    sm_scale: tl.constexpr,   # float
    H: tl.constexpr,          # int
    CK: tl.constexpr,         # int
    KP: tl.constexpr,         # int
    L_tokens: tl.constexpr,   # int
):
    # 2D grid over heads and tokens
    h = tl.program_id(0)
    t = tl.program_id(1)
    if (h >= H) or (t >= L_tokens):
        return

    # Load q vectors for this head (qn[h, :], qp[h, :])
    qn_vec = tl.load(qn_ptr + h * CK)          # [CK]
    qp_vec = tl.load(qp_ptr + h * KP)          # [KP]

    # Load Kc and Kp for this token t
    Kc_vec = tl.load(Kc_ptr + t * CK)          # [CK]
    Kp_vec = tl.load(Kp_ptr + t * KP)          # [KP]

    # Compute dot products
    dot_qn = tl.sum(qn_vec * Kc_vec, axis=0)   # scalar
    dot_qp = tl.sum(qp_vec * Kp_vec, axis=0)   # scalar

    # Scaled logits
    logits_val = sm_scale * (dot_qn + dot_qp)

    # Store logits[h, t]
    tl.store(logits_ptr + h * L_tokens + t, logits_val)


@triton.jit
def _lse_kernel(
    logits_ptr,   # *f32, [H, L_tokens] flattened
    lse_ptr,      # *f32, [H]
    inv_ln2: tl.constexpr,  # float (1 / ln(2))
    H: tl.constexpr,        # int
    L_tokens: tl.constexpr, # int
):
    # Grid over heads
    h = tl.program_id(0)
    if h >= H:
        return

    # Compute max for numerical stability
    max_val = -float("inf")
    sum_exp = 0.0
    for t in range(L_tokens):
        val = tl.load(logits_ptr + h * L_tokens + t)
        max_val = tl.maximum(max_val, val)
    # Second pass: sum exp(logits - max)
    for t in range(L_tokens):
        val = tl.load(logits_ptr + h * L_tokens + t)
        sum_exp += tl.exp(val - max_val)
    lse_val = tl.log(sum_exp) * inv_ln2  # base-2 logsumexp
    tl.store(lse_ptr + h, lse_val)


@triton.jit
def _compute_output_kernel(
    qn_ptr,        # *f32, [H, CK] flattened
    qp_ptr,        # *f32, [H, KP] flattened
    logits_ptr,    # *f32, [H, L_tokens] flattened
    lse_ptr,       # *f32, [H]
    Kc_ptr,        # *f32, [L_tokens, CK] flattened
    tok_idx_ptr,   # *i32, [L_tokens]
    out_ptr,       # *f32, [H, CK] flattened
    sm_scale: tl.constexpr,   # float
    H: tl.constexpr,          # int
    CK: tl.constexpr,         # int
    L_tokens: tl.constexpr,   # int
):
    # Grid over heads
    h = tl.program_id(0)
    if h >= H:
        return

    # lse for this head
    lse_h = tl.load(lse_ptr + h)

    # Accumulate output[h, :] = sum_t softmax(logits_scaled[h, t]) * Kc[t, :]
    for t in range(L_tokens):
        val = tl.load(logits_ptr + h * L_tokens + t)            # logits_scaled[h, t]
        soft = tl.exp(val - lse_h)                               # softmax value
        # Load Kc[t, :]
        Kc_vec = tl.load(Kc_ptr + t * CK)                        # [CK]
        # Accumulate out[h, :] += soft * Kc[t, :]
        # out_ptr is flattened [H, CK] as H*CK contiguous
        out_row_base = h * CK
        # We cannot vectorize across CK here; loop per dimension element
        for d in range(CK):
            out_ptr[out_row_base + d] += soft * Kc_vec[d]


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Shapes
        device = q_nope.device
        H = q_nope.shape[1]  # num_qo_heads
        CK = q_nope.shape[2]  # head_dim_ckv
        KP = q_pe.shape[2]    # head_dim_kpe

        # Extract per-batch token ranges
        B = q_nope.shape[0]
        # Slice ckv/kpe caches to [num_tokens, CK/KP]
        # We need to know num_tokens per batch b: L_tokens = kv_indptr[b+1] - kv_indptr[b]
        # Compute L_tokens per batch (assuming batch_size equal to number of segments)
        Ls = (kv_indptr[1:] - kv_indptr[:B]).tolist()
        # Create output and lse buffers
        output = torch.empty((B, H, CK), dtype=torch.bfloat16, device=device)
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        for b in range(B):
            L = int(Ls[b])
            # Collect token indices for this batch segment
            tok_idx = kv_indices[kv_indptr[b]: kv_indptr[b + 1]].to(torch.int32).to(device)
            # Slice cached Kc and Kp: [L, CK] and [L, KP]
            # Note: original ckv_cache/kpe_cache are [num_pages, 1, CK/KP], here we ignore the 1-sized dim and use the whole cache indexed by tok_idx
            # To reflect the original indexing, we gather by tok_idx into a local Kc/Kp of shape [L, CK/KP]
            # Since ckv_cache is [num_pages, 1, CK], we can gather with tok_idx as token positions; original code squeezes [1] out implicitly.
            # Here, we assume tok_idx maps to rows in the original cache; for the given inputs, tok_idx is within range.
            # Build local Kc/Kp by gathering rows
            Kc = ckv_cache[tok_idx]  # [L, CK], float32
            Kp = kpe_cache[tok_idx]  # [L, KP], float32

            # Prepare q vectors for this batch
            qn = q_nope[b].to(torch.float32).contiguous()  # [H, CK]
            qp = q_pe[b].to(torch.float32).contiguous()    # [H, KP]

            # Allocate logits buffer [H, L]
            logits = torch.empty((H, L), dtype=torch.float32, device=device)

            # Launch 1: compute logits_scaled
            _compute_logits_scaled_kernel[(H, L)](
                qn, qp, Kc, Kp, tok_idx, logits,
                sm_scale=float(sm_scale),
                H=H, CK=CK, KP=KP, L_tokens=L
            )

            # Compute per-head lse (base-2) in Triton
            inv_ln2 = 1.0 / math.log(2.0)
            _lse_kernel[(H,)](
                logits, lse[b], inv_ln2,
                H=H, L_tokens=L
            )

            # Compute output[h, :] = sum_t softmax(logits_scaled[h, t]) * Kc[t, :]
            out_row = torch.empty((H, CK), dtype=torch.float32, device=device)
            _compute_output_kernel[(H,)](
                qn, qp, logits, lse[b], Kc, tok_idx, out_row,
                sm_scale=float(sm_scale),
                H=H, CK=CK, L_tokens=L
            )

            # Store result for this batch
            output[b] = out_row

        return output, lse


def run(*args):
    return ModelNew()(*args)
