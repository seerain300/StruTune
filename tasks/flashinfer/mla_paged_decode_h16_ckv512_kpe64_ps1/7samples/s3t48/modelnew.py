import math
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# Triton kernel: compute full vector for one (b, h) including logits, lse (base-2), and out = attn @ Kc.
# We loop over tokens (L) in a simple for-loop (Python range in host) to avoid Triton static recursion.
@triton.jit
def compute_all_bh_kernel(qn_ptr, qp_ptr, Kc_ptr, Kp_ptr,
                          out_ptr, lse_ptr,
                          L: tl.constexpr, Dc: tl.constexpr, Dp: tl.constexpr,
                          sm_scale: tl.float32, inv_ln2: tl.float32):
    # Load qn and qp
    qn = tl.load(qn_ptr)  # [Dc]
    qp = tl.load(qp_ptr)  # [Dp]

    # Running max and sum for stable logsumexp
    m = tl.full((), -float("inf"), tl.float32)
    s = tl.full((), 0.0, tl.float32)

    # Accumulator for output vector
    out_vec = tl.zeros((Dc,), dtype=tl.float32)

    # Iterate over tokens
    for l in range(0, L):
        # Compute logits for this token: qn @ Kc[l, :]
        sum_qn_Kc = tl.zeros((), dtype=tl.float32)
        for j in range(0, Dc):
            # Kc[l, j] = *(Kc_ptr + l * Dc + j)
            kc_elem = tl.load(Kc_ptr + l * Dc + j)
            sum_qn_Kc += qn[j] * kc_elem

        # Compute q
        sum_qp_Kp = tl.zeros((), dtype=tl.float32)
        for k in range(0, Dp):
            kp_elem = tl.load(Kp_ptr + l * Dp + k)
            sum_qp_Kp += qp[k] * kp_elem

        logit_l = sum_qn_Kc + sum_qp_Kp
        logit_scaled = logit_l * sm_scale

        # Update running max and sum for logsumexp
        m_new = tl.maximum(m, logit_scaled)
        s = s * tl.exp(m - m_new) + tl.exp(logit_scaled - m_new)
        m = m_new

    # Compute base-2 logsumexp: lse = m + log(s) / ln(2)
    lse_val = m + tl.log(s) * inv_ln2
    tl.store(lse_ptr, lse_val)

    # Recompute probabilities for each token (single pass with running m and s)
    for l in range(0, L):
        sum_qn_Kc = tl.zeros((), dtype=tl.float32)
        for j in range(0, Dc):
            kc_elem = tl.load(Kc_ptr + l * Dc + j)
            sum_qn_Kc += qn[j] * kc_elem

        sum_qp_Kp = tl.zeros((), dtype=tl.float32)
        for k in range(0, Dp):
            kp_elem = tl.load(Kp_ptr + l * Dp + k)
            sum_qp_Kp += qp[k] * kp_elem

        logit_l = sum_qn_Kc + sum_qp_Kp
        logit_scaled = logit_l * sm_scale
        p = tl.exp((logit_scaled - m) * inv_ln2) / s  # base-2 exp, then normalize by s

        # Accumulate out_vec += p * Kc[l, :]
        for j in range(0, Dc):
            kc_elem = tl.load(Kc_ptr + l * Dc + j)
            out_vec[j] += p * kc_elem

    # Write output vector
    # out_ptr points to a contiguous [B*H, Dc] buffer; we compute index as (b*H + h)
    base = tl.program_id(0) * 16 + tl.program_id(1)  # tl.program_id(0)=b, tl.program_id(1)=h
    out_offset = base * Dc
    for j in range(0, Dc):
        tl.store(out_ptr + out_offset + j, out_vec[j])


def run(q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
    # Ensure CUDA tensors and dtypes
    assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda
    device = q_nope.device

    B, H, Dc = q_nope.shape
    _, _, Dp = q_pe.shape
    N = ckv_cache.shape[0]
    L_tot = kv_indices.shape[0]

    # Prepare outputs
    output = torch.empty((B, H, Dc), dtype=torch.float32, device=device)  # compute in fp32, cast later
    lse = torch.empty((B, H), dtype=torch.float32, device=device)

    # For each batch b, compute tok_idx = kv_indices[kv_indptr[b]:kv_indptr[b+1]]
    for b in range(B):
        start = int(kv_indptr[b].item())
        end = int(kv_indptr[b + 1].item())
        if start >= end:
            # No tokens in this batch element -> output zeros
            output[b].zero_()
            lse[b].zero_()
            continue

        tok_idx = kv_indices[start:end].to(torch.int32).to(device)  # [L_b]
        L_b = tok_idx.numel()

        # Gather Kc and Kp for this batch
        Kc_b = ckv_cache[tok_idx].contiguous().to(torch.float32)  # [L_b, Dc]
        Kp_b = kpe_cache[tok_idx].contiguous().to(torch.float32)  # [L_b, Dp]

        # Launch Triton kernel per (b, h)
        for h in range(H):
            # Prepare pointers
            qn = q_nope[b, h].contiguous().to(torch.float32)
            qp = q_pe[b, h].contiguous().to(torch.float32)

            # Compute linear output offset for (b, h)
            out_offset = (b * H + h) * Dc

            # Launch kernel
            compute_all_bh_kernel[(1,)](
                qn, qp, Kc_b, Kp_b,
                output.view(-1), lse[b],  # out_ptr is flattened [B*H, Dc], lse per head
                L_b, Dc, Dp,
                float(sm_scale), float(1.0 / math.log(2.0))  # sm_scale and 1/ln(2)
            )

    # Cast output to bfloat16 to match typical original behavior
    output_bf16 = output.to(torch.bfloat16)
    return output_bf16, lse


# Optional: keep these to satisfy the original signature
def get_inputs():
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16, device='cuda')
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16, device='cuda')
    ckv_cache = torch.randn([989669, 1, 512], dtype=torch.bfloat16, device='cuda')
    kpe_cache = torch.randn([989669, 1, 64], dtype=torch.bfloat16, device='cuda')
    _n = 1; _t = 8
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32).to('cuda')
    kv_indices = torch.randint(0, 989669, [8], dtype=torch.int32).to('cuda')
    sm_scale = 1.0  # float32 scalar
    return [q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale]


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        return run(*args)