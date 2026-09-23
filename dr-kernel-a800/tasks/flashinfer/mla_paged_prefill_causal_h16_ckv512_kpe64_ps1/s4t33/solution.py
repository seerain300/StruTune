import math
import torch
import triton
import triton.language as tl


# Kernel 1: dot between qn[h, :] and Kc[t, :] -> outputs a vector of size H*L
@triton.jit
def compute_dot_qn_Kc(dot_qn_ptr, qn_ptr, Kc_ptr, H: tl.constexpr, Dc: tl.constexpr, L: tl.constexpr):
    h = tl.program_id(0)  # program_id(0) covers H
    t = tl.program_id(1)  # program_id(1) covers L
    acc = 0.0
    # Loop over Dc to compute dot product
    for d in range(Dc):
        qn_val = tl.load(qn_ptr + h * Dc + d)
        Kc_val = tl.load(Kc_ptr + t * Dc + d)
        acc += qn_val * Kc_val
    index = h * L + t
    tl.store(dot_qn_ptr + index, acc)


# Kernel 2: dot between qp[h, :] and Kp[t, :] -> outputs a vector of size H*L
@triton.jit
def compute_dot_qp_Kp(dot_qp_ptr, qp_ptr, Kp_ptr, H: tl.constexpr, Dp: tl.constexpr, L: tl.constexpr):
    h = tl.program_id(0)  # program_id(0) covers H
    t = tl.program_id(1)  # program_id(1) covers L
    acc = 0.0
    # Loop over Dp to compute dot product
    for d in range(Dp):
        qp_val = tl.load(qp_ptr + h * Dp + d)
        Kp_val = tl.load(Kp_ptr + t * Dp + d)
        acc += qp_val * Kp_val
    index = h * L + t
    tl.store(dot_qp_ptr + index, acc)


# Kernel 3: per head compute logits, lse, attn (softmax), with causal mask
@triton.jit
def compute_logits_lse_attn(dot_qn_ptr, dot_qp_ptr, lse_out_ptr, attn_out_ptr,
                            H: tl.constexpr, L: tl.constexpr,
                            sm_scale: tl.constexpr, i_abs: tl.constexpr):
    h = tl.program_id(0)
    # Initialize logits vector of size L
    logits = tl.zeros((L,), dtype=tl.float32)
    # Load dot products into logits
    for t in range(L):
        index = h * L + t
        logits[t] = tl.load(dot_qn_ptr + index) + tl.load(dot_qp_ptr + index)

    # Causal mask: only tokens t <= i_abs are allowed; others set to -inf
    for t in range(L):
        if t > i_abs:
            logits[t] = -float("inf")

    # Logsumexp with scaling and base-2 log
    # Compute max for numerical stability
    m = logits[0]
    for t in range(1, L):
        m = tl.maximum(m, logits[t])
    # sum exp(logits - m) and multiply by sm_scale
    sum_exp = 0.0
    for t in range(L):
        sum_exp += tl.exp(logits[t] - m)
    sum_exp = sum_exp * sm_scale
    # lse = m + log(sum_exp) (natural log); compute log2 via ln / ln(2)
    inv_ln2 = 1.4426950408889634  # 1 / ln(2)
    lse_h = m + tl.log(sum_exp) * inv_ln2
    tl.store(lse_out_ptr + h, lse_h)

    # Compute attention: softmax over logits_scaled = logits - lse
    for t in range(L):
        logits[t] = logits[t] - lse_h
        attn_out_ptr[h * L + t] = tl.exp(logits[t])
    # Normalize: sum of attention
    sum_attn = 0.0
    for t in range(L):
        sum_attn += attn_out_ptr[h * L + t]
    for t in range(L):
        attn_out_ptr[h * L + t] = attn_out_ptr[h * L + t] / sum_attn


# Kernel 4: out[h, :] = sum_t attn[h, t] * Kc[t, :]
@triton.jit
def matmul_vec_by_mat(out_ptr, attn_ptr, Kc_ptr, H: tl.constexpr, Dc: tl.constexpr, L: tl.constexpr):
    h = tl.program_id(0)  # head
    # attn vector for this head, size L
    attn_vec = tl.zeros((L,), dtype=tl.float32)
    # Read attn vector
    for t in range(L):
        attn_vec[t] = tl.load(attn_ptr + h * L + t)
    # Accumulate output vector of size Dc
    out_vec = tl.zeros((Dc,), dtype=tl.float32)
    for d in range(Dc):
        acc = 0.0
        for t in range(L):
            acc += attn_vec[t] * tl.load(Kc_ptr + t * Dc + d)
        out_vec[d] = acc
    # Store output vector for this head
    # out_ptr is expected to be a [H, Dc] flattened as [H*Dc]
    out_idx = h * Dc + d  # but out_vec is length Dc, we store at base h*0 + offset
    # Note: Triton kernel assumes out_ptr points to the start of h-th row, i.e., out_ptr + h * Dc
    # We pass a flattened tensor and compute offset h * Dc + d in host code when launching.
    # Here we store each d element at base h*0 + d, but to be safe, we store to out_ptr base h and offset d.
    # Triton does not support multi-dimensional indexing; we rely on host to pass correct base.
    pass  # placeholder to avoid syntax issues; actual store handled in host


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Shapes and constants
        device = q_nope.device
        total_q, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]
        num_pages, _, _ = ckv_cache.shape
        assert num_qo_heads == 16, "num_qo_heads must be 16"
        assert head_dim_ckv == 512, "head_dim_ckv must be 512"
        assert head_dim_kpe == 64, "head_dim_kpe must be 64"
        H = num_qo_heads
        Dc = head_dim_ckv
        Dp = head_dim_kpe

        len_indptr = qo_indptr.shape[0]
        batch_size = len_indptr - 1
        num_kv_indices = kv_indices.shape[0]

        # Output and lse tensors
        output = torch.empty((total_q, H, Dc), dtype=torch.bfloat16, device=device)
        lse = torch.full((total_q, H), -float("inf"), dtype=torch.float32, device=device)

        # Iterate batch elements
        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())

            if q_start >= q_end:
                continue

            # tok_idx for this batch element
            tok_idx = kv_indices[kv_indptr[b]: kv_indptr[b + 1]].to(torch.int32)  # [L]
            L = tok_idx.numel()

            # Kc and Kp for this batch segment: [L, Dc] and [L, Dp]
            Kc_batch = ckv_cache[tok_idx].to(torch.float32)  # [L, Dc]
            Kp_batch = kpe_cache[tok_idx].to(torch.float32)  # [L, Dp]

            # For each query i in this batch segment
            for i in range(q_start, q_end):
                # Prepare qn[h, :] and qp[h, :] flattened
                qn_flat = q_nope[i].reshape(H * Dc).to(torch.float32)  # [H*Dc]
                qp_flat = q_pe[i].reshape(H * Dp).to(torch.float32)   # [H*Dp]

                # Intermediate buffers
                dot_qn = torch.empty((H * L,), dtype=torch.float32, device=device)
                dot_qp = torch.empty((H * L,), dtype=torch.float32, device=device)

                # Launch dot kernels: grid (H, L)
                compute_dot_qn_Kc[(H, L)](dot_qn, qn_flat, Kc_batch, H=H, Dc=Dc, L=L)
                compute_dot_qp_Kp[(H, L)](dot_qp, qp_flat, Kp_batch, H=H, Dp=Dp, L=L)

                # Per-head lse and attn
                lse_h = torch.empty((H,), dtype=torch.float32, device=device)
                attn_out = torch.empty((H * L,), dtype=torch.float32, device=device)
                for h in range(H):
                    compute_logits_lse_attn[(1,)](dot_qn, dot_qp, lse_h[h], attn_out[h * L:(h + 1) * L],
                                                  H=H, L=L, sm_scale=float(sm_scale), i_abs=i)

                # Compute out[h, :] = sum_t attn[h, t] * Kc[t, :]
                out_flat = torch.empty((H * Dc,), dtype=torch.float32, device=device)
                for h in range(H):
                    attn_vec = attn_out[h * L:(h + 1) * L].contiguous()  # [L]
                    out_vec = torch.empty((Dc,), dtype=torch.float32, device=device)
                    # Use a simple matmul-like accumulation here (Triton kernel is more efficient but we keep it simple)
                    for d in range(Dc):
                        acc = 0.0
                        for t in range(L):
                            acc += attn_vec[t].item() * float(Kc_batch[t, d].item())
                        out_vec[d] = acc
                    # Store into out_flat
                    out_flat[h * Dc:(h + 1) * Dc] = out_vec

                # Store output[i, :, :] as bfloat16
                # Convert to bfloat16
                out_bf16 = out_flat.view(H, Dc).to(torch.bfloat16)
                output[i] = out_bf16

        return output, lse


def run(*args):
    return ModelNew()(*args)
