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
        qn_val = tl.load(qn_ptr + h * Dc + d)  # qn_ptr has shape [H*Dc], contiguous
        Kc_val = tl.load(Kc_ptr + t * Dc + d)  # Kc_ptr has shape [L*Dc], contiguous
        acc += qn_val * Kc_val
    index = h * L + t
    tl.store(dot_qn_ptr + index, acc)


# Kernel 2: dot between qp[h, :] and Kp[t, :] -> outputs a vector of size H*L
@triton.jit
def compute_dot_qp_Kp(dot_qp_ptr, qp_ptr, Kp_ptr, H: tl.constexpr, Dp: tl.constexpr, L: tl.constexpr):
    h = tl.program_id(0)
    t = tl.program_id(1)
    acc = 0.0
    for d in range(Dp):
        qp_val = tl.load(qp_ptr + h * Dp + d)  # qp_ptr has shape [H*Dp], contiguous
        Kp_val = tl.load(Kp_ptr + t * Dp + d)  # Kp_ptr has shape [L*Dp], contiguous
        acc += qp_val * Kp_val
    index = h * L + t
    tl.store(dot_qp_ptr + index, acc)


# Kernel 3: per head h, compute logits vector, lse, attn
@triton.jit
def compute_logits_lse_attn(lse_out_ptr, attn_ptr, dot_qn_ptr, dot_qp_ptr, L: tl.constexpr, sm_scale: tl.constexpr, i_abs: tl.constexpr):
    h = tl.program_id(0)  # one program per head
    # Initialize max for numerical stability
    max_val = -float("inf")
    # Compute max over logits[h, :]
    for t in range(L):
        v = tl.load(dot_qn_ptr + h * L + t) + tl.load(dot_qp_ptr + h * L + t)
        if v > max_val:
            max_val = v
    # Compute sum exp(v - max)
    sum_exp = 0.0
    for t in range(L):
        v = tl.load(dot_qn_ptr + h * L + t) + tl.load(dot_qp_ptr + h * L + t)
        e = tl.exp(v - max_val)
        # Apply causal mask: only tokens t <= i_abs allowed
        if t > i_abs:
            e = 0.0
        sum_exp += e
    # logsumexp with scaling
    lse = max_val + tl.log(sum_exp) * sm_scale  # sm_scale is passed as scalar
    tl.store(lse_out_ptr + h, lse)

    # Compute attn[h, :]
    for t in range(L):
        v = tl.load(dot_qn_ptr + h * L + t) + tl.load(dot_qp_ptr + h * L + t)
        e = tl.exp(v - max_val)
        if t > i_abs:
            e = 0.0
        attn = e / sum_exp
        tl.store(attn_ptr + h * L + t, attn)


# Kernel 4: matmul of attn[h, :] with Kc[t, :] to produce out[h, :]
@triton.jit
def matmul_vec_by_mat(out_ptr, attn_ptr, Kc_ptr, H: tl.constexpr, Dc: tl.constexpr, L: tl.constexpr):
    # This kernel computes out[h, :] = sum_t attn[h, t] * Kc[t, :] for a given h
    # We’ll launch one program per head h, and loop over L and Dc
    h = tl.program_id(0)
    acc = tl.zeros((Dc,), dtype=tl.float32)
    for t in range(L):
        attn_t = tl.load(attn_ptr + h * L + t)  # scalar attn for this head and token t
        for d in range(Dc):
            kc_val = tl.load(Kc_ptr + t * Dc + d)
            acc[d] += attn_t * kc_val
    # Store acc to out_ptr[h, :]
    # out_ptr layout is [H, Dc], contiguous => index = h * Dc + d
    for d in range(Dc):
        tl.store(out_ptr + h * Dc + d, acc[d])


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        """
        Mimics the original run function's logic using Triton kernels.
        Assumes:
          - q_nope: [total_q, num_qo_heads=16, head_dim_ckv=512]
          - q_pe: [total_q, num_qo_heads=16, head_dim_kpe=64]
          - ckv_cache: [num_pages, 512]
          - kpe_cache: [num_pages, 64]
          - qo_indptr: [len_indptr], e.g., [0, total_q]
          - kv_indptr: [len_indptr], e.g., [0, num_kv_indices]
          - kv_indices: [num_kv_indices], int32 indices into ckv_cache
          - sm_scale: float32 scalar
        """
        device = q_nope.device
        total_q, H, Dc = q_nope.shape
        _, _, Dp = q_pe.shape
        assert H == 16, "num_qo_heads must be 16"
        assert Dc == 512, "head_dim_ckv must be 512"
        assert Dp == 64, "head_dim_kpe must be 64"

        # Build output tensor and lse
        output = torch.empty((total_q, H, Dc), dtype=torch.bfloat16, device=device)
        lse = torch.full((total_q, H), -float("inf"), dtype=torch.float32, device=device)

        # Process batch segments
        b = 0
        while b < qo_indptr.shape[0] - 1:
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            if q_start >= q_end:
                b += 1
                continue

            # tok_idx for this batch element: tokens mapping to cache rows
            tok_idx = kv_indices[kv_indptr[b]: kv_indptr[b + 1]].to(torch.int32)  # [L]
            L = tok_idx.numel()

            # Extract Kc and Kp for this segment
            Kc_batch = ckv_cache[tok_idx].to(torch.float32)  # [L, Dc]
            Kp_batch = kpe_cache[tok_idx].to(torch.float32)  # [L, Dp]

            # Prepare flattened qn and qp for Triton
            qn_flat = q_nope[q_start:q_end].reshape((q_end - q_start) * H * Dc).to(torch.float32)  # [T*H*Dc], T = q_end - q_start
            qp_flat = q_pe[q_start:q_end].reshape((q_end - q_start) * H * Dp).to(torch.float32)   # [T*H*Dp]

            # Allocate intermediate buffers
            dot_qn = torch.empty((H * L * (q_end - q_start)), dtype=torch.float32, device=device)  # [T*H*L]
            dot_qp = torch.empty((H * L * (q_end - q_start)), dtype=torch.float32, device=device)  # [T*H*L]

            # Launch dot kernels: one per (h, t) pair -> grid (H, L)
            # We need to produce dot for each query in [q_start, q_end). We'll compute per i and store results.
            # To do this, we loop i and launch kernels per i.
            for i in range(q_start, q_end):
                # qn for this query: reshape as (H, Dc) flattened, then (H*Dc)
                qn_cur = q_nope[i]  # [H, Dc]
                qp_cur = q_pe[i]    # [H, Dp]
                qn_flat_cur = qn_cur.reshape(H * Dc).to(torch.float32)
                qp_flat_cur = qp_cur.reshape(H * Dp).to(torch.float32)

                dot_qn_i = torch.empty((H * L,), dtype=torch.float32, device=device)
                dot_qp_i = torch.empty((H * L,), dtype=torch.float32, device=device)

                # Launch compute_dot_qn_Kc and compute_dot_qp_Kp for this i
                # We need to set grid (H, L). Triton accepts grid via kernel launch: (H, L)
                compute_dot_qn_Kc[(H, L)](dot_qn_i, qn_flat_cur, Kc_batch.reshape(L * Dc).contiguous(), H=H, Dc=Dc, L=L)
                compute_dot_qp_Kp[(H, L)](dot_qp_i, qp_flat_cur, Kp_batch.reshape(L * Dp).contiguous(), H=H, Dp=Dp, L=L)

                # Store into dot_qn/dot_qp buffers at offset (i - q_start)
                offset = (i - q_start) * H * L
                dot_qn[offset:] = dot_qn_i
                dot_qp[offset:] = dot_qp_i

            # Now, for each i, compute lse and attn, then out[h, :]
            for i in range(q_start, q_end):
                offset = (i - q_start) * H * L
                dot_qn_i = dot_qn[offset:offset + H * L]
                dot_qp_i = dot_qp[offset:offset + H * L]

                # Allocate per-head lse and attn
                lse_h = torch.empty((H,), dtype=torch.float32, device=device)
                attn_flat = torch.empty((H * L,), dtype=torch.float32, device=device)

                # Launch compute_logits_lse_attn per head
                compute_logits_lse_attn[(H,)](lse_h, attn_flat, dot_qn_i, dot_qp_i, L=L, sm_scale=sm_scale, i_abs=i)

                # Convert attn_flat to [H, L]
                attn_2d = attn_flat.view(H, L)  # [H, L]

                # Compute out[h, :] = sum_t attn[h, t] * Kc[t, :]
                for h_idx in range(H):
                    out_vec = torch.empty((Dc,), dtype=torch.float32, device=device)
                    # Kc rows are [L, Dc]; attn_2d[h_idx, :] has L scalars
                    for t in range(L):
                        attn_t = attn_2d[h_idx, t]
                        # Sum over Dc: out_vec += attn_t * Kc_batch[t, :]
                        # Kc_batch[t, :] is contiguous over Dc
                        row = Kc_batch[t]  # [Dc]
                        out_vec += attn_t * row
                    # Store to output at [i, h_idx, :]
                    output[i, h_idx] = out_vec.to(torch.bfloat16)

                    # Also store lse[i, h_idx]
                    lse[i, h_idx] = lse_h[h_idx]

                # If you need lse only, we already filled it; no additional kernel needed here.

            b += 1

        return output, lse


def run(*args):
    return ModelNew()(*args)
