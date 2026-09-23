import torch
import math
import triton
import triton.language as tl


# Kernel 1: matmul_qn_kc => computes A[h, l] = sum_d qn[h, d] * Kc[l, d]
@triton.jit
def matmul_qn_kc(qn_ptr, kc_ptr, out_ptr,
                 H: tl.constexpr, L: tl.constexpr, D: tl.constexpr,
                 BLOCK_H: tl.constexpr, BLOCK_D: tl.constexpr):
    h = tl.program_id(0)
    l = tl.program_id(1)
    acc = tl.zeros((), dtype=tl.float32)
    # Reduce over D
    for d_start in range(0, D, BLOCK_D):
        d = d_start + tl.arange(0, BLOCK_D)
        d_mask = d < D
        # qn[h, d] as vector
        qn_vec = tl.load(qn_ptr + h * D + d, mask=d_mask, other=0.0)
        # Kc[l, d] as vector
        kc_vec = tl.load(kc_ptr + l * D + d, mask=d_mask, other=0.0)
        # Dot product of qn_vec and kc_vec
        acc += tl.sum(qn_vec * kc_vec, axis=0)
    tl.store(out_ptr + h * L + l, acc)


# Kernel 2: matmul_qp_kp => computes B[h, l] = sum_p qp[h, p] * Kp[l, p]
@triton.jit
def matmul_qp_kp(qp_ptr, kp_ptr, out_ptr,
                 H: tl.constexpr, L: tl.constexpr, P: tl.constexpr,
                 BLOCK_H: tl.constexpr, BLOCK_P: tl.constexpr):
    h = tl.program_id(0)
    l = tl.program_id(1)
    acc = tl.zeros((), dtype=tl.float32)
    for p_start in range(0, P, BLOCK_P):
        p = p_start + tl.arange(0, BLOCK_P)
        p_mask = p < P
        qp_vec = tl.load(qp_ptr + h * P + p, mask=p_mask, other=0.0)
        kp_vec = tl.load(kp_ptr + l * P + p, mask=p_mask, other=0.0)
        acc += tl.sum(qp_vec * kp_vec, axis=0)
    tl.store(out_ptr + h * L + l, acc)


# Kernel 3: add_logits => out[h, l] = A[h, l] + B[h, l]
@triton.jit
def add_logits(a_ptr, b_ptr, out_ptr,
               H: tl.constexpr, L: tl.constexpr):
    h = tl.program_id(0)
    l = tl.program_id(1)
    a = tl.load(a_ptr + h * L + l)
    b = tl.load(b_ptr + h * L + l)
    tl.store(out_ptr + h * L + l, a + b)


# Kernel 4: scale_logits => out[h, l] = D[h, l] * sm_scale
@triton.jit
def scale_logits(in_ptr, out_ptr, sm_scale,
                 H: tl.constexpr, L: tl.constexpr):
    h = tl.program_id(0)
    l = tl.program_id(1)
    x = tl.load(in_ptr + h * L + l)
    y = x * sm_scale
    tl.store(out_ptr + h * L + l, y)


# Kernel 5: apply_mask => set -inf where j <= query_abs_pos; keep otherwise
@triton.jit
def apply_mask(logits_ptr, masked_ptr,
               H: tl.constexpr, L: tl.constexpr, query_abs_pos):
    h = tl.program_id(0)
    l = tl.program_id(1)
    val = tl.load(logits_ptr + h * L + l)
    keep = l > query_abs_pos
    val = tl.where(keep, val, -float("inf"))
    tl.store(masked_ptr + h * L + l, val)


# Kernel 6: row_logsumexp => compute lse[h] = log(sum exp(masked[h, :])) / ln(2)
@triton.jit
def row_logsumexp(masked_ptr, lse_ptr,
                  H: tl.constexpr, L: tl.constexpr, inv_ln2):
    h = tl.program_id(0)
    # First pass: max
    m = -float("inf")
    for l in range(0, L):
        val = tl.load(masked_ptr + h * L + l)
        m = tl.maximum(m, val)
    # Second pass: sumexp
    sumexp = 0.0
    for l in range(0, L):
        val = tl.load(masked_ptr + h * L + l)
        sumexp += tl.exp(val - m)
    lse = tl.log(sumexp) + m  # logsumexp
    tl.store(lse_ptr + h, lse * inv_ln2)


# Kernel 7: softmax_row_triton => compute softmax per row using lse[h]
@triton.jit
def softmax_row_triton(masked_ptr, lse_ptr, soft_ptr,
                        H: tl.constexpr, L: tl.constexpr):
    h = tl.program_id(0)
    denom = 0.0
    for l in range(0, L):
        val = tl.load(masked_ptr + h * L + l)
        e = tl.exp(val - tl.load(lse_ptr + h))
        denom += e
    for l in range(0, L):
        val = tl.load(masked_ptr + h * L + l)
        soft = tl.exp(val - tl.load(lse_ptr + h)) / denom
        tl.store(soft_ptr + h * L + l, soft)


# Kernel 8: matmul_attn_kc => out[h, d] = sum_l softmax[h, l] * Kc[l, d]
@triton.jit
def matmul_attn_kc(soft_ptr, kc_ptr, out_ptr,
                   H: tl.constexpr, L: tl.constexpr, D: tl.constexpr,
                   BLOCK_D: tl.constexpr):
    h = tl.program_id(0)
    d = tl.program_id(1)
    acc = tl.zeros((), dtype=tl.float32)
    for l in range(0, L):
        soft = tl.load(soft_ptr + h * L + l)
        kc_vec = tl.load(kc_ptr + l * D + d)  # scalar load per d
        acc += soft * kc_vec
    tl.store(out_ptr + h * D + d, acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Prepare constants
        total_q, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]
        num_pages = ckv_cache.shape[0]
        assert num_qo_heads == 16
        assert head_dim_ckv == 512
        assert head_dim_kpe == 64

        device = q_nope.device
        # Compute Kc_all and Kp_all
        Kc_all = ckv_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, D]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, P]

        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        batch_size = qo_indptr.shape[0] - 1

        # Loop over batches
        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            L = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            tok_idx = kv_indices[int(kv_indptr[b].item()):int(kv_indptr[b + 1].item())].to(torch.int32).to(device)
            Kc = Kc_all[tok_idx].contiguous()  # [L, D]
            Kp = Kp_all[tok_idx].contiguous()  # [L, P]
            Q = q_end - q_start

            if Q <= 0 or L <= 0:
                continue

            q_nope_b = q_nope[q_start:q_end].to(torch.float32).contiguous()  # [Q, H, D]
            q_pe_b = q_pe[q_start:q_end].to(torch.float32).contiguous()      # [Q, H, P]

            # Precompute query_abs_pos range
            prefix_len = L - Q  # number of previously cached tokens
            for i in range(Q):
                query_abs_pos = prefix_len + i

                # Prepare qn: [H, D], qp: [H, P]
                qn = q_nope_b[i].transpose(0, 1).contiguous()  # [H, D]
                qp = q_pe_b[i].transpose(0, 1).contiguous()   # [H, P]

                # Matmul A = qn @ Kc.T -> [H, L]
                A = torch.empty((num_qo_heads, L), dtype=torch.float32, device=device)
                grid1 = (num_qo_heads, L)
                matmul_qn_kc[grid1](qn, Kc, A, H=num_qo_heads, L=L, D=head_dim_ckv,
                                    BLOCK_H=16, BLOCK_D=64)

                # Matmul B = qp @ Kp.T -> [H, L]
                B = torch.empty((num_qo_heads, L), dtype=torch.float32, device=device)
                grid2 = (num_qo_heads, L)
                matmul_qp_kp[grid2](qp, Kp, B, H=num_qo_heads, L=L, P=head_dim_kpe,
                                    BLOCK_H=16, BLOCK_P=64)

                # Add A + B -> [H, L]
                C = torch.empty((num_qo_heads, L), dtype=torch.float32, device=device)
                add_logits[grid1](A, B, C, H=num_qo_heads, L=L)

                # Scale by sm_scale
                D = torch.empty((num_qo_heads, L), dtype=torch.float32, device=device)
                scale_logits[grid1](C, D, sm_scale, H=num_qo_heads, L=L)

                # Apply mask: keep where j > query_abs_pos
                Masked = torch.empty((num_qo_heads, L), dtype=torch.float32, device=device)
                apply_mask[grid1](D, Masked, H=num_qo_heads, L=L, query_abs_pos=query_abs_pos)

                # Row-wise logsumexp / ln(2)
                row_logsumexp[(num_qo_heads,)](Masked, lse[q_start + i], H=num_qo_heads, L=L, inv_ln2=1.0 / math.log(2.0))

                # Softmax per row using lse
                Soft = torch.empty((num_qo_heads, L), dtype=torch.float32, device=device)
                softmax_row_triton[(num_qo_heads,)](Masked, lse[q_start + i], Soft, H=num_qo_heads, L=L)

                # Final output: Soft @ Kc -> [H, D]
                OutRow = torch.empty((num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
                grid3 = (num_qo_heads, head_dim_ckv)
                matmul_attn_kc[grid3](Soft, Kc, OutRow, H=num_qo_heads, L=L, D=head_dim_ckv,
                                      BLOCK_D=64)

                output[q_start + i] = OutRow.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
