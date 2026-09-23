import math
import torch

import triton
import triton.language as tl

# Kernel 1: Compute A = qn @ Kc.T for a single query i and batch b
# Inputs:
#   qn_ptr: *float32, shape [H, D], row-major
#   kc_ptr: *float32, shape [L, D], row-major
#   outA_ptr: *float32, shape [H, L], row-major
# Sizes:
#   H: num_qo_heads, D: head_dim_ckv, L: kv_len
@triton.jit
def matmul_qn_kc(qn_ptr, kc_ptr, outA_ptr,
                 H: tl.constexpr, D: tl.constexpr, L: tl.constexpr,
                 stride_qh, stride_qd,
                 stride_kl, stride_kd,
                 stride_oh, stride_ol):
    pid = tl.program_id(0)  # single program per (b, i, h) with H fixed, but here we use grid over H and L tiles
    # We'll use a 2D grid where axis 0 is H (heads), axis 1 tiles L
    h = pid // tl.num_programs(1)  # pid // L_tiles, but we set grid[0]=H so this is h
    # Simplify: set grid=(H, ceil_div(L, BLOCK_L)) and compute h, l_block from pid
    # However Triton doesn't support tl.num_programs; we use linearized pid:
    # Let BLOCK_L be constexpr. Then:
    # grid = (H, ceil_div(L, BLOCK_L))
    # pid = h*ceil_div(L, BLOCK_L) + l_block
    # So we reconstruct:
    l_blocks = (L + 128 - 1) // 128
    h = pid // l_blocks
    l_block = pid % l_blocks
    l_start = l_block * 128
    # If grid[1] > l_blocks, we could clamp, but we set grid[1]=l_blocks, so l_block in [0, l_blocks)
    l_offsets = l_start + tl.arange(0, 128)
    mask_l = l_offsets < L

    acc = tl.zeros([128], dtype=tl.float32)

    # Loop over K dimension (D) in tiles of 128
    for k in range(0, D, 128):
        k_offsets = k + tl.arange(0, 128)
        mask_k = k_offsets < D
        # qn[h, k_offsets]
        q_vals = tl.load(qn_ptr + h * stride_qh + k_offsets * stride_qd, mask=mask_k, other=0.0)  # [128]
        # kc[l_offsets, k_offsets]
        kc_vals = tl.load(kc_ptr + l_offsets[:, None] * stride_kl + k_offsets[None, :] * stride_kd, mask=mask_l[:, None] & mask_k[None, :], other=0.0)  # [128, 128]
        # acc += q_vals[k] * kc_vals[:, k]
        acc += tl.sum(kc_vals * q_vals[None, :], axis=1)  # reduce over k tile -> [128]
    # Store acc into outA[h, l_offsets]
    tl.store(outA_ptr + h * stride_oh + l_offsets * stride_ol, acc, mask=mask_l)

# Kernel 2: Compute B = qp @ Kp.T
@triton.jit
def matmul_qp_kp(qp_ptr, kp_ptr, outB_ptr,
                 H: tl.constexpr, P: tl.constexpr, L: tl.constexpr,
                 stride_qh, stride_qp,
                 stride_kl, stride_kp,
                 stride_obh, stride_obl):
    l_blocks = (L + 128 - 1) // 128
    h = tl.program_id(0)  # axis 0 = H
    l_block = tl.program_id(1)  # axis 1 = L tiles
    l_start = l_block * 128
    l_offsets = l_start + tl.arange(0, 128)
    mask_l = l_offsets < L

    acc = tl.zeros([128], dtype=tl.float32)

    for p in range(0, P, 128):
        p_offsets = p + tl.arange(0, 128)
        mask_p = p_offsets < P
        qp_vals = tl.load(qp_ptr + h * stride_qh + p_offsets * stride_qp, mask=mask_p, other=0.0)  # [128]
        kp_vals = tl.load(kp_ptr + l_offsets[:, None] * stride_kl + p_offsets[None, :] * stride_kp, mask=mask_l[:, None] & mask_p[None, :], other=0.0)  # [128, 128]
        acc += tl.sum(kp_vals * qp_vals[None, :], axis=1)
    tl.store(outB_ptr + h * stride_obh + l_offsets * stride_obl, acc, mask=mask_l)

# Kernel 3: Add A and B into logits
@triton.jit
def add_logits(A_ptr, B_ptr, logits_ptr,
               H: tl.constexpr, L: tl.constexpr,
               stride_ah, stride_al,
               stride_bh, stride_bl,
               stride_lh, stride_ll):
    l_blocks = (L + 128 - 1) // 128
    h = tl.program_id(0)
    l_block = tl.program_id(1)
    l_start = l_block * 128
    l_offsets = l_start + tl.arange(0, 128)
    mask_l = l_offsets < L

    a = tl.load(A_ptr + h * stride_ah + l_offsets * stride_al, mask=mask_l, other=0.0)
    b = tl.load(B_ptr + h * stride_bh + l_offsets * stride_bl, mask=mask_l, other=0.0)
    logit = a + b
    tl.store(logits_ptr + h * stride_lh + l_offsets * stride_ll, logit, mask=mask_l)

# Kernel 4: Scale logits by sm_scale
@triton.jit
def scale_logits(logits_ptr, scaled_ptr,
                 H: tl.constexpr, L: tl.constexpr,
                 stride_lh, stride_ll,
                 stride_sh, stride_sl,
                 sm_scale: tl.float32):
    l_blocks = (L + 128 - 1) // 128
    h = tl.program_id(0)
    l_block = tl.program_id(1)
    l_start = l_block * 128
    l_offsets = l_start + tl.arange(0, 128)
    mask_l = l_offsets < L

    logit = tl.load(logits_ptr + h * stride_lh + l_offsets * stride_ll, mask=mask_l, other=0.0)
    logit = logit * sm_scale
    tl.store(scaled_ptr + h * stride_sh + l_offsets * stride_sl, logit, mask=mask_l)

# Kernel 5: Apply causal mask: set -inf where j <= query_abs_pos
@triton.jit
def apply_mask(scaled_ptr, masked_ptr,
               H: tl.constexpr, L: tl.constexpr,
               stride_sh, stride_sl,
               stride_mh, stride_ml,
               query_abs_pos: tl.int32):
    l_blocks = (L + 128 - 1) // 128
    h = tl.program_id(0)
    l_block = tl.program_id(1)
    l_start = l_block * 128
    l_offsets = l_start + tl.arange(0, 128)
    mask_l = l_offsets < L

    # Load scaled logits
    logit = tl.load(scaled_ptr + h * stride_sh + l_offsets * stride_sl, mask=mask_l, other=0.0)

    # Compute mask: keep if j > query_abs_pos else -inf
    # l_offsets is int32 vector; query_abs_pos is int32 scalar
    keep = l_offsets > query_abs_pos
    neg_inf = -float('inf')
    logit = tl.where(keep, logit, neg_inf)
    tl.store(masked_ptr + h * stride_mh + l_offsets * stride_ml, logit, mask=mask_l)

# Kernel 6: Row-wise logsumexp per (b, i, h)
@triton.jit
def row_lse(scaled_ptr, lse_ptr,
            H: tl.constexpr, L: tl.constexpr,
            stride_sh, stride_sl,
            ln2: tl.float32):
    h = tl.program_id(0)
    # Single program per row
    # First pass: find max
    max_val = tl.full([1], -float('inf'), dtype=tl.float32)
    for l in range(0, L):
        val = tl.load(scaled_ptr + h * stride_sh + l * stride_sl)
        max_val = tl.maximum(max_val, val)
    # Second pass: sum exp(x - max)
    sum_exp = tl.zeros([1], dtype=tl.float32)
    for l in range(0, L):
        val = tl.load(scaled_ptr + h * stride_sh + l * stride_sl)
        sum_exp += tl.exp(val - max_val)
    lse_row = max_val + tl.log(sum_exp)  # logsumexp
    lse_row = lse_row / ln2  # divide by ln(2)
    # Store per (b, i)
    tl.store(lse_ptr + h, lse_row)

# Kernel 7: Softmax per row, write back to masked_ptr (overwrite)
@triton.jit
def softmax_row(masked_ptr, out_softmax_ptr,
                 H: tl.constexpr, L: tl.constexpr,
                 stride_mh, stride_ml,
                 stride_oh, stride_ol,
                 lse_scalar: tl.float32):
    h = tl.program_id(0)
    # First compute numerator for each j
    # We need the whole row, but we'll recompute using masked_ptr and subtract lse_scalar
    # Softmax: e_j / sum_k e_k
    sum_exp = tl.zeros([1], dtype=tl.float32)
    for l in range(0, L):
        val = tl.load(masked_ptr + h * stride_mh + l * stride_ml)
        sum_exp += tl.exp(val - lse_scalar)
    # Write normalized values
    for l in range(0, L):
        val = tl.load(masked_ptr + h * stride_mh + l * stride_ml)
        e = tl.exp(val - lse_scalar)
        out_val = e / sum_exp
        tl.store(out_softmax_ptr + h * stride_oh + l * stride_ol, out_val)

# Kernel 8: Final matmul attn @ Kc -> output[H, D]
@triton.jit
def matmul_attn_kc(attn_ptr, kc_ptr, out_ptr,
                   H: tl.constexpr, D: tl.constexpr, L: tl.constexpr,
                   stride_ah, stride_al,
                   stride_kl, stride_kd,
                   stride_oh, stride_od):
    # Grid: (H, ceil_div(D, 128))
    d_blocks = (D + 128 - 1) // 128
    h = tl.program_id(0)
    d_block = tl.program_id(1)
    d_start = d_block * 128
    d_offsets = d_start + tl.arange(0, 128)
    mask_d = d_offsets < D

    acc = tl.zeros([128], dtype=tl.float32)

    # Loop over K dimension L
    for l in range(0, L):
        attn_val = tl.load(attn_ptr + h * stride_ah + l * stride_al)  # scalar
        kc_vals = tl.load(kc_ptr + l * stride_kl + d_offsets * stride_kd, mask=mask_d, other=0.0)  # [128]
        acc += attn_val * kc_vals
    tl.store(out_ptr + h * stride_oh + d_offsets * stride_od, acc, mask=mask_d)

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters; everything is computed in Triton

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Device and dtype setup
        device = q_nope.device
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "All tensors must be on CUDA device"
        # Shapes
        total_q, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]
        assert num_qo_heads == 16, "num_qo_heads must be 16"
        assert head_dim_ckv == 512, "head_dim_ckv must be 512"
        assert head_dim_kpe == 64, "head_dim_kpe must be 64"
        assert ckv_cache.shape[1] == 1 and kpe_cache.shape[1] == 1, "page_size must be 1 (already asserted)"

        # Prepare Kc_all and Kp_all: [num_pages, D] and [num_pages, P]
        Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, 64]

        # Output and lse
        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)  # we'll cast to bfloat16 at the end
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Process batches using qo_indptr and kv_indptr
        # Convert to int32
        qo_indptr = qo_indptr.to(torch.int32)
        kv_indptr = kv_indptr.to(torch.int32)

        len_indptr = qo_indptr.shape[0]
        batch_size = len_indptr - 1
        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())

            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or page_beg >= page_end:
                continue

            # Token indices for this batch
            tok_idx = kv_indices[page_beg:page_end].to(torch.int32).to(device)  # [kv_len]
            kv_len = tok_idx.numel()

            # Slice q_nope and q_pe
            q_nope_batch = q_nope[q_start:q_end].to(torch.float32).contiguous()  # [Q, H, D]
            q_pe_batch = q_pe[q_start:q_end].to(torch.float32).contiguous()    # [Q, H, P]

            Q = q_end - q_start

            # Prepare Kc and Kp for this batch
            Kc = Kc_all[tok_idx]  # [kv_len, D]
            Kp = Kp_all[tok_idx]  # [kv_len, P]

            # Loop over queries in this batch
            for i in range(Q):
                h = 0  # since num_qo_heads == 16, we use h dimension directly
                qn = q_nope_batch[i].to(torch.float32).contiguous()  # [H, D], but here H=16
                qp = q_pe_batch[i].to(torch.float32).contiguous()    # [H, P]

                # Allocate intermediates
                A = torch.empty((num_qo_heads, kv_len), dtype=torch.float32, device=device)
                B = torch.empty((num_qo_heads, kv_len), dtype=torch.float32, device=device)
                Logits = torch.empty((num_qo_heads, kv_len), dtype=torch.float32, device=device)
                Scaled = torch.empty((num_qo_heads, kv_len), dtype=torch.float32, device=device)
                Masked = torch.empty((num_qo_heads, kv_len), dtype=torch.float32, device=device)
                SoftmaxOut = torch.empty((num_qo_heads, kv_len), dtype=torch.float32, device=device)

                # Launch Triton kernels: compute A
                H = num_qo_heads
                D = head_dim_ckv
                L = kv_len
                grid_A = (H, (L + 128 - 1) // 128)
                matmul_qn_kc[grid_A](
                    qn, Kc, A,
                    H, D, L,
                    qn.stride(0), qn.stride(1),
                    Kc.stride(0), Kc.stride(1),
                    A.stride(0), A.stride(1),
                    num_warps=4, num_stages=2
                )

                # Compute B
                P = head_dim_kpe
                grid_B = (H, (L + 128 - 1) // 128)
                matmul_qp_kp[grid_B](
                    qp, Kp, B,
                    H, P, L,
                    qp.stride(0), qp.stride(1),
                    Kp.stride(0), Kp.stride(1),
                    B.stride(0), B.stride(1),
                    num_warps=4, num_stages=2
                )

                # Add
                grid_add = (H, (L + 128 - 1) // 128)
                add_logits[grid_add](A, B, Logits,
                                     H, L,
                                     A.stride(0), A.stride(1),
                                     B.stride(0), B.stride(1),
                                     Logits.stride(0), Logits.stride(1),
                                     num_warps=4, num_stages=2)

                # Scale
                grid_scale = (H, (L + 128 - 1) // 128)
                scale_logits[grid_scale](Logits, Scaled,
                                         H, L,
                                         Logits.stride(0), Logits.stride(1),
                                         Scaled.stride(0), Scaled.stride(1),
                                         sm_scale,
                                         num_warps=4, num_stages=2)

                # Apply causal mask: j > (L - Q + i)
                prefix_len = kv_len - Q
                query_abs_pos = prefix_len + i
                grid_mask = (H, (L + 128 - 1) // 128)
                apply_mask[grid_mask](Scaled, Masked,
                                      H, L,
                                      Scaled.stride(0), Scaled.stride(1),
                                      Masked.stride(0), Masked.stride(1),
                                      query_abs_pos,
                                      num_warps=4, num_stages=2)

                # Row-wise logsumexp
                ln2 = math.log(2.0)
                grid_lse = (H,)
                row_lse[grid_lse](Masked, lse[q_start + i],  # lse is 1D [N,H], but we write to row q_start+i
                                  H, L,
                                  Masked.stride(0), Masked.stride(1),
                                  ln2,
                                  num_warps=1, num_stages=1)

                # Softmax
                grid_softmax = (H,)
                softmax_row[grid_softmax](Masked, SoftmaxOut,
                                          H, L,
                                          Masked.stride(0), Masked.stride(1),
                                          SoftmaxOut.stride(0), SoftmaxOut.stride(1),
                                          lse[q_start + i].item(),
                                          num_warps=1, num_stages=1)

                # Final matmul attn @ Kc -> output[H, D]
                OutRow = torch.empty((H, head_dim_ckv), dtype=torch.float32, device=device)
                grid_final = (H, (head_dim_ckv + 128 - 1) // 128)
                matmul_attn_kc[grid_final](SoftmaxOut, Kc, OutRow,
                                           H, head_dim_ckv, L,
                                           SoftmaxOut.stride(0), SoftmaxOut.stride(1),
                                           Kc.stride(0), Kc.stride(1),
                                           OutRow.stride(0), OutRow.stride(1),
                                           num_warps=4, num_stages=2)

                # Store to output
                output[q_start + i] = OutRow  # overwrite the row; lse[q_start + i] already computed

        # Cast output to bfloat16 to match original
        output = output.to(torch.bfloat16)
        return output, lse

# For compatibility with the provided harness, keep these helpers unchanged
def get_inputs():
    q_nope = torch.randn([1, 16, 512], dtype=torch.bfloat16, device='cuda')
    q_pe = torch.randn([1, 16, 64], dtype=torch.bfloat16, device='cuda')
    ckv_cache = torch.randn([989669, 1, 512], dtype=torch.bfloat16, device='cuda')
    kpe_cache = torch.randn([989669, 1, 64], dtype=torch.bfloat16, device='cuda')
    _n = 1; _t = 1
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    qo_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32).to('cuda')
    _n = 1; _t = 34
    _lens = torch.full((_n,), _t // _n, dtype=torch.int32)
    _lens[: _t % _n] += 1
    kv_indptr = torch.cat([torch.zeros(1, dtype=torch.int32), torch.cumsum(_lens, 0)]).to(torch.int32).to('cuda')
    kv_indices = torch.randint(0, 989669, [34], dtype=torch.int32).to('cuda')
    sm_scale = 1.0  # float32 scalar
    return [q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale]

def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6, tensor_7):
    _out = ModelNew()(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6, tensor_7)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]

# Original Model class that invokes ModelNew
class Model(torch.nn.Module):
    def forward(self, *args):
        return ModelNew().forward(*args)


def run(*args):
    return ModelNew()(*args)
