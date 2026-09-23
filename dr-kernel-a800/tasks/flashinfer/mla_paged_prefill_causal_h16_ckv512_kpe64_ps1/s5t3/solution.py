import torch
import math
import triton
import triton.language as tl


# ---------------------------
# Triton kernels
# ---------------------------

@triton.jit
def matmul_qn_kc(qn_ptr, kc_ptr, out_ptr,
                 H: tl.constexpr, L: tl.constexpr, D: tl.constexpr,
                 stride_qn_h, stride_qn_d,
                 stride_kc_l, stride_kc_d,
                 stride_out_h, stride_out_l):
    # out[h, l] = sum_d qn[h, d] * kc[l, d]
    h = tl.program_id(0)
    l = tl.program_id(1)
    # accumulator
    acc = 0.0
    # loop over d
    for d in range(0, D):
        q = tl.load(qn_ptr + h * stride_qn_h + d * stride_qn_d)
        k = tl.load(kc_ptr + l * stride_kc_l + d * stride_kc_d)
        acc += q * k
    tl.store(out_ptr + h * stride_out_h + l * stride_out_l, acc)


@triton.jit
def matmul_qp_kp(qp_ptr, kp_ptr, out_ptr,
                 H: tl.constexpr, L: tl.constexpr, P: tl.constexpr,
                 stride_qp_h, stride_qp_p,
                 stride_kp_l, stride_kp_p,
                 stride_out_h, stride_out_l):
    # out[h, l] = sum_p qp[h, p] * kp[l, p]
    h = tl.program_id(0)
    l = tl.program_id(1)
    acc = 0.0
    for p in range(0, P):
        q = tl.load(qp_ptr + h * stride_qp_h + p * stride_qp_p)
        k = tl.load(kp_ptr + l * stride_kp_l + p * stride_kp_p)
        acc += q * k
    tl.store(out_ptr + h * stride_out_h + l * stride_out_l, acc)


@triton.jit
def add_logits(a_ptr, b_ptr, out_ptr,
               H: tl.constexpr, L: tl.constexpr,
               stride_a_h, stride_a_l,
               stride_b_h, stride_b_l,
               stride_out_h, stride_out_l):
    h = tl.program_id(0)
    l = tl.program_id(1)
    a = tl.load(a_ptr + h * stride_a_h + l * stride_a_l)
    b = tl.load(b_ptr + h * stride_b_h + l * stride_b_l)
    tl.store(out_ptr + h * stride_out_h + l * stride_out_l, a + b)


@triton.jit
def scale_logits(x_ptr, out_ptr, scale,
                 H: tl.constexpr, L: tl.constexpr,
                 stride_x_h, stride_x_l,
                 stride_out_h, stride_out_l):
    h = tl.program_id(0)
    l = tl.program_id(1)
    x = tl.load(x_ptr + h * stride_x_h + l * stride_x_l)
    y = x * scale
    tl.store(out_ptr + h * stride_out_h + l * stride_out_l, y)


@triton.jit
def apply_mask(x_ptr, out_ptr, mask_ptr, H: tl.constexpr, L: tl.constexpr,
               stride_x_h, stride_x_l,
               stride_out_h, stride_out_l,
               stride_mask_h, stride_mask_l):
    # mask[h, l] = True if l > query_abs_pos, else False
    h = tl.program_id(0)
    l = tl.program_id(1)
    x = tl.load(x_ptr + h * stride_x_h + l * stride_x_l)
    mval = tl.load(mask_ptr + h * stride_mask_h + l * stride_mask_l)  # 0.0 or 1.0
    # if not masked, keep; else set to -inf
    is_masked = mval == 0.0
    y = tl.where(is_masked, x, -float("inf"))
    tl.store(out_ptr + h * stride_out_h + l * stride_out_l, y)


@triton.jit
def row_logsumexp(x_ptr, lse_ptr, H: tl.constexpr, L: tl.constexpr,
                  stride_x_h, stride_x_l):
    # compute lse[h] = log(sum_j exp(x[h, j] - m)) + m, where m = max_j x[h, j]
    h = tl.program_id(0)
    max_val = -float("inf")
    for j in range(0, L):
        x = tl.load(x_ptr + h * stride_x_h + j * stride_x_l)
        max_val = tl.maximum(max_val, x)
    sum_exp = 0.0
    for j in range(0, L):
        x = tl.load(x_ptr + h * stride_x_h + j * stride_x_l)
        e = tl.exp(x - max_val)
        sum_exp += e
    lse = tl.log(sum_exp) + max_val  # log2 via multiply by 1/ln(2) later in host
    tl.store(lse_ptr + h, lse)


@triton.jit
def softmax_row_triton(x_ptr, lse_ptr, soft_ptr, H: tl.constexpr, L: tl.constexpr,
                       stride_x_h, stride_x_l,
                       stride_soft_h, stride_soft_l):
    h = tl.program_id(0)
    lse = tl.load(lse_ptr + h)
    for j in range(0, L):
        x = tl.load(x_ptr + h * stride_x_h + j * stride_x_l)
        s = tl.exp(x - lse)
        tl.store(soft_ptr + h * stride_soft_h + j * stride_soft_l, s)


@triton.jit
def matmul_attn_kc(soft_ptr, kc_ptr, out_ptr,
                   H: tl.constexpr, L: tl.constexpr, D: tl.constexpr,
                   stride_soft_h, stride_soft_l,
                   stride_kc_l, stride_kc_d,
                   stride_out_h, stride_out_d):
    # out[h, d] = sum_l soft[h, l] * kc[l, d]
    h = tl.program_id(0)
    d = tl.program_id(1)
    acc = 0.0
    for l in range(0, L):
        s = tl.load(soft_ptr + h * stride_soft_h + l * stride_soft_l)
        k = tl.load(kc_ptr + l * stride_kc_l + d * stride_kc_d)
        acc += s * k
    tl.store(out_ptr + h * stride_out_h + d * stride_out_d, acc)


# ---------------------------
# Triton launcher for ModelNew
# ---------------------------

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Device setup
        device = q_nope.device
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda
        # Prepare Kc_all and Kp_all: squeeze the "1" dimension
        Kc_all = ckv_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, D]
        Kp_all = kpe_cache.squeeze(1).contiguous().to(torch.float32)  # [num_pages, P]

        total_q, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]
        assert num_qo_heads == 16
        assert head_dim_ckv == 512
        assert head_dim_kpe == 64

        # output and lse initialization
        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        # Process each batch b
        len_indptr = qo_indptr.shape[0]
        batch_size = len_indptr - 1
        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            if q_start >= q_end:
                continue

            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            if page_beg >= page_end:
                continue

            kv_len = page_end - page_beg
            tok_idx = kv_indices[page_beg:page_end].to(torch.int32).to(device)  # [L]

            Kc = Kc_all[tok_idx].contiguous()  # [L, D]
            Kp = Kp_all[tok_idx].contiguous()  # [L, P]

            q_len = q_end - q_start

            # Iterate queries within this batch
            for i in range(q_len):
                # Slice q_nope and q_pe for this query and all heads
                # q_nope: [N, H, D] -> q_start + i -> H slices across D
                qn = q_nope[q_start + i]  # [H, D], bfloat16
                qp = q_pe[q_start + i]    # [H, P], bfloat16

                # Cast to float32 for compute
                qn_f = qn.contiguous().to(torch.float32)    # [H, D]
                qp_f = qp.contiguous().to(torch.float32)    # [H, P]

                Dmat = Kc.shape[1]  # D = 512
                Lmat = Kc.shape[0]  # L = len(kv_indices[page_beg:page_end])
                Pmat = Kp.shape[1]  # P = 64

                # Allocate intermediate buffers (float32)
                A = torch.empty((num_qo_heads, Lmat), dtype=torch.float32, device=device)  # [H, L]
                B = torch.empty((num_qo_heads, Lmat), dtype=torch.float32, device=device)  # [H, L]
                C = torch.empty((num_qo_heads, Lmat), dtype=torch.float32, device=device)  # [H, L]
                D_scaled = torch.empty((num_qo_heads, Lmat), dtype=torch.float32, device=device)  # [H, L]
                Mask = torch.empty((num_qo_heads, Lmat), dtype=torch.float32, device=device)  # [H, L], 0.0 or 1.0

                # Compute query absolute position
                query_abs_pos = (Lmat - q_len) + i  # per original code

                # Launch matmul for qn @ Kc.T -> A
                grid_A = (num_qo_heads, Lmat)
                matmul_qn_kc[grid_A](
                    qn_f, Kc,
                    A,
                    H=num_qo_heads, L=Lmat, D=Dmat,
                    stride_qn_h=qn_f.stride(0), stride_qn_d=qn_f.stride(1),
                    stride_kc_l=Kc.stride(0), stride_kc_d=Kc.stride(1),
                    stride_out_h=A.stride(0), stride_out_l=A.stride(1)
                )

                # Launch matmul for qp @ Kp.T -> B
                grid_B = (num_qo_heads, Lmat)
                matmul_qp_kp[grid_B](
                    qp_f, Kp,
                    B,
                    H=num_qo_heads, L=Lmat, P=Pmat,
                    stride_qp_h=qp_f.stride(0), stride_qp_p=qp_f.stride(1),
                    stride_kp_l=Kp.stride(0), stride_kp_p=Kp.stride(1),
                    stride_out_h=B.stride(0), stride_out_l=B.stride(1)
                )

                # Add A + B -> C
                grid_add = (num_qo_heads, Lmat)
                add_logits[grid_add](
                    A, B, C,
                    H=num_qo_heads, L=Lmat,
                    stride_a_h=A.stride(0), stride_a_l=A.stride(1),
                    stride_b_h=B.stride(0), stride_b_l=B.stride(1),
                    stride_out_h=C.stride(0), stride_out_l=C.stride(1)
                )

                # Scale C by sm_scale
                grid_scale = (num_qo_heads, Lmat)
                scale_logits[grid_scale](
                    C, D_scaled,
                    sm_scale,
                    H=num_qo_heads, L=Lmat,
                    stride_x_h=C.stride(0), stride_x_l=C.stride(1),
                    stride_out_h=D_scaled.stride(0), stride_out_l=D_scaled.stride(1)
                )

                # Build mask: [H, L], True if l > query_abs_pos
                # torch.where for mask: 1.0 if True else 0.0
                # Create boolean mask tensor for device
                row_mask = torch.arange(Lmat, device=device) > query_abs_pos  # [L]
                # Broadcast to [H, L]
                # Convert to float32 0/1
                # Use torch operations for mask preparation (host side), Triton applies it.
                mask_vals = torch.where(row_mask, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))
                # Make sure it's broadcastable; we'll pass D_scaled and mask as pointers and rely on contiguous layout
                # Note: Here we need Mask with shape [H, L] of 0.0/1.0
                # Create per-row mask
                for h in range(num_qo_heads):
                    mask_row = mask_vals if row_mask else mask_vals  # same for all h
                    # Store in Mask[h, :]
                    # Since Mask is [H, L], we write row h
                    # To avoid recomputing, fill entire Mask with mask_vals repeated H times
                    # But Triton kernel expects a pointer, so we create a 2D tensor
                    # Create 2D Mask tensor explicitly
                    Mask[h, :] = mask_vals
                    # Also store absolute 0.0/1.0 values in Mask for Triton to read
                    # We already created Mask as float32, filled above via assignment; we can set the whole tensor using torch indexing:
                    # Since we can't write inside Triton from Python per-row, we compute and pass it directly.

                    # However, to keep Triton kernels self-contained, we can compute Mask directly inside Triton by recomputing arange, but Triton kernels here
                    # don't take mask as input; apply_mask kernel expects mask_ptr. To avoid inconsistency, we compute Mask as a tensor and pass it.
                    # Above lines were illustrative; we actually set Mask as filled before kernel call. Let's correct by filling Mask tensor.
                    # Fill Mask tensor before kernel call using broadcasting:
                    # We need to assign mask_vals to each row; we can do it once:
                    # But since we already have A, B, C, D_scaled, we need to fill Mask. Let's do it:
                    # We can't fill row-wise here; better approach: compute mask_vals and write into Mask[h, :] by broadcasting.
                    # torch.zeros((num_qo_heads, Lmat), device=device), then fill rows.
                    # But we already created Mask as empty. Let's fix by creating Mask properly:
                    # We need to fill Mask with ones for positions > query_abs_pos, zeros otherwise. Triton doesn't need it pre-filled; apply_mask uses mask_vals.
                    # We'll pass Mask as a float32 tensor of 0.0/1.0 computed via torch.where.

                # The above Mask creation can be simplified by using torch.arange and broadcasting:
                # Prepare mask tensor for Triton: shape [H, L], dtype float32, 0.0 or 1.0
                # Use torch to build it:
                # row index h is not needed for mask; mask depends only on l > query_abs_pos.
                # We can broadcast a 1D mask to [H, L].
                # Create a 2D mask tensor:
                # Note: We need to create Mask of shape [H, L] with identical 1/0 per row. This is acceptable since mask depends only on l.
                # Generate ones and zeros for all rows:
                # But Triton kernel apply_mask expects mask[h, l], we can pass a 2D tensor of shape [H, L] filled with mask_vals.
                # Since mask_vals is [L], we can repeat it H times along rows: torch.stack([mask_vals]*H, dim=0) would create a [H, L] tensor.
                # Let's create Mask properly:
                # We'll compute Mask as a torch tensor:
                # However, to keep consistency, we'll compute Mask once using torch broadcasting:
                # Create a boolean mask vector of length L, then expand to [H, L] and convert to float32.
                l_vec = torch.arange(Lmat, device=device)
                bool_mask_vec = l_vec > query_abs_pos  # [L], boolean
                # Create a 2D boolean mask and convert to float32
                # We need to broadcast to [H, L]; create a dummy h index and expand:
                # Since we don't have h, we can broadcast to [1, L] and expand to [H, L] by unsqueeze:
                # But we need to repeat across H dimension. In Triton, we'll pass a tensor of shape [H, L].
                # Let's construct Mask tensor as zeros and fill using torch indexing per row. However, since Triton expects a pointer, we can simply create Mask as:
                # Mask = torch.empty((num_qo_heads, Lmat), dtype=torch.float32, device=device)
                # Fill Mask with mask_vals for each row:
                # We can do it in a small loop:
                for h in range(num_qo_heads):
                    Mask[h, :] = mask_vals

                # Now apply mask in Triton: we need a pointer to Mask. But we haven't filled Mask? We did above. Let's reconfirm:
                # We created Mask as empty and filled in a loop. The loop runs H times, so Mask is filled.
                # We need to pass it to apply_mask kernel.

                # Apply mask: D_scaled -> out_masked
                D_out = torch.empty_like(D_scaled)  # [H, L]
                grid_mask = (num_qo_heads, Lmat)
                apply_mask[grid_mask](
                    D_scaled, D_out, Mask,
                    H=num_qo_heads, L=Lmat,
                    stride_x_h=D_scaled.stride(0), stride_x_l=D_scaled.stride(1),
                    stride_out_h=D_out.stride(0), stride_out_l=D_out.stride(1),
                    stride_mask_h=Mask.stride(0), stride_mask_l=Mask.stride(1)
                )

                # Row-wise logsumexp over masked logits -> lse[h]
                grid_lse = (num_qo_heads,)
                row_logsumexp[grid_lse](
                    D_out, lse,
                    H=num_qo_heads, L=Lmat,
                    stride_x_h=D_out.stride(0), stride_x_l=D_out.stride(1)
                )

                # Softmax per row: D_out (masked) -> softmax
                Soft = torch.empty((num_qo_heads, Lmat), dtype=torch.float32, device=device)
                grid_softmax = (num_qo_heads,)
                softmax_row_triton[grid_softmax](
                    D_out, lse, Soft,
                    H=num_qo_heads, L=Lmat,
                    stride_x_h=D_out.stride(0), stride_x_l=D_out.stride(1),
                    stride_soft_h=Soft.stride(0), stride_soft_l=Soft.stride(1)
                )

                # Final matmul: softmax[h, :] @ Kc[L, D] -> out[h, D]
                Out = torch.empty((num_qo_heads, Dmat), dtype=torch.float32, device=device)
                grid_final = (num_qo_heads, Dmat)
                matmul_attn_kc[grid_final](
                    Soft, Kc, Out,
                    H=num_qo_heads, L=Lmat, D=Dmat,
                    stride_soft_h=Soft.stride(0), stride_soft_l=Soft.stride(1),
                    stride_kc_l=Kc.stride(0), stride_kc_d=Kc.stride(1),
                    stride_out_h=Out.stride(0), stride_out_d=Out.stride(1)
                )

                # Store output for this query i across all heads
                # output shape [total_q, H, D], we write at row q_start + i
                output[q_start + i] = Out.to(torch.bfloat16)

                # lse per head for this query
                # lse[q_start + i] = lse[h] (lse is vector of length H)
                # Assign each head
                for h in range(num_qo_heads):
                    lse[q_start + i, h] = lse[h]

        return output, lse


def run(*args):
    return ModelNew()(*args)
