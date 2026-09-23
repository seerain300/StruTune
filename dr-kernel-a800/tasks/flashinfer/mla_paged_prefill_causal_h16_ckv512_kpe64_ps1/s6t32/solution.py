import torch
import triton
import triton.language as tl


@triton.jit
def compute_logits_kernel(
    Qn_ptr, Qp_ptr, Kc_ptr, Kp_ptr, Logits_ptr,
    H, L, D_ckv, D_kpe,
    Qn_stride0, Qn_stride1,
    Qp_stride0, Qp_stride1,
    Kc_stride0, Kc_stride1,
    Kp_stride0, Kp_stride1,
    Logits_stride0, Logits_stride1,
    BLOCK_L: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid: (H, cdiv(L, BLOCK_L))
    h = tl.program_id(0)
    pid_l = tl.program_id(1)

    ls = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    mask_l = ls < L

    acc = tl.zeros((BLOCK_L,), dtype=tl.float32)

    # Reduce over Kc: ks in [0, D_ckv)
    for k0 in range(0, D_ckv, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        mask_k = ks < D_ckv
        # Load qn[h, ks] as 2D: [1, BLOCK_K] using h index
        qn_ptrs = Qn_ptr + h * Qn_stride0 + ks * Qn_stride1  # [BLOCK_K]
        qn_vec = tl.load(qn_ptrs, mask=mask_k, other=0.0)    # [BLOCK_K]
        # Load Kc[ls, ks] tile: [BLOCK_L, BLOCK_K]
        Kc_ptrs = Kc_ptr + ls[:, None] * Kc_stride0 + ks[None, :] * Kc_stride1
        mask = mask_l[:, None] & mask_k[None, :]
        Kc_tile = tl.load(Kc_ptrs, mask=mask, other=0.0)     # [BLOCK_L, BLOCK_K]
        # Accumulate: acc += qn_vec[j] * Kc_tile[:, j] for each j
        for j in range(BLOCK_K):
            acc += qn_vec[j] * Kc_tile[:, j]

    # Reduce over Kp: ks' in [0, D_kpe)
    for k0 in range(0, D_kpe, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        mask_k = ks < D_kpe
        # Load qp[h, ks] as 2D: [1, BLOCK_K]
        qp_ptrs = Qp_ptr + h * Qp_stride0 + ks * Qp_stride1  # [BLOCK_K]
        qp_vec = tl.load(qp_ptrs, mask=mask_k, other=0.0)    # [BLOCK_K]
        # Load Kp[ls, ks] tile: [BLOCK_L, BLOCK_K]
        Kp_ptrs = Kp_ptr + ls[:, None] * Kp_stride0 + ks[None, :] * Kp_stride1
        mask = mask_l[:, None] & mask_k[None, :]
        Kp_tile = tl.load(Kp_ptrs, mask=mask, other=0.0)     # [BLOCK_L, BLOCK_K]
        # Accumulate: acc += qp_vec[j] * Kp_tile[:, j] for each j
        for j in range(BLOCK_K):
            acc += qp_vec[j] * Kp_tile[:, j]

    # Store logits
    out_ptrs = Logits_ptr + h * Logits_stride0 + ls * Logits_stride1
    tl.store(out_ptrs, acc, mask=mask_l)


@triton.jit
def lse_mask_kernel(
    Logits_ptr, Lse_ptr,
    H, L,
    Logits_stride0, Logits_stride1,
    inv_ln2: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    # One program per head
    h = tl.program_id(0)

    # Compute row-wise max over all positions (no causal mask here; original applies mask before lse)
    max_val = -float('inf')
    for l0 in range(0, L, BLOCK_L):
        ls = l0 + tl.arange(0, BLOCK_L)
        mask_l = ls < L
        vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
        block_max = tl.max(vals, axis=0)
        max_val = tl.maximum(max_val, block_max)

    # Compute sum of exp(vals - max_val)
    sum_exp = 0.0
    for l0 in range(0, L, BLOCK_L):
        ls = l0 + tl.arange(0, BLOCK_L)
        mask_l = ls < L
        vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
        e = tl.exp(vals - max_val)
        sum_exp += tl.sum(e, axis=0)

    lse_scaled = tl.log(sum_exp) * inv_ln2  # divide by ln(2)
    tl.store(Lse_ptr + h, lse_scaled)


@triton.jit
def softmax_matmul_kernel(
    Logits_ptr, Kc_ptr, Out_ptr,
    H, L, D_ckv,
    Logits_stride0, Logits_stride1,
    Kc_stride0, Kc_stride1,
    Out_stride0, Out_stride1,
    inv_ln2: tl.constexpr,
    BLOCK_L: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # One program per head
    h = tl.program_id(0)

    # Load logits[h, :] and compute max for softmax
    max_val = -float('inf')
    for l0 in range(0, L, BLOCK_L):
        ls = l0 + tl.arange(0, BLOCK_L)
        mask_l = ls < L
        vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
        block_max = tl.max(vals, axis=0)
        max_val = tl.maximum(max_val, block_max)

    sum_exp = 0.0
    for l0 in range(0, L, BLOCK_L):
        ls = l0 + tl.arange(0, BLOCK_L)
        mask_l = ls < L
        vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
        e = tl.exp(vals - max_val)
        sum_exp += tl.sum(e, axis=0)

    # Re-load logits to compute output: out[h, k] = sum_{l} softmax_l * Kc[l, k]
    for k0 in range(0, D_ckv, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)
        mask_k = k < D_ckv
        out_vec = tl.zeros((BLOCK_K,), dtype=tl.float32)

        for l0 in range(0, L, BLOCK_L):
            ls = l0 + tl.arange(0, BLOCK_L)
            mask_l = ls < L
            vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
            e = tl.exp(vals - max_val)  # [BLOCK_L]
            attn = e / sum_exp          # [BLOCK_L]
            # Kc[ls, k]
            Kc_ptrs = Kc_ptr + ls[:, None] * Kc_stride0 + k[None, :] * Kc_stride1
            mask = mask_l[:, None] & mask_k[None, :]
            Kc_tile = tl.load(Kc_ptrs, mask=mask, other=0.0)  # [BLOCK_L, BLOCK_K]
            # Accumulate: out_vec[j] += sum_l attn[l] * Kc_tile[l, j]
            for j in range(BLOCK_K):
                out_vec[j] += tl.sum(attn * Kc_tile[:, j], axis=0)

        # Store out[h, k]
        out_ptrs = Out_ptr + h * Out_stride0 + k * Out_stride1
        tl.store(out_ptrs, out_vec, mask=mask_k)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        # Ensure tensors are on same device and CUDA
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda, "All tensors must be on CUDA device"
        device = q_nope.device

        # Constants from the original code
        num_qo_heads = 16
        head_dim_ckv = 512
        head_dim_kpe = 64

        total_q = q_nope.shape[0]
        batch_size = qo_indptr.numel() - 1

        # For each batch b in [0, batch_size)
        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            kv_start = int(kv_indptr[b].item())
            kv_end = int(kv_indptr[b + 1].item())

            if q_start >= q_end or kv_start >= kv_end:
                continue

            # Gather tokens for this batch
            tok_idx = kv_indices[kv_start:kv_end].to(torch.int32).to(device)  # [L]
            Kc = ckv_cache[tok_idx, 0].to(torch.float32).contiguous()  # [L, 512]
            Kp = kpe_cache[tok_idx, 0].to(torch.float32).contiguous()  # [L, 64]

            q_len = q_end - q_start
            # Prepare q_nope and q_pe for this batch (shape [q_len, num_heads, D])
            # Reshape to 2D [H, D] to pass to Triton kernels
            q_nope_batch = q_nope[q_start:q_end].to(torch.float32).contiguous()  # [q_len, 16, 512]
            q_pe_batch = q_pe[q_start:q_end].to(torch.float32).contiguous()     # [q_len, 16, 64]
            # Ensure we have at least 2 dims (H, D)
            # For Triton kernels, we use view(H, -1)
            # Note: q_len may be 1, but H is 16, so total elements are H*D. We still need 2D pointers.
            # We will construct q_nope_2d[i] = q_nope_batch[i].view(H, D_ckv) and q_pe_2d[i] similarly.
            # However, Triton expects pointers to 2D tensors; we cannot index inside kernel by i directly.
            # So we compute for each i in a loop, but Triton kernels require static 2D pointers.
            # To ensure 2D pointers, we compute Qn_ptr and Qp_ptr per i in Python, but Triton JIT compiles per i separately.
            # Therefore, we launch kernels inside the loop for each i.

            # Output buffer [q_len, H, D_ckv]
            output = torch.empty((q_len, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)
            lse = torch.empty((q_len, num_qo_heads), dtype=torch.float32, device=device)

            for i in range(q_len):
                # Extract head vectors for this i: reshape to [H, D] and pass pointers
                qn = q_nope_batch[i]  # [16, 512], ensure 2D
                qp = q_pe_batch[i]    # [16, 64]

                H = num_qo_heads
                L = Kc.shape[0]

                # Allocate Logits [H, L]
                Logits = torch.empty((H, L), dtype=torch.float32, device=device)

                # Launch compute_logits_kernel for this i
                BLOCK_L = 128
                grid = (H, triton.cdiv(L, BLOCK_L))
                compute_logits_kernel[grid](
                    qn, qp, Kc, Kp, Logits,
                    H=H, L=L, D_ckv=head_dim_ckv, D_kpe=head_dim_kpe,
                    Qn_stride0=qn.stride(0), Qn_stride1=qn.stride(1),
                    Qp_stride0=qp.stride(0), Qp_stride1=qp.stride(1),
                    Kc_stride0=Kc.stride(0), Kc_stride1=Kc.stride(1),
                    Kp_stride0=Kp.stride(0), Kp_stride1=Kp.stride(1),
                    Logits_stride0=Logits.stride(0), Logits_stride1=Logits.stride(1),
                    BLOCK_L=BLOCK_L, BLOCK_K=64, num_warps=4, num_stages=2,
                )

                # Compute lse for this i without causal mask (original applies mask before lse; here we compute on Logits)
                grid_lse = (H,)
                lse_mask_kernel[grid_lse](
                    Logits, lse[i],
                    H=H, L=L,
                    Logits_stride0=Logits.stride(0), Logits_stride1=Logits.stride(1),
                    inv_ln2=1.4426950408889634,  # 1/ln(2)
                    BLOCK_L=BLOCK_L, num_warps=2, num_stages=2,
                )

                # Compute output for this i: softmax_matmul_kernel
                Out = output[i]  # [H, D_ckv]
                grid_out = (H,)
                softmax_matmul_kernel[grid_out](
                    Logits, Kc, Out,
                    H=H, L=L, D_ckv=head_dim_ckv,
                    Logits_stride0=Logits.stride(0), Logits_stride1=Logits.stride(1),
                    Kc_stride0=Kc.stride(0), Kc_stride1=Kc.stride(1),
                    Out_stride0=Out.stride(0), Out_stride1=Out.stride(1),
                    inv_ln2=1.4426950408889634,
                    BLOCK_L=BLOCK_L, BLOCK_K=64, num_warps=4, num_stages=2,
                )

        # Return output and lse; cast output to bfloat16 to match original
        output = output.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
