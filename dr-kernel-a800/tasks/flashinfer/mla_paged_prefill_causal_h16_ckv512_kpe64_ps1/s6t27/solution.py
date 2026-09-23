import math
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

    # Load q vectors for head h (shape broadcasted as [1, 1, D])
    # We pass Qn_ptr with stride0=H and stride1=D to form [1, 1, D] pointer arithmetic.
    qn_base = tl.load(Qn_ptr + h * Qn_stride0, mask=True, other=0.0)  # [D_ckv]
    qn_vals = tl.load(Qn_ptr + h * Qn_stride0 + tl.arange(0, D_ckv) * Qn_stride1, mask=True, other=0.0)  # [D_ckv]

    qh = tl.load(Qp_ptr + h * Qp_stride0, mask=True, other=0.0)  # [D_kpe]
    qp_vals = tl.load(Qp_ptr + h * Qp_stride0 + tl.arange(0, D_kpe) * Qp_stride1, mask=True, other=0.0)  # [D_kpe]

    acc = tl.zeros((BLOCK_L,), dtype=tl.float32)

    # Accumulate over Kc
    for k0 in range(0, D_ckv, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        mask_k = ks < D_ckv
        qn_chunk = tl.load(Qn_ptr + h * Qn_stride0 + ks * Qn_stride1, mask=mask_k, other=0.0)  # [BLOCK_K]
        Kc_ptrs = Kc_ptr + ls * Kc_stride0 + ks * Kc_stride1  # [BLOCK_L, BLOCK_K]
        Kc_vals = tl.load(Kc_ptrs, mask=mask_l[:, None] & mask_k[None, :], other=0.0)  # [BLOCK_L, BLOCK_K]
        # Dot: (BLOCK_L, BLOCK_K) @ (BLOCK_K,) -> (BLOCK_L,)
        acc += tl.sum(Kc_vals * qn_chunk[None, :], axis=1)

    # Accumulate over Kp
    for k0 in range(0, D_kpe, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        mask_k = ks < D_kpe
        qp_chunk = tl.load(Qp_ptr + h * Qp_stride0 + ks * Qp_stride1, mask=mask_k, other=0.0)  # [BLOCK_K]
        Kp_ptrs = Kp_ptr + ls * Kp_stride0 + ks * Kp_stride1  # [BLOCK_L, BLOCK_K]
        Kp_vals = tl.load(Kp_ptrs, mask=mask_l[:, None] & mask_k[None, :], other=0.0)  # [BLOCK_L, BLOCK_K]
        acc += tl.sum(Kp_vals * qp_chunk[None, :], axis=1)

    # Store acc to Logits[h, ls]
    out_ptrs = Logits_ptr + h * Logits_stride0 + ls * Logits_stride1
    tl.store(out_ptrs, acc, mask=mask_l)


@triton.jit
def masked_lse_kernel(
    Logits_ptr, Mask_ptr, LSE_ptr,
    H, L, inv_ln2,
    Logits_stride0, Logits_stride1,
    Mask_stride0,
):
    # One program per head
    h = tl.program_id(0)
    # Row-wise max with mask
    max_val = -float('inf')
    for l0 in range(0, L, 1):  # Triton prefers tl.constexpr loops; we loop scalar
        ls = l0
        mask_l = ls < L
        val = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
        max_val = tl.maximum(max_val, val)

    # Sum of exp over masked values
    sum_exp = 0.0
    for l0 in range(0, L, 1):
        ls = l0
        mask_l = ls < L
        # Load mask
        mask_val = tl.load(Mask_ptr + ls * Mask_stride0, mask=True, other=0)
        val = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
        val = tl.where(mask_val != 0, val, -float('inf'))
        sum_exp += tl.exp(val - max_val)

    lse_scaled = tl.log(sum_exp) * inv_ln2
    tl.store(LSE_ptr + h, lse_scaled)


@triton.jit
def masked_softmax_matmul_kernel_write(
    Logits_ptr, Kc_ptr, Output_ptr,
    H, L, D_ckv,
    Kc_stride0, Kc_stride1,
    Output_stride0, Output_stride1,
    BLOCK_L: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # One program per (i, h)
    h = tl.program_id(0)
    i = tl.program_id(1)

    # Compute per-head max (for softmax stability)
    max_val = -float('inf')
    for l0 in range(0, L, BLOCK_L):
        ls = l0 + tl.arange(0, BLOCK_L)
        mask_l = ls < L
        vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
        block_max = tl.max(vals, axis=0)
        max_val = tl.maximum(max_val, block_max)

    # Compute sum_exp = sum(exp(vals - max_val)) with causal mask from Logits (implicitly handled via vals after mask application).
    sum_exp = 0.0
    for l0 in range(0, L, BLOCK_L):
        ls = l0 + tl.arange(0, BLOCK_L)
        mask_l = ls < L
        vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
        e = tl.exp(vals - max_val)
        sum_exp += tl.sum(e, axis=0)

    # Prepare Output[h, :] = 0
    out_row = tl.zeros((D_ckv,), dtype=tl.float32)

    # Compute output: out[h, :] = sum_l softmax[l] * Kc[l, :]
    for l0 in range(0, L, BLOCK_L):
        ls = l0 + tl.arange(0, BLOCK_L)
        mask_l = ls < L
        vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
        e = tl.exp(vals - max_val)  # raw exp for softmax
        # Convert to softmax probabilities
        # Note: masked_lse_kernel would have set lse; here we assume sum_exp computed. For masked_softmax, we need -inf for non-causal.
        # We recompute softmax probabilities by hand using sum_exp.
        # softmax = e / sum_exp
        softmax_vals = e / sum_exp  # vector over BLOCK_L

        # Accumulate into out_row
        for k0 in range(0, D_ckv, BLOCK_K):
            ks = k0 + tl.arange(0, BLOCK_K)
            mask_k = ks < D_ckv
            Kc_ptrs = Kc_ptr + ls * Kc_stride0 + ks * Kc_stride1  # [BLOCK_L, BLOCK_K]
            Kc_vals = tl.load(Kc_ptrs, mask=mask_l[:, None] & mask_k[None, :], other=0.0)  # [BLOCK_L, BLOCK_K]
            # out_row += sum_l softmax_vals[l] * Kc[l, ks]
            # Broadcast softmax_vals over ks: (BLOCK_L,) * (BLOCK_L, BLOCK_K) -> (BLOCK_K,)
            out_row += tl.sum(softmax_vals[:, None] * Kc_vals, axis=0)

    # Store out_row to Output[i, h, :]
    out_ptrs = Output_ptr + i * Output_stride0 + h * Output_stride1
    # Output tensor is (total_q, H, D_ckv), but we keep it (H, D_ckv) for simplicity. Adjust stride accordingly:
    # Since we launch grid=(H, total_q), we need to store at position (i, h). Use Output_stride0 for i, Output_stride1 for h.
    # Store entire row:
    for k in range(0, D_ckv):
        tl.store(out_ptrs + k, out_row[k])


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        device = q_nope.device
        assert q_nope.dtype == torch.bfloat16 and q_pe.dtype == torch.bfloat16
        assert qo_indptr.dim() == 1 and kv_indptr.dim() == 1 and kv_indices.dim() == 1

        # Gather shapes
        total_q = q_nope.shape[0]  # batch of queries (likely 1 in provided get_inputs)
        H = q_nope.shape[1]
        D_ckv = q_nope.shape[2]
        D_kpe = q_pe.shape[2]
        num_pages = ckv_cache.shape[0]
        assert ckv_cache.shape[1] == 1 and kpe_cache.shape[1] == 1
        assert q_nope.shape == (total_q, H, D_ckv)
        assert q_pe.shape == (total_q, H, D_kpe)
        assert H == 16 and D_ckv == 512 and D_kpe == 64

        # Compute number of batches and per-batch q ranges
        batch_size = qo_indptr.numel() - 1
        qo_indptr_cpu = qo_indptr.cpu()
        kv_indptr_cpu = kv_indptr.cpu()

        # Prepare output and lse buffers
        output = torch.empty((total_q, H, D_ckv), dtype=torch.float32, device=device)
        lse = torch.empty((total_q, H), dtype=torch.float32, device=device)

        for b in range(batch_size):
            q_start = int(qo_indptr_cpu[b].item())
            q_end = int(qo_indptr_cpu[b + 1].item())

            # KV block for this batch b
            kv_len = int(kv_indptr_cpu[b + 1].item()) - int(kv_indptr_cpu[b].item())
            # Gather tokens used for K/V
            tok_idx = kv_indices[b:b + kv_len].to(torch.int32).to(device)
            Kc = ckv_cache[tok_idx, 0].to(torch.float32)  # [kv_len, 512]
            Kp = kpe_cache[tok_idx, 0].to(torch.float32)  # [kv_len, 64]

            # Current query segment
            q_nope_batch = q_nope[q_start:q_end].to(torch.float32)  # [q_len, 16, 512]
            q_pe_batch = q_pe[q_start:q_end].to(torch.float32)      # [q_len, 16, 64]
            q_len = q_end - q_start

            # For each query position i in this batch
            for i in range(q_len):
                qn = q_nope_batch[i]  # [16, 512]
                qp = q_pe_batch[i]    # [16, 64]

                # Allocate intermediate
                Logits = torch.empty((H, kv_len), dtype=torch.float32, device=device)

                # Launch compute_logits_kernel
                BLOCK_L = 128
                BLOCK_K = 32
                grid = (H, triton.cdiv(kv_len, BLOCK_L))
                compute_logits_kernel[grid](
                    qn, qp, Kc, Kp, Logits,
                    H, kv_len, D_ckv, D_kpe,
                    qn.stride(0), qn.stride(1),
                    qp.stride(0), qp.stride(1),
                    Kc.stride(0), Kc.stride(1),
                    Kp.stride(0), Kp.stride(1),
                    Logits.stride(0), Logits.stride(1),
                    BLOCK_L=BLOCK_L, BLOCK_K=BLOCK_K,
                )

                # Build causal mask: for each l, causal if l <= (kv_len - q_len + i)
                abs_pos = kv_len - q_len + i  # absolute position of current query
                mask = torch.arange(kv_len, device=device, dtype=torch.int32)
                mask = (mask <= abs_pos).to(torch.int32)  # 1 for causal, 0 for non-causal

                # Launch masked_lse_kernel
                lse_i = torch.empty((H,), dtype=torch.float32, device=device)
                masked_lse_kernel[(H,)](
                    Logits, mask,
                    H, kv_len, 1.4426950408889634,  # 1 / ln(2)
                    Logits.stride(0), Logits.stride(1),
                    mask.stride(0),
                )
                lse[q_start + i] = lse_i  # store per head

                # Launch masked_softmax_matmul_kernel_write
                Output_i = torch.empty((H, D_ckv), dtype=torch.float32, device=device)
                masked_softmax_matmul_kernel_write[(H,)](
                    Logits, Kc, Output_i,
                    H, kv_len, D_ckv,
                    Kc.stride(0), Kc.stride(1),
                    Output_i.stride(0), Output_i.stride(1),
                    BLOCK_L=BLOCK_L, BLOCK_K=BLOCK_K,
                )
                output[q_start + i] = Output_i  # store per head

        # Return as original: output bfloat16, lse float32
        return output.to(torch.bfloat16), lse


def run(*args):
    return ModelNew()(*args)
