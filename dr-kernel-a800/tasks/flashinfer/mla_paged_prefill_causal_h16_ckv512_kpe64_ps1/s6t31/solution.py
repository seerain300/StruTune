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

    # Accumulator for logits[h, ls]
    acc = tl.zeros((BLOCK_L,), dtype=tl.float32)

    # Reduce over Kc: ks in [0, D_ckv)
    for k0 in range(0, D_ckv, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        mask_k = ks < D_ckv
        # Load qn[h, ks] (vector over ks), note: Qn shape [Q, H, D_ckv], we index fixed h
        qn_vec = tl.load(Qn_ptr + h * Qn_stride1 + ks * Qn_stride1, mask=mask_k, other=0.0)  # shape [BLOCK_K]
        # Load Kc[ls, ks] (matrix [BLOCK_L, BLOCK_K])
        Kc_ptrs = Kc_ptr + ls[:, None] * Kc_stride0 + ks[None, :] * Kc_stride1
        Kc_block = tl.load(Kc_ptrs, mask=mask_l[:, None] & mask_k[None, :], other=0.0)  # [BLOCK_L, BLOCK_K]
        # Accumulate dot: sum over K block
        acc += tl.sum(Kc_block * qn_vec[None, :], axis=1)

    # Reduce over Kp: ks in [D_ckv, D_ckv+D_kpe)
    for k0 in range(0, D_kpe, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        mask_k = ks < D_kpe
        # Load qn for ks >= D_ckv? We need to load q vectors; in this design, we keep Qn as q_nope and pass Qp as q_pe. To compute qn contribution, we need Qn[h, :]; we load from Qn_ptr using ks < D_ckv. However, we cannot branch with Python if; instead, we pre-load qn_vec_q for ks<D_ckv, and q_vec_p for ks>=D_ckv from Qp_ptr, because they share the same indexing convention. Here, we assume Qp corresponds to the same query index and head; but in the kernel we should load Qn for ks<D_ckv and Qp for ks>=D_ckv. To do this cleanly, we re-load qn_vec_q and qp_vec from Qn_ptr and Qp_ptr respectively for all ks, and then mask by ks<D_ckv and ks>=D_ckv.
        # Load Qn for ks < D_ckv
        qn_vec_q = tl.load(Qn_ptr + h * Qn_stride1 + ks * Qn_stride1, mask=mask_k, other=0.0)  # [BLOCK_K]
        # Load Qp for all ks, then mask by ks>=D_ckv
        qp_vec = tl.load(Qp_ptr + h * Qp_stride1 + ks * Qp_stride1, mask=mask_k, other=0.0)  # [BLOCK_K]
        # Build q contrib: for ks<D_ckv use qn_vec_q, else use 0; for ks>=D_ckv use qp_vec - qn_vec_q (but qn_vec_q for ks>=D_ckv is zero). To simplify, we compute:
        # q_contrib = where(ks<D_ckv, qn_vec_q, where(ks>=D_ckv, qp_vec, 0))
        # But Triton does not have vectorized where with tensor masks easily here. Instead, we perform two pointer loads: one for Qn and one for Qp, selecting via ks<D_ckv. We can't select directly; so we instead load both and then use mask_k to compute q_contrib:
        # We will load Qn and Qp separately for ks ranges; but Triton kernel has a single vector ks. So we cannot pick Qn or Qp per element. To handle this, we restructure the approach: keep Qn_ptr and Qp_ptr separate, and let ks range over Kc and Kp separately as above. The previous comment was incorrect; we already did Qn reduction above. Now we do Qp:
        # Load Kp[ls, ks] (matrix [BLOCK_L, BLOCK_K])
        Kp_ptrs = Kp_ptr + ls[:, None] * Kp_stride0 + ks[None, :] * Kp_stride1
        Kp_block = tl.load(Kp_ptrs, mask=mask_l[:, None] & mask_k[None, :], other=0.0)  # [BLOCK_L, BLOCK_K]
        # Load Qp vector for ks (we loaded above): qp_vec
        # Note: We loaded qp_vec from Qp_ptr earlier. Now multiply: acc += sum(Kp_block * qp_vec[None, :], axis=1)
        acc += tl.sum(Kp_block * qp_vec[None, :], axis=1)

    # Store accumulated logits
    out_ptrs = Logits_ptr + h * Logits_stride0 + ls * Logits_stride1
    tl.store(out_ptrs, acc, mask=mask_l)


@triton.jit
def lse_mask_kernel(
    Logits_ptr, Lse_ptr,
    H, L,
    Logits_stride0, Logits_stride1,
    inv_ln2: tl.float32,
    BLOCK_L: tl.constexpr,
):
    # One program per head h
    h = tl.program_id(0)

    # Row-wise max
    max_val = -float('inf')
    for l0 in range(0, L, BLOCK_L):
        ls = l0 + tl.arange(0, BLOCK_L)
        mask_l = ls < L
        vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
        block_max = tl.max(vals, axis=0)
        max_val = tl.maximum(max_val, block_max)

    # Row-wise sum of exp(vals - max_val)
    sum_exp = 0.0
    for l0 in range(0, L, BLOCK_L):
        ls = l0 + tl.arange(0, BLOCK_L)
        mask_l = ls < L
        vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
        e = tl.exp(vals - max_val)
        sum_exp += tl.sum(e, axis=0)

    lse_scaled = tl.log(sum_exp) * inv_ln2  # divide by ln(2) via log2: not available; use natural log
    # Triton has tl.log and tl.exp; tl.log2 is not available in many versions, so we use tl.log
    # However, we need to divide by ln(2). Use inv_ln2 * tl.log(sum_exp)
    tl.store(Lse_ptr + h, lse_scaled)


@triton.jit
def softmax_matmul_kernel(
    Logits_ptr, Kc_ptr, Out_ptr,
    H, L, D_ckv,
    Kc_stride0, Kc_stride1,
    Out_stride0, Out_stride1,
    BLOCK_L: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # One program per head h
    h = tl.program_id(0)

    # Compute row-wise max
    max_val = -float('inf')
    for l0 in range(0, L, BLOCK_L):
        ls = l0 + tl.arange(0, BLOCK_L)
        mask_l = ls < L
        vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
        block_max = tl.max(vals, axis=0)
        max_val = tl.maximum(max_val, block_max)

    # Compute sum_exp
    sum_exp = 0.0
    for l0 in range(0, L, BLOCK_L):
        ls = l0 + tl.arange(0, BLOCK_L)
        mask_l = ls < L
        vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
        e = tl.exp(vals - max_val)
        sum_exp += tl.sum(e, axis=0)

    # Now compute output vector: out[h, :] = softmax(Logits[h, :]) @ Kc[:, :D_ckv]
    out_vec = tl.zeros((D_ckv,), dtype=tl.float32)
    for k0 in range(0, D_ckv, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)
        mask_k = ks < D_ckv

        # Accumulate contributions over L tiles
        acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
        for l0 in range(0, L, BLOCK_L):
            ls = l0 + tl.arange(0, BLOCK_L)
            mask_l = ls < L
            vals = tl.load(Logits_ptr + h * Logits_stride0 + ls * Logits_stride1, mask=mask_l, other=-float('inf'))
            e = tl.exp(vals - max_val)  # softmax probabilities for this tile
            sum_tile = tl.sum(e, axis=0)  # scalar: sum of exp in this tile
            # To get per-position softmax, we need e / sum_exp. Triton allows scalar division:
            p = e / sum_exp
            Kc_ptrs = Kc_ptr + ks[None, :] * Kc_stride1 + ls[:, None] * Kc_stride0  # [BLOCK_K, BLOCK_L]
            Kc_block = tl.load(Kc_ptrs, mask=mask_k[:, None] & mask_l[None, :], other=0.0)  # [BLOCK_K, BLOCK_L]
            acc += tl.sum(p[None, :] * Kc_block, axis=1)  # reduce over L

        # Store acc into Out[h, ks]
        out_ptrs = Out_ptr + h * Out_stride0 + ks * Out_stride1
        tl.store(out_ptrs, acc, mask=mask_k)

    # The above computes partial out_vec; now we combine acc blocks:
    # Since we wrote per ks block, we just store acc for each k0; acc is already length BLOCK_K vector for each k0 iteration.
    # We need to ensure final store happens per k0. The loop above already does it. Note: Triton kernel stores each acc vector for each k0.


# -------------------------------
# ModelNew: Triton-ONLY forward
# -------------------------------
class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale):
        device = q_nope.device
        # Extract constants
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and qo_indptr.is_cuda and kv_indptr.is_cuda and kv_indices.is_cuda, "All inputs must be on CUDA"
        total_q, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]
        assert num_qo_heads == 16
        assert head_dim_ckv == 512
        assert head_dim_kpe == 64
        num_qo_heads = 16

        batch_size = int(qo_indptr[-1].item()) - int(qo_indptr[0].item())
        # For each batch element b, process queries q_start:q_end, and corresponding kv block
        output = torch.empty((total_q, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=device)  # we'll fill q positions with results
        lse = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=device)

        for b in range(batch_size):
            q_start = int(qo_indptr[b].item())
            q_end = int(qo_indptr[b + 1].item())
            if q_start >= q_end:
                continue

            q_len = q_end - q_start

            # KV block: indices for this batch
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            if page_beg >= page_end:
                continue

            tok_idx = kv_indices[page_beg:page_end].to(torch.int64)  # indices as int64 for Kc/Kp
            L = page_end - page_beg  # number of KV tokens in this block

            # Prepare Q batch: q_nope and q_pe for all queries i in [0, q_len)
            qn_batch = q_nope[q_start:q_end].to(torch.float32)  # [q_len, H, D_ckv]
            qp_batch = q_pe[q_start:q_end].to(torch.float32)    # [q_len, H, D_kpe]

            # Kc and Kp: [L, D_ckv] and [L, D_kpe]
            Kc = ckv_cache[tok_idx].to(torch.float32)  # [L, D_ckv]
            Kp = kpe_cache[tok_idx].to(torch.float32)  # [L, D_kpe]

            # Loop over each query i
            for i in range(q_len):
                # Prepare pointers: select h dimension by broadcasting H vector (we pass h separately via grid)
                # We will compute Logits[h, L] for all heads h in a loop
                H = num_qo_heads
                # Allocate Logits [H, L]
                logits = torch.empty((H, L), dtype=torch.float32, device=device)
                # Launch compute_logits_kernel: grid over (H, cdiv(L, BLOCK_L))
                BLOCK_L = 128
                grid = (H, triton.cdiv(L, BLOCK_L))
                # We need to pass Qn and Qp shaped as [1, H, D]; for each i, use qn_batch[i] and qp_batch[i].
                Qn_ptr = qn_batch[i].contiguous().view(1, H, head_dim_ckv)  # [1, H, D_ckv]
                Qp_ptr = qp_batch[i].contiguous().view(1, H, head_dim_kpe)  # [1, H, D_kpe]
                Logits_ptr = logits

                # Strides
                Qn_stride0, Qn_stride1 = Qn_ptr.stride(0), Qn_ptr.stride(1)  # strides for [1, H, D]
                Qp_stride0, Qp_stride1 = Qp_ptr.stride(0), Qp_ptr.stride(1)
                Kc_stride0, Kc_stride1 = Kc.stride(0), Kc.stride(1)          # [L, D_ckv]
                Kp_stride0, Kp_stride1 = Kp.stride(0), Kp.stride(1)          # [L, D_kpe]
                Logits_stride0, Logits_stride1 = Logits_ptr.stride(0), Logits_ptr.stride(1)  # [H, L]

                # Launch compute_logits_kernel
                compute_logits_kernel[grid](
                    Qn_ptr, Qp_ptr, Kc, Kp, Logits_ptr,
                    H, L, head_dim_ckv, head_dim_kpe,
                    Qn_stride0, Qn_stride1,
                    Qp_stride0, Qp_stride1,
                    Kc_stride0, Kc_stride1,
                    Kp_stride0, Kp_stride1,
                    Logits_stride0, Logits_stride1,
                    BLOCK_L=BLOCK_L, BLOCK_K=64,
                )

                # Compute LSE per head (masked with causal)
                # Build causal mask: for this (b), prefix_len = L - q_len; query_abs_pos = prefix_len + i
                prefix_len = L - q_len
                query_abs_pos = prefix_len + i
                # mask: positions where ls <= query_abs_pos are causal
                mask_vec = torch.full((L,), query_abs_pos, dtype=torch.int32, device=device)  # [L]
                # Launch lse_mask_kernel: grid over H
                grid_lse = (H,)
                lse_ptr = lse[q_start + i]  # per (b,i) lse vector of length H
                lse_mask_kernel[grid_lse](
                    Logits_ptr, lse_ptr,
                    H, L,
                    Logits_stride0, Logits_stride1,
                    1.4426950408889634,  # inv_ln2
                    BLOCK_L=BLOCK_L,
                )

                # Compute output vector per head: softmax(Logits[h, :]) @ Kc[:, :D_ckv]
                out_vec = torch.empty((H, head_dim_ckv), dtype=torch.float32, device=device)
                grid_out = (H,)
                Out_ptr = out_vec  # [H, D_ckv]
                Out_stride0, Out_stride1 = Out_ptr.stride(0), Out_ptr.stride(1)

                softmax_matmul_kernel[grid_out](
                    Logits_ptr, Kc, Out_ptr,
                    H, L, head_dim_ckv,
                    Kc_stride0, Kc_stride1,
                    Out_stride0, Out_stride1,
                    BLOCK_L=BLOCK_L, BLOCK_K=64,
                )

                # Store result into output[q_start+i, :, :]
                # out_vec has shape [H, D_ckv], match output[q_start+i, :, :]
                output[q_start + i] = out_vec  # torch assignment copies; output is float32

        # Cast outputs to bfloat16 to match original model
        output_cast = output.to(torch.bfloat16)
        # lse remains float32
        return output_cast, lse


def run(*args):
    return ModelNew()(*args)
