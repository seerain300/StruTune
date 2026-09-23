import torch
import math

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


@triton.jit
def _block_attention_gqa_kernel(
    q_ptr,         # *float32, [M, D] contiguous per q_idx (we iterate q_idx inside)
    k_ptr,         # *float32, [N, GH, D] expanded by GQA ratio
    v_ptr,         # *float32, [N, GH, D] expanded by GQA ratio
    out_ptr,       # *bfloat16, [M, G, D] output per q_idx and head
    lse_ptr,       # *float32, [M, G] logsumexp per q_idx and head
    qo_indptr_ptr, # *int32, shape [2] (q_start, q_end) for this block
    kv_indptr_ptr, # *int32, shape [2] (kv_start, kv_end) for this block
    M: tl.constexpr,     # number of queries in this block
    N: tl.constexpr,     # number of KV tokens in this block
    G: tl.constexpr,     # num_qo_heads (e.g., 32)
    GH: tl.constexpr,    # num_kv_heads (e.g., 8)
    D: tl.constexpr,     # head_dim (e.g., 128)
    SM_SCALE: tl.constexpr,  # scaling factor (e.g., 1/sqrt(D))
    R: tl.constexpr,       # GQA ratio = G // GH (e.g., 4)
    BLOCK_D: tl.constexpr,    # tile for D, e.g., 128
    BLOCK_N: tl.constexpr,    # tile for N, e.g., 128
):
    # Load block ranges
    q_start = tl.load(qo_indptr_ptr + 0)
    q_end = tl.load(qo_indptr_ptr + 1)
    kv_start = tl.load(kv_indptr_ptr + 0)
    kv_end = tl.load(kv_indptr_ptr + 1)

    # Iterate over each query in the block
    for i in range(0, M):
        q_idx = q_start + i

        # Load q vector for this q_idx: q layout is [total_q, G, D], contiguous.
        # For each head g, we compute output; we need q[q_idx, g, :].
        # To keep kernel simple, we load q as [G, D] by iterating g; but here q is passed as [M, D] per block
        # In practice, we'll pass q per q_idx as 1D vector; better: pass q per q_idx as [G, D] but flatten.
        # Given the original call, we expect q as [total_q, G, D]. We access via q_ptr + q_idx * (G * D) + g * D.
        # However, to simplify, we pass q as [M, D] in forward; so we can load q vector directly.
        # Simpler approach: in forward, we pass q[q_start:q_end] to kernel and interpret as [M, D].

        # Compute q vector for this q_idx (flattened across G: q_ptr points to [M, D])
        q_vec = tl.zeros((D,), dtype=tl.float32)
        for d0 in range(0, D, BLOCK_D):
            d_offsets = d0 + tl.arange(0, BLOCK_D)
            mask_d = d_offsets < D
            # q_ptr points to flattened [M, D]; element for q_idx is base q_idx * D
            q_sub = tl.load(q_ptr + q_idx * D + d_offsets, mask=mask_d, other=0.0)
            q_vec[d0:d0+BLOCK_D] = q_sub

        # For each head g in 0..G-1
        for g in range(0, G):
            lse_val = tl.full((), -float('inf'), tl.float32)

            # Compute logits over expanded K/V heads: total H_eff = GH * R
            H_eff = GH * R
            logits = tl.zeros((N,), dtype=tl.float32)

            # Causal bound
            q_add = q_idx + 1

            # Loop over expanded K/V heads (0..H_eff-1)
            # Note: k_ptr points to [N, H_eff, D]; v_ptr points to [N, H_eff, D]
            for h_eff in range(0, H_eff):
                # Map expanded head to original kv head
                h = h_eff // R  # since we repeat each kv head R times contiguously
                # Base pointer for this (j, h)
                # For j in 0..N-1, we compute pointer as k_ptr + j * (H_eff * D) + h * D
                # v similarly
                # We'll loop j from 0..N-1
                for j in range(0, N):
                    # Skip if causal mask is violated
                    if j >= q_add:
                        continue
                    k_base = k_ptr + j * (H_eff * D) + h * D
                    v_base = v_ptr + j * (H_eff * D) + h * D

                    # Load K vector for this (j, h)
                    k_vec = tl.zeros((D,), dtype=tl.float32)
                    for d0 in range(0, D, BLOCK_D):
                        d_offsets = d0 + tl.arange(0, BLOCK_D)
                        mask_d = d_offsets < D
                        k_sub = tl.load(k_base + d_offsets, mask=mask_d, other=0.0)
                        k_vec[d0:d0+BLOCK_D] = k_sub

                    # Dot product q_vec · k_vec
                    dot = 0.0
                    for d in range(0, D):
                        dot += q_vec[d] * k_vec[d]
                    logits[j] = dot * SM_SCALE

            # Apply causal mask explicitly
            for j in range(0, N):
                if j >= q_add:
                    logits[j] = -float('inf')

            # LSE for this head
            lse_val = tl.logsumexp(logits)
            # Store lse at [q_idx, g]
            tl.store(lse_ptr + q_idx * G + g, lse_val)

            # Softmax
            max_score = tl.max(logits)
            soft = tl.exp(logits - max_score)
            sum_soft = tl.sum(soft)
            soft = soft / sum_soft  # [N]

            # Output accumulation for this head
            out_base = out_ptr + q_idx * (G * D) + g * D
            out_vec = tl.zeros((D,), dtype=tl.float32)
            for j in range(0, N):
                if j >= q_add:
                    continue
                v_base = v_ptr + j * (H_eff * D) + (h_eff // R) * D
                v_vec = tl.zeros((D,), dtype=tl.float32)
                for d0 in range(0, D, BLOCK_D):
                    d_offsets = d0 + tl.arange(0, BLOCK_D)
                    mask_d = d_offsets < D
                    v_sub = tl.load(v_base + d_offsets, mask=mask_d, other=0.0)
                    v_vec[d0:d0+BLOCK_D] = v_sub
                out_vec += soft[j] * v_vec

            # Store output vector for this head
            for d0 in range(0, D, BLOCK_D):
                d_offsets = d0 + tl.arange(0, BLOCK_D)
                mask_d = d_offsets < D
                out_sub = out_vec[d0:d0+BLOCK_D].to(tl.bfloat16)
                tl.store(out_base + d_offsets, out_sub, mask=mask_d)


class ModelNew(torch.nn.Module):
    def forward(self, q, k, v, qo_indptr, kv_indptr, sm_scale):
        # Ensure Triton and CUDA
        assert TRITON_AVAILABLE, "Triton is not available."
        assert q.is_cuda and k.is_cuda and v.is_cuda, "Input tensors must be on CUDA."

        device = q.device
        # Make tensors contiguous and in float32 for compute
        q = q.contiguous().to(torch.float32)          # [total_q, G, D]
        # Expand K/V by GQA ratio on host to match original behavior exactly
        G = q.shape[1]
        GH = k.shape[1]
        D = q.shape[2]
        R = G // GH  # GQA ratio, e.g., 4
        k_expanded = k.contiguous().to(torch.float32).repeat_interleave(R, dim=1)  # [total_kv, GH*R, D]
        v_expanded = v.contiguous().to(torch.float32).repeat_interleave(R, dim=1)  # [total_kv, GH*R, D]

        # Output buffers
        output = torch.empty((q.shape[0], G, D), dtype=torch.bfloat16, device=device)
        lse = torch.empty((q.shape[0], G), dtype=torch.float32, device=device)

        # Handle one block: load qo_indptr and kv_indptr for b=0
        qo_indptr_b = qo_indptr[0:2].to(torch.int32)
        kv_indptr_b = kv_indptr[0:2].to(torch.int32)

        total_q, G, D = q.shape
        total_kv, GH, _ = k.shape  # GH is not used after expansion; we use expanded shapes

        # Number of queries and KV tokens in this block
        M = int((qo_indptr_b[1] - qo_indptr_b[0]).item())
        N = int((kv_indptr_b[1] - kv_indptr_b[0]).item())

        # Pass q per block as [M, D] to simplify kernel access
        # We slice q along the first dimension: q_block has shape [M, D]
        # But Triton expects a flat pointer; so we pass flattened q values for the block.
        # To do that, we need to flatten q[q_start:q_end, :, :] into [M, D].
        # We can build q_block via slicing and flattening in forward:
        q_block = q[qo_indptr_b[0]:qo_indptr_b[1]].reshape(M, D).contiguous()

        BLOCK_D = 128
        BLOCK_N = 128

        # Launch Triton kernel: one program handles the whole block
        _block_attention_gqa_kernel[(1,)](
            q_block,                            # flattened q for this block: [M, D]
            k_expanded,                        # [N, GH*R, D]
            v_expanded,                        # [N, GH*R, D]
            output,                            # [M, G, D]
            lse,                               # [M, G]
            qo_indptr_b,                       # [2] int32
            kv_indptr_b,                       # [2] int32
            M=M, N=N, G=G, GH=GH, D=D, SM_SCALE=sm_scale, R=R,
            BLOCK_D=BLOCK_D, BLOCK_N=BLOCK_N,
        )

        return output, lse


def run(*args):
    return ModelNew()(*args)
