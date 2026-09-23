import math
import torch

import triton
import triton.language as tl


@triton.jit
def _compute_bf16_out_and_lse_kernel(
    q_nope_rows_ptr,       # *bf16, shape [H, D1] (pre-sliced per batch and head)
    q_pe_rows_ptr,         # *bf16, shape [H, D2] (pre-sliced per batch and head)
    Kc_all_ptr,            # *bf16, shape [N, D1]
    Kp_all_ptr,            # *bf16, shape [N, D2]
    kv_indices_ptr,        # *int32, shape [L_tokens]
    output_ptr,            # *bf16, shape [B, H, D1]
    lse_ptr,               # *f32,   shape [B, H]
    B: tl.int32,
    H: tl.int32,
    D1: tl.int32,          # head_dim_ckv (512)
    D2: tl.int32,          # head_dim_kpe (64)
    L_tokens: tl.int32,    # number of tokens in this batch element
    sm_scale: tl.float32,  # scaling factor for logits
    BLOCK_H: tl.constexpr, # tile for head_dim accumulation (128)
):
    pid = tl.program_id(axis=0)  # 0 .. (B*H - 1)
    b = pid // H
    head = pid % H

    # Casts and dtype setup
    # We'll operate in fp32 for math; output is bf16 as original.
    # Load q vectors for this head (already pre-sliced to length D1 and D2 respectively)
    # q_nope_rows_ptr points to [D1], q_pe_rows_ptr points to [D2]
    qn = tl.load(q_nope_rows_ptr + tl.arange(0, D1), mask=tl.arange(0, D1) < D1, other=0.0).to(tl.float32)  # [D1]
    qp = tl.load(q_pe_rows_ptr + tl.arange(0, D2), mask=tl.arange(0, D2) < D2, other=0.0).to(tl.float32)  # [D2]

    # Allocate accumulators
    logits = tl.zeros((L_tokens, D1), dtype=tl.float32)
    out_vec = tl.zeros((D1,), dtype=tl.float32)

    # Loop over tokens; compute logits and then final output
    # Note: L_tokens could be 0; we assume forward handles this and doesn't launch when L_tokens==0
    for t in range(0, L_tokens):
        token_idx = tl.load(kv_indices_ptr + t)  # int32
        # Load K rows (bf16), cast to fp32 for math
        Kc_row = tl.load(Kc_all_ptr + token_idx * D1 + tl.arange(0, D1), mask=tl.arange(0, D1) < D1, other=0.0).to(tl.float32)  # [D1]
        Kp_row = tl.load(Kp_all_ptr + token_idx * D2 + tl.arange(0, D2), mask=tl.arange(0, D2) < D2, other=0.0).to(tl.float32)  # [D2]

        # Compute two dot products: qn @ Kc_row.T and qp @ Kp_row.T
        # We implement these via tiled inner products over D1 and D2 respectively.
        # Initialize accumulators for dot products
        dot1 = tl.zeros((), dtype=tl.float32)
        dot2 = tl.zeros((), dtype=tl.float32)

        # Dot over D1
        for off in range(0, D1, BLOCK_H):
            offs = off + tl.arange(0, BLOCK_H)
            mask = offs < D1
            qn_block = qn[offs]
            Kc_block = Kc_row[offs]
            # dot = sum_i qn_block[i] * Kc_block[i]
            dot1 += tl.sum(qn_block * Kc_block, axis=0)
        # Dot over D2
        for off in range(0, D2, BLOCK_H):
            offs = off + tl.arange(0, BLOCK_H)
            mask = offs < D2
            qp_block = qp[offs]
            Kp_block = Kp_row[offs]
            dot2 += tl.sum(qp_block * Kp_block, axis=0)

        logits[t, :] = (dot1 + dot2) * sm_scale

    # Compute lse per head: logsumexp(logits) / ln(2)
    lse_row = tl.max(logits, axis=0)  # scalar per head
    sum_exp = tl.sum(tl.exp(logits - lse_row), axis=0)  # scalar
    lse_val = tl.log(2.0) * (lse_row + tl.log(sum_exp))
    tl.store(lse_ptr + b * H + head, lse_val)

    # Compute attn and final output: out = attn @ Kc_all selected rows
    # attn[t] = exp(logits[t]) / sum_exp
    for t in range(0, L_tokens):
        token_idx = tl.load(kv_indices_ptr + t)
        Kc_row = tl.load(Kc_all_ptr + token_idx * D1 + tl.arange(0, D1), mask=tl.arange(0, D1) < D1, other=0.0).to(tl.float32)
        attn_t = tl.exp(logits[t, :] - lse_row) / sum_exp  # vector over D1
        out_vec += attn_t * Kc_row  # elementwise multiply and sum along D1 accumulates

    # Store output in bf16
    tl.store(output_ptr + b * H * D1 + head * D1 + tl.arange(0, D1), out_vec.to(tl.bfloat16), mask=tl.arange(0, D1) < D1)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        """
        q_nope: [B, H, D1] bfloat16
        q_pe:   [B, H, D2] bfloat16
        ckv_cache: [N, 1, D1] bfloat16 (N=num_pages)
        kpe_cache: [N, 1, D2] bfloat16
        kv_indptr: [B+1] int32
        kv_indices: [M] int32, M = sum of kv_indptr
        sm_scale: float32 scalar
        """
        assert q_nope.shape[1] == 16, "num_qo_heads must be 16"
        B, H, D1 = q_nope.shape
        _, _, D2 = q_pe.shape
        N = ckv_cache.shape[0]
        assert ckv_cache.shape[1] == 1 and kpe_cache.shape[1] == 1
        assert D1 == 512 and D2 == 64, "Fixed dimensions: head_dim_ckv=512, head_dim_kpe=64"
        assert q_nope.is_cuda and q_pe.is_cuda and ckv_cache.is_cuda and kpe_cache.is_cuda and kv_indices.is_cuda, "Triton requires CUDA tensors"

        device = q_nope.device
        output = torch.empty((B, H, D1), dtype=torch.bfloat16, device=device)
        lse = torch.empty((B, H), dtype=torch.float32, device=device)

        # For each batch element, compute L_tokens and pass pre-sliced q vectors to the kernel
        # We'll pre-slice q_nope_rows[b, head, :] and q_pe_rows[b, head, :] to 1D before launching.
        for b in range(B):
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            L_tokens = end - start
            if L_tokens <= 0:
                # No KV in this batch element: output zeros, lse = -inf
                output[b].zero_()
                lse[b] = -float("inf")
                continue

            # Prepare pre-sliced q vectors
            # q_nope_rows: [H, D1] for this batch element, but Triton kernel expects [H, D1] per (b, head)
            # Since we loop over b, we need one head at a time; but forward() launches per (b, head), so
            # we will reconstruct q_nope_rows_ptr and q_pe_rows_ptr inside the loop using torch slicing.
            # However, we cannot pass dynamic slices into Triton kernel as function args; we precompute them
            # and pass as pointers. We'll do that in a helper call that returns pointers to slices.
            # To keep it simple, we'll compute q_rows on host as 1D tensors and pass their data_ptr.

            # We can prepare q_rows for the current b, for all heads, and pass them into the kernel via
            # a small wrapper that allocates temporary 1D tensors per (b, head) and launches kernel.
            # But Triton can't take host-side Python loops for every head inside forward; instead, we
            # launch grid = (B*H,) and compute q_rows inside the kernel (as we attempted).
            # Since Triton kernel signature is fixed, we'll go back to the approach of launching per (b, head)
            # and passing q_rows directly. We'll do that by re-launching for each (b, head) combination.
            # The forward() will iterate heads and launch the kernel with appropriate pointers.

            # Launch Triton kernel for each head
            for head in range(H):
                # Precompute q_rows as 1D tensors: [D1] and [D2]
                q_nope_rows = q_nope[b, head, :].contiguous().to(torch.bfloat16)
                q_pe_rows = q_pe[b, head, :].contiguous().to(torch.bfloat16)

                # Now, launch the kernel for this (b, head). We need to pass q_rows pointers.
                # Triton can't accept torch tensors with non-constexpr sizes directly; we pass them as flat data.
                # We'll create temporary 1D tensors for q_rows (already 1D) and pass their data_ptr.
                # However, Triton requires pointers; torch.Tensor.data_ptr() is not accessible in Python.
                # Therefore, we'll pass q_nope_rows and q_pe_rows as 1D tensors to the kernel by
                # slicing q_nope and q_pe appropriately and creating 1D views, then passing them as pointers.

                # Create 1D views
                qn_flat = q_nope_rows
                qp_flat = q_pe_rows

                # We need pointers to these 1D tensors. Triton kernel expects pointer to the start.
                # We'll invoke the kernel as follows:
                _compute_bf16_out_and_lse_kernel[(B * H,)](
                    qn_flat,              # q_nope_rows_ptr: 1D [D1]
                    qp_flat,              # q_pe_rows_ptr:   1D [D2]
                    ckv_cache,            # Kc_all_ptr
                    kpe_cache,            # Kp_all_ptr
                    kv_indices[start:end],  # kv_indices subset for this batch element
                    output,               # output_ptr
                    lse,                  # lse_ptr
                    B, H, D1, D2, L_tokens, sm_scale,
                    BLOCK_H=128,
                )

        return output, lse


def run(*args):
    return ModelNew()(*args)
