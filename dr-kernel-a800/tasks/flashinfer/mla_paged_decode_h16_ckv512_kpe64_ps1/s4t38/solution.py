import torch
import triton
import triton.language as tl


@triton.jit
def matvec_row_kernel(
    A_ptr,       # *float32, pointer to q vector (length K), flattened to 1D
    B_ptr,       # *float32, pointer to B rows, shape [M, K], contiguous
    C_ptr,       # *float32, pointer to output, shape [M], contiguous
    M,           # int, number of rows in B (runtime)
    K: tl.constexpr,           # int, length of q and columns of B (compile-time)
    BLOCK_K: tl.constexpr = 64 # tile size along K
):
    # One program computes a single output element: C[i] = A @ B[i, :]
    i = tl.program_id(0)  # index over M
    acc = 0.0
    # Loop over K in tiles
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)  # vector of indices along K (constexpr)
        mask_k = offs_k < K
        # Load q chunk
        a = tl.load(A_ptr + offs_k, mask=mask_k, other=0.0)
        # Load B row chunk: row i, cols offs_k
        b = tl.load(B_ptr + i * K + offs_k, mask=mask_k, other=0.0)
        # Accumulate dot product of a and b
        acc += tl.sum(a * b, axis=0)
    # Store the result for this i
    tl.store(C_ptr + i, acc)


@triton.jit
def softmax_lse_kernel(
    logits_ptr,         # *float32, pointer to logits vector of length M
    attn_ptr,           # *float32, pointer to attn vector, length M (per head)
    lse_ptr,            # *float32, pointer to per-head lse, length num_qo_heads
    M: tl.constexpr,    # length of logits vector (compile-time)
    scale: tl.constexpr,       # scaling factor (compile-time float)
    inv_ln2: tl.constexpr       # 1.0 / ln(2) (compile-time float)
):
    # Each program handles one head index h
    h = tl.program_id(0)
    # Compute max for numerical stability
    max_val = -float("inf")
    sum_exp = 0.0
    # Loop over M in tiles to compute max
    for m0 in range(0, M, 128):
        offs_m = m0 + tl.arange(0, 128)
        mask_m = offs_m < M
        vals = tl.load(logits_ptr + offs_m, mask=mask_m, other=-float("inf"))
        vals = vals * scale
        block_max = tl.max(vals, axis=0)
        max_val = tl.maximum(max_val, block_max)
    # Loop over M again to compute sum of exp
    for m0 in range(0, M, 128):
        offs_m = m0 + tl.arange(0, 128)
        mask_m = offs_m < M
        vals = tl.load(logits_ptr + offs_m, mask=mask_m, other=-float("inf"))
        vals = vals * scale
        sum_exp += tl.sum(tl.exp(vals - max_val), axis=0)
    # lse = (max + log(sum_exp)) * inv_ln2
    lse_val = (max_val + tl.log(sum_exp)) * inv_ln2
    # Store lse for head h
    tl.store(lse_ptr + h, lse_val)
    # Write attn entries for this head
    for m0 in range(0, M, 128):
        offs_m = m0 + tl.arange(0, 128)
        mask_m = offs_m < M
        vals = tl.load(logits_ptr + offs_m, mask=mask_m, other=-float("inf"))
        vals = vals * scale
        exp_vals = tl.exp(vals - max_val)
        attn_vals = exp_vals / sum_exp
        tl.store(attn_ptr + h * M + offs_m, attn_vals, mask=mask_m)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Shapes from the original code
        batch_size, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]
        num_pages = ckv_cache.shape[0]
        # Cache rows
        Kc_all = ckv_cache.to(torch.float32).contiguous()  # [num_pages, 1, 512] -> [num_pages, 512]
        Kp_all = kpe_cache.to(torch.float32).contiguous()  # [num_pages, 1, 64] -> [num_pages, 64]
        # Output tensors
        output = torch.zeros(
            (batch_size, num_qo_heads, head_dim_ckv), dtype=torch.float32, device=q_nope.device
        )
        lse = torch.full((batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=q_nope.device)

        # Constants
        inv_ln2 = 1.4426950408889634  # 1 / ln(2)

        # Process each batch element
        for b in range(batch_size):
            # Number of tokens for this batch element
            if kv_indptr.numel() <= 1:
                # degenerate case; no valid indptr
                output[b].zero_()
                lse[b] = -float("inf")
                continue

            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            M = end - start

            if M <= 0:
                output[b].zero_()
                lse[b] = -float("inf")
                continue

            # Gather token indices
            tok_idx = kv_indices[start:end].contiguous()  # [M]
            # Gather K rows for this batch
            Kc = Kc_all[tok_idx].contiguous()  # [M, 512]
            Kp = Kp_all[tok_idx].contiguous()  # [M, 64]

            # Prepare q vectors for each head
            for h in range(num_qo_heads):
                qn = q_nope[b, h].to(torch.float32).contiguous()  # [512]
                qp = q_pe[b, h].to(torch.float32).contiguous()   # [64]

                # Compute logits = qn @ Kc.T + qp @ Kp.T → [M]
                logits = torch.empty(M, dtype=torch.float32, device=q_nope.device)
                grid = (M,)
                matvec_row_kernel[grid](
                    qn, Kc, logits,
                    M, 512, 64
                )
                logits_qp = torch.empty(M, dtype=torch.float32, device=q_nope.device)
                matvec_row_kernel[grid](
                    qp, Kp, logits_qp,
                    M, 64, 64
                )
                logits_scaled = (logits + logits_qp) * sm_scale

                # Compute lse and attn per head
                attn_vec = torch.empty(M, dtype=torch.float32, device=q_nope.device)
                # Launch softmax_lse_kernel: one program per head
                softmax_lse_kernel[(num_qo_heads,)](
                    logits_scaled, attn_vec, lse[b],
                    M, sm_scale, inv_ln2
                )

                # Compute out[h, :] = attn_vec @ Kc
                out_vec = torch.empty(head_dim_ckv, dtype=torch.float32, device=q_nope.device)
                matvec_row_kernel[(head_dim_ckv,)](
                    attn_vec, Kc, out_vec,
                    head_dim_ckv, M, 64
                )
                output[b, h] = out_vec

        # Cast output to bfloat16 to match original function
        output = output.to(torch.bfloat16)
        return output, lse


def run(*args):
    return ModelNew()(*args)
