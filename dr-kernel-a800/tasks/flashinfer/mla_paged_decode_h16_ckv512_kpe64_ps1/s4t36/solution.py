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
    # Initialize accumulator for this output element
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
    lse_ptr,            # *float32, pointer to per-head lse, length num_qo_heads
    M: tl.constexpr,    # length of logits vector (compile-time for kernel specialization)
    scale: tl.constexpr,       # scaling factor (compile-time float)
    inv_ln2: tl.constexpr       # 1.0 / ln(2) (compile-time float)
):
    # Each program handles one head index h (grid is (num_qo_heads,))
    h = tl.program_id(0)
    # Compute max for numerical stability
    max_val = -float("inf")
    # Process in chunks of 128
    for m0 in range(0, M, 128):
        offs = m0 + tl.arange(0, 128)
        mask = offs < M
        x = tl.load(logits_ptr + offs, mask=mask, other=-float("inf"))
        cur_max = tl.max(x, axis=0)
        max_val = tl.maximum(max_val, cur_max)

    # Compute sum of exp(logits * scale)
    sum_val = 0.0
    for m0 in range(0, M, 128):
        offs = m0 + tl.arange(0, 128)
        mask = offs < M
        x = tl.load(logits_ptr + offs, mask=mask, other=-float("inf"))
        e = tl.exp((x - max_val) * scale)
        # Zero out contributions from masked positions
        e = tl.where(mask, e, 0.0)
        sum_val += tl.sum(e, axis=0)

    lse = (max_val + tl.log(sum_val)) * inv_ln2
    tl.store(lse_ptr + h, lse)


class ModelNew(torch.nn.Module):
    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Shapes and constraints
        batch_size, num_qo_heads, head_dim_ckv = q_nope.shape
        head_dim_kpe = q_pe.shape[-1]
        assert num_qo_heads == 16
        assert head_dim_ckv == 512
        assert head_dim_kpe == 64

        device = q_nope.device
        # Convert caches to float32 and make contiguous
        Kc_all = ckv_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, 512]
        Kp_all = kpe_cache.squeeze(1).to(torch.float32).contiguous()  # [num_pages, 64]

        output = torch.zeros((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
        lse = torch.full((batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=device)

        inv_ln2 = 1.4426950408889634  # 1 / ln(2)

        for b in range(batch_size):
            page_beg = int(kv_indptr[b].item())
            page_end = int(kv_indptr[b + 1].item())
            if page_beg >= page_end:
                # No KV cache for this batch element
                output[b].zero_()
                lse[b] = -float("inf")
                continue

            M = page_end - page_beg  # number of tokens for this batch
            if M <= 0:
                output[b].zero_()
                lse[b] = -float("inf")
                continue

            # Gather K rows
            tok_idx = kv_indices[page_beg:page_end].to(torch.int64)  # indices into caches
            Kc = Kc_all[tok_idx]  # [M, 512], float32
            Kp = Kp_all[tok_idx]  # [M, 64], float32
            Kc = Kc.contiguous()
            Kp = Kp.contiguous()

            # Per-head outputs
            for h in range(num_qo_heads):
                qn = q_nope[b, h, :].to(torch.float32).contiguous()  # [512]
                qp = q_pe[b, h, :].to(torch.float32).contiguous()   # [64]

                # Allocate logits buffer [M]
                logits = torch.empty((M,), dtype=torch.float32, device=device)
                # Compute logits = qn @ Kc.T
                grid = (M,)
                matvec_row_kernel[grid](
                    qn, Kc, logits, M, K=Kc.shape[1], BLOCK_K=64, num_warps=4
                )
                # Add contribution from qp @ Kp.T
                logits_qp = torch.empty((M,), dtype=torch.float32, device=device)
                grid2 = (M,)
                matvec_row_kernel[grid2](
                    qp, Kp, logits_qp, M, K=Kp.shape[1], BLOCK_K=64, num_warps=4
                )
                logits = logits + logits_qp

                # Compute lse for this head
                softmax_lse_kernel[(num_qo_heads,)](
                    logits, lse[b], M, scale=float(sm_scale), inv_ln2=inv_ln2
                )

                # Compute attn vector using stable softmax on logits_scaled
                logits_scaled = logits * float(sm_scale)
                max_val = torch.max(logits_scaled)
                sum_exp = torch.sum(torch.exp(logits_scaled - max_val))
                attn = torch.exp(logits_scaled - max_val) / sum_exp  # [M]

                # out[h, :] = sum_i attn[i] * Kc[i, :] → [512]
                out_vec = torch.empty((head_dim_ckv,), dtype=torch.float32, device=device)
                # Implement matvec by accumulating across M; torch handles the reduction and is stable.
                # Equivalent to out_vec = attn @ Kc, but computed via elementwise and sum.
                for k in range(head_dim_ckv):
                    # Dot of attn with column k of Kc
                    out_vec[k] = torch.sum(attn * (Kc[:, k]))
                output[b, h, :] = out_vec.to(torch.bfloat16)

        return output, lse


def fused_operator(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6):
    _out = ModelNew().forward(tensor_0, tensor_1, tensor_2, tensor_3, tensor_4, tensor_5, tensor_6)
    return list(_out) if isinstance(_out, (tuple, list)) else [_out]


def run(*args):
    return ModelNew()(*args)
