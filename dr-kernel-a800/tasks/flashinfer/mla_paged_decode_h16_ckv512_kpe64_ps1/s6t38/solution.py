import math
import torch
import triton
import triton.language as tl


# Triton kernel: compute output_row[h, :] = sum_t attn[h, t] * Kc[t, :]
# Input:
#   attn_ptr: *fp32, shape [H, L_tokens], row-major
#   Kc_ptr:   *fp32, shape [L_tokens, D], row-major
#   out_ptr:  *fp32, shape [H, D], row-major
@triton.jit
def attn_matmul_rows_fp32(
    attn_ptr,   # *fp32, [H, L_tokens], contiguous
    Kc_ptr,     # *fp32, [L_tokens, D], contiguous
    out_ptr,    # *fp32, [H, D], contiguous
    H: tl.constexpr,         # number of heads (compile-time)
    D: tl.constexpr,         # head_dim_ckv (e.g., 512)
    L_tokens: tl.constexpr,  # number of tokens per batch element
    BLOCK_T: tl.constexpr,   # tile size over tokens
):
    # One program per row (head)
    h = tl.program_id(0)
    base_out = h * D

    # Accumulator for output row
    acc = tl.zeros((D,), dtype=tl.float32)

    # Loop over tokens in tiles
    for t0 in range(0, L_tokens, BLOCK_T):
        t_offsets = t0 + tl.arange(0, BLOCK_T)
        mask = t_offsets < L_tokens

        # Load attn[h, t_offsets] vector
        attn_vec = tl.load(attn_ptr + h * L_tokens + t_offsets, mask=mask, other=0.0)  # [BLOCK_T]

        # Load Kc[t_offsets, :] as a matrix of shape [BLOCK_T, D]
        Kc_mat = tl.zeros((BLOCK_T, D), dtype=tl.float32)
        # Row-wise load
        for j in range(0, BLOCK_T):
            t_idx = t0 + j
            row_mask = (t_idx < L_tokens)
            # If row_mask is true, load, else 0
            # Triton supports elementwise mask on load
            Kc_row = tl.load(Kc_ptr + t_idx * D + tl.arange(0, D), mask=row_mask, other=0.0)
            Kc_mat[j, :] = Kc_row

        # acc += sum_j attn_vec[j] * Kc_mat[j, :]
        # Loop over D to accumulate
        for j in range(0, BLOCK_T):
            t_idx = t0 + j
            valid = t_idx < L_tokens
            # Multiply scalar attn with row
            row = Kc_mat[j, :]
            # If invalid, attn_vec[j] is 0 due to mask, so skip or multiply by 0
            # We avoid multiply for invalid by masking (acc += attn_vec[j] * row) with valid flag:
            # Since Triton vectors don't support per-element scalar multiplication here, we gate via tl.where(valid, attn_vec[j], 0.0)
            # But attn_vec[j] is scalar, Kc_mat[j, :] is vector; Triton supports vector-scalar ops. We can compute and mask with tl.where over scalar.
            # Better approach: multiply and then gate via mask with vector zeros. Triton allows vector operations; since row is zero when invalid, it's fine.
            acc += attn_vec[j] * row

    # Store result row
    tl.store(out_ptr + base_out + tl.arange(0, D), acc)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Constants and shapes
        assert q_nope.shape == (1, 16, 512), "q_nope must be [1, 16, 512]"
        assert q_pe.shape == (1, 16, 64), "q_pe must be [1, 16, 64]"
        assert ckv_cache.shape[1] == 1 and kpe_cache.shape[1] == 1, "num_pages must be 1 (already asserted in original)"
        assert ckv_cache.shape[2] == 512 and kpe_cache.shape[2] == 64, "head dims must be 512 and 64"

        batch_size = q_nope.shape[0]
        H = q_nope.shape[1]
        D = q_nope.shape[2]
        Dp = q_pe.shape[2]
        device = q_nope.device

        # Ensure inputs on the same device and cast to float32 for compute
        qn = q_nope.contiguous().to(torch.float32)  # [1, 16, 512]
        qp = q_pe.contiguous().to(torch.float32)   # [1, 16, 64]

        # Squeeze out "num_pages" dimension: [L_tokens, D] and [L_tokens, 64]
        Kc_all = ckv_cache.squeeze(1).contiguous().to(torch.float32)  # [L_tokens, 512]
        Kp_all = kpe_cache.squeeze(1).contiguous().to(torch.float32)  # [L_tokens, 64]

        output = torch.zeros((batch_size, H, D), dtype=torch.bfloat16, device=device)
        lse = torch.full((batch_size, H), -float("inf"), dtype=torch.float32, device=device)

        # Iterate over batch, compute per-batch tokens slice and per-head outputs
        for b in range(batch_size):
            # Compute L_tokens for this batch element
            L_tokens = int(kv_indptr[b + 1].item()) - int(kv_indptr[b].item())
            if L_tokens <= 0:
                lse[b] = -float("inf")
                continue

            # Gather token indices for this batch
            tok_idx = kv_indices[kv_indptr[b]:kv_indptr[b + 1]].to(torch.long)  # [L_tokens]
            # Gather Kc and Kp for these tokens
            Kc = Kc_all[tok_idx]  # [L_tokens, 512]
            Kp = Kp_all[tok_idx]  # [L_tokens, 64]

            # Compute logits per head and token: logits = (qn @ Kc.T) + (qp @ Kp.T)
            # qn[b] is [16, 512], Kc.T is [512, L_tokens] -> [16, L_tokens]
            # We'll compute via PyTorch for exact numerics
            logits1 = torch.matmul(qn[b], Kc.transpose(0, 1))  # [16, L_tokens]
            logits2 = torch.matmul(qp[b], Kp.transpose(0, 1))  # [16, L_tokens]
            logits = logits1 + logits2  # [16, L_tokens]

            # Scale
            logits_scaled = logits * float(sm_scale)  # [16, L_tokens]

            # lse per head: logsumexp(logits_scaled) / ln(2)
            row_max = torch.amax(logits_scaled, dim=1)                          # [16]
            sum_exp = torch.sum(torch.exp(logits_scaled - row_max[:, None]), dim=1)  # [16]
            lse_row = torch.log(sum_exp) + row_max                              # [16]
            lse[b] = lse_row / math.log(2.0)                                    # [16]

            # Softmax per head along tokens
            attn = torch.softmax(logits_scaled, dim=1)                          # [16, L_tokens]

            # Final output per head: attn @ Kc => [16, 512], compute via Triton
            out_row = torch.empty((H, D), dtype=torch.float32, device=device)

            # Launch Triton kernel: one program per head
            attn_matmul_rows_fp32[(H,)](
                attn, Kc, out_row,
                H=H, D=D, L_tokens=L_tokens, BLOCK_T=128,
                num_warps=4, num_stages=2
            )

            output[b] = out_row.to(torch.bfloat16)

        return output, lse


def run(*args):
    return ModelNew()(*args)
