import math
import torch
import triton
import triton.language as tl


@triton.jit
def matvec_row_kernel(
    A_ptr,       # *float32, pointer to [1, K] contiguous, but we pass it as [K] with row offset 0
    B_ptr,       # *float32, pointer to [M, K] contiguous
    C_ptr,       # *float32, pointer to [1, M] contiguous
    K: tl.constexpr,          # int, e.g., 512
    M: tl.constexpr,          # int, number of rows in B (compile-time for kernel)
    BLOCK_K: tl.constexpr     # tile size along K, e.g., 64 or 128
):
    # One program computes the output row at row_out = 0
    row_out = 0
    # Column offsets for tiles
    offs_k = tl.arange(0, BLOCK_K)
    # Accumulator for output vector
    acc = tl.zeros((M,), dtype=tl.float32)

    # Reduce across K in tiles
    for k_start in tl.static_range(0, K, BLOCK_K):
        k_idx = k_start + offs_k  # [BLOCK_K]
        mask_k = k_idx < K

        # Load A[k_idx] as a vector: A_ptr + k_idx (A is [1, K] row, but pointer is 1D; treat as row 0)
        a_vec = tl.load(A_ptr + k_idx, mask=mask_k, other=0.0)

        # Load corresponding B[:, k_idx] as a [M, BLOCK_K] matrix: rows 0..M-1, columns k_idx
        b_ptrs = B_ptr + (tl.arange(0, M)[:, None] * K) + k_idx[None, :]
        # We need a 2D mask: rows in [0, M) and columns in [k_start, k_start+BLOCK_K)
        mask_b = (tl.arange(0, M)[:, None] < M) & (k_idx[None, :] < K)
        b_tile = tl.load(b_ptrs, mask=mask_b, other=0.0)

        # Accumulate: dot product of a_vec with each row of b_tile
        # Convert a_vec to [1, BLOCK_K] to broadcast over rows
        acc += tl.sum(b_tile * a_vec[None, :], axis=1)

    # Store the accumulated output vector to C at row 0
    tl.store(C_ptr + row_out * M + tl.arange(0, M), acc)


@triton.jit
def softmax_lse_kernel(
    x_ptr,          # *float32, pointer to logits_scaled vector of length M
    out_lse_ptr,    # *float32, pointer to scalar per head
    attn_ptr,       # *float32, pointer to attn vector of length M
    M: tl.constexpr,   # int, vector length (compile-time for kernel)
    sm_scale: tl.constexpr   # float32 scalar, we treat as constexpr for simplicity
):
    # Compute max for numerical stability
    max_val = tl.max(tl.load(x_ptr + tl.arange(0, M)), axis=0)

    # Compute sum of exp(logits - max)
    exps = tl.exp(tl.load(x_ptr + tl.arange(0, M)) - max_val)
    sum_exp = tl.sum(exps, axis=0)

    # lse = max + log(sum_exp) / log(2)
    lse_val = (max_val + tl.log(sum_exp)) / math.log(2.0)

    # Store lse per head
    tl.store(out_lse_ptr, lse_val)

    # Compute attn and store
    attn_vec = exps / sum_exp
    tl.store(attn_ptr + tl.arange(0, M), attn_vec)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale):
        # Device setup
        device = q_nope.device
        dtype_qn_qp = torch.float32
        dtype_kc = torch.float32

        batch_size = q_nope.shape[0]
        num_qo_heads = q_nope.shape[1]
        head_dim_ckv = q_nope.shape[2]
        head_dim_kpe = q_pe.shape[2]

        # Prepare cached matrices without squeezing (to avoid extra device copies); we'll gather per batch
        # We'll do computations in float32
        # output tensor
        output = torch.empty((batch_size, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device)
        lse = torch.empty((batch_size, num_qo_heads), dtype=torch.float32, device=device)

        for b in range(batch_size):
            # Determine valid token range for batch b
            # kv_indptr: [0, ..., kv_indptr[-1]]
            start = int(kv_indptr[b].item())
            end = int(kv_indptr[b + 1].item())
            if start >= end:
                # No valid tokens for this batch element
                lse[b].zero_()
                continue

            # Gather token indices
            tokens = kv_indices[start:end]  # int32 tensor of shape [M]
            M = tokens.numel()

            # Gather Kc and Kp rows from caches
            # Cast to float32 for compute
            Kc_rows = ckv_cache[tokens.long(), 0, :].to(dtype_kc)  # [M, 512]
            Kp_rows = kpe_cache[tokens.long(), 0, :].to(dtype_kc)  # [M, 64]

            # Prepare output accumulators (float32)
            logits512 = torch.empty((M,), dtype=torch.float32, device=device)
            logits64 = torch.empty((M,), dtype=torch.float32, device=device)

            # Compute qn @ Kc.T and qp @ Kp.T using Triton matvec_row_kernel
            # For qn @ Kc.T: A = q_nope[b, :, :], B = Kc_rows.T
            # Note: We need to pass A as 1D [K], but q_nope is [H, K]. We take head h=0 for now; but original code
            # depends on head h; since H is 16, we loop h or we can create A per head. Here we implement per head:
            # However, Triton kernel expects a single row. We will compute per head by copying q_nope[b, h, :] into A.

            # We'll compute per head h:
            # For each head, we set A_ptr to q_nope[b, h, :] and B_ptr to Kc_rows.T
            # To pass A as [K], we create a 1D tensor and copy q_nope[b, h, :].contiguous()
            for h in range(num_qo_heads):
                # A: q_nope[b, h, :] as 1D float32
                qn = q_nope[b, h, :].contiguous().to(dtype_qn_qp)  # [512]
                qp = q_pe[b, h, :].contiguous().to(dtype_qn_qp)    # [64]

                # A_ptr for qn
                A_qn = qn
                # B_ptr for Kc.T: reshape Kc_rows [M, 512] to [K, M] layout and pass pointer
                # But Triton kernel expects B_ptr to be [M, K] contiguous. We'll use Kc_rows.T for matmul.
                # We need to pass a contiguous [M, K] matrix. We can use Kc_rows.T.contiguous() as B_ptr for matvec_row_kernel.
                # However, Triton kernel expects B_ptr to be [M, K] contiguous. Here, Kc_rows is [M, 512] contiguous along columns.
                # We will pass Kc_rows directly as [M, K] and rely on indexing.
                # B_qn = Kc_rows.T: to get [K, M], we can transpose and make contiguous.
                B_qn = Kc_rows.transpose(0, 1).contiguous()  # [512, M]
                # Launch matvec_row_kernel for qn @ Kc.T
                # Output C_qn is [1, M], we store into logits512
                C_qn = torch.empty((1, M), dtype=torch.float32, device=device)
                matvec_row_kernel[(1,)](
                    A_qn, B_qn, C_qn, K=head_dim_ckv, M=M, BLOCK_K=128, num_warps=4
                )
                # Read C_qn[0, :] and store in logits512
                logits512 = C_qn[0]  # shape [M]

                # Now compute qp @ Kp.T similarly
                A_qp = qp
                B_qp = Kp_rows.transpose(0, 1).contiguous()  # [64, M]
                C_qp = torch.empty((1, M), dtype=torch.float32, device=device)
                matvec_row_kernel[(1,)](
                    A_qp, B_qp, C_qp, K=head_dim_kpe, M=M, BLOCK_K=64, num_warps=2
                )
                logits64 = C_qp[0]  # shape [M]

                # Total logits
                total_logits = logits512 + logits64  # [M], float32

                # Multiply by sm_scale
                scaled_logits = total_logits * sm_scale  # float32

                # Allocate attn vector and lse scalar
                attn = torch.empty((M,), dtype=torch.float32, device=device)
                out_lse = torch.empty((1,), dtype=torch.float32, device=device)

                # Launch softmax_lse_kernel: it writes attn and lse
                softmax_lse_kernel[(1,)](
                    scaled_logits, out_lse, attn, M=M, sm_scale=sm_scale, num_warps=2
                )

                # Compute out[h, :] = attn @ Kc (still use Triton matvec_row_kernel)
                # We need A = attn [M], B = Kc_rows [M, 512], output C[1, 512]
                # attn is 1D [M], Kc_rows is [M, 512]
                # Triton matvec_row_kernel expects A as 1D [K]; but we need A of length M. For out, A should be [K] as well.
                # The out is a vector of length 512 (K), so we use A=attn.sum over M? That's not right.
                # Instead, we implement out as a simple torch op for correctness since M can be runtime; or write a matvec for A as [K].
                # To keep Triton-only, we implement a matvec_row_kernel for this by creating a dummy A of length 512 (all zeros) plus attn contribution? That's not helpful.
                # So, we will compute out using torch.matmul to ensure correctness and avoid further Triton compilation issues for this step.
                # Note: This step is small and M is moderate; torch.matmul will be fine.
                out_vec = attn @ Kc_rows  # [512], float32
                output[b, h, :] = out_vec.to(torch.bfloat16)

                # Store lse per head
                lse[b, h] = out_lse.item()  # scalar float32

        return output, lse


def run(*args):
    return ModelNew()(*args)
