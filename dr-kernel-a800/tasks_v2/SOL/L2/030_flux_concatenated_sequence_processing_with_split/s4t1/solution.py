import torch
import triton
import triton.language as tl


# Fused concatenation + matmul:
# Computes processed[m, :] = concatenated[m, :] @ process_weight.T, where concatenated is implicitly built:
# For m in [0, B*(T+I)):
#   - If m < B*T: concatenated[m] = encoder_hidden_states[b, m%B*T, :]
#   - Else: concatenated[m] = hidden_states[b, m-B*T, :]
# Then processed = [B*(T+I), H]
@triton.jit
def matmul_concat_kernel(
    enc_ptr, img_ptr, weight_t_ptr, processed_ptr,
    B, T, I, H,
    stride_eb, stride_et, stride_eh,
    stride_ib, stride_it, stride_ih,
    stride_wb, stride_wh,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Tile over M (rows) and N (columns)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    M = B * (T + I)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # output rows
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # output columns (hidden_dim)

    m_mask = offs_m < M
    n_mask = offs_n < H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K = H
    for k in range(0, H, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        k_mask = offs_k < H

        # Determine batch and seq type for each m
        total_seqs = T + I
        b = offs_m // total_seqs
        seq_type = offs_m % total_seqs  # 0 -> encoder, 1 -> image

        # Masks per row
        mask_e = (seq_type == 0) & (b < B) & m_mask
        mask_i = (seq_type == 1) & (b < B) & m_mask

        # Compute source indices:
        # For encoder: t = offs_m % T
        # For image: t' = offs_m - B*T
        t_e = offs_m % T
        t_i = offs_m - B * T

        # Build A tiles for both sources and load with masks
        # Encoders: A_row = enc[b, t_e, offs_k]
        # Images:   A_row = img[b, t_i, offs_k]
        # We'll use masked loads to avoid OOB.
        # For enc, invalid rows (mask_e == False) will contribute zeros; similarly for img.
        # Note: we can't directly branch per row; we rely on masked loads and sum.

        # Compute batch broadcast
        batch_b = b[:, None]  # [BLOCK_M, 1]

        # Pointers for encoder and image
        a_e_ptrs = enc_ptr + (batch_b * stride_eb + t_e[:, None] * stride_et + offs_k[None, :] * stride_eh)
        a_e = tl.load(a_e_ptrs, mask=(mask_e[:, None] & k_mask[None, :]), other=0.0)

        a_i_ptrs = img_ptr + (batch_b * stride_ib + t_i[:, None] * stride_it + offs_k[None, :] * stride_ih)
        a_i = tl.load(a_i_ptrs, mask=(mask_i[:, None] & k_mask[None, :]), other=0.0)

        # A for this K-block
        a = a_e + a_i  # since mask_e and mask_i are disjoint for a given m

        # B tile: weight_t_ptr[offs_k, offs_n] = process_weight.T[k, n]
        b_ptrs = weight_t_ptr + (offs_k[:, None] * stride_wb + offs_n[None, :] * stride_wh)
        b = tl.load(b_ptrs, mask=(k_mask[:, None] & n_mask[None, :]), other=0.0)

        # Accumulate
        acc += tl.dot(a, b)

    # Store results into processed
    # processed is [M, H], contiguous; we can use row-major offsets
    processed_row_ptrs = processed_ptr + (offs_m[:, None] * H + offs_n[None, :])  # since processed_ptr points to [M, H]
    tl.store(processed_row_ptrs, acc, mask=(m_mask[:, None] & n_mask[None, :]))


# Per-batch copy kernel for encoder: write rows [b*T : (b+1)*T, :] from processed into processed_encoder[b]
@triton.jit
def copy_rows_kernel(
    src_ptr, dest_ptr,
    B, T, I, H,
    stride_srcm, stride_srch,
    stride_destb, stride_destt, stride_desth,
    b: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)  # tiles over rows [0, T)
    pid_n = tl.program_id(1)  # tiles over columns [0, H)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < T
    n_mask = offs_n < H

    # Source rows in processed: start_row = b*T
    start_row = b * T
    src_rows = start_row + offs_m
    src_ptrs = src_ptr + (src_rows[:, None] * stride_srcm + offs_n[None, :] * stride_srch)
    # Destination: processed_encoder[b] has shape [T, H]
    dest_ptrs = dest_ptr + (b * stride_destb + offs_m[:, None] * stride_destt + offs_n[None, :] * stride_desth)

    vals = tl.load(src_ptrs, mask=(m_mask[:, None] & n_mask[None, :]), other=0.0)
    tl.store(dest_ptrs, vals, mask=(m_mask[:, None] & n_mask[None, :]))


# Per-batch copy kernel for hidden: write rows [(b+1)*T + b*I : (b+1)*(T+I), :] from processed into processed_hidden[b]
@triton.jit
def copy_rows_kernel_hidden(
    src_ptr, dest_ptr,
    B, T, I, H,
    stride_srcm, stride_srch,
    stride_destb, stride_destt, stride_desth,
    b: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < I
    n_mask = offs_n < H

    # Start row for hidden in this batch
    start_row = (b + 1) * T + b * I
    src_rows = start_row + offs_m
    src_ptrs = src_ptr + (src_rows[:, None] * stride_srcm + offs_n[None, :] * stride_srch)

    dest_ptrs = dest_ptr + (b * stride_destb + offs_m[:, None] * stride_destt + offs_n[None, :] * stride_desth)

    vals = tl.load(src_ptrs, mask=(m_mask[:, None] & n_mask[None, :]), other=0.0)
    tl.store(dest_ptrs, vals, mask=(m_mask[:, None] & n_mask[None, :]))


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-only implementation that:
        - Fuses concatenation into the matmul kernel (no torch.cat).
        - Performs per-batch copying to produce outputs (no torch slicing).
        - Matches the original function's signature and returns (processed_encoder, processed_hidden).
        """
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3, "Inputs must be [B, S, H]"
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        total_seqs = T + I
        M = B * total_seqs

        # Ensure contiguous for simple stride usage
        enc = encoder_hidden_states.contiguous()
        img = hidden_states.contiguous()
        weight = process_weight.contiguous()  # [H, H]

        # Allocate processed [M, H]
        processed = torch.empty((M, H), dtype=enc.dtype, device=enc.device)

        # Launch fused matmul kernel
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(H, BLOCK_N))
        matmul_concat_kernel[grid](
            enc, img, weight, processed,
            B, T, I, H,
            enc.stride(0), enc.stride(1), enc.stride(2),
            img.stride(0), img.stride(1), img.stride(2),
            weight.stride(0), weight.stride(1),
            num_warps=4, num_stages=2,
        )

        # Allocate outputs
        processed_encoder = torch.empty((B, T, H), dtype=enc.dtype, device=enc.device)
        processed_hidden = torch.empty((B, I, H), dtype=enc.dtype, device=enc.device)

        # Per-batch copies
        for b in range(B):
            grid_copy_e = (triton.cdiv(T, BLOCK_M), triton.cdiv(H, BLOCK_N))
            copy_rows_kernel[grid_copy_e](
                processed, processed_encoder[b],
                B, T, I, H,
                processed.stride(0), processed.stride(1),
                processed_encoder[b].stride(0), processed_encoder[b].stride(1), processed_encoder[b].stride(2),
                b,
                num_warps=4, num_stages=2,
            )
            grid_copy_h = (triton.cdiv(I, BLOCK_M), triton.cdiv(H, BLOCK_N))
            copy_rows_kernel_hidden[grid_copy_h](
                processed, processed_hidden[b],
                B, T, I, H,
                processed.stride(0), processed.stride(1),
                processed_hidden[b].stride(0), processed_hidden[b].stride(1), processed_hidden[b].stride(2),
                b,
                num_warps=4, num_stages=2,
            )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
