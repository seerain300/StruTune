import torch
import triton
import triton.language as tl


@triton.jit
def out_row_kernel(
    enc_ptr,        # *ptr to [B, T, H]
    img_ptr,        # *ptr to [B, I, H]
    out_ptr,        # *ptr to [M_total, H], M_total = B*(T+I)
    B, T, I, H,     # sizes
    stride_b_e, stride_t_e, stride_h_e,  # encoder strides
    stride_b_i, stride_i_i, stride_h_i,  # image strides
    stride_m_out, stride_h_out,          # out strides (row-major [M_total, H])
    BLOCK_N: tl.constexpr,               # tile over H (columns)
    BLOCK_K: tl.constexpr,               # tile over reduction K
):
    # One program computes one output row m
    m = tl.program_id(axis=0)
    # Compute batch and seq within the batch
    batch = m // (T + I)
    seq = m % (T + I)

    # Select source tensor and row index
    # If seq < T: from encoder, else from image at seq - T
    # Note: seq is in [0, T+I-1], so seq < T is always valid.
    # Build pointers for the row
    if seq < T:
        ptr_row = enc_ptr + batch * stride_b_e + seq * stride_t_e + tl.arange(0, H) * stride_h_e
        a_row = tl.load(ptr_row, mask=tl.arange(0, H) < H, other=0.0)  # load entire row
    else:
        src_row = seq - T
        ptr_row = img_ptr + batch * stride_b_i + src_row * stride_i_i + tl.arange(0, H) * stride_h_i
        a_row = tl.load(ptr_row, mask=tl.arange(0, H) < H, other=0.0)  # load entire row

    # Now compute out[m, :] = a_row @ process_weight.T
    # We'll iterate over H in tiles of BLOCK_N and reduce over K tiles of BLOCK_K.
    # Accumulator is a vector of length H (float32).
    acc = tl.zeros((H,), dtype=tl.float32)

    for n_start in range(0, H, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < H

        # For each n in tile, accumulate dot with a_row
        for k_start in range(0, H, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            mask_k = offs_k < H

            # Load weight_T tile: weight_T is [H, H], column tile at n tile, row tile at k tile
            # weight_T[k, n] for k in k_tile, n in n_tile
            w_ptrs = weight_T_ptr + offs_k[:, None] * stride_m_w + offs_n[None, :] * stride_h_w
            w_tile = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

            # Dot: sum over k of w_tile[k, :] * a_row[k]
            # Multiply tile by a_row and reduce along axis=0 (K axis)
            prod = w_tile * a_row[None, :][mask_k[:, None], :]  # broadcast a_row along K
            acc[offs_n] += tl.sum(prod, axis=0)

    # Store the computed row to out[m, :]
    out_ptrs = out_ptr + m * stride_m_out + tl.arange(0, H) * stride_h_out
    tl.store(out_ptrs, acc, mask=tl.arange(0, H) < H)


@triton.jit
def copy_rows_split_kernel(
    src_ptr,        # *ptr to [M_total, H]
    dst_ptr,        # *ptr to [B, S, H], where S is T (encoder) or I (hidden)
    B, T, I, H, S,  # dims
    start_row,      # starting row in src for dst batch
    stride_m_src, stride_h_src,
    stride_b_dst, stride_s_dst, stride_h_dst,
    BLOCK_N: tl.constexpr,
):
    # Grid: (B, S, ceil(H / BLOCK_N))
    b = tl.program_id(axis=0)
    s = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=2)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < H

    src_row = start_row + b * S + s
    src_ptrs = src_ptr + src_row * stride_m_src + offs_n * stride_h_src
    vals = tl.load(src_ptrs, mask=mask_n, other=0.0)

    dst_row_ptrs = dst_ptr + b * stride_b_dst + s * stride_s_dst + offs_n * stride_h_dst
    tl.store(dst_row_ptrs, vals, mask=mask_n)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        # Ensure inputs are contiguous and on same device/dtype
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        assert encoder_hidden_states.shape[2] == H, "encoder_hidden_states hidden_dim must match hidden_states"
        assert process_weight.shape[0] == H and process_weight.shape[1] == H, "process_weight must be [H, H]"
        # Make sure tensors are contiguous
        enc = encoder_hidden_states.contiguous()
        img = hidden_states.contiguous()
        # Move process_weight to [H, H] contiguous
        weight = process_weight.contiguous()  # [H, H], no bias
        # Create weight^T for kernel: [H, H]
        weight_T = weight.t().contiguous()  # [H, H], we'll pass pointers as-is

        M_total = B * (T + I)

        # Output buffer [M_total, H], dtype float32 (compute in fp32 for stability)
        out = torch.empty((M_total, H), dtype=torch.float32, device=enc.device)

        # Launch Triton kernel: one program per output row
        grid = (M_total,)
        out_row_kernel[grid](
            enc, img, out,
            B, T, I, H,
            enc.stride(0), enc.stride(1), enc.stride(2),
            img.stride(0), img.stride(1), img.stride(2),
            out.stride(0), out.stride(1),
            BLOCK_N=128, BLOCK_K=64,
            num_warps=4, num_stages=2,
        )

        # Split into encoder and hidden outputs using PyTorch slicing (safe and simple)
        processed_encoder_rows = out[:B * T, :]
        processed_hidden_rows = out[B * T:, :]

        # Reshape to [B, T, H] and [B, I, H]
        processed_encoder = processed_encoder_rows.view(B, T, H)
        processed_hidden = processed_hidden_rows.view(B, I, H)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
