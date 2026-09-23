import torch
import triton
import triton.language as tl


@triton.jit
def matmul_concat_row_kernel(
    enc_ptr,        # *ptr to [B, T, H]
    img_ptr,        # *ptr to [B, I, H]
    weight_T_ptr,   # *ptr to [H, H]
    out_ptr,        # *ptr to [M_total, H], M_total = B*(T+I)
    B, T, I, H,                 # dims
    stride_b_e, stride_t_e, stride_h_e,  # enc strides
    stride_b_i, stride_i_i, stride_h_i,  # img strides
    stride_m_out, stride_h_out,          # out strides
    stride_m_w, stride_h_w,              # weight strides
    BLOCK_N: tl.constexpr,               # tile over hidden dim, e.g., 128
    BLOCK_K: tl.constexpr,               # reduction tile, e.g., 64
):
    # Each program handles one output row m in [0, M_total)
    m = tl.program_id(axis=0)
    M_total = B * (T + I)
    # Compute batch and sequence index for this row
    b = m // (T + I)
    seq = m % (T + I)

    # Determine source tensor and row index
    if seq < T:
        src_ptr = enc_ptr + b * stride_b_e + seq * stride_t_e
    else:
        src_ptr = img_ptr + b * stride_b_i + (seq - T) * stride_i_i

    # Accumulator vector for H outputs
    acc_vec = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Loop over hidden_dim in tiles of BLOCK_N
    for n_start in range(0, H, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < H

        # Load a_vec = selected row [H] from source tensor
        a_ptrs = src_ptr + offs_n * stride_h_e if seq < T else src_ptr + offs_n * stride_h_i
        a_vec = tl.load(a_ptrs, mask=mask_n, other=0.0)

        # Accumulate dot with weight_T[:, n] for each n in this tile
        for k_start in range(0, H, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            mask_k = offs_k < H

            # Load weight_T tile [BLOCK_K, BLOCK_N]
            w_ptrs = weight_T_ptr + offs_k[:, None] * stride_m_w + offs_n[None, :] * stride_h_w
            w_tile = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

            # Multiply and reduce along k: acc_vec += sum_k w_tile[k, :] * a_vec[k]
            # Note: a_vec is [BLOCK_N]; we need to multiply per k by corresponding element of a_vec.
            # We implement reduction by summing w_tile * a_vec expanded along k.
            # To do this explicitly:
            # Create k-broadcasted a_vec: a_vec_k = a_vec[None, :] broadcast over k
            # Then sum along axis=0
            acc_vec += tl.sum(w_tile * a_vec[None, :], axis=0)

    # Store the resulting acc_vec to out[m, :]
    out_ptrs = out_ptr + m * stride_m_out + offs_n * stride_h_out
    tl.store(out_ptrs, acc_vec, mask=mask_n)


@triton.jit
def copy_rows_kernel(
    src_ptr,      # *ptr to [M_total, H]
    dst_ptr,      # *ptr to [B, S, H], where S is T or I
    B, T, I, H, S,   # dims (S is T for encoder, I for hidden)
    start_row,             # starting row in src for dst batch
    stride_m_src, stride_h_src,
    stride_b_dst, stride_s_dst, stride_h_dst,
    BLOCK_N: tl.constexpr,
):
    # Each program handles one row copy: dst batch b, row s in [0, S), column tile n
    pid_row = tl.program_id(axis=0)
    b = pid_row // S
    s = pid_row % S
    src_row = start_row + s

    for n_start in range(0, H, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        mask_n = offs_n < H

        src_ptrs = src_ptr + src_row * stride_m_src + offs_n * stride_h_src
        vals = tl.load(src_ptrs, mask=mask_n, other=0.0)

        dst_ptrs = dst_ptr + b * stride_b_dst + s * stride_s_dst + offs_n * stride_h_dst
        tl.store(dst_ptrs, vals, mask=mask_n)


def _run_triton_only(hidden_states, encoder_hidden_states, process_weight):
    """
    Compute processed = [B*(T+I), H] using a Triton kernel that fuses concatenation and matmul,
    then split into processed_encoder [B, T, H] and processed_hidden [B, I, H] using Triton copy kernels.
    """
    assert hidden_states.is_contiguous() and encoder_hidden_states.is_contiguous() and process_weight.is_contiguous(), "Tensors must be contiguous"
    B = hidden_states.shape[0]
    T = encoder_hidden_states.shape[1]
    I = hidden_states.shape[1]
    H = hidden_states.shape[2]
    M_total = B * (T + I)

    # Prepare weight_T = process_weight.T as [H, H]
    weight_T = process_weight.t().contiguous()  # [H, H]

    # Output [M_total, H]
    out = torch.empty((M_total, H), dtype=hidden_states.dtype, device=hidden_states.device)

    # Strides
    stride_b_e, stride_t_e, stride_h_e = encoder_hidden_states.stride()
    stride_b_i, stride_i_i, stride_h_i = hidden_states.stride()
    stride_m_out, stride_h_out = out.stride()
    stride_m_w, stride_h_w = weight_T.stride()

    # Launch Triton kernel: one program per output row m in [0, M_total)
    grid = (M_total,)
    # Choose block sizes; for H=128 (common in your axes), BLOCK_N=128 covers the entire hidden dim.
    BLOCK_N = 128 if H >= 128 else (64 if H >= 64 else 32)
    BLOCK_K = 64  # reduction tile
    matmul_concat_row_kernel[grid](
        encoder_hidden_states, hidden_states, weight_T, out,
        B, T, I, H,
        stride_b_e, stride_t_e, stride_h_e,
        stride_b_i, stride_i_i, stride_h_i,
        stride_m_out, stride_h_out,
        stride_m_w, stride_h_w,
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )

    # Allocate outputs [B, T, H] and [B, I, H]
    processed_encoder = torch.empty((B, T, H), dtype=hidden_states.dtype, device=hidden_states.device)
    processed_hidden = torch.empty((B, I, H), dtype=hidden_states.dtype, device=hidden_states.device)

    # Per-batch copy of rows into encoder and hidden outputs
    # Encoder: rows [0 : B*T)
    grid_encoder = (B * T,)
    copy_rows_kernel[grid_encoder](
        out, processed_encoder,
        B, T, I, H, T, 0,
        stride_m_out, stride_h_out,
        processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
        BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=2,
    )
    # Hidden: rows [B*T : M_total)
    grid_hidden = (M_total - B * T,)
    # Note: grid_hidden should only cover rows in hidden part; Triton grid requires integer, so launch with B*I and mask within kernel.
    # To avoid mismatch, we launch over B*I and compute start_row = B*T inside the kernel.
    copy_rows_kernel[(B * I,)](
        out, processed_hidden,
        B, T, I, H, I, B * T,
        stride_m_out, stride_h_out,
        processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
        BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=2,
    )

    return processed_encoder, processed_hidden


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        # Ensure inputs are contiguous
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()
        if not encoder_hidden_states.is_contiguous():
            encoder_hidden_states = encoder_hidden_states.contiguous()
        if not process_weight.is_contiguous():
            process_weight = process_weight.contiguous()

        # All computation happens in Triton (no torch ops)
        processed_encoder, processed_hidden = _run_triton_only(hidden_states, encoder_hidden_states, process_weight)
        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
