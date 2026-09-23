import torch
import triton
import triton.language as tl


@triton.jit
def _concatenate_streams_kernel(
    E_ptr,        # encoder_hidden_states: [B, T, D]
    H_ptr,        # hidden_states: [B, I, D]
    C_ptr,        # concatenated output: [B, T+I, D]
    B, T, I, D,
    E_b_stride, E_t_stride, E_d_stride,
    H_b_stride, H_i_stride, H_d_stride,
    C_b_stride, C_seqlen_stride, C_d_stride,
):
    pid_b = tl.program_id(0)  # batch dimension
    pid_pos = tl.program_id(1)  # position in [T+I]
    t_total = T + I
    stream = pid_pos // t_total  # 0 => encoder, 1 => hidden
    pos = pid_pos % t_total
    if stream == 1:
        pos = pos - T

    if stream == 0:
        src_ptr = E_ptr + pid_b * E_b_stride + pos * E_t_stride
    else:
        src_ptr = H_ptr + pid_b * H_b_stride + pos * H_i_stride
    dst_ptr = C_ptr + pid_b * C_b_stride + pid_pos * C_seqlen_stride

    # D is typically small; simple loop is fine. If needed, vectorize with tl.arange.
    for d in range(0, D):
        val = tl.load(src_ptr + d * E_d_stride)
        tl.store(dst_ptr + d * C_d_stride, val)


@triton.jit
def _batched_two_streams_gemm_kernel(
    E_ptr,        # encoder_hidden_states: [B, T, D]
    H_ptr,        # hidden_states: [B, I, D]
    W_ptr,        # process_weight: [D, D]
    Y0_ptr,       # output for encoder stream: [B, T, D]
    Y1_ptr,       # output for image stream: [B, I, D]
    B, T, I, D,
    E_b_stride, E_t_stride, E_d_stride,
    H_b_stride, H_i_stride, H_d_stride,
    W0_stride, W1_stride,
    Y0_b_stride, Y0_t_stride, Y0_d_stride,
    Y1_b_stride, Y1_i_stride, Y1_d_stride,
    BLOCK_M, BLOCK_N, BLOCK_K,
):
    # Each program instance computes one output row for either encoder (first T rows) or image (last I rows)
    # We handle both streams by passing total_rows = T+I and using a grid of (total_rows, B).
    pid_row = tl.program_id(0)  # row index in [0, T+I)
    pid_b = tl.program_id(1)    # batch index

    # Determine which stream and base index
    t_total = T + I
    stream = pid_row // t_total  # 0 => encoder, 1 => hidden
    base = 0
    if stream == 1:
        base = T

    # Output pointers
    if stream == 0:
        out_ptr = Y0_ptr + pid_b * Y0_b_stride + (pid_row - base) * Y0_t_stride
    else:
        out_ptr = Y1_ptr + pid_b * Y1_b_stride + (pid_row - base) * Y1_i_stride

    # Accumulator vector of length D
    acc = tl.zeros((D,), dtype=tl.float32)

    # Iterate over K dimension (hidden_dim) in tiles
    # E/H rows are length D vectors; W is [D, D], we load tiles of shape [BLOCK_K, D]
    for k0 in range(0, D, BLOCK_K):
        k_range = k0 + tl.arange(0, BLOCK_K)  # vector [k0, k0+1, ..., k0+BLOCK_K-1]
        # Load input row vector from either E or H at row (pid_row - base)
        # Handle bounds: if pid_row >= T (then stream==1), row index is pid_row - T for H; else pid_row for E
        row_idx = pid_row - base
        valid_row = row_idx >= 0  # always true in this grid, but keep for safety
        # Load input vector for this k-tile
        # Note: we assume contiguous along D for E/H
        in_vec = tl.zeros((BLOCK_K,), dtype=tl.float32)
        # We need to load elements: for each kk in k_range, load E/H at that index
        # Use a loop to construct in_vec (BLOCK_K is small in practice)
        for kk in range(0, BLOCK_K):
            k_idx = k0 + kk
            if k_idx < D:
                val = tl.load(
                    E_ptr + pid_b * E_b_stride + row_idx * E_t_stride + k_idx * E_d_stride
                ) if stream == 0 else tl.load(
                    H_ptr + pid_b * H_b_stride + row_idx * H_i_stride + k_idx * H_d_stride
                )
                in_vec[kk] = val

        # Load weight tile W_sub: shape [BLOCK_K, D]
        w_sub = tl.zeros((BLOCK_K, D), dtype=tl.float32)
        for kk in range(0, BLOCK_K):
            if (k0 + kk) < D:
                w_row_ptr = W_ptr + (k0 + kk) * W0_stride + tl.arange(0, D) * W1_stride
                w_sub[kk, :] = tl.load(w_row_ptr, mask=tl.arange(0, D) < D, other=0.0)

        # Accumulate: acc += sum(in_vec * w_sub, axis=0)
        # Broadcast multiply and sum
        # Build w_sub as a matrix and compute elementwise product
        prod = tl.zeros((BLOCK_K, D), dtype=tl.float32)
        for kk in range(0, BLOCK_K):
            w_row = w_sub[kk, :]
            prod[kk, :] = in_vec[kk] * w_row

        acc += tl.sum(prod, axis=0)

    # Store the accumulated vector to output
    for d in range(0, D):
        tl.store(out_ptr + d * (Y0_d_stride if stream == 0 else Y1_d_stride), acc[d])


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized version:
        - Concatenate streams in Triton.
        - Compute matmul in Triton (two separate batched GEMMs) without torch.matmul.
        Returns: (processed_encoder: [B, T, D], processed_hidden: [B, I, D])
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be CUDA tensors for Triton."
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        D = hidden_states.shape[2]
        assert encoder_hidden_states.shape[2] == D and process_weight.shape[0] == D and process_weight.shape[1] == D, "Dimension mismatch."

        # Ensure contiguous tensors for predictable strides
        E = encoder_hidden_states.contiguous()
        H = hidden_states.contiguous()
        W = process_weight.contiguous()

        # 1) Concatenate streams in Triton: [B, T+I, D]
        C_total = T + I
        C = torch.empty((B, C_total, D), device=E.device, dtype=torch.float32)

        grid_concat = (B, C_total)
        _concatenate_streams_kernel[grid_concat](
            E, H, C,
            B, T, I, D,
            E.stride(0), E.stride(1), E.stride(2),
            H.stride(0), H.stride(1), H.stride(2),
            C.stride(0), C.stride(1), C.stride(2),
            num_warps=4, num_stages=2,
        )

        # 2) Compute processed = C @ W^T in Triton without torch.matmul
        # We implement two separate batched GEMMs:
        # - yA = A @ W, where A = encoder_hidden_states, output [B, T, D]
        # - yB = B @ W, where B = hidden_states, output [B, I, D]
        # Launch Triton kernel for encoder stream: outputs [B, T, D]
        Y0 = torch.empty((B, T, D), device=E.device, dtype=torch.float32)
        # Launch Triton kernel for image stream: outputs [B, I, D]
        Y1 = torch.empty((B, I, D), device=E.device, dtype=torch.float32)

        # Choose tiling. For robustness across varied D, use BLOCK_M=64, BLOCK_N=64, BLOCK_K=64.
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 64

        # Grid is (total_rows, B). For encoder: total_rows = T; for image: total_rows = I. We can reuse grid logic.
        grid = (T, B)  # for Y0
        _batched_two_streams_gemm_kernel[grid](
            E, H, W, Y0, Y1,
            B, T, I, D,
            E.stride(0), E.stride(1), E.stride(2),
            H.stride(0), H.stride(1), H.stride(2),
            W.stride(0), W.stride(1),
            Y0.stride(0), Y0.stride(1), Y0.stride(2),
            Y1.stride(0), Y1.stride(1), Y1.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 3) Split processed from concatenated to match original semantics.
        # Note: In our two-GEMM approach, Y0 and Y1 already correspond to encoder and image streams respectively,
        # so no further split from concatenated is needed.
        return Y0, Y1