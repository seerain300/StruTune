import torch
import triton
import triton.language as tl


@triton.jit
def _concatenate_rows_kernel(
    src_a_ptr,  # *float32, points to [B, T, D]
    src_b_ptr,  # *float32, points to [B, I, D]
    dst_ptr,    # *float32, points to [B, T+I, D]
    B: tl.constexpr, T: tl.constexpr, I: tl.constexpr, D: tl.constexpr,
    src_a_stride_b, src_a_stride_i, src_a_stride_d,
    src_b_stride_b, src_b_stride_i, src_b_stride_d,
    dst_stride_b, dst_stride_s, dst_stride_d,
    TOTAL_S: tl.constexpr,  # TOTAL_S = T + I
    segment_id: tl.constexpr,  # which segment along sequence: 0 -> encoder, 1 -> image
    BLOCK_M: tl.constexpr,     # number of rows per segment (typically 1)
):
    # Grid: (B, TOTAL_S, segment_id) but we use segment_id to pick source and destination slice.
    # Each program handles exactly one output row for one batch:
    b = tl.program_id(0)
    s = tl.program_id(1)  # output row index within [0, TOTAL_S)
    # Decide whether this s belongs to encoder (first T) or image (last I)
    is_encoder = s < T
    # Compute the source row index in the corresponding input:
    # If segment_id == 0 (encoder): s is output row; if segment_id == 1 (image): src_i = s - T
    # However, since we pass TOTAL_S and segment_id, we can infer from is_encoder.
    if is_encoder:
        src_i = s
        src_ptr = src_a_ptr
        src_stride_b = src_a_stride_b
        src_stride_i = src_a_stride_i
        src_stride_d = src_a_stride_d
    else:
        src_i = s - T
        src_ptr = src_b_ptr
        src_stride_b = src_b_stride_b
        src_stride_i = src_b_stride_i
        src_stride_d = src_b_stride_d

    # Destination index in output tensor
    dst_s = s
    dst_ptr_row = dst_ptr + b * dst_stride_b + dst_s * dst_stride_s

    # Copy along D dimension
    # We iterate d from 0 to D-1
    for d in range(0, D):
        val = tl.load(src_ptr + b * src_stride_b + src_i * src_stride_i + d * src_stride_d)
        tl.store(dst_ptr_row + d * dst_stride_d, val)


@triton.jit
def _matmul_rowwise_kernel(
    X_ptr,      # *float32, [B, S, D] where S=T+I
    W_ptr,      # *float32, [D, D]
    Y_ptr,      # *float32, [B, S, D]
    B: tl.constexpr, S: tl.constexpr, D: tl.constexpr,
    X_stride_b, X_stride_s, X_stride_d,
    W_stride_d0, W_stride_d1,
    Y_stride_b, Y_stride_s, Y_stride_d,
    BLOCK_K: tl.constexpr,
):
    # One program per (b, s)
    b = tl.program_id(0)
    s = tl.program_id(1)

    # Vector accumulator for the output row
    acc = tl.zeros([D], dtype=tl.float32)

    # Iterate over hidden_dim D in tiles
    for k0 in range(0, D, BLOCK_K):
        # Load input row segment X[b, s, k0:k0+BLOCK_K]
        k_range = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_range < D
        x_row = tl.load(X_ptr + b * X_stride_b + s * X_stride_s + k_range * X_stride_d, mask=mask_k, other=0.0)

        # Load corresponding submatrix W[k_range, 0:D] as [BLOCK_K, D]
        # Build 2D pointer for W: rows = k_range, cols = 0:D
        w_ptrs = W_ptr + k_range[:, None] * W_stride_d0 + tl.arange(0, D)[None, :] * W_stride_d1
        mask_w = (k_range[:, None] < D) & (tl.arange(0, D)[None, :] < D)
        w_sub = tl.load(w_ptrs, mask=mask_w, other=0.0)

        # Accumulate: acc += sum over k_tile of x_row[k] * w_sub[k, :]
        # Cast to float32 for accumulation
        x_row = x_row.to(tl.float32)
        w_sub = w_sub.to(tl.float32)
        acc += tl.sum(x_row[:, None] * w_sub, axis=0)

    # Store the result row
    y_row_ptr = Y_ptr + b * Y_stride_b + s * Y_stride_s
    for d in range(0, D):
        tl.store(y_row_ptr + d * Y_stride_d, acc[d])


@triton.jit
def _split_copy_rows_kernel(
    src_ptr,   # *float32, [B, TOTAL_S, D]
    dst_ptr,   # *float32, destination tensor [B, T or I, D]
    B: tl.constexpr, TOTAL_S: tl.constexpr, D: tl.constexpr,
    src_stride_b, src_stride_s, src_stride_d,
    dst_stride_b, dst_stride_i, dst_stride_d,
    N: tl.constexpr,  # N = T or I for the destination
    segment_id: tl.constexpr,  # 0 -> split first N rows (encoder), 1 -> split last N rows (image)
):
    b = tl.program_id(0)
    i = tl.program_id(1)
    if segment_id == 0 and i < N:
        s = i
        src_row = src_ptr + b * src_stride_b + s * src_stride_s
        dst_row = dst_ptr + b * dst_stride_b + i * dst_stride_i
        for d in range(0, D):
            val = tl.load(src_row + d * src_stride_d)
            tl.store(dst_row + d * dst_stride_d, val)
    elif segment_id == 1 and i < N:
        s = i + N  # split last I rows from [T+I]
        src_row = src_ptr + b * src_stride_b + s * src_stride_s
        dst_row = dst_ptr + b * dst_stride_b + i * dst_stride_i
        for d in range(0, D):
            val = tl.load(src_row + d * src_stride_d)
            tl.store(dst_row + d * dst_stride_d, val)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only implementation of:
          concatenated = cat([encoder_hidden_states, hidden_states], dim=1)  # [B, T+I, D]
          processed = concatenated @ process_weight.T                       # [B, T+I, D]
          processed_encoder = processed[:, :T, :]
          processed_hidden = processed[:, T:, :]
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be CUDA for Triton kernels."
        B = hidden_states.shape[0]
        I = hidden_states.shape[1]
        T = encoder_hidden_states.shape[1]
        D = hidden_states.shape[2]
        TOTAL_S = T + I

        # Ensure contiguous tensors
        hs = hidden_states.contiguous()
        ehs = encoder_hidden_states.contiguous()
        W = process_weight.contiguous()

        # 1) Concatenate in Triton: dst [B, TOTAL_S, D]
        dst = torch.empty((B, TOTAL_S, D), dtype=torch.float32, device=hs.device)

        # Launch concatenation kernel: grid over (B, TOTAL_S, segments). We only have two segments (encoder and image).
        grid_concat = (B, TOTAL_S, 2)
        _concatenate_rows_kernel[grid_concat](
            ehs, hs, dst,
            B=B, T=T, I=I, D=D,
            src_a_stride_b=ehs.stride(0), src_a_stride_i=ehs.stride(1), src_a_stride_d=ehs.stride(2),
            src_b_stride_b=hs.stride(0), src_b_stride_i=hs.stride(1), src_b_stride_d=hs.stride(2),
            dst_stride_b=dst.stride(0), dst_stride_s=dst.stride(1), dst_stride_d=dst.stride(2),
            TOTAL_S=TOTAL_S,
            segment_id=0, BLOCK_M=1,
            num_warps=1, num_stages=1,
        )

        # 2) Matmul in Triton: processed = dst @ W.T  -> [B, TOTAL_S, D]
        processed = torch.empty((B, TOTAL_S, D), dtype=torch.float32, device=hs.device)
        _matmul_rowwise_kernel[(B, TOTAL_S)](
            dst, W,
            processed,
            B=B, S=TOTAL_S, D=D,
            X_stride_b=dst.stride(0), X_stride_s=dst.stride(1), X_stride_d=dst.stride(2),
            W_stride_d0=W.stride(0), W_stride_d1=W.stride(1),
            Y_stride_b=processed.stride(0), Y_stride_s=processed.stride(1), Y_stride_d=processed.stride(2),
            BLOCK_K=64,  # reasonable tile size; D is typically a multiple of 64 in provided workloads
            num_warps=2, num_stages=2,
        )

        # 3) Split back in Triton
        processed_encoder = torch.empty((B, T, D), dtype=torch.float32, device=hs.device)
        processed_hidden = torch.empty((B, I, D), dtype=torch.float32, device=hs.device)

        # Split encoder: first T rows
        grid_split0 = (B, T)
        _split_copy_rows_kernel[grid_split0](
            processed,
            processed_encoder,
            B=B, TOTAL_S=TOTAL_S, D=D,
            src_stride_b=processed.stride(0), src_stride_s=processed.stride(1), src_stride_d=processed.stride(2),
            dst_stride_b=processed_encoder.stride(0), dst_stride_i=processed_encoder.stride(1), dst_stride_d=processed_encoder.stride(2),
            N=T, segment_id=0,
            num_warps=1, num_stages=1,
        )

        # Split hidden: last I rows
        grid_split1 = (B, I)
        _split_copy_rows_kernel[grid_split1](
            processed,
            processed_hidden,
            B=B, TOTAL_S=TOTAL_S, D=D,
            src_stride_b=processed.stride(0), src_stride_s=processed.stride(1), src_stride_d=processed.stride(2),
            dst_stride_b=processed_hidden.stride(0), dst_stride_i=processed_hidden.stride(1), dst_stride_d=processed_hidden.stride(2),
            N=I, segment_id=1,
            num_warps=1, num_stages=1,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
