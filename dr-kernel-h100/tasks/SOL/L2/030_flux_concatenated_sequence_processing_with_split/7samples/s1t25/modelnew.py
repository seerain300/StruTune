import torch
import triton
import triton.language as tl


@triton.jit
def concat_seq_kernel(
    E_ptr, H_ptr, OUT_ptr,
    B, T, I, D,
    E_b_stride, E_t_stride, E_d_stride,
    H_b_stride, H_i_stride, H_d_stride,
    OUT_b_stride, OUT_s_stride, OUT_d_stride,
):
    pid_b = tl.program_id(0)  # batch id
    pid_pos = tl.program_id(1)  # position in [0, T+I)

    # Determine source tensor and index
    M_total = T + I
    if pid_pos < T:
        src = 0  # encoder
        pos_idx = pid_pos
    else:
        src = 1  # hidden
        pos_idx = pid_pos - T

    # Compute source pointers
    if src == 0:
        src_ptr = E_ptr + pid_b * E_b_stride + pos_idx * E_t_stride
    else:
        src_ptr = H_ptr + pid_b * H_b_stride + (pos_idx) * H_i_stride

    # Compute destination pointer
    dst_ptr = OUT_ptr + pid_b * OUT_b_stride + pid_pos * OUT_s_stride

    # Copy D features
    # We assume E/H are contiguous along D, but use strides for generality.
    for d in range(0, D):
        val = tl.load(src_ptr + d * E_d_stride)
        tl.store(dst_ptr + d * OUT_d_stride, val)


@triton.jit
def batched_matmul_kernel(
    E_ptr, Wt_ptr, Y_ptr,
    B, T, I, D,
    E_b_stride, E_t_stride, E_d_stride,
    Wt_k_stride, Wt_d_stride,  # Wt is [D, D], strides over k and n
    Y_b_stride, Y_m_stride, Y_d_stride,
):
    pid_b = tl.program_id(0)  # batch id
    pid_m = tl.program_id(1)  # output row id in [0, T+I)

    # Accumulator for output row
    acc = tl.zeros((D,), dtype=tl.float32)

    # Loop over K = D for accumulation
    for k in range(0, D):
        e_val = tl.load(E_ptr + pid_b * E_b_stride + pid_m * E_t_stride + k * E_d_stride)
        # Wt[k, n] indexed as [k, n] with strides
        for n in range(0, D):
            w_val = tl.load(Wt_ptr + k * Wt_k_stride + n * Wt_d_stride)
            acc[n] += e_val * w_val

    # Store the row
    y_row_ptr = Y_ptr + pid_b * Y_b_stride + pid_m * Y_m_stride
    for n in range(0, D):
        tl.store(y_row_ptr + n * Y_d_stride, acc[n])


@triton.jit
def split_seq_kernel(
    Y_ptr, OUT_ptr,
    B, T, I, D,
    Y_b_stride, Y_s_stride, Y_d_stride,
    OUT_b_stride, OUT_d_stride,
    ROWS: tl.constexpr,  # number of rows to copy (T or I)
    BLOCK_D: tl.constexpr,  # tile for D dimension
):
    pid_b = tl.program_id(0)  # batch
    pid_idx = tl.program_id(1)  # row index in [0, ROWS)

    # Source row index in Y
    src_row = pid_idx  # since Y has T+I rows, we will call with ROWS=T or ROWS=I

    # Destination pointer in OUT
    dst_ptr = OUT_ptr + pid_b * OUT_b_stride
    src_ptr = Y_ptr + pid_b * Y_b_stride + src_row * Y_s_stride

    # Copy across D dimension in tiles
    for d in range(0, D, BLOCK_D):
        offs = d + tl.arange(0, BLOCK_D)
        mask = offs < D
        vals = tl.load(src_ptr + offs * Y_d_stride, mask=mask, other=0.0)
        tl.store(dst_ptr + offs * OUT_d_stride, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original:
        - Concatenates along sequence dimension in Triton.
        - Computes linear projection in Triton (batched matmul).
        - Splits in Triton.
        """
        assert hidden_states.ndim == 3 and encoder_hidden_states.ndim == 3
        assert process_weight.ndim == 2
        B, T, D = encoder_hidden_states.shape
        Bi, I, Di = hidden_states.shape
        assert B == Bi, "Batch sizes must match"
        assert D == Di, "Hidden dims must match"
        assert process_weight.shape[1] == D and process_weight.shape[0] == D, "process_weight must be [D, D]"

        # Make inputs contiguous for predictable strides
        E = encoder_hidden_states.contiguous()
        H = hidden_states.contiguous()
        W = process_weight.contiguous()

        # 1) Concatenate in Triton: out [B, T+I, D]
        M_total = T + I
        out = torch.empty((B, M_total, D), device=E.device, dtype=torch.float32)

        grid_concat = (B, M_total)
        concat_seq_kernel[grid_concat](
            E, H, out,
            B, T, I, D,
            E.stride(0), E.stride(1), E.stride(2),
            H.stride(0), H.stride(1), H.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            num_warps=1, num_stages=1,
        )

        # 2) Compute processed = out @ process_weight.T in Triton: [B, T+I, D]
        Wt = process_weight.transpose(0, 1).contiguous()  # [D, D]
        processed = torch.empty((B, M_total, D), device=E.device, dtype=torch.float32)

        grid_matmul = (B, M_total)
        batched_matmul_kernel[grid_matmul](
            out, Wt, processed,
            B, T, I, D,
            out.stride(0), out.stride(1), out.stride(2),
            Wt.stride(0), Wt.stride(1),
            processed.stride(0), processed.stride(1), processed.stride(2),
            num_warps=1, num_stages=1,
        )

        # 3) Split in Triton into encoder and image streams
        processed_encoder = torch.empty((B, T, D), device=E.device, dtype=torch.float32)
        processed_hidden = torch.empty((B, I, D), device=E.device, dtype=torch.float32)

        grid_encoder = (B, T)
        grid_hidden = (B, I)

        # Use BLOCK_D=128 for copying; masked loads/stores handle tail if D < 128
        split_seq_kernel[grid_encoder](
            processed, processed_encoder,
            B, T, I, D,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(2),
            ROWS=T, BLOCK_D=128,
            num_warps=1, num_stages=1,
        )

        split_seq_kernel[grid_hidden](
            processed, processed_hidden,
            B, T, I, D,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(2),
            ROWS=I, BLOCK_D=128,
            num_warps=1, num_stages=1,
        )

        return processed_encoder, processed_hidden