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
    C_b_stride, C_row_stride, C_d_stride,
):
    # 3D grid: (batch, row_in_concatenated, feature)
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)  # row index in the concatenated sequence
    pid_d = tl.program_id(2)  # feature index (along D)

    # Determine source tensor and row index
    # If pid_m < T: source = E at row pid_m; else source = H at row pid_m - T
    if pid_m < T:
        src_ptr = E_ptr + pid_b * E_b_stride + pid_m * E_t_stride
    else:
        src_row = pid_m - T
        src_ptr = H_ptr + pid_b * H_b_stride + src_row * H_i_stride

    dst_ptr = C_ptr + pid_b * C_b_stride + pid_m * C_row_stride

    val = tl.load(src_ptr + pid_d * E_d_stride)
    tl.store(dst_ptr + pid_d * C_d_stride, val)


@triton.jit
def _split_rows_kernel(
    Y_ptr,        # processed: [B, T+I, D]
    OUT0_ptr,     # processed_encoder: [B, T, D]
    OUT1_ptr,     # processed_hidden: [B, I, D]
    B, T, I, D,
    Y_b_stride, Y_row_stride, Y_d_stride,
    OUT0_b_stride, OUT0_row_stride, OUT0_d_stride,
    OUT1_b_stride, OUT1_row_stride, OUT1_d_stride,
):
    # 3D grid: (batch, output_row, feature)
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)  # output row index (0..T-1 for OUT0, T..T+I-1 for OUT1)
    pid_d = tl.program_id(2)

    # Map pid_m to source row in Y: OUT0 uses rows 0..T-1; OUT1 uses rows T..T+I-1
    src_row = pid_m

    # Load and store
    y_ptr = Y_ptr + pid_b * Y_b_stride + src_row * Y_row_stride
    out0_ptr = OUT0_ptr + pid_b * OUT0_b_stride + pid_m * OUT0_row_stride

    # Load from Y and store to OUT0
    val = tl.load(y_ptr + pid_d * Y_d_stride)
    tl.store(out0_ptr + pid_d * OUT0_d_stride, val)

    # For OUT1, pid_m in [T, T+I-1] => src_row = pid_m; OUT1_ptr handles different strides
    out1_ptr = OUT1_ptr + pid_b * OUT1_b_stride + (pid_m - T) * OUT1_row_stride
    val = tl.load(y_ptr + pid_d * Y_d_stride)
    tl.store(out1_ptr + pid_d * OUT1_d_stride, val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized variant:
        - Concatenate two streams in Triton.
        - Perform linear projection using torch.matmul (robust and correct).
        - Split result into two streams in Triton.
        Returns (processed_encoder, processed_hidden)
        """
        # Ensure dtype is float32 for robustness; original code uses default float32
        # Inputs: [B, T, D], [B, I, D], [D, D]
        # We assume contiguous tensors for simplicity
        E = encoder_hidden_states.contiguous()
        H = hidden_states.contiguous()
        W = process_weight.contiguous()

        B, T, D = E.shape
        B2, I, D2 = H.shape
        assert B == B2 and D == D2, "Batch and feature dimensions must match."

        # 1) Concatenate encoder and hidden sequences along sequence dimension using Triton
        C_total = T + I
        C = torch.empty((B, C_total, D), device=E.device, dtype=torch.float32)

        grid_concat = (B, C_total, D)
        _concatenate_streams_kernel[grid_concat](
            E, H, C,
            B, T, I, D,
            E.stride(0), E.stride(1), E.stride(2),
            H.stride(0), H.stride(1), H.stride(2),
            C.stride(0), C.stride(1), C.stride(2),
            num_warps=4, num_stages=2,
        )

        # 2) Linear projection using torch.matmul (no bias) -> robust and correct
        # Original expects W of shape [D, D], right multiply: C @ W.T
        processed = torch.matmul(C, W.t())

        # 3) Split back into two streams using Triton
        processed_encoder = torch.empty((B, T, D), device=processed.device, dtype=torch.float32)
        processed_hidden = torch.empty((B, I, D), device=processed.device, dtype=torch.float32)

        grid_split = (B, T, D)
        _split_rows_kernel[grid_split](
            processed, processed_encoder, processed_hidden,
            B, T, I, D,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden