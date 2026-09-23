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
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)  # batch
    pid_pos = tl.program_id(1)  # position in [T+I]
    t_total = T + I
    stream = pid_pos // t_total  # 0 => encoder rows, 1 => hidden rows
    pos = pid_pos % t_total
    if stream == 1:
        pos = pos - T  # map to hidden seq index

    src_ptr = E_ptr + pid_b * E_b_stride + pos * E_t_stride if stream == 0 else H_ptr + pid_b * H_b_stride + pos * H_i_stride
    dst_ptr = C_ptr + pid_b * C_b_stride + pid_pos * C_seqlen_stride

    # Vectorized copy along D
    for d in range(0, D, BLOCK_D):
        offs = d + tl.arange(0, BLOCK_D)
        mask = offs < D
        vals = tl.load(src_ptr + offs * E_d_stride, mask=mask, other=0.0)
        tl.store(dst_ptr + offs * C_d_stride, vals, mask=mask)


@triton.jit
def _split_seq_kernel(
    Y_ptr,        # processed concatenated: [B, T+I, D]
    Y0_ptr,       # processed_encoder: [B, T, D]
    Y1_ptr,       # processed_hidden: [B, I, D]
    B, T, I, D,
    Y_b_stride, Y_seqlen_stride, Y_d_stride,
    Y0_b_stride, Y0_d_stride,
    Y1_b_stride, Y1_d_stride,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)  # batch
    pid_row = tl.program_id(1)  # row within [T] or [I]
    # Encoder rows
    for d in range(0, D, BLOCK_D):
        offs = d + tl.arange(0, BLOCK_D)
        mask = offs < D
        src = Y_ptr + pid_b * Y_b_stride + pid_row * Y_seqlen_stride + offs * Y_d_stride
        dst = Y0_ptr + pid_b * Y0_b_stride + pid_row * Y0_d_stride + offs * Y0_d_stride
        vals = tl.load(src, mask=mask, other=0.0)
        tl.store(dst, vals, mask=mask)
    # Hidden rows (shift by T)
    for d in range(0, D, BLOCK_D):
        offs = d + tl.arange(0, BLOCK_D)
        mask = offs < D
        src = Y_ptr + pid_b * Y_b_stride + (pid_row + T) * Y_seqlen_stride + offs * Y_d_stride
        dst = Y1_ptr + pid_b * Y1_b_stride + pid_row * Y1_d_stride + offs * Y1_d_stride
        vals = tl.load(src, mask=mask, other=0.0)
        tl.store(dst, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized variant:
        - Concatenates encoder_hidden_states and hidden_states along sequence dimension in Triton.
        - Computes batched matmul in PyTorch (torch.matmul) to ensure correctness.
        - Splits the processed tensor back into encoder and hidden outputs using a Triton kernel.
        """
        # Ensure inputs are on CUDA and contiguous
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All tensors must be on CUDA device"
        E = encoder_hidden_states.contiguous()
        H = hidden_states.contiguous()
        W = process_weight.contiguous()

        B, T, D = E.shape
        _, I, D2 = H.shape
        assert D == D2, "hidden_dim must match between encoder_hidden_states and hidden_states"
        assert W.shape == (D, D), "process_weight must have shape [hidden_dim, hidden_dim]"

        # 1) Concatenate in Triton: [B, T+I, D]
        C_total = T + I
        C = torch.empty((B, C_total, D), device=E.device, dtype=torch.float32)

        grid_concat = (B, C_total)
        _concatenate_streams_kernel[grid_concat](
            E, H, C,
            B, T, I, D,
            E.stride(0), E.stride(1), E.stride(2),
            H.stride(0), H.stride(1), H.stride(2),
            C.stride(0), C.stride(1), C.stride(2),
            BLOCK_D=64, num_warps=4, num_stages=2,
        )

        # 2) Batched matmul: processed = C @ W.T (no bias)
        # C: [B, T+I, D], W.T: [D, D]
        processed = torch.matmul(C, W.t())

        # 3) Split using Triton: yA = first T rows, yB = next I rows
        processed_encoder = torch.empty((B, T, D), device=E.device, dtype=torch.float32)
        processed_hidden = torch.empty((B, I, D), device=E.device, dtype=torch.float32)

        grid_split = (B, T)  # each program instance handles one row copy; second half is done by iterating D tiles
        # Note: The split kernel loops over D in tiles; launching over (B, T) is sufficient.
        _split_seq_kernel[grid_split](
            processed, processed_encoder, processed_hidden,
            B, T, I, D,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(2),
            BLOCK_D=64, num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
