import torch
import triton
import triton.language as tl

@triton.jit
def cat_rows_kernel(
    e_ptr, i_ptr, out_ptr,
    B, T, I, H,
    stride_eb, stride_et, stride_eh,
    stride_ib, stride_it, stride_ih,
    stride_ob, stride_om, stride_on,
    BLOCK_N: tl.constexpr,
):
    # Each program handles one (batch, row) pair
    b = tl.program_id(0)
    p = tl.program_id(1)
    M = T + I

    # Bounds check
    if (b >= B) or (p >= M):
        return

    # Compute source pointer for the row
    if p < T:
        src_ptr = e_ptr + b * stride_eb + p * stride_et
    else:
        src_ptr = i_ptr + b * stride_ib + (p - T) * stride_it

    # Output pointer for the row in concatenated matrix
    out_row_ptr = out_ptr + b * stride_ob + p * stride_om

    # Copy H elements from src to out_row
    for col in range(0, H, BLOCK_N):
        offs = col + tl.arange(0, BLOCK_N)
        mask = offs < H
        vals = tl.load(src_ptr + offs * stride_eh, mask=mask, other=0.0)
        tl.store(out_row_ptr + offs * stride_on, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized forward:
        - Concatenate encoder_hidden_states and hidden_states along sequence dimension using a Triton kernel.
        - Apply linear projection using torch.matmul.
        - Split results back into separate encoder and image streams.
        """
        # Ensure CUDA tensors and matching shapes
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All tensors must be on CUDA device."
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        assert encoder_hidden_states.shape[0] == B and hidden_states.shape[2] == H and process_weight.shape == (H, H), "Dimension mismatch."

        # Make inputs contiguous
        e = encoder_hidden_states.contiguous()
        i = hidden_states.contiguous()
        w = process_weight.contiguous()

        # Allocate concatenated matrix [B, M, H], M = T + I
        M = T + I
        X_cat = torch.empty((B, M, H), dtype=e.dtype, device=e.device)

        # Launch cat_rows_kernel: grid = (B, M)
        grid_cat = (B, M)
        # Choose BLOCK_N as a reasonable vector width; since we loop over H, it's fine to use 128.
        cat_rows_kernel[grid_cat](
            e, i, X_cat,
            B, T, I, H,
            e.stride(0), e.stride(1), e.stride(2),
            i.stride(0), i.stride(1), i.stride(2),
            X_cat.stride(0), X_cat.stride(1), X_cat.stride(2),
            BLOCK_N=128,
        )

        # Apply linear projection using torch.matmul: processed = X_cat @ w
        processed = torch.matmul(X_cat, w)

        # Split into encoder and hidden streams
        processed_encoder = processed[:, :T, :]
        processed_hidden = processed[:, T:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
