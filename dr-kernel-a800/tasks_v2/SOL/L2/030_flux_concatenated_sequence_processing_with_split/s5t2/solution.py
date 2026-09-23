import torch
import triton
import triton.language as tl


@triton.jit
def _batched_gemm_kernel(
    A_ptr,          # *f32, concatenated [B, T, H], contiguous
    B_ptr,          # *f32, process_weight_T [H, H], contiguous
    Out_ptr,        # *f32, output [B, T, H], contiguous
    B_dim, T_dim, H_dim,
    stride_A_b, stride_A_t, stride_A_h,
    stride_B_k, stride_B_h,  # B is [H, H]
    stride_Out_b, stride_Out_t, stride_Out_h,
    BLOCK_T: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 3D grid: (B, ceil_div(T, BLOCK_T), ceil_div(H, BLOCK_H))
    b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_h = tl.program_id(2)

    t_offsets = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)

    mask_t = t_offsets < T_dim
    mask_h = h_offsets < H_dim

    # Accumulator for this tile
    acc = tl.zeros((BLOCK_T, BLOCK_H), dtype=tl.float32)

    # Loop over K (hidden dimension)
    for k_start in range(0, H_dim, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < H_dim

        # Load A tile: [BLOCK_T, BLOCK_K] from A[b, t_offsets, k_offsets]
        a_ptrs = A_ptr + b * stride_A_b + t_offsets[:, None] * stride_A_t + k_offsets[None, :] * stride_A_h
        a_mask = mask_t[:, None] & mask_k[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B tile: [BLOCK_K, BLOCK_H] from B[k_offsets, h_offsets]
        b_ptrs = B_ptr + k_offsets[:, None] * stride_B_k + h_offsets[None, :] * stride_B_h
        b_mask = mask_k[:, None] & mask_h[None, :]
        b_tile = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a, b_tile)

    # Store result to Out[b, t_offsets, h_offsets]
    out_ptrs = Out_ptr + b * stride_Out_b + t_offsets[:, None] * stride_Out_t + h_offsets[None, :] * stride_Out_h
    out_mask = mask_t[:, None] & mask_h[None, :]
    tl.store(out_ptrs, acc, mask=out_mask)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized forward:
        - Concatenates encoder_hidden_states and hidden_states along sequence dimension (PyTorch data movement).
        - Applies linear projection process_weight.T to the concatenated sequence (Triton GEMM).
        - Splits the result back into processed_encoder and processed_hidden.
        """
        # Ensure inputs are on the same device and contiguous
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be CUDA tensors"
        B = encoder_hidden_states.shape[0]
        Stext = encoder_hidden_states.shape[1]
        H = encoder_hidden_states.shape[2]
        Bhs = hidden_states.shape[0]
        assert Bhs == B, "batch sizes must match"
        assert hidden_states.shape[2] == H, "hidden_dim must match"
        assert process_weight.shape[0] == H and process_weight.shape[1] == H, "process_weight must be [H, H]"

        # Concatenate along sequence dimension (PyTorch)
        T = Stext + hidden_states.shape[1]
        concatenated = torch.cat([encoder_hidden_states, hidden_states], dim=1).contiguous()

        # Prepare process_weight_T (PyTorch transpose, contiguous)
        process_weight_T = process_weight.transpose(0, 1).contiguous()
        if process_weight_T.dtype != concatenated.dtype:
            process_weight_T = process_weight_T.to(concatenated.dtype)

        # Allocate output [B, T, H]
        out = torch.empty((B, T, H), device=concatenated.device, dtype=concatenated.dtype)

        # Tiling parameters (conservative defaults)
        BLOCK_T = 64
        BLOCK_H = 64
        BLOCK_K = 32

        # Launch Triton GEMM kernel
        grid = (B, triton.cdiv(T, BLOCK_T), triton.cdiv(H, BLOCK_H))
        _batched_gemm_kernel[grid](
            concatenated, process_weight_T, out,
            B, T, H,
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            process_weight_T.stride(0), process_weight_T.stride(1),
            out.stride(0), out.stride(1), out.stride(2),
            BLOCK_T=BLOCK_T, BLOCK_H=BLOCK_H, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # Split back into two streams via slicing (views)
        processed_encoder = out[:, :Stext, :]
        processed_hidden = out[:, Stext:, :]
        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
