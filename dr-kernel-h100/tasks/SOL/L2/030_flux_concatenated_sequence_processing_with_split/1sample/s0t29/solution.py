import torch
import triton
import triton.language as tl


@triton.jit
def _batch_right_gemm_kernel(
    A_ptr,  # *f32, [M, H], M = B*(T+I)
    W_ptr,  # *f32, [H, H] (right-multiply by W^T)
    C_ptr,  # *f32, [M, H]
    M,      # int: number of rows in A (batch * (T+I))
    H,      # int: hidden_dim
    A_stride_m, A_stride_h,   # strides for A: A.stride(0), A.stride(1)
    W_stride_k, W_stride_h,   # strides for W: W.stride(0), W.stride(1)
    C_stride_m, C_stride_h,   # strides for C: C.stride(0), C.stride(1)
    BLOCK_M: tl.constexpr,    # number of rows to process per program (here 1)
    BLOCK_N: tl.constexpr,    # columns tile
    BLOCK_K: tl.constexpr,    # reduction tile
):
    # Each program handles one output row 'm' and a tile of columns [cols : cols+BLOCK_N]
    m = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    # Initialize accumulator
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Loop over K dimension in chunks
    for k0 in range(0, H, BLOCK_K):
        ks = k0 + tl.arange(0, BLOCK_K)  # reduction indices
        # Load A row m for ks, shape [BLOCK_K]
        a_row = tl.load(
            A_ptr + m * A_stride_m + ks * A_stride_h,
            mask=ks < H,
            other=0.0
        )
        # Load W sub-block [ks, cols], shape [BLOCK_K, BLOCK_N]
        w_block = tl.load(
            W_ptr + ks[:, None] * W_stride_k + cols[None, :] * W_stride_h,
            mask=(ks[:, None] < H) & (cols[None, :] < H),
            other=0.0
        )
        # Accumulate: acc[cols] += sum_k a_row[k] * w_block[k, :]
        acc += tl.sum(a_row[:, None] * w_block, axis=0)

    # Store results
    tl.store(C_ptr + m * C_stride_m + cols * C_stride_h, acc, mask=cols < H)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Validate device/dtype
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All inputs must be on CUDA device"
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3, "Inputs must be 3D tensors [B, S, H]"
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        assert process_weight.shape == (H, H), "process_weight must have shape [H, H]"
        # Ensure contiguity for predictable strides
        enc = encoder_hidden_states.contiguous()
        hid = hidden_states.contiguous()
        W = process_weight.contiguous()

        # Step 1: Concatenate along sequence dimension (simple and robust)
        concatenated = torch.cat([enc, hid], dim=1)  # shape [B, T+I, H]
        M = concatenated.shape[1] * concatenated.shape[0]  # B*(T+I)
        A = concatenated.reshape(M, H).contiguous()  # [M, H]

        # Allocate C for output: [M, H]
        C = torch.empty((M, H), dtype=torch.float32, device=hidden_states.device)

        # Launch Triton kernel: grid over (M rows, tiles of H columns)
        grid = (M, triton.cdiv(H, 128))
        _batch_right_gemm_kernel[grid](
            A, W, C,
            M, H,
            A.stride(0), A.stride(1),
            W.stride(0), W.stride(1),
            C.stride(0), C.stride(1),
            BLOCK_M=1, BLOCK_N=128, BLOCK_K=64,
            num_warps=4, num_stages=2
        )

        # Step 2: Split C back into two streams using torch (reliable and simple)
        # C has shape [M, H] and M = B*(T+I)
        # Split along dim=0 (rows) into first B*T rows and next B*I rows
        processed_encoder = C[:B * T, :].reshape(B, T, H)
        processed_hidden = C[B * T:, :].reshape(B, I, H)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
