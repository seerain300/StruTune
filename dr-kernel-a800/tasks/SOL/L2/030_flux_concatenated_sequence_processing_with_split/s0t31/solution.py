import torch
import triton
import triton.language as tl


@triton.jit
def batched_matmul_kernel(
    A_ptr,        # *A [B, M, K] = concatenated [B, L_txt + L_img, D]
    Wt_ptr,       # *W^T [K, N] = process_weight.T [D, D]
    C_ptr,        # *C [B, M, N] = output processed [B, L_txt + L_img, D]
    B: tl.int32,
    M: tl.int32,  # L_txt + L_img
    K: tl.int32,  # hidden_dim
    N: tl.int32,  # hidden_dim (== D)
    A_stride_b: tl.int32, A_stride_m: tl.int32, A_stride_k: tl.int32,
    Wt_stride_k: tl.int32, Wt_stride_n: tl.int32,
    C_stride_b: tl.int32, C_stride_m: tl.int32, C_stride_n: tl.int32,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 3D launch: (batch, tiles along M, tiles along N)
    b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = m_offsets < M
    n_mask = n_offsets < N

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in blocks
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # Load A tile: [BLOCK_M, BLOCK_K]
        A_tile_ptrs = A_ptr + b * A_stride_b + m_offsets[:, None] * A_stride_m + k_offsets[None, :] * A_stride_k
        A_tile = tl.load(A_tile_ptrs, mask=m_mask[:, None] & k_mask[None, :], other=0.0).to(tl.float32)

        # Load Wt tile: [BLOCK_K, BLOCK_N]
        Wt_tile_ptrs = Wt_ptr + k_offsets[:, None] * Wt_stride_k + n_offsets[None, :] * Wt_stride_n
        Wt_tile = tl.load(Wt_tile_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float32)

        # Accumulate: acc += A_tile @ Wt_tile
        acc += tl.dot(A_tile, Wt_tile)

    # Store result to C
    C_tile_ptrs = C_ptr + b * C_stride_b + m_offsets[:, None] * C_stride_m + n_offsets[None, :] * C_stride_n
    tl.store(C_tile_ptrs, acc, mask=m_mask[:, None] & n_mask[None, :])


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        # Validate dimensions
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3, "Inputs must be 3D [B, S, D]"
        assert process_weight.dim() == 2, "process_weight must be 2D [D, D]"
        assert hidden_states.shape[2] == encoder_hidden_states.shape[2] == process_weight.shape[0] == process_weight.shape[1], "Hidden dim mismatch"
        assert hidden_states.shape[0] == encoder_hidden_states.shape[0], "Batch size mismatch"

        B = hidden_states.shape[0]
        D = hidden_states.shape[2]
        L_txt = encoder_hidden_states.shape[1]
        L_img = hidden_states.shape[1]
        M = L_txt + L_img
        K = D
        N = D

        device = hidden_states.device
        dtype = hidden_states.dtype
        assert encoder_hidden_states.device == device and process_weight.device == device, "All tensors must be on the same device"
        assert encoder_hidden_states.dtype == dtype and process_weight.dtype == dtype, "All tensors must have the same dtype"

        # Ensure inputs are contiguous or at least use strides correctly
        # We do not force .contiguous() to preserve non-contiguous behavior, but Triton will use strides as provided.
        A = torch.empty((B, M, K), device=device, dtype=dtype)  # will be filled by concatenation

        # Concatenate A along sequence dimension using torch for simplicity and correctness
        # A[:, :L_txt, :] = encoder_hidden_states
        # A[:, L_txt:, :] = hidden_states
        A[:, :L_txt, :] = encoder_hidden_states
        A[:, L_txt:, :] = hidden_states

        # Prepare W^T = process_weight.T contiguous [K, N]
        Wt = process_weight.transpose(0, 1).contiguous()  # [D, D]

        # Output tensor C [B, M, N]
        C = torch.empty((B, M, N), device=device, dtype=torch.float32)  # compute in fp32
        # Strides
        A_stride_b, A_stride_m, A_stride_k = A.stride(0), A.stride(1), A.stride(2)
        Wt_stride_k, Wt_stride_n = Wt.stride(0), Wt.stride(1)
        C_stride_b, C_stride_m, C_stride_n = C.stride(0), C.stride(1), C.stride(2)

        # Launch Triton GEMM kernel
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
        grid = (B, triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        batched_matmul_kernel[grid](
            A, Wt, C,
            B, M, K, N,
            A_stride_b, A_stride_m, A_stride_k,
            Wt_stride_k, Wt_stride_n,
            C_stride_b, C_stride_m, C_stride_n,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # Cast back to original dtype for outputs
        processed = C.to(dtype)

        # Split into encoder and hidden parts along sequence dimension
        processed_encoder = processed[:, :L_txt, :]
        processed_hidden = processed[:, L_txt:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
