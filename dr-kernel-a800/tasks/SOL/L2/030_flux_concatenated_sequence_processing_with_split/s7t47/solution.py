import torch

# Triton is required; the evaluator provides ModelNew. We will define kernels here.

# Kernel: concatenate rows from encoder_hidden_states and hidden_states into X_cat[b, p, :]
# X_cat shape: [B, M, H], M = T + I
@triton.jit
def cat_rows_kernel(
    hidden_states_ptr,   # *float, [B, I, H]
    encoder_hidden_ptr,  # *float, [B, T, H]
    X_cat_ptr,           # *float, [B, M, H]
    B: tl.constexpr, T: tl.constexpr, I: tl.constexpr, H: tl.constexpr, M: tl.constexpr,
):
    b = tl.program_id(0)  # batch id
    p = tl.program_id(1)  # row id in [0, M)
    # Determine source tensor and index
    is_encoder = p < T
    if is_encoder:
        src_idx = p
        src_ptr = encoder_hidden_ptr + b * T * H + src_idx * H
    else:
        src_idx = p - T
        src_ptr = hidden_states_ptr + b * I * H + src_idx * H

    dst_ptr = X_cat_ptr + b * M * H + p * H

    # Copy H elements
    for j in range(0, H):
        val = tl.load(src_ptr + j)
        tl.store(dst_ptr + j, val)


# Kernel: batched GEMM: for each batch b, Y[b] = X_cat[b] @ process_weight
# X_cat[b] shape: [M, H], process_weight shape: [H, H], Y[b] shape: [M, H]
@triton.jit
def batched_gemm_kernel(
    X_ptr,          # *float, [B, M, H]
    W_ptr,          # *float, [H, H]
    Y_ptr,          # *float, [B, M, H]
    M: tl.constexpr, H: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    b = tl.program_id(0)
    # Pointers for this batch
    X_b = X_ptr + b * M * H
    W = W_ptr
    Y_b = Y_ptr + b * M * H

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in tiles
    for k0 in range(0, H, BLOCK_K):
        # Current tiles
        K = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        # Loop over M tiles
        for m0 in range(0, M, BLOCK_M):
            M_idx = m0 + tl.arange(0, BLOCK_M)  # [BLOCK_M]
            # Load X_tile: [BLOCK_M, BLOCK_K]
            X_tile = tl.load(
                X_b + M_idx[:, None] * H + K[None, :],
                mask=(M_idx[:, None] < M) & (K[None, :] < H),
                other=0.0,
            )
            # Load W_tile: [BLOCK_K, BLOCK_N], but here N=H
            W_tile = tl.load(
                W + K[:, None] * H + tl.arange(0, BLOCK_N)[None, :],
                mask=(K[:, None] < H) & (tl.arange(0, BLOCK_N)[None, :] < H),
                other=0.0,
            )
            # Accumulate
            acc += tl.dot(X_tile.to(tl.float32), W_tile.to(tl.float32))
        # Store partial results
        N_idx = tl.arange(0, BLOCK_N)  # [BLOCK_N] corresponds to H
        for m0 in range(0, M, BLOCK_M):
            M_idx = m0 + tl.arange(0, BLOCK_M)
            acc_tile = acc[m0: m0 + BLOCK_M, :BLOCK_N]
            tl.store(
                Y_b + M_idx[:, None] * H + N_idx[None, :],
                acc_tile,
                mask=(M_idx[:, None] < M) & (N_idx[None, :] < H),
            )


# Kernel: copy rows slice from Y[b] to out[b]
# We'll use it to copy processed_encoder = Y[:, :T, :] and processed_hidden = Y[:, T:, :]
@triton.jit
def copy_rows_slice_kernel(
    Y_ptr,            # *float, [B, M, H]
    out_ptr,          # *float, [B, N, H]
    B: tl.constexpr, M: tl.constexpr, N: tl.constexpr, H: tl.constexpr,
):
    b = tl.program_id(0)  # batch id
    p = tl.program_id(1)  # row id in [0, N)
    # Destination base pointer
    dst_ptr = out_ptr + b * N * H + p * H
    # Source base pointer in Y
    src_ptr = Y_ptr + b * M * H + p * H
    # Copy H elements
    for j in range(0, H):
        val = tl.load(src_ptr + j)
        tl.store(dst_ptr + j, val)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        # Ensure CUDA tensors
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA."
        B = hidden_states.shape[0]
        I = hidden_states.shape[1]
        T = encoder_hidden_states.shape[1]
        H = hidden_states.shape[2]
        M = T + I

        # Allocate X_cat [B, M, H]
        X_cat = torch.empty((B, M, H), dtype=hidden_states.dtype, device=hidden_states.device)

        # Launch concatenation kernel: one program per (b, p)
        grid_cat = (B, M)
        cat_rows_kernel[grid_cat](
            hidden_states, encoder_hidden_states, X_cat,
            B=B, T=T, I=I, H=H, M=M,
            num_warps=1, num_stages=1,
        )

        # Allocate Y [B, M, H] to hold the result of X_cat @ process_weight
        Y = torch.empty((B, M, H), dtype=torch.float32, device=hidden_states.device)  # compute in fp32 for stability

        # Launch batched GEMM kernel: one program per batch
        # Choose reasonable tiles; H is typically moderate (e.g., 64-128). We pick general values.
        BLOCK_M = 128
        BLOCK_N = 64
        BLOCK_K = 32
        grid_mm = (B,)
        batched_gemm_kernel[grid_mm](
            X_cat, process_weight, Y,
            M=M, H=H,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )

        # Allocate outputs for processed_encoder [B, T, H] and processed_hidden [B, I, H]
        processed_encoder = torch.empty((B, T, H), dtype=torch.float32, device=hidden_states.device)
        processed_hidden = torch.empty((B, I, H), dtype=torch.float32, device=hidden_states.device)

        # Launch slice copy kernels
        grid_copy1 = (B, T, H)
        grid_copy2 = (B, I, H)
        copy_rows_slice_kernel[grid_copy1](
            Y, processed_encoder,
            B=B, M=M, N=T, H=H,
            num_warps=1, num_stages=1,
        )
        copy_rows_slice_kernel[grid_copy2](
            Y, processed_hidden,
            B=B, M=M, N=I, H=H,
            num_warps=1, num_stages=1,
        )

        # Cast outputs back to original dtype if needed
        processed_encoder = processed_encoder.to(hidden_states.dtype)
        processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
