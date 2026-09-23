import torch
import triton
import triton.language as tl


@triton.jit
def cat_rows_kernel(
    e_ptr,  # pointer to encoder_hidden_states [B, T, H]
    i_ptr,  # pointer to hidden_states [B, I, H]
    out_ptr,  # pointer to X_cat [B, M, H]
    B, T, I, H, M,
):
    # 2D grid: (B, M)
    b = tl.program_id(0)
    p = tl.program_id(1)

    # Compute base offsets assuming contiguous layout:
    # For a 3D tensor [B, L, H] contiguous, stride on L is H, on H is 1.
    # So row offset for index (b, p, :) is b * (L*H) + p * H.
    if p < T:
        src_off = b * (T * H) + p * H
    else:
        src_off = b * (I * H) + (p - T) * H

    out_off = b * (M * H) + p * H

    # Copy H elements
    for j in range(0, H):
        val = tl.load(e_ptr + src_off + j) if p < T else tl.load(i_ptr + src_off + j)
        tl.store(out_ptr + out_off + j, val)


@triton.jit
def batched_gemm_kernel(
    X_ptr,  # [M, H]
    W_ptr,  # [H, H]
    Y_ptr,  # [M, H]
    M, H,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # One program per batch performing GEMM: Y = X @ W
    # We tile over M and N (here N=H) and loop over K=H in BLOCK_K.
    for m in range(0, M, BLOCK_M):
        m_offsets = m + tl.arange(0, BLOCK_M)
        for n in range(0, H, BLOCK_N):
            n_offsets = n + tl.arange(0, BLOCK_N)
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k in range(0, H, BLOCK_K):
                k_offsets = k + tl.arange(0, BLOCK_K)
                # Load X tile [BLOCK_M, BLOCK_K] from X_ptr
                x_ptrs = X_ptr + m_offsets[:, None] * H + k_offsets[None, :]
                # Load W tile [BLOCK_K, BLOCK_N] from W_ptr
                w_ptrs = W_ptr + k_offsets[:, None] * H + n_offsets[None, :]
                mask_x = (m_offsets[:, None] < M) & (k_offsets[None, :] < H)
                mask_w = (k_offsets[:, None] < H) & (n_offsets[None, :] < H)
                x = tl.load(x_ptrs, mask=mask_x, other=0.0).to(tl.float32)
                w = tl.load(w_ptrs, mask=mask_w, other=0.0).to(tl.float32)
                acc += tl.dot(x, w)
            # Store acc into Y
            y_ptrs = Y_ptr + m_offsets[:, None] * H + n_offsets[None, :]
            mask_y = (m_offsets[:, None] < M) & (n_offsets[None, :] < H)
            tl.store(y_ptrs, acc, mask=mask_y)


@triton.jit
def slice_copy_rows_kernel(
    Y_ptr,  # [B, M, H]
    out_ptr,  # target [B, N, H] (N=T or N=I)
    B, M, N, H,
    src_start: tl.constexpr,
):
    # Copy rows [src_start:src_start+N] from Y to out
    # Grid: (B, N, H)
    b = tl.program_id(0)
    r = tl.program_id(1)
    j = tl.program_id(2)
    src_row = src_start + r
    y_offset = b * (M * H) + src_row * H + j
    out_offset = b * (N * H) + r * H + j
    val = tl.load(Y_ptr + y_offset)
    tl.store(out_ptr + out_offset, val)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only implementation:
        - Concatenate along sequence dim in Triton.
        - Perform batched linear projection in Triton GEMM.
        - Slice the result into encoder and hidden streams via Triton copy.
        Returns (processed_encoder, processed_hidden).
        """
        # Ensure inputs are on CUDA and contiguous
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA"
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        process_weight = process_weight.contiguous()

        B = hidden_states.shape[0]
        I = hidden_states.shape[1]
        T = encoder_hidden_states.shape[1]
        H = hidden_states.shape[2]
        M = T + I

        # Allocate X_cat [B, M, H] contiguous
        X_cat = torch.empty((B, M, H), device=hidden_states.device, dtype=hidden_states.dtype)

        # Launch cat_rows_kernel: grid (B, M)
        grid_cat = (B, M)
        cat_rows_kernel[grid_cat](
            encoder_hidden_states, hidden_states, X_cat,
            B, T, I, H, M,
            num_warps=1, num_stages=1,
        )

        # Allocate Y [B, M, H]
        Y = torch.empty((B, M, H), device=hidden_states.device, dtype=hidden_states.dtype)

        # Launch batched GEMM kernel: one program per batch
        grid_mm = (B,)
        # Choose tile sizes suitable for typical H in provided workloads
        BLOCK_M = 128
        BLOCK_N = 64
        BLOCK_K = 32
        batched_gemm_kernel[grid_mm](
            X_cat, process_weight, Y,
            M, H,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )

        # Allocate outputs
        processed_encoder = torch.empty((B, T, H), device=hidden_states.device, dtype=hidden_states.dtype)
        processed_hidden = torch.empty((B, I, H), device=hidden_states.device, dtype=hidden_states.dtype)

        # Copy slices via Triton kernel
        grid_copy1 = (B, T, H)
        grid_copy2 = (B, I, H)

        # Copy first T rows from Y to processed_encoder
        slice_copy_rows_kernel[grid_copy1](
            Y, processed_encoder,
            B, M, T, H,
            src_start=0,
            num_warps=1, num_stages=1,
        )
        # Copy next I rows from Y to processed_hidden
        slice_copy_rows_kernel[grid_copy2](
            Y, processed_hidden,
            B, M, I, H,
            src_start=T,
            num_warps=1, num_stages=1,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
