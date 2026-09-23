import torch
import triton
import triton.language as tl


@triton.jit
def _concatenate_sequences_kernel(
    encoder_ptr, hidden_ptr, out_ptr,
    B, T, I, H,
):
    # One program per batch
    b = tl.program_id(0)
    # Pointer to the first element of this batch in each tensor
    encoder_batch_ptr = encoder_ptr + b * T * H
    hidden_batch_ptr = hidden_ptr + b * I * H
    out_batch_ptr = out_ptr + b * (T + I) * H

    # Loop over concatenated sequence length
    for l in range(0, T + I):
        # If l < T: load from encoder, else: load from hidden
        src_ptr = encoder_batch_ptr + l * H if l < T else hidden_batch_ptr + (l - T) * H
        row_ptr = out_batch_ptr + l * H
        # Load one row of length H; mask not needed since loop bounds are exact
        x = tl.load(src_ptr)
        tl.store(row_ptr, x)


@triton.jit
def _batched_matmul_per_row_kernel(
    A_ptr, W_ptr, C_ptr,
    B, T, I, H,  # runtime sizes
    M,  # M = B*(T + I)
):
    # One program per output row m
    m = tl.program_id(0)
    # A is [M, H], W is [H, H], C is [M, H]
    # Compute row m of C = A[m, :] @ W
    acc = tl.zeros((H,), dtype=tl.float32)
    # Iterate over K dimension in tiles of BLOCK_K
    BLOCK_K = 128
    for k0 in range(0, H, BLOCK_K):
        k_idx = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_idx < H
        # Load A[m, k:k+BLOCK_K]
        a_row_ptrs = A_ptr + m * H + k_idx
        a = tl.load(a_row_ptrs, mask=mask_k, other=0.0)
        # Load W[k:k+BLOCK_K, :]
        w_ptrs = W_ptr + k_idx[:, None] * H + tl.arange(0, H)[None, :]  # [BLOCK_K, H]
        w = tl.load(w_ptrs, mask=(mask_k[:, None]), other=0.0)
        # Accumulate dot: sum over K tile
        acc += tl.sum(a[:, None] * w, axis=0)
    # Store the result row to C[m, :]
    C_row_ptrs = C_ptr + m * H + tl.arange(0, H)
    tl.store(C_row_ptrs, acc)


@triton.jit
def _split_into_encoder_hidden_kernel(
    C_ptr, encoder_ptr, hidden_ptr,
    B, T, I, H,
):
    # One program per batch
    b = tl.program_id(0)
    M = B * (T + I)

    # Copy first T rows to encoder
    for l in range(0, T):
        m = l
        row_ptrs = C_ptr + m * H + tl.arange(0, H)
        vals = tl.load(row_ptrs)
        tl.store(encoder_ptr + b * T * H + l * H, vals)

    # Copy remaining I rows to hidden, shifted by T
    for l in range(0, I):
        m = T + l
        row_ptrs = C_ptr + m * H + tl.arange(0, H)
        vals = tl.load(row_ptrs)
        tl.store(hidden_ptr + b * I * H + l * H, vals)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only implementation:
        - Concatenate sequences along sequence dimension using Triton.
        - Perform batched GEMM using a Triton kernel (one program per row).
        - Split outputs into encoder and hidden streams using Triton.
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA device for Triton kernels."

        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        assert encoder_hidden_states.shape[2] == H and process_weight.shape[0] == H and process_weight.shape[1] == H, "Dimension mismatch."

        # 1) Concatenate sequences: [B, T+I, H] using Triton
        out_cat = torch.empty((B, T + I, H), dtype=torch.float32, device=hidden_states.device)
        _concatenate_sequences_kernel[(B,)](
            encoder_hidden_states, hidden_states, out_cat,
            B, T, I, H,
            num_warps=1, num_stages=1
        )

        # 2) GEMM: C = out_cat @ process_weight.T using Triton
        # out_cat is [M, H], process_weight is [H, H], we want C[M, H]
        M = B * (T + I)
        A = out_cat
        W = process_weight  # shape [H, H]
        C = torch.empty((M, H), dtype=torch.float32, device=hidden_states.device)
        _batched_matmul_per_row_kernel[(M,)](
            A, W, C,
            B, T, I, H, M,
            num_warps=1, num_stages=1
        )

        # 3) Split into [B, T, H] and [B, I, H] using Triton
        processed_encoder = torch.empty((B, T, H), dtype=torch.float32, device=hidden_states.device)
        processed_hidden = torch.empty((B, I, H), dtype=torch.float32, device=hidden_states.device)
        _split_into_encoder_hidden_kernel[(B,)](
            C, processed_encoder, processed_hidden,
            B, T, I, H,
            num_warps=1, num_stages=1
        )

        return processed_encoder, processed_hidden