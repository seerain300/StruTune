import torch
import triton
import triton.language as tl


@triton.jit
def batched_matmul_concat_kernel(
    X_ptr,          # *fp32, pointer to input X: [M_total, H], contiguous
    W_ptr,          # *fp32, pointer to weight W: [H, H], contiguous
    Y_ptr,          # *fp32, pointer to output Y: [M_total, H], contiguous
    M_total: tl.int32,  # total number of rows in X across all batches
    H: tl.int32,        # number of columns in X and rows in W
    stride_xm: tl.int32,  # stride of X in rows (usually H)
    stride_yn: tl.int32,  # stride of Y in columns (usually 1)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 2D launch grid over output tiles
    pid_m = tl.program_id(0)  # tile index over M_total
    pid_n = tl.program_id(1)  # tile index over H

    # Compute row/col offsets for this program
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension (hidden_dim H)
    for k0 in range(0, H, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)

        # Pointers for A (X) block: shape [BLOCK_M, BLOCK_K]
        a_ptrs = X_ptr + m_offsets[:, None] * stride_xm + k_offsets[None, :]
        # Pointers for B (W) block: shape [BLOCK_K, BLOCK_N]
        # W is [H, H], so B[k, n] is W[k, n]
        b_ptrs = W_ptr + k_offsets[:, None] * H + n_offsets[None, :]

        # Masks for edges
        a_mask = (m_offsets[:, None] < M_total) & (k_offsets[None, :] < H)
        b_mask = (k_offsets[:, None] < H) & (n_offsets[None, :] < H)

        # Load tiles
        A = tl.load(a_ptrs, mask=a_mask, other=0.0)
        B = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Accumulate
        acc += tl.dot(A, B)

    # Store results to Y: [M_total, H]
    y_ptrs = Y_ptr + m_offsets[:, None] * stride_xm + n_offsets[None, :]
    y_mask = (m_offsets[:, None] < M_total) & (n_offsets[None, :] < H)
    tl.store(y_ptrs, acc, mask=y_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized version of the original run function.
        - Concatenates encoder_hidden_states and hidden_states along sequence dimension.
        - Applies a dense linear projection via Triton GEMM: (cat([E, I])) @ process_weight.T
        - Splits the result back into processed_encoder and processed_hidden.
        """
        # Ensure inputs are on CUDA for Triton
        if not (hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda):
            # Move to default CUDA device if available
            device = torch.device('cuda', 0) if torch.cuda.is_available() else torch.device('cpu')
            hidden_states = hidden_states.to(device)
            encoder_hidden_states = encoder_hidden_states.to(device)
            process_weight = process_weight.to(device)

        # Compute shapes
        batch_size = hidden_states.shape[0]
        text_seq_len = encoder_hidden_states.shape[1]
        img_seq_len = hidden_states.shape[1]
        hidden_dim = hidden_states.shape[2]
        assert encoder_hidden_states.shape[2] == hidden_dim, "Hidden dims must match."
        assert process_weight.shape == (hidden_dim, hidden_dim), "process_weight must be [hidden_dim, hidden_dim]."

        # Per-batch concatenation into a single large X_total
        # We will store per-batch rows sequentially into a single [B * (T+I), H] matrix X_total
        total_rows = batch_size * (text_seq_len + img_seq_len)

        # Make sure inputs are contiguous
        E = encoder_hidden_states.contiguous()  # [B, T, H]
        I = hidden_states.contiguous()          # [B, I, H]

        # Prepare X_total, Y_total as contiguous [total_rows, H], [total_rows, H]
        X_total = torch.empty((total_rows, hidden_dim), dtype=E.dtype, device=device)
        Y_total = torch.empty((total_rows, hidden_dim), dtype=E.dtype, device=device)

        # Helper to fill X_total for a given batch b
        def fill_x_total(b, E_b, I_b, start_row):
            # Concatenate along sequence dimension: [T+I, H]
            X_b = torch.cat([E_b, I_b], dim=1)  # [T+I, H]
            # Write into X_total starting at row 'start_row'
            # Note: PyTorch supports advanced indexing with 1D vector for rows
            X_total[start_row:start_row + X_b.shape[0], :] = X_b  # [rows, H] assignment

        # Fill X_total for all batches
        start_row = 0
        for b in range(batch_size):
            fill_x_total(b, E[b], I[b], start_row)
            start_row += (text_seq_len + img_seq_len)

        # Ensure W is contiguous [H, H]
        W = process_weight.contiguous()  # [H, H]

        # Launch Triton kernel on X_total and W, produce Y_total
        # Grid: (ceil_div(total_rows, BLOCK_M), ceil_div(H, BLOCK_N))
        BLOCK_M = 128
        BLOCK_N = 64
        BLOCK_K = 32

        grid = (triton.cdiv(total_rows, BLOCK_M), triton.cdiv(hidden_dim, BLOCK_N))
        batched_matmul_concat_kernel[grid](
            X_total, W, Y_total,
            total_rows, hidden_dim,
            X_total.stride(0),  # stride_xm = H (since X_total[row, :] advances by H)
            Y_total.stride(1),  # stride_yn = 1 (column stride)
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # Now slice Y_total back per batch to get per-batch results
        processed_encoder_list = []
        processed_hidden_list = []
        for b in range(batch_size):
            m_local = text_seq_len + img_seq_len
            start_row = b * m_local
            end_row = start_row + m_local
            Y_b = Y_total[start_row:end_row, :]  # [m_local, H]
            # Split back: first T rows are encoder, next I rows are image
            processed_encoder = Y_b[:text_seq_len, :]  # [T, H]
            processed_hidden = Y_b[text_seq_len:, :]   # [I, H]
            processed_encoder_list.append(processed_encoder)
            processed_hidden_list.append(processed_hidden)

        # Convert list to tensors
        processed_encoder = torch.stack(processed_encoder_list, dim=0)  # [B, T, H]
        processed_hidden = torch.stack(processed_hidden_list, dim=0)    # [B, I, H]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
