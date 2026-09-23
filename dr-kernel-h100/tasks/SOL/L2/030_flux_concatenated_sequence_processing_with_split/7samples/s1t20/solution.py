import torch
import triton
import triton.language as tl


@triton.jit
def _concatenate_seqs_kernel(
    inA_ptr,  # encoder_hidden_states: [B, T, D]
    inB_ptr,  # hidden_states: [B, I, D]
    out_ptr,  # concatenated: [B, T+I, D]
    B: tl.constexpr, T: tl.constexpr, I: tl.constexpr, D: tl.constexpr,
    sA_b, sA_t, sA_d,  # strides for inA
    sB_b, sB_i, sB_d,  # strides for inB
    sO_b, sO_s, sO_d,  # strides for out
    BLOCK_D: tl.constexpr,
):
    # Grid: (B, T+I, ceil_div(D, BLOCK_D))
    pid_b = tl.program_id(0)
    pid_seq = tl.program_id(1)
    pid_tile = tl.program_id(2)

    # Determine whether this seq index belongs to encoder or hidden part
    is_encoder = pid_seq < T

    # Vector of D indices for this tile
    d_offsets = pid_tile * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = d_offsets < D

    if is_encoder:
        # Load from encoder_hidden_states[b, t, :]
        row_offset = pid_b * sA_b + pid_seq * sA_t
        x = tl.load(inA_ptr + row_offset + d_offsets * sA_d, mask=mask_d, other=0.0)
        # Store to out[b, pid_seq, :]
        out_row_offset = pid_b * sO_b + pid_seq * sO_s
        tl.store(out_ptr + out_row_offset + d_offsets * sO_d, x, mask=mask_d)
    else:
        # Load from hidden_states[b, i, :]
        i_idx = pid_seq - T
        row_offset = pid_b * sB_b + i_idx * sB_i
        x = tl.load(inB_ptr + row_offset + d_offsets * sB_d, mask=mask_d, other=0.0)
        # Store to out[b, pid_seq, :]
        out_row_offset = pid_b * sO_b + pid_seq * sO_s
        tl.store(out_ptr + out_row_offset + d_offsets * sO_d, x, mask=mask_d)


@triton.jit
def _row_matmul_no_bias_kernel(
    X_ptr,    # input [B, L, D], where L is T for encoder or I for hidden
    W_ptr,    # weight [D, D]
    Y_ptr,    # output [B, L, D]
    B: tl.constexpr, L: tl.constexpr, D: tl.constexpr,
    sX_b, sX_l, sX_d,  # strides for X
    sW0, sW1,          # strides for W
    sY_b, sY_l, sY_d,  # strides for Y
    BLOCK_N: tl.constexpr,  # tile over output columns
    BLOCK_K: tl.constexpr,  # tile over input feature dim K
):
    # Grid: (B, L, ceil_div(D, BLOCK_N))
    pid_b = tl.program_id(0)
    pid_l = tl.program_id(1)
    pid_n_tile = tl.program_id(2)

    n_offsets = pid_n_tile * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = n_offsets < D

    # Accumulator for this output row
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Loop over K (input feature dimension) in tiles
    for k0 in range(0, D, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < D

        # Load the input row vector x = X[b, l, k_offsets]
        x = tl.load(X_ptr + pid_b * sX_b + pid_l * sX_l + k_offsets * sX_d, mask=mask_k, other=0.0)
        x = x.to(tl.float32)  # ensure fp32 accumulation

        # Load a submatrix of W: W[k_offsets, n_offsets] -> shape [BLOCK_K, BLOCK_N]
        w = tl.load(
            W_ptr + k_offsets[:, None] * sW0 + n_offsets[None, :] * sW1,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        )
        w = w.to(tl.float32)

        # Accumulate: acc += sum_k x[k] * w[k, :]
        # For each kk in BLOCK_K, multiply and reduce along n
        for kk in range(BLOCK_K):
            # Guard: if k_offsets[kk] >= D, x[kk] is already 0 due to mask_k, but we still guard by mask_k
            if mask_k[kk]:
                acc += x[kk] * w[kk, :]

    # Store the accumulated result to Y[b, l, :]
    tl.store(Y_ptr + pid_b * sY_b + pid_l * sY_l + n_offsets * sY_d, acc, mask=mask_n)


def _triton_concatenate_and_split(B, T, I, D, encoder_hidden_states, hidden_states, process_weight):
    """
    Triton-only implementation:
    1) Concatenate encoder_hidden_states and hidden_states along sequence dimension -> [B, T+I, D] using Triton.
    2) Apply linear projection via Triton row-wise matmul without bias: [B, T+I, D] @ [D, D] -> [B, T+I, D].
    3) Split back into [B, T, D] and [B, I, D].
    Returns (processed_encoder, processed_hidden).
    """
    # Ensure contiguous tensors
    A = encoder_hidden_states.contiguous()
    Bhs = hidden_states.contiguous()
    W = process_weight.contiguous()

    # Allocate concatenated tensor
    total_seq = T + I
    concatenated = torch.empty((B, total_seq, D), dtype=torch.float32, device=A.device)

    # Launch concatenation kernel
    BLOCK_D = 128  # tile along D; 128 works well for typical D up to 4096
    grid_concat = (B, total_seq, triton.cdiv(D, BLOCK_D))
    _concatenate_seqs_kernel[grid_concat](
        A, Bhs, concatenated,
        B, T, I, D,
        A.stride(0), A.stride(1), A.stride(2),
        Bhs.stride(0), Bhs.stride(1), Bhs.stride(2),
        concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
        BLOCK_D=BLOCK_D,
        num_warps=4, num_stages=2,
    )

    # Apply linear projection via Triton row-wise matmul: concatenated @ W -> concatenated
    # Output Y has same shape as concatenated
    Y = torch.empty_like(concatenated)
    grid_matmul = (B, total_seq, triton.cdiv(D, 64))  # 64 tiles along N dimension
    _row_matmul_no_bias_kernel[grid_matmul](
        concatenated, W, Y,
        B, total_seq, D,
        concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
        W.stride(0), W.stride(1),
        Y.stride(0), Y.stride(1), Y.stride(2),
        BLOCK_N=64, BLOCK_K=32,  # N and K tiles; K=32 ensures all columns covered when D is multiple of 32
        num_warps=4, num_stages=2,
    )

    # Split back: take first T rows for encoder, last I rows for hidden
    processed_encoder = Y[:, :T, :]
    processed_hidden = Y[:, T:, :]
    return processed_encoder, processed_hidden


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Ensure inputs are on CUDA for Triton; if not, move them (evaluation assumes CUDA)
        if not hidden_states.is_cuda:
            hidden_states = hidden_states.cuda()
        if not encoder_hidden_states.is_cuda:
            encoder_hidden_states = encoder_hidden_states.cuda()
        if not process_weight.is_cuda:
            process_weight = process_weight.cuda()

        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        D = hidden_states.shape[2]

        # Run Triton-only computation
        processed_encoder, processed_hidden = _triton_concatenate_and_split(B, T, I, D, encoder_hidden_states, hidden_states, process_weight)
        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
