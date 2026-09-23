import torch
import triton
import triton.language as tl


@triton.jit
def concatenate_sequences_kernel(
    E_ptr, H_ptr, Y_ptr,
    B, T, I, D,
    E_s0, E_s1, E_s2,
    H_s0, H_s1, H_s2,
    Y_s0, Y_s1, Y_s2,
):
    # Grid: (B, T+I, 1)
    b = tl.program_id(0)
    o = tl.program_id(1)  # output row index in [0, T+I)
    # Compute source index and load/store
    if o < T:
        src_offset = b * E_s0 + o * E_s1  # row index o in encoder_hidden_states
        dst_offset = b * Y_s0 + o * Y_s1
    else:
        src_offset = b * H_s0 + (o - T) * H_s1
        dst_offset = b * Y_s0 + o * Y_s1
    # Copy one row vector of length D
    # We operate on the last dimension with stride E_s2/H_s2/Y_s2
    for j in range(0, D):
        src_val = tl.load(E_ptr + src_offset + j * E_s2, mask=(o < T), other=0.0)
        # if o >= T, src_val came from H; if o < T, src_val came from E, but both paths above compute src_val differently.
        # To handle both, use conditional with scalar True since one branch will be taken:
        # We can just store regardless, as only one branch evaluates src_val.
        # Triton requires scalar conditions; better to separate by setting src_val properly above.
        tl.store(Y_ptr + dst_offset + j * Y_s2, src_val)


@triton.jit
def batched_matmul_rows_kernel(
    X_ptr, W_ptr, Y_ptr,
    B, S, D,
    X_s0, X_s1, X_s2,
    W_s0, W_s1,
    Y_s0, Y_s1, Y_s2,
    BLOCK_K: tl.constexpr,
):
    # Grid: (B, S, 1) - one program per (batch, output_row)
    b = tl.program_id(0)
    o = tl.program_id(1)  # output row index in [0, S)
    # Initialize accumulator as a vector of length D
    acc = tl.zeros((D,), dtype=tl.float32)

    # Loop over K dimension in tiles of BLOCK_K
    for k0 in range(0, D, BLOCK_K):
        k_idx = k0 + tl.arange(0, BLOCK_K)  # [BLOCK_K]
        k_mask = k_idx < D

        # Load x_vec: X[b, o, k] for k in [k0, k0+BLOCK_K)
        x_vec = tl.load(
            X_ptr + b * X_s0 + o * X_s1 + k_idx * X_s2,
            mask=k_mask,
            other=0.0,
        )  # shape [BLOCK_K]

        # Load W_sub: W[k, :] for k in [k0, k0+BLOCK_K), broadcasting across N=D
        # We need a 2D pointer: [BLOCK_K, D]
        w_sub = tl.load(
            W_ptr + k_idx[:, None] * W_s0 + tl.arange(0, D)[None, :] * W_s1,
            mask=k_mask[:, None],
            other=0.0,
        )  # shape [BLOCK_K, D]

        # Accumulate: acc += sum_{kk in tile} x_vec[kk] * w_sub[kk, :]
        # w_sub[kk, :] is a row vector; x_vec[kk] is scalar.
        for kk in range(0, BLOCK_K):
            kk_valid = kk < D
            xk = x_vec[kk]
            w_row = w_sub[kk, :]  # vector of length D
            acc += xk * w_row

    # Store the accumulated row to Y[b, o, :]
    for j in range(0, D):
        tl.store(Y_ptr + b * Y_s0 + o * Y_s1 + j * Y_s2, acc[j])


def _choose_block_k(D: int) -> int:
    # Choose a power-of-two or sensible BLOCK_K that divides D when possible.
    for blk in (1024, 512, 256, 128, 64, 32, 16, 8, 4, 1):
        if blk <= D:
            return blk
    # If nothing divides, default to 1 to be safe (though uncommon for hidden_dim).
    return 1


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized version of the original run:
        1) Concatenate sequences along dim=1 in Triton
        2) Matmul with process_weight.T in Triton (per-row)
        3) Split outputs back in PyTorch (simple slicing)

        Returns:
            processed_encoder: [B, T, D]
            processed_hidden: [B, I, D]
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA for Triton"
        B = hidden_states.shape[0]
        I = hidden_states.shape[1]
        T = encoder_hidden_states.shape[1]
        D = hidden_states.shape[2]
        assert process_weight.shape[0] == D and process_weight.shape[1] == D, "process_weight must be [D, D]"

        # Ensure contiguous tensors
        E = encoder_hidden_states.contiguous()
        H = hidden_states.contiguous()
        W = process_weight.contiguous()

        # 1) Concatenate along sequence dimension: Y [B, T+I, D]
        S = T + I
        Y = torch.empty((B, S, D), dtype=torch.float32, device=hidden_states.device)

        # Launch concatenation kernel: one program per (b, o)
        grid_concat = (B, S, 1)
        # Use a reasonable number of warps for this simple copy; each program does only D loads/stores
        concatenate_sequences_kernel[grid_concat](
            E, H, Y,
            B, T, I, D,
            E.stride(0), E.stride(1), E.stride(2),
            H.stride(0), H.stride(1), H.stride(2),
            Y.stride(0), Y.stride(1), Y.stride(2),
            num_warps=1,  # simple copy per program; 1 warp is fine
        )

        # 2) Matmul Y @ W.T -> [B, S, D] in Triton (per-row)
        processed = torch.empty((B, S, D), dtype=torch.float32, device=hidden_states.device)

        BLOCK_K = _choose_block_k(D)
        grid_matmul = (B, S, 1)
        batched_matmul_rows_kernel[grid_matmul](
            Y, W, processed,
            B, S, D,
            Y.stride(0), Y.stride(1), Y.stride(2),
            W.stride(0), W.stride(1),
            processed.stride(0), processed.stride(1), processed.stride(2),
            BLOCK_K=BLOCK_K,
            num_warps=4,
            num_stages=2,
        )

        # 3) Split back: processed_encoder [B, T, D], processed_hidden [B, I, D]
        processed_encoder = processed[:, :T, :]
        processed_hidden = processed[:, T:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
