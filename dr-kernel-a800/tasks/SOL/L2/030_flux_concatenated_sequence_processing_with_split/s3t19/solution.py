import torch
import triton
import triton.language as tl


@triton.jit
def _fused_concat_gemm_split(
    enc_ptr,       # *ptr to encoder_hidden_states: [B, T, D]
    hst_ptr,       # *ptr to hidden_states: [B, I, D]
    wt_ptr,        # *ptr to process_weight.T: [D, D]
    out_enc_ptr,   # *ptr to processed_encoder: [B, T, D]
    out_hst_ptr,   # *ptr to processed_hidden: [B, I, D]
    B: tl.constexpr,
    T: tl.constexpr,
    I: tl.constexpr,
    D: tl.constexpr,
    BLOCK_P: tl.constexpr,  # tile size along sequence
    BLOCK_N: tl.constexpr,  # tile size along feature
):
    # Grid: (B, ceil(P/BLOCK_P), ceil(D/BLOCK_N)), but since we compute both outputs,
    # we keep only (B, ceil(D/BLOCK_N)) and iterate p and i inside the kernel.
    b = tl.program_id(0)
    pid_n = tl.program_id(1)
    n_start = pid_n * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    mask_n = n_offsets < D

    # Accumulator for the output tile [P_tile, BLOCK_N], but we will store to two outputs.
    # We'll loop over p (both encoder and hidden) and compute acc per row and store.
    # First handle encoder stream: rows p in [0, T)
    for p_i in range(0, BLOCK_P):
        p = p_i  # scalar p within the tile
        # Only process if p < T
        if p < T:
            acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
            # Reduction over K (hidden dimension)
            for k in range(0, D):
                # Select value from encoder or hidden based on p
                enc_val = tl.load(enc_ptr + b * T * D + p * D + k, mask=True, other=0.0)
                # For hidden, p would be out of range, but we gate with if p < T. So enc_val is valid for encoder rows.
                # We need a value for hidden rows; however, since we're currently on an encoder row, we only use enc_val.
                # Compute contribution: acc += enc_val * wt[k, n_offsets]
                wt_row = tl.load(wt_ptr + k * D + n_offsets, mask=mask_n, other=0.0)
                acc += enc_val * wt_row
            # Store to processed_encoder[b, p, n_offsets]
            tl.store(out_enc_ptr + b * T * D + p * D + n_offsets, acc, mask=mask_n)

    # Now handle hidden stream: rows p in [T, T+I)
    for p_i in range(0, BLOCK_P):
        p = T + p_i  # scalar p within the tile
        if p < (T + I):
            acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
            # Reduction over K (hidden dimension)
            for k in range(0, D):
                # For hidden rows, use value from hidden_states
                hst_val = tl.load(hst_ptr + b * I * D + (p - T) * D + k, mask=True, other=0.0)
                wt_row = tl.load(wt_ptr + k * D + n_offsets, mask=mask_n, other=0.0)
                acc += hst_val * wt_row
            # Store to processed_hidden[b, p - T, n_offsets]
            tl.store(out_hst_ptr + b * I * D + (p - T) * D + n_offsets, acc, mask=mask_n)


def run(
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    process_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Triton-fused version of the original run function:
    - Builds concatenated input (conceptually) without torch.cat
    - Performs the linear projection via a Triton kernel (outer-product style)
    - Writes directly into the two output streams
    """
    B = hidden_states.shape[0]
    I = hidden_states.shape[1]
    T = encoder_hidden_states.shape[1]
    D = hidden_states.shape[2]
    assert encoder_hidden_states.shape[0] == B and encoder_hidden_states.shape[2] == D
    assert process_weight.shape[0] == D and process_weight.shape[1] == D

    # Ensure tensors are contiguous and on CUDA
    enc = encoder_hidden_states.contiguous()
    hst = hidden_states.contiguous()
    wt_T = process_weight.t().contiguous()  # process_weight.T

    # Outputs (fp32 for accumulation)
    processed_encoder = torch.empty((B, T, D), dtype=torch.float32, device=enc.device)
    processed_hidden = torch.empty((B, I, D), dtype=torch.float32, device=hst.device)

    # Choose tiles; small to be robust for varied shapes
    BLOCK_P = 32  # sequence tile
    BLOCK_N = 64  # feature tile

    # Launch Triton kernel: grid over batch and feature tiles
    grid = (B, triton.cdiv(D, BLOCK_N))
    _fused_concat_gemm_split[grid](
        enc, hst, wt_T, processed_encoder, processed_hidden,
        B=B, T=T, I=I, D=D,
        BLOCK_P=BLOCK_P, BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=2
    )

    # Return as originally expected (the original code likely uses fp32; if not, cast accordingly)
    # The original code uses matmul on fp32 tensors, so returning fp32 is consistent.
    return processed_encoder, processed_hidden


class ModelNew(torch.nn.Module):
    def forward(self, *args):
        # Forward uses Triton kernels only; no torch.cat, matmul, or tensor mm.
        return run(*args)


def run(*args):
    return ModelNew()(*args)
