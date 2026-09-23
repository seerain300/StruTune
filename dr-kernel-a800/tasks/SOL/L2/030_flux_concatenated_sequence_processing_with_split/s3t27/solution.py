import torch
import triton
import triton.language as tl


@triton.jit
def _concat_gemm_split_kernel(
    enc_ptr,        # *ptr to encoder_hidden_states: [B, T, D]
    hst_ptr,        # *ptr to hidden_states: [B, I, D]
    wt_ptr,         # *ptr to WT: [D, D] (process_weight.T)
    out_enc_ptr,    # *ptr to output encoder stream: [B, T, D]
    out_hst_ptr,    # *ptr to output hidden stream: [B, I, D]
    B: tl.constexpr,
    T: tl.constexpr,
    I: tl.constexpr,
    D: tl.constexpr,
    BLOCK_POS: tl.constexpr,  # tile over P = T + I
    BLOCK_F: tl.constexpr,    # tile over D
    BLOCK_K: tl.constexpr,    # reduction chunk over D
):
    # Grid: (B, ceil((T+I)/BLOCK_POS), ceil(D/BLOCK_F))
    b = tl.program_id(0)
    pid_pos = tl.program_id(1)
    pid_f = tl.program_id(2)

    pos_start = pid_pos * BLOCK_POS
    f_start = pid_f * BLOCK_F

    pos_offsets = pos_start + tl.arange(0, BLOCK_POS)  # indices 0..T+I-1
    f_offsets = f_start + tl.arange(0, BLOCK_F)       # indices 0..D-1

    mask_pos = pos_offsets < (T + I)
    mask_f = f_offsets < D

    acc = tl.zeros((BLOCK_POS, BLOCK_F), dtype=tl.float32)

    # Reduction over hidden dimension D
    for k0 in range(0, D, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < D

        # Build X rows per pos: either from encoder or hidden
        # For each row r in tile:
        #   if pos_offsets[r] < T: X[b, p, k] = enc[b, p, k]
        #   else:                 X[b, p, k] = hst[b, p-T, k]
        x_rows = tl.zeros((BLOCK_POS, BLOCK_K), dtype=tl.float32)

        for r in range(BLOCK_POS):
            p = pos_offsets[r]
            # if p < T, use encoder; else use hidden at p - T
            if p < T:
                x_row = tl.load(enc_ptr + b * T * D + p * D + k_offsets, mask=mask_k, other=0.0)
            else:
                p_hidden = p - T
                x_row = tl.load(hst_ptr + b * I * D + p_hidden * D + k_offsets, mask=mask_k, other=0.0)
            x_rows[r, :] = x_row

        # WT tile [BK, BF]: WT[k, f]
        wt_tile = tl.load(wt_ptr + k_offsets[:, None] * D + f_offsets[None, :], mask=mask_k[:, None] & mask_f[None, :], other=0.0)

        # Accumulate: (BP, BK) @ (BK, BF) -> (BP, BF)
        acc += tl.dot(x_rows, wt_tile)

    # Store results to corresponding outputs
    for r in range(BLOCK_POS):
        p = pos_offsets[r]
        valid = mask_pos[r]
        if valid:
            if p < T:
                out_row_ptr = out_enc_ptr + b * T * D + p * D + f_offsets
                tl.store(out_row_ptr, acc[r, :], mask=mask_f)
            else:
                p_hidden = p - T
                out_row_ptr = out_hst_ptr + b * I * D + p_hidden * D + f_offsets
                tl.store(out_row_ptr, acc[r, :], mask=mask_f)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized forward:
        - Single Triton kernel computes concatenated linear projection and splits outputs.
        - No torch.cat, no torch.matmul, no slicing in forward. All computation is in Triton.
        Returns (processed_encoder, processed_hidden) with shapes [B, T, D] and [B, I, D], respectively.
        """
        # Triton requires CUDA tensors
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA for Triton."
        device = hidden_states.device

        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        D = hidden_states.shape[2]
        P = T + I

        # Allocate outputs (fp32 for stability)
        processed_encoder = torch.empty((B, T, D), dtype=torch.float32, device=device)
        processed_hidden = torch.empty((B, I, D), dtype=torch.float32, device=device)

        # Prepare WT = process_weight.T in fp32
        WT = process_weight.t().contiguous().to(torch.float32)

        # Launch single Triton kernel
        BLOCK_POS = 128  # tile over P = T + I
        BLOCK_F = 64     # tile over D
        BLOCK_K = 64     # reduction chunk over D

        grid = (B, triton.cdiv(P, BLOCK_POS), triton.cdiv(D, BLOCK_F))
        _concat_gemm_split_kernel[grid](
            encoder_hidden_states, hidden_states, WT,
            processed_encoder, processed_hidden,
            B, T, I, D,
            BLOCK_POS=BLOCK_POS, BLOCK_F=BLOCK_F, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
