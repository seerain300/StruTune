import torch
import triton
import triton.language as tl


@triton.jit
def _full_process_kernel(
    encoder_ptr,    # *float32, shape [B, T, K]
    hidden_ptr,     # *float32, shape [B, P, K]
    weight_ptr,     # *float32, shape [K, K]
    out_enc_ptr,    # *float32, shape [B, T, K]
    out_hid_ptr,    # *float32, shape [B, P, K]
    B: tl.constexpr, T: tl.constexpr, P: tl.constexpr, K: tl.constexpr,
    total: tl.constexpr,  # T + P
    BLOCK_L: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 3D grid: (B, cdiv(total, BLOCK_L), cdiv(K, BLOCK_K))
    b_id = tl.program_id(0)
    l_block = tl.program_id(1)
    k_block = tl.program_id(2)

    l_idx = l_block * BLOCK_L + tl.arange(0, BLOCK_L)  # [BLOCK_L]
    k_idx = k_block * BLOCK_K + tl.arange(0, BLOCK_K)  # [BLOCK_K]

    mask_l = l_idx < total
    mask_k = k_idx < K

    # For each l in this tile, compute source and output
    for i in range(BLOCK_L):
        l = l_idx[i]
        valid = l < total

        # Determine source tensor based on whether this position belongs to encoder stream
        is_encoder = l < T
        if is_encoder:
            # source = encoder_hidden_states[b_id, l, :]
            src_ptr = encoder_ptr + b_id * (T * K) + l * K + k_idx
        else:
            # source = hidden_states[b_id, l - T, :]
            src_idx = l - T
            src_ptr = hidden_ptr + b_id * (P * K) + src_idx * K + k_idx

        # Load source vector for this row
        src = tl.load(src_ptr, mask=mask_k & valid, other=0.0)  # [BLOCK_K]

        # Multiply by weight.T: weight[k, k'] where k' in k_idx
        acc = tl.zeros((BLOCK_K,), dtype=tl.float32)
        for k_start in range(0, K, BLOCK_K):
            k_offsets = k_start + tl.arange(0, BLOCK_K)  # [BLOCK_K]
            mask_k2 = k_offsets < K
            w_vec = tl.load(weight_ptr + k_idx * K + k_offsets, mask=mask_k2, other=0.0)
            acc += src * w_vec

        # Store to appropriate output
        if is_encoder:
            out_row = out_enc_ptr + b_id * (T * K) + l * K + k_idx
            tl.store(out_row, acc, mask=mask_k & valid)
        else:
            p = l - T
            out_row = out_hid_ptr + b_id * (P * K) + p * K + k_idx
            tl.store(out_row, acc, mask=mask_k & valid)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only forward:
        - Produces processed_encoder [B, T, K] and processed_hidden [B, P, K] directly.
        - Implements (concatenated @ process_weight.T) without any torch concatenation/slicing.
        """
        # Ensure CUDA tensors
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All tensors must be on CUDA."
        B, P, K = hidden_states.shape
        T, _, _ = encoder_hidden_states.shape
        assert hidden_states.shape[2] == K
        assert encoder_hidden_states.shape[2] == K
        assert process_weight.shape == (K, K)
        assert hidden_states.device == encoder_hidden_states.device == process_weight.device

        # Cast to float32 for Triton
        if hidden_states.dtype != torch.float32:
            hidden_states = hidden_states.float()
        if encoder_hidden_states.dtype != torch.float32:
            encoder_hidden_states = encoder_hidden_states.float()
        if process_weight.dtype != torch.float32:
            process_weight = process_weight.float()

        # Allocate outputs
        processed_encoder = torch.empty((B, T, K), device=hidden_states.device, dtype=torch.float32)
        processed_hidden = torch.empty((B, P, K), device=hidden_states.device, dtype=torch.float32)

        total = T + P

        # Launch Triton kernel: 3D grid over (B, sequence tiles, K tiles)
        # Use moderate tile sizes; robust across shapes.
        BLOCK_L = 64
        BLOCK_K = 64
        grid = (B, triton.cdiv(total, BLOCK_L), triton.cdiv(K, BLOCK_K))
        _full_process_kernel[grid](
            encoder_hidden_states, hidden_states, process_weight.t(),
            processed_encoder, processed_hidden,
            B, T, P, K, total,
            BLOCK_L=BLOCK_L, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
