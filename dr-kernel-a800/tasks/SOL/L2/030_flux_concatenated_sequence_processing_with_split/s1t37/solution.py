import torch
import triton
import triton.language as tl


@triton.jit
def _concat_kernel(
    e_ptr, h_ptr, out_ptr,
    B, T, P, K,
    e_stride_b, e_stride_t, e_stride_k,
    h_stride_b, h_stride_p, h_stride_k,
    out_stride_b, out_stride_l, out_stride_k,
    BLOCK_K: tl.constexpr,
):
    # 3D grid: (batch, l index in T+P, K tile)
    b = tl.program_id(0)
    l = tl.program_id(1)
    k_block = tl.program_id(2)

    k_offsets = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offsets < K

    # pointers
    e_off = b * e_stride_b + l * e_stride_t + k_offsets * e_stride_k
    h_off = b * h_stride_b + (l - T) * h_stride_p + k_offsets * h_stride_k
    out_off = b * out_stride_b + l * out_stride_l + k_offsets * out_stride_k

    # choose source based on l
    use_encoder = l < T
    src_ptr = tl.where(use_encoder, e_ptr + e_off, h_ptr + h_off)
    val = tl.load(src_ptr, mask=k_mask, other=0.0)
    tl.store(out_ptr + out_off, val, mask=k_mask)


@triton.jit
def _split_encoder_kernel(
    c_ptr, out_ptr,
    B, T, K,
    c_stride_b, c_stride_t, c_stride_k,
    out_stride_b, out_stride_t, out_stride_k,
    BLOCK_K: tl.constexpr,
):
    b = tl.program_id(0)
    t = tl.program_id(1)
    k_block = tl.program_id(2)

    k_offsets = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offsets < K

    c_off = b * c_stride_b + t * c_stride_t + k_offsets * c_stride_k
    out_off = b * out_stride_b + t * out_stride_t + k_offsets * out_stride_k
    val = tl.load(c_ptr + c_off, mask=k_mask, other=0.0)
    tl.store(out_ptr + out_off, val, mask=k_mask)


@triton.jit
def _split_hidden_kernel(
    c_ptr, out_ptr,
    B, T, P, K,
    c_stride_b, c_stride_l, c_stride_k,
    out_stride_b, out_stride_p, out_stride_k,
    BLOCK_K: tl.constexpr,
):
    b = tl.program_id(0)
    p = tl.program_id(1)
    k_block = tl.program_id(2)

    k_offsets = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offsets < K

    # l = T + p
    l = T + p
    c_off = b * c_stride_b + l * c_stride_l + k_offsets * c_stride_k
    out_off = b * out_stride_b + p * out_stride_p + k_offsets * out_stride_k
    val = tl.load(c_ptr + c_off, mask=k_mask, other=0.0)
    tl.store(out_ptr + out_off, val, mask=k_mask)


def _triton_concat(encoder_hidden_states: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
    """
    Concatenate [B, T, K] and [B, P, K] along sequence dimension using Triton, producing [B, T+P, K].
    """
    assert encoder_hidden_states.is_cuda and hidden_states.is_cuda, "Inputs must be on CUDA for Triton."
    B, T, K = encoder_hidden_states.shape
    B2, P2, K2 = hidden_states.shape
    assert B == B2 and K == K2, "Batch and feature dims must match for concatenation."
    P = P2

    Acat = torch.empty((B, T + P, K), device=encoder_hidden_states.device, dtype=torch.float32)

    BLOCK_K = 128
    grid = (B, T + P, triton.cdiv(K, BLOCK_K))
    _concat_kernel[grid](
        encoder_hidden_states, hidden_states, Acat,
        B, T, P, K,
        encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
        hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
        Acat.stride(0), Acat.stride(1), Acat.stride(2),
        BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return Acat


def _triton_split(processed: torch.Tensor, T: int, P: int, K: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Split processed [B, T+P, K] into (processed_encoder [B, T, K], processed_hidden [B, P, K]) using Triton.
    """
    B = processed.shape[0]
    processed_encoder = torch.empty((B, T, K), device=processed.device, dtype=torch.float32)
    processed_hidden = torch.empty((B, P, K), device=processed.device, dtype=torch.float32)

    BLOCK_K = 128
    grid_e = (B, T, triton.cdiv(K, BLOCK_K))
    _split_encoder_kernel[grid_e](
        processed, processed_encoder,
        B, T, K,
        processed.stride(0), processed.stride(1), processed.stride(2),
        processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
        BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )

    grid_h = (B, P, triton.cdiv(K, BLOCK_K))
    _split_hidden_kernel[grid_h](
        processed, processed_hidden,
        B, T, P, K,
        processed.stride(0), processed.stride(1), processed.stride(2),
        processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
        BLOCK_K=BLOCK_K,
        num_warps=4, num_stages=2,
    )
    return processed_encoder, processed_hidden


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Drop-in replacement that calls the original run and performs concatenation/splitting with Triton.
        - Concatenation: Triton kernel (no torch.cat).
        - Linear projection: torch.matmul (as in original), since signature must be preserved.
        - Splitting: Triton kernels (no torch slicing).
        """
        # Ensure tensors are on CUDA for Triton
        if not (hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda):
            # Fallback: keep original behavior if not on CUDA
            text_seq_len = encoder_hidden_states.shape[1]
            img_seq_len = hidden_states.shape[1]
            concatenated = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            processed = torch.matmul(concatenated, process_weight.t())
            processed_encoder = processed[:, :text_seq_len, :]
            processed_hidden = processed[:, text_seq_len:, :]
            return processed_encoder, processed_hidden

        # Triton-based concatenation
        Acat = _triton_concat(encoder_hidden_states, hidden_states)

        # Linear projection using torch (preserves original semantics)
        processed = torch.matmul(Acat, process_weight.t())

        # Triton-based splitting
        T = encoder_hidden_states.shape[1]
        P = hidden_states.shape[1]
        K = hidden_states.shape[2]
        processed_encoder, processed_hidden = _triton_split(processed, T, P, K)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
