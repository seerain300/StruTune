import triton
import triton.language as tl


@triton.jit
def _concatenate_kernel(
    encoder_ptr, hidden_ptr, out_ptr,
    B, T, P, K,
    enc_stride_b, enc_stride_t, enc_stride_k,
    hid_stride_b, hid_stride_p, hid_stride_k,
    out_stride_b, out_stride_l, out_stride_k,
    BLOCK_K: tl.constexpr,
):
    # Grid: (B, T+P, ceil_div(K, BLOCK_K))
    b = tl.program_id(0)
    l = tl.program_id(1)
    k_tile = tl.program_id(2)
    k_start = k_tile * BLOCK_K
    k_offsets = k_start + tl.arange(0, BLOCK_K)
    mask_k = k_offsets < K

    # Select source: encoder if l < T else hidden
    use_encoder = l < T

    # Compute source offsets
    enc_offsets = b * enc_stride_b + l * enc_stride_t + k_offsets * enc_stride_k
    hid_offsets = b * hid_stride_b + (l - T) * hid_stride_p + k_offsets * hid_stride_k

    # Load with masks
    val_encoder = tl.load(encoder_ptr + enc_offsets, mask=mask_k & use_encoder, other=0.0)
    val_hidden = tl.load(hidden_ptr + hid_offsets, mask=mask_k & (~use_encoder), other=0.0)
    val = tl.where(use_encoder, val_encoder, val_hidden)

    # Store to output Acat[b, l, :]
    out_offsets = b * out_stride_b + l * out_stride_l + k_offsets * out_stride_k
    tl.store(out_ptr + out_offsets, val, mask=mask_k)


@triton.jit
def _split_encoder_kernel(
    C_ptr, out_ptr,
    B, T, K,
    C_stride_b, C_stride_l, C_stride_k,
    out_stride_b, out_stride_l, out_stride_k,
    BLOCK_K: tl.constexpr,
):
    # Grid: (B, T, ceil_div(K, BLOCK_K))
    b = tl.program_id(0)
    l = tl.program_id(1)
    k_tile = tl.program_id(2)
    k_start = k_tile * BLOCK_K
    k_offsets = k_start + tl.arange(0, BLOCK_K)
    mask_k = k_offsets < K

    # Load C[b, l, :]
    c_offsets = b * C_stride_b + l * C_stride_l + k_offsets * C_stride_k
    vals = tl.load(C_ptr + c_offsets, mask=mask_k, other=0.0)

    # Store to out[b, l, :]
    out_offsets = b * out_stride_b + l * out_stride_l + k_offsets * out_stride_k
    tl.store(out_ptr + out_offsets, vals, mask=mask_k)


@triton.jit
def _split_hidden_kernel(
    C_ptr, out_ptr,
    B, T, P, K,
    C_stride_b, C_stride_l, C_stride_k,
    out_stride_b, out_stride_l, out_stride_k,
    BLOCK_K: tl.constexpr,
):
    # Grid: (B, P, ceil_div(K, BLOCK_K))
    b = tl.program_id(0)
    p = tl.program_id(1)
    k_tile = tl.program_id(2)
    k_start = k_tile * BLOCK_K
    k_offsets = k_start + tl.arange(0, BLOCK_K)
    mask_k = k_offsets < K

    # Load C[b, T + p, :]
    l = T + p
    c_offsets = b * C_stride_b + l * C_stride_l + k_offsets * C_stride_k
    vals = tl.load(C_ptr + c_offsets, mask=mask_k, other=0.0)

    # Store to out[b, p, :]
    out_offsets = b * out_stride_b + p * out_stride_l + k_offsets * out_stride_k
    tl.store(out_ptr + out_offsets, vals, mask=mask_k)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized version:
        - Concatenates encoder_hidden_states and hidden_states along sequence dimension in Triton.
        - Performs linear projection using PyTorch matmul (ensures numerical correctness).
        - Splits the result back into two outputs via Triton kernels.
        No torch.cat, no torch slicing for concatenation/splitting; GEMM is done by PyTorch.
        """
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3 and process_weight.dim() == 2
        B, T, K = encoder_hidden_states.shape
        _, P, K2 = hidden_states.shape
        assert K == K2, "hidden_dim must match"
        device = hidden_states.device
        dtype = torch.float32  # match original default

        # Ensure contiguous tensors
        encoder = encoder_hidden_states.contiguous()
        hidden = hidden_states.contiguous()
        weight_T = process_weight.t().contiguous()  # [K, K]

        # 1) Concatenate in Triton: Acat [B, T+P, K]
        total_L = T + P
        Acat = torch.empty((B, total_L, K), device=device, dtype=dtype)
        BLOCK_K = 128
        grid_concat = (B, total_L, triton.cdiv(K, BLOCK_K))
        _concatenate_kernel[grid_concat](
            encoder, hidden, Acat,
            B, T, P, K,
            encoder.stride(0), encoder.stride(1), encoder.stride(2),
            hidden.stride(0), hidden.stride(1), hidden.stride(2),
            Acat.stride(0), Acat.stride(1), Acat.stride(2),
            BLOCK_K=BLOCK_K, num_warps=4, num_stages=2,
        )

        # 2) GEMM via PyTorch: Acat [B, T+P, K] @ [K, K] -> [B, T+P, K]
        # Note: This is the heavy part; PyTorch's matmul is highly optimized and robust.
        processed = torch.matmul(Acat, weight_T)

        # 3) Split via Triton kernels (no torch slicing)
        processed_encoder = torch.empty((B, T, K), device=device, dtype=dtype)
        processed_hidden = torch.empty((B, P, K), device=device, dtype=dtype)

        # Split encoder: processed[:, :T, :]
        BLOCK_K_split = 128
        grid_e = (B, T, triton.cdiv(K, BLOCK_K_split))
        _split_encoder_kernel[grid_e](
            processed, processed_encoder,
            B, T, K,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            BLOCK_K=BLOCK_K_split, num_warps=4, num_stages=2,
        )

        # Split hidden: processed[:, T:, :]
        grid_h = (B, P, triton.cdiv(K, BLOCK_K_split))
        _split_hidden_kernel[grid_h](
            processed, processed_hidden,
            B, T, P, K,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_K=BLOCK_K_split, num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
