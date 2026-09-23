import torch
import triton
import triton.language as tl


# Triton kernel: concatenate encoder_hidden_states and hidden_states along sequence dim
# Inputs:
#   E: [B, T, K]
#   H: [B, P, K]
#   Ac: [B, T+P, K] (output, must be allocated by host)
# Grid: (B, T+P, K) -> each program handles one (b, l) vector across K
@triton.jit
def _concatenate_kernel(
    E_ptr, H_ptr, Ac_ptr,
    B, T, P, K,
    E_stride_b, E_stride_t, E_stride_k,
    H_stride_b, H_stride_p, H_stride_k,
    Ac_stride_b, Ac_stride_l, Ac_stride_k,
    BLOCK_K: tl.constexpr
):
    b = tl.program_id(0)
    l = tl.program_id(1)
    k = tl.program_id(2)

    # Compute vector of k indices
    ks = k + tl.arange(0, BLOCK_K)
    mask_k = ks < K

    # If l < T: source is encoder E[b, l, :], else source is hidden H[b, l-T, :]
    is_encoder = l < T

    # Pointers
    E_row_ptr = E_ptr + b * E_stride_b + l * E_stride_t + ks * E_stride_k
    # For hidden, source position is l - T
    H_src_l = l - T
    H_row_ptr = H_ptr + b * H_stride_b + H_src_l * H_stride_p + ks * H_stride_k
    Ac_row_ptr = Ac_ptr + b * Ac_stride_b + l * Ac_stride_l + ks * Ac_stride_k

    # Select source based on is_encoder
    # Triton lacks tl.where for pointer arithmetic; we use a mask multiply by boolean cast.
    # We'll load both and then select with is_encoder
    E_vals = tl.load(E_row_ptr, mask=mask_k, other=0.0)
    H_vals = tl.load(H_row_ptr, mask=mask_k, other=0.0)
    # If is_encoder: Ac = E_vals else Ac = H_vals
    Ac_vals = tl.where(is_encoder, E_vals, H_vals)

    tl.store(Ac_row_ptr, Ac_vals, mask=mask_k)


# Triton kernel: GEMM on Acat [B, T+P, K] and W [K, K], write C [B, T+P, K]
# Grid: (B, T+P, K) -> each program computes one output row m for given (b, l), reducing over K.
@triton.jit
def _gemm_row_kernel(
    A_ptr, W_ptr, C_ptr,
    B, T, P, K,
    A_stride_b, A_stride_l, A_stride_k,
    W_stride_k, W_stride_w,  # W is [K, K]; strides for k (rows) and w (cols)
    C_stride_b, C_stride_l, C_stride_k,
    BLOCK_K: tl.constexpr
):
    b = tl.program_id(0)
    l = tl.program_id(1)
    k_out = tl.program_id(2)

    # Accumulator for one output row
    acc = tl.zeros([BLOCK_K], dtype=tl.float32)

    # Loop over K in chunks
    for kk in range(0, K, BLOCK_K):
        ks = kk + tl.arange(0, BLOCK_K)
        mask_k = ks < K

        # Load A row for m = b*(T+P) + l
        m = b * (T + P) + l
        A_row_ptr = A_ptr + m * A_stride_b + ks * A_stride_k
        A_vals = tl.load(A_row_ptr, mask=mask_k, other=0.0)

        # Load W chunk as [BLOCK_K, BLOCK_K]
        W_block = tl.load(
            W_ptr + ks[:, None] * W_stride_k + tl.arange(0, BLOCK_K)[None, :] * W_stride_w,
            mask=mask_k[:, None],
            other=0.0
        )

        # Accumulate
        acc += tl.dot(A_vals, W_block)  # A_vals: [BLOCK_K], W_block: [BLOCK_K, BLOCK_K]

    # Store result to C at row m
    C_row_ptr = C_ptr + m * C_stride_b + k_out * C_stride_k
    tl.store(C_row_ptr, acc, mask=True)  # k_out always < K


# Triton kernel: copy slice of C[:, :T, :] into processed_encoder [B, T, K]
# Grid: (B, T, K)
@triton.jit
def _split_encoder_kernel(
    C_ptr, PE_ptr,
    B, T, K,
    C_stride_b, C_stride_l, C_stride_k,
    PE_stride_b, PE_stride_t, PE_stride_k,
    BLOCK_K: tl.constexpr
):
    b = tl.program_id(0)
    t = tl.program_id(1)
    k = tl.program_id(2)

    ks = k + tl.arange(0, BLOCK_K)
    mask_k = ks < K

    C_row_ptr = C_ptr + b * C_stride_b + t * C_stride_l + ks * C_stride_k
    PE_row_ptr = PE_ptr + b * PE_stride_b + t * PE_stride_t + ks * PE_stride_k

    vals = tl.load(C_row_ptr, mask=mask_k, other=0.0)
    tl.store(PE_row_ptr, vals, mask=mask_k)


# Triton kernel: copy slice of C[:, T:, :] into processed_hidden [B, P, K]
# Grid: (B, P, K)
@triton.jit
def _split_hidden_kernel(
    C_ptr, PH_ptr,
    B, T, P, K,
    C_stride_b, C_stride_l, C_stride_k,
    PH_stride_b, PH_stride_p, PH_stride_k,
    BLOCK_K: tl.constexpr
):
    b = tl.program_id(0)
    p = tl.program_id(1)
    k = tl.program_id(2)

    ks = k + tl.arange(0, BLOCK_K)
    mask_k = ks < K

    # Source l = T + p
    l = T + p
    C_row_ptr = C_ptr + b * C_stride_b + l * C_stride_l + ks * C_stride_k
    PH_row_ptr = PH_ptr + b * PH_stride_b + p * PH_stride_p + ks * PH_stride_k

    vals = tl.load(C_row_ptr, mask=mask_k, other=0.0)
    tl.store(PH_row_ptr, vals, mask=mask_k)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only implementation:
          - Concatenate (encoder, image) via Triton kernel
          - Linear projection via Triton GEMM kernel
          - Split into encoder and image outputs via Triton kernels
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA"
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        P = hidden_states.shape[1]
        K = hidden_states.shape[2]
        # Ensure dtype is float32 for numerical consistency
        if hidden_states.dtype != torch.float32:
            hidden_states = hidden_states.float()
        if encoder_hidden_states.dtype != torch.float32:
            encoder_hidden_states = encoder_hidden_states.float()
        if process_weight.dtype != torch.float32:
            process_weight = process_weight.float()

        # 1) Triton concatenation: Acat = cat([encoder_hidden_states, hidden_states], dim=1)
        Acat = torch.empty((B, T + P, K), device=hidden_states.device, dtype=torch.float32)

        E = encoder_hidden_states
        H = hidden_states

        # Strides
        E_stride_b, E_stride_t, E_stride_k = E.stride(0), E.stride(1), E.stride(2)
        H_stride_b, H_stride_p, H_stride_k = H.stride(0), H.stride(1), H.stride(2)
        Ac_stride_b, Ac_stride_l, Ac_stride_k = Acat.stride(0), Acat.stride(1), Acat.stride(2)

        BLOCK_K = 128  # tile along K
        grid_concat = (B, T + P, K)
        _concatenate_kernel[grid_concat](
            E, H, Acat,
            B, T, P, K,
            E_stride_b, E_stride_t, E_stride_k,
            H_stride_b, H_stride_p, H_stride_k,
            Ac_stride_b, Ac_stride_l, Ac_stride_k,
            BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # 2) Triton GEMM: C = Acat @ process_weight.T
        Wt = process_weight.t().contiguous()  # [K, K]
        C = torch.empty_like(Acat)  # [B, T+P, K]

        A_stride_b, A_stride_l, A_stride_k = Acat.stride(0), Acat.stride(1), Acat.stride(2)
        W_stride_k, W_stride_w = Wt.stride(0), Wt.stride(1)  # Wt is [K, K]
        C_stride_b, C_stride_l, C_stride_k = C.stride(0), C.stride(1), C.stride(2)

        grid_gemm = (B, T + P, K)
        _gemm_row_kernel[grid_gemm](
            Acat, Wt, C,
            B, T, P, K,
            A_stride_b, A_stride_l, A_stride_k,
            W_stride_k, W_stride_w,
            C_stride_b, C_stride_l, C_stride_k,
            BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # 3) Triton split: processed_encoder = C[:, :T, :], processed_hidden = C[:, T:, :]
        processed_encoder = torch.empty((B, T, K), device=hidden_states.device, dtype=torch.float32)
        processed_hidden = torch.empty((B, P, K), device=hidden_states.device, dtype=torch.float32)

        # Split encoder
        BLOCK_K_split = 128
        grid_e = (B, T, K)
        _split_encoder_kernel[grid_e](
            C, processed_encoder,
            B, T, K,
            C.stride(0), C.stride(1), C.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            BLOCK_K=BLOCK_K_split,
            num_warps=4, num_stages=2
        )

        # Split hidden
        grid_h = (B, P, K)
        _split_hidden_kernel[grid_h](
            C, processed_hidden,
            B, T, P, K,
            C.stride(0), C.stride(1), C.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_K=BLOCK_K_split,
            num_warps=4, num_stages=2
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
