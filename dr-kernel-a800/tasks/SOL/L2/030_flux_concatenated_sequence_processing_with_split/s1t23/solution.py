import torch
import triton
import triton.language as tl


@triton.jit
def _concatenation_kernel(
    encoder_ptr,        # *f32, [B, T, K]
    hidden_ptr,         # *f32, [B, P, K]
    output_ptr,         # *f32, [B, T+P, K]
    B: tl.constexpr, T: tl.constexpr, P: tl.constexpr, K: tl.constexpr,
    BLOCK_K: tl.constexpr
):
    # grid = (B, T+P)
    b = tl.program_id(0)
    l = tl.program_id(1)

    # Compute whether we take from encoder or hidden
    # l in [0, T+P)
    take_encoder = l < T

    # Offsets for K dimension
    offs_k = tl.arange(0, BLOCK_K)
    mask_k = offs_k < K

    # Compute source and dest pointers
    if take_encoder:
        src = encoder_ptr + b * T * K + l * K + offs_k
        dest = output_ptr + b * (T + P) * K + l * K + offs_k
    else:
        src = hidden_ptr + b * P * K + (l - T) * K + offs_k
        dest = output_ptr + b * (T + P) * K + l * K + offs_k

    # Load and store with mask over K
    vals = tl.load(src, mask=mask_k, other=0.0)
    tl.store(dest, vals, mask=mask_k)


@triton.jit
def _split_encoder_kernel(
    input_ptr,          # *f32, [B, T+P, K]
    output_ptr,         # *f32, [B, T, K]
    B: tl.constexpr, T: tl.constexpr, K: tl.constexpr,
    BLOCK_K: tl.constexpr
):
    # grid = (B, T)
    b = tl.program_id(0)
    t = tl.program_id(1)
    offs_k = tl.arange(0, BLOCK_K)
    mask_k = offs_k < K

    src = input_ptr + b * (T + P) * K + t * K + offs_k
    dest = output_ptr + b * T * K + t * K + offs_k

    vals = tl.load(src, mask=mask_k, other=0.0)
    tl.store(dest, vals, mask=mask_k)


@triton.jit
def _split_hidden_kernel(
    input_ptr,          # *f32, [B, T+P, K]
    output_ptr,         # *f32, [B, P, K]
    B: tl.constexpr, P: tl.constexpr, K: tl.constexpr,
    BLOCK_K: tl.constexpr
):
    # grid = (B, P)
    b = tl.program_id(0)
    p = tl.program_id(1)
    offs_k = tl.arange(0, BLOCK_K)
    mask_k = offs_k < K

    # Input starts at column T for hidden part
    src = input_ptr + b * (T + P) * K + (T + p) * K + offs_k
    dest = output_ptr + b * P * K + p * K + offs_k

    vals = tl.load(src, mask=mask_k, other=0.0)
    tl.store(dest, vals, mask=mask_k)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters; we rely on provided tensors in forward

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-backed forward:
        - Concatenates encoder_hidden_states and hidden_states along sequence dimension (Triton).
        - Performs linear projection via torch.matmul (to ensure numerical correctness).
        - Splits back into two streams using Triton.
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All tensors must be on CUDA for Triton kernels."

        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        P = hidden_states.shape[1]
        K = hidden_states.shape[2]
        assert encoder_hidden_states.shape == (B, T, K), "encoder_hidden_states shape must be [B, T, K]"
        assert hidden_states.shape == (B, P, K), "hidden_states shape must be [B, P, K]"
        assert process_weight.shape == (K, K), "process_weight shape must be [K, K]"
        assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, "Use float32 tensors."

        # 1) Concatenate along sequence dimension using Triton
        Acat = torch.empty((B, T + P, K), device=hidden_states.device, dtype=hidden_states.dtype)

        # We'll use a reasonable BLOCK_K; since K varies, we pick 128 to cover typical sizes. Masks guard tails.
        BLOCK_K = 128

        grid = (B, T + P)
        _concatenation_kernel[grid](
            encoder_hidden_states, hidden_states, Acat,
            B=B, T=T, P=P, K=K, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # 2) Linear projection: Acat [B, T+P, K] @ process_weight.T [K, K]
        # Use torch.matmul for robust, correct results. This is the heavy op.
        # Note: process_weight is [K, K]; we need [K, K].
        W_t = process_weight.transpose(0, 1).contiguous()  # [K, K]
        processed = torch.matmul(Acat, W_t)  # [B, T+P, K]

        # 3) Split back into encoder and image streams using Triton
        processed_encoder = torch.empty((B, T, K), device=hidden_states.device, dtype=hidden_states.dtype)
        processed_hidden = torch.empty((B, P, K), device=hidden_states.device, dtype=hidden_states.dtype)

        grid_e = (B, T)
        _split_encoder_kernel[grid_e](
            processed, processed_encoder,
            B=B, T=T, K=K, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        grid_h = (B, P)
        _split_hidden_kernel[grid_h](
            processed, processed_hidden,
            B=B, P=P, K=K, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
