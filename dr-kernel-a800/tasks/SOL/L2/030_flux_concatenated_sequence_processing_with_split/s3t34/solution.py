import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_encoder_elementwise(
    X_ptr,       # *ptr to encoder_hidden_states: [B, T, D], contiguous
    WT_ptr,      # *ptr to process_weight.T: [D, D], contiguous
    Out_ptr,     # *ptr to processed_encoder: [B, T, D], contiguous
    B: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
):
    # Grid: (B, T, D)
    b = tl.program_id(0)
    p = tl.program_id(1)
    d = tl.program_id(2)

    # Accumulator in the same dtype as inputs
    acc = 0.0

    # Reduction over k in [0, D)
    for k in range(0, D):
        x_val = tl.load(X_ptr + b * T * D + p * D + k)
        wt_val = tl.load(WT_ptr + k * D + d)
        acc += x_val * wt_val

    tl.store(Out_ptr + b * T * D + p * D + d, acc)


@triton.jit
def _gemm_hidden_elementwise(
    X_ptr,       # *ptr to hidden_states: [B, I, D], contiguous
    WT_ptr,      # *ptr to process_weight.T: [D, D], contiguous
    Out_ptr,     # *ptr to processed_hidden: [B, I, D], contiguous
    B: tl.constexpr,
    I: tl.constexpr,
    D: tl.constexpr,
):
    # Grid: (B, I, D)
    b = tl.program_id(0)
    p = tl.program_id(1)
    d = tl.program_id(2)

    acc = 0.0
    for k in range(0, D):
        x_val = tl.load(X_ptr + b * I * D + p * D + k)
        wt_val = tl.load(WT_ptr + k * D + d)
        acc += x_val * wt_val

    tl.store(Out_ptr + b * I * D + p * D + d, acc)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Ensure CUDA tensors and contiguity
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be CUDA tensors for Triton."
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        D = hidden_states.shape[2]

        enc = encoder_hidden_states.contiguous()
        hst = hidden_states.contiguous()
        WT = process_weight.t().contiguous()  # [D, D]

        # Allocate outputs
        processed_encoder = torch.empty((B, T, D), dtype=enc.dtype, device=enc.device)
        processed_hidden = torch.empty((B, I, D), dtype=hst.dtype, device=hst.device)

        # Launch Triton kernels: one program per output element
        grid_enc = (B, T, D)
        _gemm_encoder_elementwise[grid_enc](enc, WT, processed_encoder, B, T, D, num_warps=1, num_stages=1)

        grid_hid = (B, I, D)
        _gemm_hidden_elementwise[grid_hid](hst, WT, processed_hidden, B, I, D, num_warps=1, num_stages=1)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
