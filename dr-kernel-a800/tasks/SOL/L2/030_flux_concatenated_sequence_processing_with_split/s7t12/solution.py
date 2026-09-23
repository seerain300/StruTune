import torch
import triton
import triton.language as tl


@triton.jit
def cat_rows_kernel(
    encoder_ptr,  # *const float, shape [B, T, H]
    hidden_ptr,   # *const float, shape [B, I, H]
    out_ptr,      # *float, shape [B, T+I, H]
    T: tl.constexpr,  # int
    I: tl.constexpr,  # int
    H: tl.constexpr,  # int
    stride_e_b, stride_e_t, stride_e_h,
    stride_i_b, stride_i_i, stride_i_h,
    stride_o_b, stride_o_m, stride_o_h,
):
    # grid: (B, T+I)
    b = tl.program_id(0)
    p = tl.program_id(1)
    # Choose source based on p
    if p < T:
        src_ptr = encoder_ptr + b * stride_e_b + p * stride_e_t
    else:
        src_ptr = hidden_ptr + b * stride_i_b + (p - T) * stride_i_i
    dst_ptr = out_ptr + b * stride_o_b + p * stride_o_m
    # Copy H elements
    for h in range(0, H):
        val = tl.load(src_ptr + h * stride_e_h if p < T else hidden_ptr + b * stride_i_b + (p - T) * stride_i_i + h * stride_i_h)
        tl.store(dst_ptr + h * stride_o_h, val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        # Ensure inputs are CUDA tensors
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All inputs must be CUDA tensors."
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3 and process_weight.dim() == 2, "Invalid input shapes."
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        M = T + I

        # Allocate concatenated matrix X_cat: [B, M, H]
        X_cat = torch.empty((B, M, H), device=hidden_states.device, dtype=hidden_states.dtype)

        # Launch cat_rows_kernel: grid = (B, M)
        grid_cat = (B, M)
        cat_rows_kernel[grid_cat](
            encoder_hidden_states, hidden_states, X_cat,
            T, I, H,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            X_cat.stride(0), X_cat.stride(1), X_cat.stride(2),
            num_warps=1, num_stages=1,
        )

        # Linear projection using torch (GPU) to avoid complex Triton GEMM issues
        # process_weight is [H, H]; we need X_cat @ process_weight
        processed = torch.matmul(X_cat, process_weight)

        # Split into encoder and hidden streams via host slicing
        processed_encoder = [processed[b, :T, :] for b in range(B)]
        processed_hidden = [processed[b, T:, :] for b in range(B)]

        # Ensure dtype consistency
        processed_encoder = [pe.to(hidden_states.dtype) for pe in processed_encoder]
        processed_hidden = [ph.to(hidden_states.dtype) for ph in processed_hidden]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
