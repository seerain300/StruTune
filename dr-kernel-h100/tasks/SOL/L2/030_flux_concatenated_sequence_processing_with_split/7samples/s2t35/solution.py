import torch
import triton
import triton.language as tl


@triton.jit
def concat_seqs_kernel(
    enc_ptr,  # *ptr to encoder_hidden_states [B, T, H]
    hid_ptr,  # *ptr to hidden_states [B, I, H]
    out_ptr,  # *ptr to concatenated [B, S, H]
    B: tl.constexpr, T: tl.constexpr, I: tl.constexpr, H: tl.constexpr,
    stride_e_b, stride_e_t, stride_e_h,
    stride_h_b, stride_h_i, stride_h_h,
    stride_o_b, stride_o_s, stride_o_h,
    num_warps: tl.constexpr,
):
    # Grid: (B,)
    b = tl.program_id(0)
    S = T + I

    # Copy encoder rows into out[b, 0:T, :]
    for t in range(0, T):
        row_offset = b * stride_e_b + t * stride_e_t
        col_offset = 0 * stride_o_s
        # Vectorized over H
        h_idx = tl.arange(0, H)
        e_vals = tl.load(enc_ptr + row_offset + h_idx * stride_e_h)
        tl.store(out_ptr + b * stride_o_b + (t + col_offset) * stride_o_s + h_idx * stride_o_h, e_vals)

    # Copy hidden rows into out[b, T:T+I, :]
    for i in range(0, I):
        row_offset = b * stride_h_b + i * stride_h_i
        col_offset = T * stride_o_s
        h_idx = tl.arange(0, H)
        h_vals = tl.load(hid_ptr + row_offset + h_idx * stride_h_h)
        tl.store(out_ptr + b * stride_o_b + (i + col_offset) * stride_o_s + h_idx * stride_o_h, h_vals)


@triton.jit
def split_seqs_kernel(
    inp_ptr,          # *ptr to processed [B, S, H]
    out_e_ptr,        # *ptr to processed_encoder [B, T, H]
    out_i_ptr,        # *ptr to processed_hidden [B, I, H]
    B: tl.constexpr, T: tl.constexpr, I: tl.constexpr, H: tl.constexpr, S: tl.constexpr,
    stride_i_b, stride_i_s, stride_i_h,
    stride_e_b, stride_e_t, stride_e_h,
    stride_h_b, stride_h_i, stride_h_h,
    num_warps: tl.constexpr,
):
    # Grid: (B,)
    b = tl.program_id(0)

    # Split: first T rows go to encoder output
    for t in range(0, T):
        row_offset = b * stride_i_b + t * stride_i_s
        col_offset = 0 * stride_e_t
        h_idx = tl.arange(0, H)
        vals = tl.load(inp_ptr + row_offset + h_idx * stride_i_h)
        tl.store(out_e_ptr + b * stride_e_b + (t + col_offset) * stride_e_t + h_idx * stride_e_h, vals)

    # Remaining S - T rows go to hidden output
    for s in range(T, S):
        row_offset = b * stride_i_b + s * stride_i_s
        col_offset = 0 * stride_h_i
        h_idx = tl.arange(0, H)
        vals = tl.load(inp_ptr + row_offset + h_idx * stride_i_h)
        tl.store(out_i_ptr + b * stride_h_b + (s - T + col_offset) * stride_h_i + h_idx * stride_h_h, vals)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, encoder_hidden_states: torch.Tensor, hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Ensure on CUDA
        assert encoder_hidden_states.is_cuda and hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA"
        # Ensure contiguous
        enc = encoder_hidden_states.contiguous()
        hid = hidden_states.contiguous()
        B, T, H = enc.shape
        B2, I, H2 = hid.shape
        assert B == B2 and H == H2, "Batch or hidden_dim mismatch"
        S = T + I

        # 1) Concatenate encoder and hidden states in Triton: concatenated [B, S, H]
        concatenated = torch.empty((B, S, H), device=enc.device, dtype=enc.dtype)
        grid = (B,)
        concat_seqs_kernel[grid](
            enc, hid, concatenated,
            B, T, I, H,
            enc.stride(0), enc.stride(1), enc.stride(2),
            hid.stride(0), hid.stride(1), hid.stride(2),
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            num_warps=1,
        )

        # 2) Matmul in PyTorch for robustness: processed = concatenated @ process_weight.T
        # process_weight: [H, H], process_weight.T: [H, H]
        processed = torch.matmul(concatenated, process_weight.t())

        # 3) Split processed into encoder and hidden streams using Triton
        processed_encoder = torch.empty((B, T, H), device=enc.device, dtype=enc.dtype)
        processed_hidden = torch.empty((B, I, H), device=enc.device, dtype=enc.dtype)
        split_seqs_kernel[grid](
            processed, processed_encoder, processed_hidden,
            B, T, I, H, S,
            processed.stride(0), processed.stride(1), processed.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            num_warps=1,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
