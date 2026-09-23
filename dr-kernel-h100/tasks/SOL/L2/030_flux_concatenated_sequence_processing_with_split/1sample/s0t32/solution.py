import torch
import triton
import triton.language as tl

# Triton kernel: concatenate encoder_hidden_states and hidden_states along the sequence axis into out_cat
@triton.jit
def _concatenate_seq_kernel(
    e_ptr,  # *ptr to encoder_hidden_states [B, T, H]
    h_ptr,  # *ptr to hidden_states [B, I, H]
    out_ptr,  # *ptr to out_cat [B, T+I, H]
    B, T, I, H,
    e_stride_b, e_stride_t, e_stride_h,
    h_stride_b, h_stride_i, h_stride_h,
    out_stride_b, out_stride_l, out_stride_h,
    BLOCK_L: tl.constexpr,
):
    b = tl.program_id(0)
    # Compute the number of concatenated steps
    total = T + I
    # Loop over concatenated positions with a block of L
    for l_start in range(0, total, BLOCK_L):
        l = l_start + tl.arange(0, BLOCK_L)
        mask_l = l < total
        # For each l, if l < T -> copy from encoder, else copy from hidden at (l - T)
        # We'll do scalar per-l loop to keep code simple and safe
        for i in range(0, BLOCK_L):
            curr_l = l[i]
            valid = mask_l[i]
            if valid:
                # If curr_l < T, copy from encoder; else copy from hidden
                if curr_l < T:
                    # e_ptr[b, curr_l, :]
                    # Pointer arithmetic: e_ptr + b*e_stride_b + curr_l*e_stride_t + h*e_stride_h
                    # Vectorize over H
                    h_idx = tl.arange(0, H)
                    e_vals = tl.load(e_ptr + b * e_stride_b + curr_l * e_stride_t + h_idx * e_stride_h)
                    # Store to out_ptr[b, curr_l, :]
                    tl.store(out_ptr + b * out_stride_b + curr_l * out_stride_l + h_idx * out_stride_h, e_vals)
                else:
                    idx_in_h = curr_l - T
                    h_vals = tl.load(h_ptr + b * h_stride_b + idx_in_h * h_stride_i + tl.arange(0, H) * h_stride_h)
                    tl.store(out_ptr + b * out_stride_b + curr_l * out_stride_l + tl.arange(0, H) * out_stride_h, h_vals)


# Triton kernel: split processed [B, T+I, H] into two: encoder [B, T, H] and hidden [B, I, H]
@triton.jit
def _split_streams_kernel(
    in_ptr,  # processed [B, T+I, H]
    out_e_ptr,  # processed_encoder [B, T, H]
    out_h_ptr,  # processed_hidden [B, I, H]
    B, T, I, H,
    in_stride_b, in_stride_l, in_stride_h,
    e_stride_b, e_stride_t, e_stride_h,
    h_stride_b, h_stride_i, h_stride_h,
):
    b = tl.program_id(0)
    # Copy first T rows to encoder
    for l in range(0, T):
        h_idx = tl.arange(0, H)
        vals = tl.load(in_ptr + b * in_stride_b + l * in_stride_l + h_idx * in_stride_h)
        tl.store(out_e_ptr + b * e_stride_b + l * e_stride_t + h_idx * e_stride_h, vals)
    # Copy remaining rows to hidden
    start = T
    for i in range(0, I):
        l = start + i
        h_idx = tl.arange(0, H)
        vals = tl.load(in_ptr + b * in_stride_b + l * in_stride_l + h_idx * in_stride_h)
        tl.store(out_h_ptr + b * h_stride_b + i * h_stride_i + h_idx * h_stride_h, vals)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        hidden_states: [batch, img_seq_len, hidden_dim]
        encoder_hidden_states: [batch, text_seq_len, hidden_dim]
        process_weight: [hidden_dim, hidden_dim] (no bias)
        Returns: (processed_encoder_hidden_states, processed_hidden)
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be CUDA tensors"
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        device = hidden_states.device

        # Allocate concatenated tensor
        out_cat = torch.empty((B, T + I, H), dtype=hidden_states.dtype, device=device)

        # Launch concatenation kernel: one program per batch
        _concatenate_seq_kernel[(B,)](
            encoder_hidden_states, hidden_states, out_cat,
            B, T, I, H,
            *encoder_hidden_states.stride(), *hidden_states.stride(), *out_cat.stride(),
            BLOCK_L=256,  # large block, masked by T+I
            num_warps=1, num_stages=1
        )

        # Matmul using torch (robust and correct across diverse shapes)
        # processed = out_cat @ process_weight.T
        processed = torch.matmul(out_cat, process_weight.t())

        # Allocate outputs
        processed_encoder = torch.empty((B, T, H), dtype=processed.dtype, device=device)
        processed_hidden = torch.empty((B, I, H), dtype=processed.dtype, device=device)

        # Split using Triton kernel: one program per batch
        _split_streams_kernel[(B,)](
            processed, processed_encoder, processed_hidden,
            B, T, I, H,
            *processed.stride(), *processed_encoder.stride(), *processed_hidden.stride(),
            num_warps=1, num_stages=1
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
