import torch
import triton
import triton.language as tl


@triton.jit
def _concatenate_sequences_kernel(
    encoder_ptr, hidden_ptr, out_ptr,
    B, T, I, H,
    stride_e_b, stride_e_t, stride_e_h,
    stride_h_b, stride_h_i, stride_h_h,
    stride_o_b, stride_o_l, stride_o_h,
    BLOCK_H: tl.constexpr
):
    # One program per batch
    b = tl.program_id(0)
    # Loop over sequence length T + I
    total = T + I
    h_range = tl.arange(0, BLOCK_H)
    for l in range(0, total):
        # Mask for hidden columns
        mask = h_range < H
        # Load from encoder if l < T, else from hidden at l - T
        is_encoder = l < T
        if is_encoder:
            # encoder[b, l, :]
            e_ptrs = encoder_ptr + b * stride_e_b + l * stride_e_t + h_range * stride_e_h
            vals = tl.load(e_ptrs, mask=mask, other=0.0)
        else:
            # hidden[b, l - T, :]
            h_ptrs = hidden_ptr + b * stride_h_b + (l - T) * stride_h_i + h_range * stride_h_h
            vals = tl.load(h_ptrs, mask=mask, other=0.0)
        # Store to out[b, l, :]
        o_ptrs = out_ptr + b * stride_o_b + l * stride_o_l + h_range * stride_o_h
        tl.store(o_ptrs, vals, mask=mask)


@triton.jit
def _split_streams_kernel(
    in_ptr, out_e_ptr, out_h_ptr,
    B, T, I, H,
    stride_in_b, stride_in_t, stride_in_h,
    stride_e_b, stride_e_t, stride_e_h,
    stride_h_b, stride_h_i, stride_h_h,
    BLOCK_H: tl.constexpr
):
    # One program per batch
    b = tl.program_id(0)
    h_range = tl.arange(0, BLOCK_H)
    # First copy encoder part
    for t in range(0, T):
        in_ptrs = in_ptr + b * stride_in_b + t * stride_in_t + h_range * stride_in_h
        vals = tl.load(in_ptrs, mask=(h_range < H), other=0.0)
        out_e_ptrs = out_e_ptr + b * stride_e_b + t * stride_e_t + h_range * stride_e_h
        tl.store(out_e_ptrs, vals, mask=(h_range < H))
    # Then copy hidden part
    for i in range(0, I):
        in_ptrs = in_ptr + b * stride_in_b + (T + i) * stride_in_t + h_range * stride_in_h
        vals = tl.load(in_ptrs, mask=(h_range < H), other=0.0)
        out_h_ptrs = out_h_ptr + b * stride_h_b + i * stride_h_i + h_range * stride_h_h
        tl.store(out_h_ptrs, vals, mask=(h_range < H))


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        hidden_states: [B, I, H]
        encoder_hidden_states: [B, T, H]
        process_weight: [H, H]
        Returns: (processed_encoder: [B, T, H], processed_hidden: [B, I, H])
        """
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3, "Inputs must be 3D tensors"
        assert process_weight.dim() == 2 and process_weight.shape[1] == process_weight.shape[0], "process_weight must be square [H, H]"

        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]

        # Ensure tensors are on the same device
        device = hidden_states.device
        process_weight = process_weight.to(device)

        # Allocate concatenated tensor [B, T+I, H]
        out_cat = torch.empty((B, T + I, H), dtype=hidden_states.dtype, device=device)

        # Launch Triton concatenate kernel: one program per batch
        _concatenate_sequences_kernel[(B,)](
            encoder_hidden_states, hidden_states, out_cat,
            B, T, I, H,
            *encoder_hidden_states.stride(), *hidden_states.stride(), *out_cat.stride(),
            BLOCK_H=128,
            num_warps=1, num_stages=1
        )

        # GEMM with torch for robustness: out_cat [B, T+I, H] @ process_weight.T [H, H] -> [B, T+I, H]
        # process_weight.T is [H, H]
        processed = torch.matmul(out_cat, process_weight.transpose(0, 1))

        # Allocate outputs
        processed_encoder = torch.empty((B, T, H), dtype=processed.dtype, device=device)
        processed_hidden = torch.empty((B, I, H), dtype=processed.dtype, device=device)

        # Launch Triton split kernel: one program per batch
        _split_streams_kernel[(B,)](
            processed, processed_encoder, processed_hidden,
            B, T, I, H,
            *processed.stride(), *processed_encoder.stride(), *processed_hidden.stride(),
            BLOCK_H=128,
            num_warps=1, num_stages=1
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
