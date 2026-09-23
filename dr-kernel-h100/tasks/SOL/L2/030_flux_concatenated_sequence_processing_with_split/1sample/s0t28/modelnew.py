import torch
import triton
import triton.language as tl


@triton.jit
def _process_concat_linear_per_row_kernel(
    enc_ptr,          # *f32, [B, T, H]
    hid_ptr,          # *f32, [B, I, H]
    weight_ptr,       # *f32, [H, H]
    enc_out_ptr,      # *f32, [B, T, H]
    hid_out_ptr,      # *f32, [B, I, H]
    T: tl.constexpr,  # int
    I: tl.constexpr,  # int
    H: tl.constexpr,  # int (hidden_dim)
):
    # One program per batch
    b = tl.program_id(0)

    # Loop over the combined sequence length
    L = T + I
    for s in range(L):
        # Determine source: encoder if s < T, else hidden at s - T
        is_encoder = s < T

        # Vector of column indices for hidden_dim
        h_vec = tl.arange(0, H)

        # Accumulator for output row
        out = tl.zeros((H,), dtype=tl.float32)

        # Multiply input row by process_weight (right-multiply: input_row @ W.T)
        if is_encoder:
            # Load input row: enc[b, s, :]
            # Pointer arithmetic: enc has strides (stride_b, stride_t, stride_h)
            # We'll assume inputs are contiguous in hidden_dim; otherwise use generic pointer math.
            # For simplicity and robustness, assume all tensors are contiguous.
            in_row_ptr = enc_ptr + b * H * T + s * H
            in_vals = tl.load(in_row_ptr + h_vec)  # [H]
            # Load weight row: weight[h, :] for h in [0, H)
            # weight is [H, H], row-major, so offset h * H + col
            for h in range(H):
                w_row_ptr = weight_ptr + h * H + h_vec
                w_row = tl.load(w_row_ptr)  # [H]
                out += in_vals[h] * w_row
        else:
            # Load input row: hid[b, s - T, :]
            in_row_ptr = hid_ptr + b * H * I + (s - T) * H
            in_vals = tl.load(in_row_ptr + h_vec)  # [H]
            for h in range(H):
                w_row_ptr = weight_ptr + h * H + h_vec
                w_row = tl.load(w_row_ptr)  # [H]
                out += in_vals[h] * w_row

        # Store result into appropriate output tensor
        if is_encoder:
            out_ptr = enc_out_ptr + b * T * H + s * H
            tl.store(out_ptr + h_vec, out)
        else:
            out_ptr = hid_out_ptr + b * I * H + (s - T) * H
            tl.store(out_ptr + h_vec, out)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-only implementation:
        - Concatenate along sequence dimension (logically handled by per-row kernel)
        - Apply linear projection (right-multiply by process_weight) per sequence row
        - Split back into encoder and hidden outputs
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA"
        assert hidden_states.shape[0] == encoder_hidden_states.shape[0], "Batch size must match"
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        assert process_weight.shape[0] == H and process_weight.shape[1] == H, "process_weight must be [H, H]"

        # Ensure contiguous tensors (robustness)
        enc = encoder_hidden_states.contiguous()
        hid = hidden_states.contiguous()
        weight = process_weight.contiguous()

        # Allocate outputs (float32 compute)
        encoder_out = torch.empty((B, T, H), dtype=torch.float32, device=hidden_states.device)
        hidden_out = torch.empty((B, I, H), dtype=torch.float32, device=hidden_states.device)

        # Launch Triton kernel: one program per batch
        grid = (B,)
        _process_concat_linear_per_row_kernel[grid](
            enc, hid, weight, encoder_out, hidden_out,
            T=T, I=I, H=H,
            num_warps=4, num_stages=2
        )

        return encoder_out, hidden_out