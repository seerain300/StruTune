import torch
import triton
import triton.language as tl


@triton.jit
def concat_linear_split_kernel(
    encoder_hidden_states, hidden_states, process_weight_T, processed_concat,
    B, T, I, H,
    stride_e_n, stride_e_s, stride_e_h,
    stride_h_n, stride_h_s, stride_h_h,
    stride_w_k, stride_w_h,  # process_weight_T is [H, H], strides for K and H
    stride_out_n, stride_out_s, stride_out_h,
    total_seq,
    BLOCK_H: tl.constexpr,
):
    # Grid: (B, total_seq, 1) -> each program handles one (n, s)
    n = tl.program_id(0)
    s = tl.program_id(1)
    # Compute output vector for all H (we assume BLOCK_H >= H in the host code)
    h_offsets = tl.arange(0, BLOCK_H)
    mask_h = h_offsets < H

    # Determine input source: if s < T, take from encoder; else from hidden at index s - T
    use_encoder = s < T

    # Accumulator for output vector
    out = tl.zeros([BLOCK_H], dtype=tl.float32)

    # Loop over K (input features) in tiles of BLOCK_K; for correctness, we keep it simple
    # and iterate K from 0 to H (since process_weight_T is [H, H]).
    # We'll use a while loop to avoid depending on a specific BLOCK_K.
    k = 0
    while k < H:
        # Load input row: either encoder[n, s, k] or hidden[n, s - T, k]
        if use_encoder:
            # address = n*stride_e_n + s*stride_e_s + k*stride_e_h
            input_val = tl.load(
                encoder_hidden_states + n * stride_e_n + s * stride_e_s + k * stride_e_h,
                mask=True,  # scalar, always in bounds
                other=0.0
            )
        else:
            # address = n*stride_h_n + (s - T)*stride_h_s + k*stride_h_h
            input_val = tl.load(
                hidden_states + n * stride_h_n + (s - T) * stride_h_s + k * stride_h_h,
                mask=True,
                other=0.0
            )

        # Load weight row for k: process_weight_T[k, h] with strides (k over H, h over K)
        # For each h in BLOCK_H, load weight[k, h] and accumulate
        h_vec = h_offsets
        # weight[k, h] address = k*stride_w_k + h_vec*stride_w_h
        w = tl.load(
            process_weight_T + k * stride_w_k + h_vec * stride_w_h,
            mask=mask_h,
            other=0.0
        )
        # Accumulate: out[h] += input_val * w[h]
        out += input_val * w
        k += 1

    # Store the result to processed_concat[n, s, h] for h in [0, H)
    tl.store(
        processed_concat + n * stride_out_n + s * stride_out_s + h_offsets * stride_out_h,
        out,
        mask=mask_h
    )


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        # Shapes
        B = hidden_states.shape[0]
        I = hidden_states.shape[1]
        T = encoder_hidden_states.shape[1]
        H = hidden_states.shape[2]
        assert encoder_hidden_states.shape[2] == H, "hidden_dim must match between inputs and weight"

        # Make tensors contiguous and float32 for predictable numerics
        encoder_hidden_states = encoder_hidden_states.contiguous().to(torch.float32)
        hidden_states = hidden_states.contiguous().to(torch.float32)
        process_weight_T = process_weight.t().contiguous().to(torch.float32)  # [H, H]

        total_seq = T + I
        # Allocate output [B, total_seq, H]
        processed_concat = torch.empty((B, total_seq, H), device=hidden_states.device, dtype=torch.float32)

        # Strides
        stride_e_n, stride_e_s, stride_e_h = encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2)
        stride_h_n, stride_h_s, stride_h_h = hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2)
        stride_w_k, stride_w_h = process_weight_T.stride(0), process_weight_T.stride(1)
        stride_out_n, stride_out_s, stride_out_h = processed_concat.stride(0), processed_concat.stride(1), processed_concat.stride(2)

        # Launch Triton kernel: grid over (B, total_seq, 1)
        # Use BLOCK_H as H (host ensures H is reasonable, e.g., 128/256). We set BLOCK_H=H for exact coverage.
        # Note: If H is very large, consider increasing BLOCK_H or splitting H across tiles.
        # For safety and correctness, we set BLOCK_H to the next power-of-two >= H or simply H.
        # Here we set BLOCK_H = H to avoid multiple tiles and keep math exact.
        # Triton accepts dynamic values for BLOCK_H in jitted kernel. We choose 128 or 256 when possible.
        # To keep it simple and correct, we choose BLOCK_H=H (torch scalar is fine as constexpr argument).
        # If H is not a constexpr friendly, pass a reasonable tile: choose 128 for H<=128, 256 for H<=256, else 128.

        # We will use BLOCK_H = H for exactness; Triton handles it as a runtime value here.
        BLOCK_H = H

        grid = (B, total_seq, 1)

        concat_linear_split_kernel[grid](
            encoder_hidden_states, hidden_states, process_weight_T, processed_concat,
            B, T, I, H,
            stride_e_n, stride_e_s, stride_e_h,
            stride_h_n, stride_h_s, stride_h_h,
            stride_w_k, stride_w_h,
            stride_out_n, stride_out_s, stride_out_h,
            total_seq,
            BLOCK_H=BLOCK_H,
            num_warps=4,
            num_stages=2,
        )

        # Split outputs
        processed_encoder = processed_concat[:, :T, :]
        processed_hidden = processed_concat[:, T:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
