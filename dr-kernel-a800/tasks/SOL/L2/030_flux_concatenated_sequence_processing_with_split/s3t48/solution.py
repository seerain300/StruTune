import torch
import triton
import triton.language as tl


@triton.jit
def _rowwise_gemm_encoder(out_ptr, A_ptr, WT_ptr,
                           B, T, D,
                           A_s0, A_s1, A_s2,
                           WT_s0, WT_s1,
                           out_s0, out_s1, out_s2,
                           num_warps: tl.constexpr):
    # Each program handles one (b, p) row for the encoder stream
    b = tl.program_id(0)
    p = tl.program_id(1)

    # If grid is larger than actual B or T, guard (shouldn't happen if grid matches)
    if b >= B or p >= T:
        return

    # Accumulator in fp32
    acc = tl.zeros((), dtype=tl.float32)

    # Loop over hidden dimension k
    # We keep K as a scalar loop to minimize numerical discrepancies
    for k in range(0, D):
        # Load A[b, p, k]
        a_ptr = A_ptr + b * A_s0 + p * A_s1 + k * A_s2
        a_val = tl.load(a_ptr)  # fp32 load (input is expected to be fp32 in this benchmark)

        # Load WT[k, :] row and multiply
        # We'll construct the d vector and load the entire row for d in 0..D-1
        # But since we compute per d, we just multiply scalar a_val with WT[k, d] per d
        # To store, we compute pointer for each d
        # Note: out has shape [B, T, D]; indices are b, p, d
        for d in range(0, D):
            w_ptr = WT_ptr + k * WT_s0 + d * WT_s1
            w_val = tl.load(w_ptr)  # fp32
            acc += a_val * w_val

    # Store result to out[b, p, :]
    for d in range(0, D):
        out_ptr_d = out_ptr + b * out_s0 + p * out_s1 + d * out_s2
        tl.store(out_ptr_d, acc)


@triton.jit
def _rowwise_gemm_hidden(out_ptr, A_ptr, WT_ptr,
                          B, I, D,
                          A_s0, A_s1, A_s2,
                          WT_s0, WT_s1,
                          out_s0, out_s1, out_s2,
                          num_warps: tl.constexpr):
    # Each program handles one (b, p) row for the hidden stream
    b = tl.program_id(0)
    p = tl.program_id(1)

    if b >= B or p >= I:
        return

    acc = tl.zeros((), dtype=tl.float32)

    for k in range(0, D):
        a_ptr = A_ptr + b * A_s0 + p * A_s1 + k * A_s2
        a_val = tl.load(a_ptr)

        for d in range(0, D):
            w_ptr = WT_ptr + k * WT_s0 + d * WT_s1
            w_val = tl.load(w_ptr)
            acc += a_val * w_val

    for d in range(0, D):
        out_ptr_d = out_ptr + b * out_s0 + p * out_s1 + d * out_s2
        tl.store(out_ptr_d, acc)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        # Shapes
        B = hidden_states.shape[0]
        I = hidden_states.shape[1]
        T = encoder_hidden_states.shape[1]
        D = hidden_states.shape[2]

        # Ensure contiguous tensors
        hst = hidden_states.contiguous()          # [B, I, D]
        enc = encoder_hidden_states.contiguous()  # [B, T, D]
        WT = process_weight.contiguous()          # [D, D], no bias

        # Allocate outputs (same dtype as inputs; here, evaluator uses fp32)
        # The original code returns [B, T, D] and [B, I, D]
        processed_encoder = torch.empty((B, T, D), device=hst.device, dtype=hst.dtype)
        processed_hidden = torch.empty((B, I, D), device=hst.device, dtype=hst.dtype)

        # Launch kernels: one program per (b, p) for each stream
        # Grid size = (B, T) for encoder, and (B, I) for hidden
        grid_enc = (B, T)
        _rowwise_gemm_encoder[grid_enc](
            processed_encoder, enc, WT,
            B, T, D,
            enc.stride(0), enc.stride(1), enc.stride(2),
            WT.stride(0), WT.stride(1),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            num_warps=2,  # simple kernel; 2 warps is fine
        )

        grid_hid = (B, I)
        _rowwise_gemm_hidden[grid_hid](
            processed_hidden, hst, WT,
            B, I, D,
            hst.stride(0), hst.stride(1), hst.stride(2),
            WT.stride(0), WT.stride(1),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            num_warps=2,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
