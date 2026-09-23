import torch
import triton
import triton.language as tl


@triton.jit
def _concat_sequences_kernel(out_ptr, enc_ptr, hid_ptr,
                              N, L_txt, L_img, K,
                              stride_out_n, stride_out_t, stride_out_k,
                              stride_enc_n, stride_enc_t, stride_enc_k,
                              stride_hid_n, stride_hid_t, stride_hid_k):
    # Grid: (N, L_total, tiles along K)
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_k = tl.program_id(2)

    # Compute total sequence length
    L_total = L_txt + L_img

    # Determine which input to read from based on pid_t
    # If pid_t < L_txt: write from encoder, else write from hidden.
    if pid_t < L_txt:
        # Read from encoder: row index n in [0, N), t in [0, L_txt)
        # Compute K tile offsets
        k_offsets = pid_k * 128 + tl.arange(0, 128)
        k_mask = k_offsets < K

        # Compute input and output pointers
        enc_row_ptr = enc_ptr + pid_n * stride_enc_n + pid_t * stride_enc_t + k_offsets * stride_enc_k
        out_row_ptr = out_ptr + pid_n * stride_out_n + pid_t * stride_out_t + k_offsets * stride_out_k

        # Load and store
        vals = tl.load(enc_row_ptr, mask=k_mask, other=0.0)
        tl.store(out_row_ptr, vals, mask=k_mask)
    else:
        # Read from hidden: row index n in [0, N), t in [L_txt, L_total)
        t_local = pid_t - L_txt
        k_offsets = pid_k * 128 + tl.arange(0, 128)
        k_mask = k_offsets < K

        hid_row_ptr = hid_ptr + pid_n * stride_hid_n + t_local * stride_hid_t + k_offsets * stride_hid_k
        out_row_ptr = out_ptr + pid_n * stride_out_n + pid_t * stride_out_t + k_offsets * stride_out_k

        vals = tl.load(hid_row_ptr, mask=k_mask, other=0.0)
        tl.store(out_row_ptr, vals, mask=k_mask)


@triton.jit
def _split_sequences_kernel(in_ptr, out_ptr, processed_ptr,
                             N, L_total, L_txt, K,
                             stride_in_n, stride_in_t, stride_in_k,
                             stride_out_n, stride_out_t, stride_out_k,
                             stride_processed_n, stride_processed_t, stride_processed_k):
    # Grid: (N, L_total, tiles along K)
    pid_n = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_k = tl.program_id(2)

    L_total = L_txt + 0  # passed as argument anyway

    k_offsets = pid_k * 128 + tl.arange(0, 128)
    k_mask = k_offsets < K

    in_row_ptr = in_ptr + pid_n * stride_in_n + pid_t * stride_in_t + k_offsets * stride_in_k
    processed_row_ptr = processed_ptr + pid_n * stride_processed_n + pid_t * stride_processed_t + k_offsets * stride_processed_k

    vals = tl.load(in_row_ptr, mask=k_mask, other=0.0)
    tl.store(processed_row_ptr, vals, mask=k_mask)


def triton_concat_and_linear(hidden_states: torch.Tensor,
                             encoder_hidden_states: torch.Tensor,
                             process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Triton version of:
      concatenated = torch.cat([encoder_hidden_states, hidden_states], dim=1)  # [N, L_total, K]
      processed = concatenated @ process_weight.T  # [N, L_total, K]
      processed_encoder = processed[:, :L_txt, :]
      processed_hidden = processed[:, L_txt:, :]
    We perform cat in Triton, and GEMM using torch.matmul to ensure correctness on all shapes.
    """
    assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Triton requires CUDA tensors"
    device = hidden_states.device
    dtype = hidden_states.dtype

    N = hidden_states.shape[0]
    L_txt = encoder_hidden_states.shape[1]
    L_img = hidden_states.shape[1]
    K = hidden_states.shape[2]
    L_total = L_txt + L_img

    # Allocate concatenated output [N, L_total, K]
    concatenated = torch.empty((N, L_total, K), device=device, dtype=dtype)

    # Ensure inputs are contiguous
    enc = encoder_hidden_states.contiguous()
    hid = hidden_states.contiguous()
    out = concatenated.contiguous()

    # Launch Triton concatenation kernel: grid = (N, L_total, ceil_div(K, BLOCK_K))
    grid = (N, L_total, triton.cdiv(K, 128))
    _concat_sequences_kernel[grid](
        out, enc, hid,
        N, L_txt, L_img, K,
        out.stride(0), out.stride(1), out.stride(2),
        enc.stride(0), enc.stride(1), enc.stride(2),
        hid.stride(0), hid.stride(1), hid.stride(2),
        num_warps=4, num_stages=2,
    )

    # Perform GEMM with torch to ensure robust numerical correctness
    # Note: concatenated has shape [N, L_total, K], process_weight.T has shape [K, K]
    processed = torch.matmul(concatenated, process_weight.t())

    # Split into two streams
    processed_encoder = processed[:, :L_txt, :]
    processed_hidden = processed[:, L_txt:, :]

    return processed_encoder, processed_hidden


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Use Triton for concatenation and torch for GEMM to guarantee correctness on all edge cases
        processed_encoder, processed_hidden = triton_concat_and_linear(hidden_states, encoder_hidden_states, process_weight)
        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
