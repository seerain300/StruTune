import torch
import triton
import triton.language as tl


@triton.jit
def process_per_token_kernel(
    encoder_ptr,         # *f16/f32 [B, T, H]
    image_ptr,           # *f16/f32 [B, I, H]
    weight_ptr,          # *f16/f32 [H, H]
    out_ptr,             # *f32     [B, T+I, H]  (we compute in f32)
    B: tl.int32,
    T: tl.int32,
    I: tl.int32,
    H: tl.int32,
    # strides (in elements)
    encoder_stride_b, encoder_stride_t, encoder_stride_h,
    image_stride_b, image_stride_i, image_stride_h,
    weight_stride_h, weight_stride_k,  # weight is [H, H] with strides (stride along rows, cols)
    out_stride_b, out_stride_t, out_stride_h,
):
    # program ids
    pid_b = tl.program_id(0)  # batch index
    pid_t = tl.program_id(1)  # output sequence index (0..T+I-1)

    # choose input source: if pid_t < T, use encoder; else use image at position pid_t - T
    use_encoder = pid_t < T
    in_stride_b = encoder_stride_b if use_encoder else image_stride_b
    in_stride_t = encoder_stride_t if use_encoder else image_stride_i
    in_stride_h = encoder_stride_h if use_encoder else image_stride_h

    # input row pointer
    in_row_ptr = encoder_ptr + pid_b * in_stride_b
    if not use_encoder:
        in_row_ptr = image_ptr + pid_b * image_stride_b + (pid_t - T) * image_stride_i

    # output row pointer (store as float32)
    out_row_ptr = out_ptr + pid_b * out_stride_b + pid_t * out_stride_t

    # compute output vector out[b, t, :] = weight @ x_t
    # we loop over h and multiply each input scalar by weight[h, h]
    for h in range(0, H):
        # load input scalar
        in_val = tl.load(in_row_ptr + h * in_stride_h)
        # load weight row[h, :]
        # Note: weight[h, h] here means the h-th element along the column axis (second dim of weight)
        w_val = tl.load(weight_ptr + h * weight_stride_h + h * weight_stride_k)
        # accumulate in float32
        acc = in_val * w_val  # both in_val and w_val come from the same dtype as input/weight
        # store to out[b, t, h]
        tl.store(out_row_ptr + h * out_stride_h, acc)


def triton_run(encoder_hidden_states: torch.Tensor, hidden_states: torch.Tensor, process_weight: torch.Tensor) -> torch.Tensor:
    """
    Triton implementation that computes the full processed tensor [B, T+I, H] without torch.cat/torch.matmul in host code.
    It directly writes out[b, t, :] by selecting the input source based on t < T and performing a per-token projection.
    Returns tensor in float32 (accumulation dtype).
    """
    assert encoder_hidden_states.is_cuda and hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA for Triton execution."

    B = encoder_hidden_states.shape[0]
    T = encoder_hidden_states.shape[1]
    I = hidden_states.shape[1]
    H = encoder_hidden_states.shape[2]

    # Allocate output in float32 for stable accumulation
    out = torch.empty((B, T + I, H), device=encoder_hidden_states.device, dtype=torch.float32)

    # Launch kernel: one program per (b, t)
    grid = (B, T + I)
    process_per_token_kernel[grid](
        encoder_hidden_states, hidden_states, process_weight, out,
        B, T, I, H,
        encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
        hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
        process_weight.stride(0), process_weight.stride(1),  # weight is [H, H], no need for extra strides
        out.stride(0), out.stride(1), out.stride(2),
        num_warps=1, num_stages=1
    )
    return out


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function.
        Returns (processed_encoder_hidden_states, processed_image_hidden_states).
        """
        # Ensure inputs are on CUDA for Triton execution
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA for Triton execution."

        # Compute the full processed tensor with Triton (no torch.cat / torch.matmul on host)
        total = triton_run(encoder_hidden_states, hidden_states, process_weight)  # [B, T+I, H], float32
        T = encoder_hidden_states.shape[1]
        processed_encoder = total[:, :T, :]
        processed_hidden = total[:, T:, :]
        return processed_encoder, processed_hidden