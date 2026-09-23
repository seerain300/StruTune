import torch
import triton
import triton.language as tl


@triton.jit
def concat_rows_to_A_kernel(
    enc_ptr,       # [B, T, H]
    img_ptr,       # [B, I, H]
    A_ptr,         # [M, H], M = B*(T+I)
    B, T, I, H,    # dims
    stride_b_e, stride_t_e, stride_h_e,  # encoder strides
    stride_b_i, stride_i_i, stride_h_i,  # image strides
    stride_m_a, stride_h_a,               # A strides
):
    # One program per output row m in [0, M)
    m = tl.program_id(0)

    S = T + I
    b = m // S
    seq = m % S
    is_img = seq >= T
    src_row = seq - (0 if is_img else T)

    # Compute pointers
    if is_img:
        ptr = img_ptr + b * stride_b_i + src_row * stride_i_i
    else:
        ptr = enc_ptr + b * stride_b_e + src_row * stride_t_e

    # Copy the full H-length row to A[m, :]
    offs = tl.arange(0, H)  # vector of column indices
    vals = tl.load(ptr + offs * stride_h, mask=(offs < H), other=0.0)
    A_row_ptr = A_ptr + m * stride_m_a
    tl.store(A_row_ptr + offs * stride_h_a, vals, mask=(offs < H))


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        # Ensure inputs are contiguous
        enc = encoder_hidden_states.contiguous()   # [B, T, H]
        img = hidden_states.contiguous()           # [B, I, H]
        weight = process_weight.contiguous()       # [H, H]

        B, T, H = enc.shape
        I = img.shape[1]
        M = B * (T + I)

        # Allocate A: [M, H]
        A = torch.empty((M, H), dtype=enc.dtype, device=enc.device)

        # Step 1: Concatenate rows into A using Triton
        grid_concat = (M,)
        concat_rows_to_A_kernel[grid_concat](
            enc, img, A,
            B, T, I, H,
            enc.stride(0), enc.stride(1), enc.stride(2),
            img.stride(0), img.stride(1), img.stride(2),
            A.stride(0), A.stride(1),
            num_warps=4, num_stages=2,
        )

        # Step 2: Compute processed = A @ weight.T using cuBLAS via torch.matmul
        # weight.T: [H, H]
        processed = torch.matmul(A, weight.transpose(0, 1))

        # Step 3: Split processed into encoder and hidden outputs without torch.cat
        # processed has shape [M, H], M = B*(T+I)
        processed_encoder = processed[:B * T].view(B, T, H)
        processed_hidden = processed[B * T:].view(B, I, H)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
