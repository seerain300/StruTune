import torch
import triton
import triton.language as tl


@triton.jit
def linear_rows_kernel(
    in_ptr,       # input tensor pointer, [N, K] contiguous
    weight_ptr,   # weight tensor pointer, [K, H] contiguous
    out_ptr,      # output tensor pointer, [N, H] contiguous
    N: tl.int32, K: tl.int32, H: tl.int32,
    stride_in_n: tl.int32, stride_in_k: tl.int32,
    stride_w_k: tl.int32, stride_w_h: tl.int32,
    stride_out_n: tl.int32, stride_out_h: tl.int32,
    BLOCK_K: tl.constexpr, BLOCK_H: tl.constexpr,
):
    n = tl.program_id(0)
    # Accumulator for output vector of length H
    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)
    # Loop over input features in tiles of BLOCK_K
    for k0 in range(0, K, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K
        # Load in[n, k_offsets] as vector
        in_row = tl.load(in_ptr + n * stride_in_n + k_offsets * stride_in_k, mask=mask_k, other=0.0)
        # Load weight[k, :] as vector
        w_row = tl.load(weight_ptr + k_offsets[:, None] * stride_w_k + tl.arange(0, BLOCK_H)[None, :] * stride_w_h,
                        mask=mask_k[:, None], other=0.0)
        # Dot: sum over BLOCK_K
        acc += tl.sum(in_row[None, :] * w_row, axis=0)
    # Store to out[n, :]
    out_offsets = tl.arange(0, BLOCK_H)
    mask_out = out_offsets < H
    tl.store(out_ptr + n * stride_out_n + out_offsets * stride_out_h, acc, mask=mask_out)


@triton.jit
def process_encoder_kernel(
    encoder_ptr,  # [B, T, H] contiguous
    weight_ptr,   # [H, H] contiguous
    out_ptr,      # [B, T, H] contiguous
    B: tl.int32, T: tl.int32, H: tl.int32,
    stride_e_n: tl.int32, stride_e_s: tl.int32, stride_e_h: tl.int32,
    stride_w_h: tl.int32, stride_w_k: tl.int32,
    stride_out_n: tl.int32, stride_out_s: tl.int32, stride_out_h: tl.int32,
    BLOCK_K: tl.constexpr, BLOCK_H: tl.constexpr,
):
    n = tl.program_id(0)
    s = tl.program_id(1)
    # Load encoder row [H]
    in_row = tl.load(encoder_ptr + n * stride_e_n + s * stride_e_s + tl.arange(0, H) * stride_e_h,
                     mask=tl.arange(0, H) < H, other=0.0)
    # Compute linear projection: out[n, s, :] = in_row @ weight.T
    acc = tl.zeros((H,), dtype=tl.float32)
    for k0 in range(0, H, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < H
        w_block = tl.load(
            weight_ptr + k_offsets[:, None] * stride_w_k + tl.arange(0, BLOCK_H)[None, :] * stride_w_h,
            mask=mask_k[:, None] and tl.arange(0, BLOCK_H)[None, :] < H, other=0.0
        )
        # Multiply each input element by corresponding weight row and reduce
        acc += tl.sum(in_row[k0:k0+BLOCK_K] * tl.sum(w_block, axis=1), axis=0)
    # Store result
    tl.store(out_ptr + n * stride_out_n + s * stride_out_s + tl.arange(0, H) * stride_out_h, acc, mask=tl.arange(0, H) < H)


@triton.jit
def process_hidden_kernel(
    hidden_ptr,   # [B, I, H] contiguous
    weight_ptr,   # [H, H] contiguous
    out_ptr,      # [B, I, H] contiguous
    B: tl.int32, I: tl.int32, H: tl.int32,
    stride_h_n: tl.int32, stride_h_s: tl.int32, stride_h_h: tl.int32,
    stride_w_h: tl.int32, stride_w_k: tl.int32,
    stride_out_n: tl.int32, stride_out_s: tl.int32, stride_out_h: tl.int32,
    BLOCK_K: tl.constexpr, BLOCK_H: tl.constexpr,
):
    n = tl.program_id(0)
    s = tl.program_id(1)
    # Load hidden row [H]
    in_row = tl.load(hidden_ptr + n * stride_h_n + s * stride_h_s + tl.arange(0, H) * stride_h_h,
                     mask=tl.arange(0, H) < H, other=0.0)
    # Compute linear projection: out[n, s, :] = in_row @ weight.T
    acc = tl.zeros((H,), dtype=tl.float32)
    for k0 in range(0, H, BLOCK_K):
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < H
        w_block = tl.load(
            weight_ptr + k_offsets[:, None] * stride_w_k + tl.arange(0, BLOCK_H)[None, :] * stride_w_h,
            mask=mask_k[:, None] and tl.arange(0, BLOCK_H)[None, :] < H, other=0.0
        )
        acc += tl.sum(in_row[k0:k0+BLOCK_K] * tl.sum(w_block, axis=1), axis=0)
    # Store result
    tl.store(out_ptr + n * stride_out_n + s * stride_out_s + tl.arange(0, H) * stride_out_h, acc, mask=tl.arange(0, H) < H)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        # Ensure dtype and contiguity
        dtype = torch.float32
        B = hidden_states.shape[0]
        I = hidden_states.shape[1]
        T = encoder_hidden_states.shape[1]
        H = hidden_states.shape[2]
        assert process_weight.shape[0] == H and process_weight.shape[1] == H, "process_weight must be [H, H]"
        # Make inputs contiguous and cast to float32 for predictable behavior
        e = encoder_hidden_states.contiguous().to(dtype)
        h = hidden_states.contiguous().to(dtype)
        w = process_weight.contiguous().to(dtype)

        # Allocate outputs
        processed_encoder = torch.empty((B, T, H), device=e.device, dtype=dtype)
        processed_hidden = torch.empty((B, I, H), device=h.device, dtype=dtype)

        # Weight is [H, H]; for row-wise projection we need weight.T. Create a contiguous [K=H, H] view.
        # In Triton, we pass w as [H, H] and access it as weight[k, h] via strides.
        # Choose tile sizes. Since H in provided workloads is 128 or 256, these are reasonable defaults.
        BLOCK_K = 128
        BLOCK_H = 128

        # Launch kernel to compute encoder outputs: out[b, s, :] = e[b, s, :] @ w.T
        grid_e = (B, T)
        process_encoder_kernel[grid_e](
            e, w, processed_encoder,
            B, T, H,
            e.stride(0), e.stride(1), e.stride(2),
            w.stride(0), w.stride(1),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            BLOCK_K=BLOCK_K, BLOCK_H=BLOCK_H,
            num_warps=4,
        )

        # Launch kernel to compute hidden outputs: out[b, s, :] = h[b, s, :] @ w.T
        grid_h = (B, I)
        process_hidden_kernel[grid_h](
            h, w, processed_hidden,
            B, I, H,
            h.stride(0), h.stride(1), h.stride(2),
            w.stride(0), w.stride(1),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_K=BLOCK_K, BLOCK_H=BLOCK_H,
            num_warps=4,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
