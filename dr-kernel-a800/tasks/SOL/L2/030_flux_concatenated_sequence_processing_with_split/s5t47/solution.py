import torch
import triton
import triton.language as tl


@triton.jit
def concat_kernel(
    encoder_ptr,       # *f32, [B, Stext, H_in]
    hidden_ptr,        # *f32, [B, Simg, H_in]
    out_ptr,           # *f32, [B, T, H_in], T = Stext + Simg
    B: tl.constexpr,
    Stext: tl.constexpr,
    Simg: tl.constexpr,
    H_in: tl.constexpr,
    stride_e_b: tl.constexpr,
    stride_e_t: tl.constexpr,
    stride_e_h: tl.constexpr,
    stride_h_b: tl.constexpr,
    stride_h_s: tl.constexpr,   # s = t - Stext
    stride_h_h: tl.constexpr,
    stride_o_b: tl.constexpr,
    stride_o_t: tl.constexpr,
    stride_o_h: tl.constexpr,
):
    # Grid: (B, T)
    b = tl.program_id(0)
    t = tl.program_id(1)
    if (b >= B) or (t >= T):
        return

    src_row = None
    if t < Stext:
        src_row = encoder_ptr + b * stride_e_b + t * stride_e_t
    else:
        s = t - Stext
        src_row = hidden_ptr + b * stride_h_b + s * stride_h_s

    dst_row = out_ptr + b * stride_o_b + t * stride_o_t

    # Copy H_in elements; H_in is constexpr so loop is unrolled.
    for i in range(0, H_in):
        val = tl.load(src_row + i * stride_e_h if src_row is not None else i * stride_h_h)
        tl.store(dst_row + i * stride_o_h, val)


@triton.jit
def batched_gemv_kernel_tiled(
    in_ptr,            # *f32, concatenated [B, T, H_in]
    weightT_ptr,       # *f32, process_weight.T [H_in, H_out]
    out_ptr,           # *f32, processed [B, T, H_out]
    B, T, H_in, H_out,
    stride_in_b, stride_in_t, stride_in_h,
    stride_w_k, stride_w_n,   # weight_T strides: dim-0 (k) is stride_w_k, dim-1 (n) is stride_w_n
    stride_out_b, stride_out_t, stride_out_h,
    BLOCK_K: tl.constexpr,   # tile size along input dimension
    BLOCK_N: tl.constexpr,   # tile size along output dimension (features)
):
    # Grid: (B, T, n_tiles), where n_tiles = ceil_div(H_out, BLOCK_N)
    b = tl.program_id(0)
    t = tl.program_id(1)
    n_tile = tl.program_id(2)
    if (b >= B) or (t >= T) or (n_tile >= (H_out + BLOCK_N - 1) // BLOCK_N):
        return

    # Output tile offsets
    n_start = n_tile * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    mask_n = n_offsets < H_out

    # Accumulator for this (b, t) over BLOCK_N outputs
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Loop over input dimension in chunks of BLOCK_K
    k0 = 0
    while k0 < H_in:
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < H_in

        # Load input vector chunk: in[b, t, k_offsets] -> shape [BLOCK_K]
        in_ptrs = in_ptr + b * stride_in_b + t * stride_in_t + k_offsets * stride_in_h
        in_chunk = tl.load(in_ptrs, mask=mask_k, other=0.0)

        # Load weight_T tile: [BLOCK_K, BLOCK_N]
        # Address: weightT_ptr + k_offsets[:, None]*stride_w_k + n_offsets[None, :]*stride_w_n
        w_ptrs = weightT_ptr + (k_offsets[:, None] * stride_w_k) + (n_offsets[None, :] * stride_w_n)
        w_tile = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # Accumulate: [BLOCK_N] += [BLOCK_K, BLOCK_N] @ [BLOCK_K]
        # Note: tl.dot expects two matrices; here we use broadcasting to multiply elementwise and reduce.
        # However, Triton's tl.dot requires matrix inputs. Better to explicitly multiply and reduce:
        # acc += sum_k w_tile[k, :] * in_chunk[k]
        # Implement via tl.sum over axis=0
        acc += tl.sum(w_tile * in_chunk[:, None], axis=0)

        k0 += BLOCK_K

    # Store results
    out_ptrs = out_ptr + b * stride_out_b + t * stride_out_t + n_offsets * stride_out_h
    tl.store(out_ptrs, acc, mask=mask_n)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor):
        """
        Triton-optimized version:
        - Concatenates encoder_hidden_states and hidden_states along sequence dimension in Triton.
        - Applies linear projection using a Triton GEMV kernel with tiling and vectorization: out[b, t, :] = process_weight.T @ concat[b, t, :].
        - Splits processed tensor back into separate encoder and image streams.
        """
        # Ensure tensors are CUDA and float32
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All tensors must be on CUDA for Triton."
        assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, "Use float32 tensors."

        B = hidden_states.shape[0]
        Stext = encoder_hidden_states.shape[1]
        Simg = hidden_states.shape[1]
        H_in = hidden_states.shape[2]
        H_out = process_weight.shape[1]  # process_weight: [H_in, H_out]
        T = Stext + Simg

        # Ensure contiguous
        encoder = encoder_hidden_states.contiguous()
        hidden = hidden_states.contiguous()
        weight_T = process_weight.t().contiguous()

        # Allocate concatenated
        concatenated = torch.empty((B, T, H_in), device=hidden.device, dtype=hidden.dtype)

        # Launch concat kernel
        grid_concat = (B, T)
        stride_e_b, stride_e_t, stride_e_h = encoder.stride()
        stride_h_b, stride_h_s, stride_h_h = hidden.stride()
        stride_o_b, stride_o_t, stride_o_h = concatenated.stride()
        concat_kernel[grid_concat](
            encoder, hidden, concatenated,
            B, Stext, Simg, H_in,
            stride_e_b, stride_e_t, stride_e_h,
            stride_h_b, stride_h_s, stride_h_h,
            stride_o_b, stride_o_t, stride_o_h,
            num_warps=1, num_stages=1,
        )

        # Allocate output
        processed = torch.empty((B, T, H_out), device=hidden.device, dtype=hidden.dtype)

        # Launch tiled GEMV kernel
        BLOCK_K = 128
        BLOCK_N = 128
        n_tiles = (H_out + BLOCK_N - 1) // BLOCK_N
        grid_matvec = (B, T, n_tiles)

        stride_in_b, stride_in_t, stride_in_h = concatenated.stride()
        stride_w_k, stride_w_n = weight_T.stride()
        stride_out_b, stride_out_t, stride_out_h = processed.stride()

        batched_gemv_kernel_tiled[grid_matvec](
            concatenated, weight_T, processed,
            B, T, H_in, H_out,
            stride_in_b, stride_in_t, stride_in_h,
            stride_w_k, stride_w_n,
            stride_out_b, stride_out_t, stride_out_h,
            BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=2,
        )

        # Split back into streams
        processed_encoder = processed[:, :Stext, :]
        processed_hidden = processed[:, Stext:, :]
        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
