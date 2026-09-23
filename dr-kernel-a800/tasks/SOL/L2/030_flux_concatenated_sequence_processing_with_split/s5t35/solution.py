import torch
import triton
import triton.language as tl


@triton.jit
def concat_kernel(
    encoder_ptr,      # *f32, [B, Stext, H]
    hidden_ptr,       # *f32, [B, Simg, H]
    out_ptr,          # *f32, [B, T, H], T = Stext + Simg
    B: tl.constexpr,
    Stext: tl.constexpr,
    Simg: tl.constexpr,
    H: tl.constexpr,
    stride_e_b, stride_e_t, stride_e_h,
    stride_h_b, stride_h_s, stride_h_h,
    stride_o_b, stride_o_t, stride_o_h,
):
    # 2D grid over (B, T). Copy rows from encoder or hidden into out.
    b = tl.program_id(0)
    t = tl.program_id(1)
    if (b >= B) or (t >= T):
        return
    src_ptr = None
    if t < Stext:
        src_ptr = encoder_ptr + b * stride_e_b + t * stride_e_t
    else:
        src_t = t - Stext
        src_ptr = hidden_ptr + b * stride_h_b + src_t * stride_h_s
    dst_ptr = out_ptr + b * stride_o_b + t * stride_o_t
    # Copy H dimension
    for i in range(0, H):
        val = tl.load(src_ptr + i * stride_e_h if src_ptr is not None else 0)
        tl.store(dst_ptr + i * stride_o_h, val)


@triton.jit
def batched_gemv_tiled_kernel(
    concat_ptr,       # *f32, [B, T, H_in]
    weightT_ptr,      # *f32, [H_in, H_out] where weightT = process_weight.T
    out_ptr,          # *f32, [B, T, H_out]
    B, T, H_in, H_out,
    stride_c_b, stride_c_t, stride_c_h,
    stride_w_k, stride_w_n,
    stride_o_b, stride_o_t, stride_o_h,
    BLOCK_K: tl.constexpr,  # tile along input dimension (H_in)
    BLOCK_N: tl.constexpr,  # tile along output dimension (H_out)
):
    # Grid: (B*T, ceil_div(H_out, BLOCK_N))
    pid_bt = tl.program_id(0)
    pid_n  = tl.program_id(1)

    b = pid_bt // T
    t = pid_bt % T
    if (b >= B) or (t >= T):
        return

    # Output tile offsets
    n_start = pid_n * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    mask_n = n_offsets < H_out

    # Accumulator for this (b, t) across BLOCK_N outputs
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Loop over input dimension in chunks of BLOCK_K
    k0 = 0
    while k0 < H_in:
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < H_in

        # Load input chunk: concat[b, t, k_offsets] -> [BLOCK_K]
        in_ptrs = concat_ptr + b * stride_c_b + t * stride_c_t + k_offsets * stride_c_h
        in_chunk = tl.load(in_ptrs, mask=mask_k, other=0.0)  # [BLOCK_K]

        # Load weight tile: weightT[k_offsets, n_offsets] -> [BLOCK_K, BLOCK_N]
        w_ptrs = weightT_ptr + k_offsets[:, None] * stride_w_k + n_offsets[None, :] * stride_w_n
        w_tile = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)  # [BLOCK_K, BLOCK_N]

        # Accumulate: acc += sum over k of (w_tile[k, :] * in_chunk[k])
        # Triton supports elementwise multiply and tl.sum over a given axis.
        acc += tl.sum(w_tile * in_chunk[:, None], axis=0)

        k0 += BLOCK_K

    # Store accumulated results
    out_ptrs = out_ptr + b * stride_o_b + t * stride_o_t + n_offsets * stride_o_h
    tl.store(out_ptrs, acc, mask=mask_n)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of run:
        - Concatenates encoder_hidden_states and hidden_states along sequence dimension using Triton.
        - Applies the linear projection using a batched GEMV Triton kernel tiled over output and input dims.
        - Splits back into encoder and image streams via torch slicing (no computation).
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA for Triton"

        # Ensure contiguous and dtype float32
        dtype = torch.float32
        B = hidden_states.shape[0]
        Stext = encoder_hidden_states.shape[1]
        Simg = hidden_states.shape[1]
        H = hidden_states.shape[2]
        T = Stext + Simg

        # Make inputs contiguous and cast to float32
        encoder = encoder_hidden_states.contiguous().to(dtype)
        hidden = hidden_states.contiguous().to(dtype)
        weight_T = process_weight.t().contiguous().to(dtype)  # [H, H]

        # Allocate concatenated buffer
        concatenated = torch.empty((B, T, H), device=hidden.device, dtype=dtype)

        # Launch concat kernel: grid = (B, T)
        grid_concat = (B, T)
        concat_kernel[grid_concat](
            encoder, hidden, concatenated,
            B, Stext, Simg, H,
            encoder.stride(0), encoder.stride(1), encoder.stride(2),
            hidden.stride(0), hidden.stride(1), hidden.stride(2),
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
        )

        # Prepare output tensor
        out = torch.empty((B, T, H), device=hidden.device, dtype=dtype)

        # Strides for GEMV
        c_b, c_t, c_h = concatenated.stride(0), concatenated.stride(1), concatenated.stride(2)
        w_k, w_n = weight_T.stride(0), weight_T.stride(1)
        o_b, o_t, o_h = out.stride(0), out.stride(1), out.stride(2)

        # Tiling parameters
        BLOCK_N = 128
        BLOCK_K = 128

        # Grid over (B*T, tiles of H_out)
        grid_gemm = (B * T, triton.cdiv(H, BLOCK_N))

        # Launch batched GEMV tiled kernel
        batched_gemv_tiled_kernel[grid_gemm](
            concatenated, weight_T, out,
            B, T, H, H,  # H_in == H_out in this setup
            c_b, c_t, c_h,
            w_k, w_n,
            o_b, o_t, o_h,
            BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N,
            num_warps=4, num_stages=2,
        )

        # Split back into encoder and hidden streams using torch slicing (metadata ops, no data work)
        processed_encoder = out[:, :Stext, :]
        processed_hidden = out[:, Stext:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
