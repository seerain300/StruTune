import torch
import triton
import triton.language as tl


# Kernel 1: Concatenate along sequence dimension using Triton.
# Input:
#   encoder_ptr: [B, Stext, H]
#   hidden_ptr: [B, Simg, H]
# Output:
#   out_ptr: [B, T, H], T = Stext + Simg
# Each program handles a single (b, t) and copies from encoder or hidden.
@triton.jit
def concat_kernel_simple(
    encoder_ptr,      # *f32, [B, Stext, H]
    hidden_ptr,       # *f32, [B, Simg, H]
    out_ptr,          # *f32, [B, T, H]
    B: tl.constexpr,
    Stext: tl.constexpr,
    Simg: tl.constexpr,
    H: tl.constexpr,
    stride_e_b: tl.constexpr, stride_e_t: tl.constexpr, stride_e_h: tl.constexpr,
    stride_h_b: tl.constexpr, stride_h_t: tl.constexpr, stride_h_h: tl.constexpr,
    stride_o_b: tl.constexpr, stride_o_t: tl.constexpr, stride_o_h: tl.constexpr,
):
    b = tl.program_id(0)
    t = tl.program_id(1)
    if (b >= B) or (t >= (Stext + Simg)):
        return
    if t < Stext:
        src = encoder_ptr + b * stride_e_b + t * stride_e_t
    else:
        src = hidden_ptr + b * stride_h_b + (t - Stext) * stride_h_t
    dst = out_ptr + b * stride_o_b + t * stride_o_t
    for i in range(0, H):
        val = tl.load(src + i * stride_e_h)
        tl.store(dst + i * stride_o_h, val)


# Kernel 2: Batched GEMV in 2D tiles for robustness.
# Compute: out[b, t, n:n+BLOCK_N] = sum_k concat[b, t, k] * weight_T[k, n:n+BLOCK_N]
# Each program handles one (b, t) and a tile along the output dimension.
@triton.jit
def batched_gemv_kernel_tile(
    concat_ptr,       # *f32, [B, T, H_in]
    weightT_ptr,      # *f32, [H_in, H_out] = process_weight.T
    out_ptr,          # *f32, [B, T, H_out]
    B, T, H_in, H_out,
    stride_c_b, stride_c_t, stride_c_h,
    stride_w_k, stride_w_h,
    stride_o_b, stride_o_t, stride_o_h,
    BLOCK_N: tl.constexpr,
):
    b = tl.program_id(0)
    t = tl.program_id(1)
    n_block = tl.program_id(2)
    if (b >= B) or (t >= T):
        return
    n_start = n_block * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    mask_n = n_offsets < H_out

    # Accumulator for this tile
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Loop over input dimension in chunks of size BLOCK_K to accumulate dot products
    BLOCK_K = 64
    k0 = 0
    while k0 < H_in:
        k_offsets = k0 + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < H_in

        # Load input chunk: concat[b, t, k_offsets]
        in_ptrs = concat_ptr + b * stride_c_b + t * stride_c_t + k_offsets * stride_c_h
        in_chunk = tl.load(in_ptrs, mask=mask_k, other=0.0)  # [BLOCK_K]

        # Load weight_T chunk: weightT[k_offsets, n_offsets] -> [BLOCK_K, BLOCK_N]
        w_ptrs = weightT_ptr + k_offsets[:, None] * stride_w_k + n_offsets[None, :] * stride_w_h
        w_chunk = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # Accumulate: sum over k of in_chunk[k] * w_chunk[k, :]
        acc += tl.sum(in_chunk[:, None] * w_chunk, axis=0)

        k0 += BLOCK_K

    # Store accumulated results to output
    out_ptrs = out_ptr + b * stride_o_b + t * stride_o_t + n_offsets * stride_o_h
    tl.store(out_ptrs, acc, mask=mask_n)


def run_triton_only(
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    process_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Triton-only implementation:
      1) Concatenate along sequence dimension (T = text_seq_len + img_seq_len) in Triton.
      2) Apply linear projection (GEMV) with process_weight.T in Triton.
      3) Split the processed tensor into encoder and hidden streams.
    """
    assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, \
        "All tensors must be on CUDA for Triton execution."
    assert hidden_states.dtype == torch.float32 and encoder_hidden_states.dtype == torch.float32 and \
           process_weight.dtype == torch.float32, "Use float32 tensors."
    hidden_states = hidden_states.contiguous()
    encoder_hidden_states = encoder_hidden_states.contiguous()
    process_weight = process_weight.contiguous()

    B = hidden_states.shape[0]
    H = hidden_states.shape[2]
    Stext = encoder_hidden_states.shape[1]
    Simg = hidden_states.shape[1]
    T = Stext + Simg

    # 1) Triton concatenate along sequence dimension: out_cat [B, T, H]
    out_cat = torch.empty((B, T, H), dtype=torch.float32, device=hidden_states.device)

    grid_concat = (B, T)
    concat_kernel_simple[grid_concat](
        encoder_hidden_states, hidden_states, out_cat,
        B, Stext, Simg, H,
        encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
        hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
        out_cat.stride(0), out_cat.stride(1), out_cat.stride(2),
        num_warps=1, num_stages=2,
    )

    # 2) Triton batched GEMV: out_cat [B, T, H] @ process_weight.T [H, H] -> out [B, T, H]
    H_in = H  # concat's hidden dim
    H_out = H  # weight's output dim (same as hidden dim)
    out = torch.empty((B, T, H_out), dtype=torch.float32, device=hidden_states.device)

    # Launch with 3D grid: (B, T, N_blocks). N_blocks depends on BLOCK_N; choose 64 for stability.
    BLOCK_N = 64
    def grid(meta):
        blocks_n = triton.cdiv(H_out, BLOCK_N)
        return (B, T, blocks_n)

    batched_gemv_kernel_tile[grid](
        out_cat, process_weight.transpose(0, 1), out,
        B, T, H_in, H_out,
        out_cat.stride(0), out_cat.stride(1), out_cat.stride(2),
        process_weight.transpose(0, 1).stride(0), process_weight.transpose(0, 1).stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_N=BLOCK_N,
        num_warps=4, num_stages=2,
    )

    # 3) Split into separate streams using view/slice (no computation)
    processed_encoder = out[:, :Stext, :]
    processed_hidden = out[:, Stext:, :]

    return processed_encoder, processed_hidden


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        return run_triton_only(hidden_states, encoder_hidden_states, process_weight)


def run(*args):
    return ModelNew()(*args)
