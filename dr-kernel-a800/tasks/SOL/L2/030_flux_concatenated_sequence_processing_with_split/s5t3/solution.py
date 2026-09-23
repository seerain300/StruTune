import torch
import triton
import triton.language as tl


@triton.jit
def concat_kernel(
    e_ptr,        # *f32, encoder_hidden_states [B, Stext, H]
    h_ptr,        # *f32, hidden_states [B, Simg, H]
    out_ptr,      # *f32, output concatenated [B, T, H], T = Stext + Simg
    B: tl.constexpr,    # batch size (not used in body, but for completeness)
    Stext: tl.constexpr,  # text seq len
    Simg: tl.constexpr,   # image seq len
    H: tl.constexpr,      # hidden dim
    stride_e_b: tl.constexpr,
    stride_e_t: tl.constexpr,
    stride_e_h: tl.constexpr,
    stride_h_b: tl.constexpr,
    stride_h_s: tl.constexpr,
    stride_h_h: tl.constexpr,
    stride_out_b: tl.constexpr,
    stride_out_t: tl.constexpr,
    stride_out_h: tl.constexpr,
):
    # 2D grid: (B, T), each program handles one token position t for one batch b
    pid_b = tl.program_id(0)
    t = tl.program_id(1)  # scalar t index

    # Bounds check: t in [0, Stext+Simg)
    if t >= (Stext + Simg):
        return

    # Compute source pointer based on whether t < Stext
    if t < Stext:
        src_ptr = e_ptr + pid_b * stride_e_b + t * stride_e_t + tl.arange(0, H) * stride_e_h
        values = tl.load(src_ptr, mask=tl.arange(0, H) < H, other=0.0)
        dst_ptr = out_ptr + pid_b * stride_out_b + t * stride_out_t + tl.arange(0, H) * stride_out_h
        tl.store(dst_ptr, values, mask=tl.arange(0, H) < H)
    else:
        src_t = t - Stext
        src_ptr = h_ptr + pid_b * stride_h_b + src_t * stride_h_s + tl.arange(0, H) * stride_h_h
        values = tl.load(src_ptr, mask=tl.arange(0, H) < H, other=0.0)
        dst_ptr = out_ptr + pid_b * stride_out_b + t * stride_out_t + tl.arange(0, H) * stride_out_h
        tl.store(dst_ptr, values, mask=tl.arange(0, H) < H)


@triton.jit
def batched_gemm_kernel(
    A_ptr,        # *f32, concatenated [B, T, H], used as A
    B_ptr,        # *f32, process_weight.T [H, H]
    Out_ptr,      # *f32, output [B, T, H]
    B_dim: tl.constexpr,   # actual batch size
    T_dim: tl.constexpr,   # actual T = Stext + Simg
    H_dim: tl.constexpr,   # hidden dim
    stride_A_b: tl.constexpr,
    stride_A_t: tl.constexpr,
    stride_A_h: tl.constexpr,
    stride_B_k: tl.constexpr,  # along input hidden dim
    stride_B_h: tl.constexpr,  # along output hidden dim
    stride_Out_b: tl.constexpr,
    stride_Out_t: tl.constexpr,
    stride_Out_h: tl.constexpr,
    BLOCK_T: tl.constexpr,     # tile over sequence (here we use scalar t)
    BLOCK_H: tl.constexpr,     # tile over hidden
    BLOCK_K: tl.constexpr,     # tile over hidden dim for dot-product
):
    # 3D grid: (B_dim, ceil_div(T_dim, BLOCK_T), ceil_div(H_dim, BLOCK_H))
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_h = tl.program_id(2)

    # Scalar t for this program
    t = pid_t * BLOCK_T + 0  # since BLOCK_T=1 in practice; but we keep generality via pid_t
    if t >= T_dim:
        return

    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_h = h_offsets < H_dim

    # Accumulator for one token t and a tile of hidden outputs
    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

    # Loop over K (input hidden dimension) in tiles
    for k_start in range(0, H_dim, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < H_dim

        # Load A[b, t, k] as a vector
        a_ptrs = A_ptr + pid_b * stride_A_b + t * stride_A_t + k_offsets * stride_A_h
        a = tl.load(a_ptrs, mask=mask_k, other=0.0)  # shape [BLOCK_K]

        # Load B[k, h] as [BLOCK_K, BLOCK_H]
        b_ptrs = B_ptr + k_offsets[:, None] * stride_B_k + h_offsets[None, :] * stride_B_h
        b_tile = tl.load(b_ptrs, mask=mask_k[:, None] & mask_h[None, :], other=0.0)

        # acc[h] += sum_k a[k] * b_tile[k, h]
        acc += tl.sum(a[:, None] * b_tile, axis=0)

    # Store acc to Out[b, t, h]
    out_ptrs = Out_ptr + pid_b * stride_Out_b + t * stride_Out_t + h_offsets * stride_Out_h
    tl.store(out_ptrs, acc, mask=mask_h)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only implementation:
        - Concatenates encoder_hidden_states and hidden_states along sequence dimension (Triton).
        - Applies linear projection process_weight.T to the concatenated sequence (Triton).
        - Returns split streams: processed_encoder [B, Stext, H] and processed_hidden [B, Simg, H].
        """
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3, "inputs must be [B, S, H]"
        assert process_weight.dim() == 2 and process_weight.shape[0] == process_weight.shape[1], "process_weight must be square"

        B = hidden_states.shape[0]
        Stext = encoder_hidden_states.shape[1]
        Simg = hidden_states.shape[1]
        H = hidden_states.shape[2]
        device = hidden_states.device
        dtype = hidden_states.dtype

        # Ensure contiguity
        e = encoder_hidden_states.contiguous()      # [B, Stext, H]
        h = hidden_states.contiguous()              # [B, Simg, H]
        W_T = process_weight.transpose(0, 1).contiguous()  # [H, H]

        # Allocate concatenated output [B, T, H]
        T = Stext + Simg
        out_cat = torch.empty((B, T, H), device=device, dtype=dtype)

        # Launch concat kernel: 2D grid over (B, T)
        grid_concat = (B, T)
        concat_kernel[grid_concat](
            e, h, out_cat,
            B, Stext, Simg, H,
            e.stride(0), e.stride(1), e.stride(2),
            h.stride(0), h.stride(1), h.stride(2),
            out_cat.stride(0), out_cat.stride(1), out_cat.stride(2),
            num_warps=1, num_stages=1,
        )

        # Allocate output after projection [B, T, H]
        out = torch.empty((B, T, H), device=device, dtype=dtype)

        # Launch GEMM kernel: 3D grid over (B, T, H tiles)
        # We tile only over H since each program handles one token t
        BLOCK_T = 1        # one token per program (t is scalar)
        BLOCK_H = 64
        BLOCK_K = 32
        grid_gemm = (B, T, triton.cdiv(H, BLOCK_H))
        batched_gemm_kernel[grid_gemm](
            out_cat, W_T, out,
            B, T, H,
            out_cat.stride(0), out_cat.stride(1), out_cat.stride(2),
            W_T.stride(0), W_T.stride(1),
            out.stride(0), out.stride(1), out.stride(2),
            BLOCK_T=BLOCK_T, BLOCK_H=BLOCK_H, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # Split along sequence dimension: return only the requested parts
        processed_encoder = out[:, :Stext, :]
        processed_hidden = out[:, Stext:, :]

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
