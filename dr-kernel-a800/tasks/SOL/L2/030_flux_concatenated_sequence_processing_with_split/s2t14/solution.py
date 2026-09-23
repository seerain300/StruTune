import torch
import triton
import triton.language as tl

@triton.jit
def cat_seq_kernel(
    enc_ptr,     # *float32, [B, T, H]
    hid_ptr,     # *float32, [B, I, H]
    out_ptr,     # *float32, [B, L, H]
    B: tl.constexpr,
    T: tl.constexpr,
    I: tl.constexpr,
    H: tl.constexpr,
    enc_stride_b, enc_stride_t, enc_stride_h,
    hid_stride_b, hid_stride_i, hid_stride_h,
    out_stride_b, out_stride_l, out_stride_h,
):
    # Grid: (B, L, H) with B=program_id(0) and L=program_id(1) and H=program_id(2)
    b = tl.program_id(0)
    l = tl.program_id(1)
    h = tl.program_id(2)

    # Determine source based on sequence position
    is_hidden = l >= T
    seq_index = l - T  # valid only when is_hidden

    # Compute offsets
    # Note: For b-dimension, only b is used; since grid's first dim is B, we ensure b < B via launch grid.
    if is_hidden:
        # Load from hidden: [B, I, H]
        value = tl.load(
            hid_ptr + b * hid_stride_b + (l - T) * hid_stride_i + h * hid_stride_h
        )
    else:
        # Load from encoder: [B, T, H]
        value = tl.load(
            enc_ptr + b * enc_stride_b + l * enc_stride_t + h * enc_stride_h
        )
    # Store to output: [B, L, H]
    tl.store(out_ptr + b * out_stride_b + l * out_stride_l + h * out_stride_h, value)


@triton.jit
def batched_matmul_kernel(
    A_ptr,       # *float32, [B, M, K] where M=L
    Bt_ptr,      # *float32, [K, N] where Bt = process_weight.T, N=H
    C_ptr,       # *float32, [B, M, N]
    B: tl.constexpr,
    M: tl.constexpr,   # L
    N: tl.constexpr,   # H
    K: tl.constexpr,   # H
    A_stride_b, A_stride_m, A_stride_k,
    Bt_stride_k, Bt_stride_n,
    C_stride_b, C_stride_m, C_stride_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 3D launch: (B, tiles along M, tiles along N)
    b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    offs_m = m_start + tl.arange(0, BLOCK_M)[:, None]  # (BM, 1)
    offs_n = n_start + tl.arange(0, BLOCK_N)[None, :]  # (1, BN)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    k = 0
    while k < K:
        offs_k = k + tl.arange(0, BLOCK_K)  # (BK,)

        # Masks for bounds
        mask_a = (offs_m < M)[:, None] & (offs_k[None, :] < K)
        mask_bt = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        mask_c = (offs_m < M)[:, None] & (offs_n[None, :] < N)

        # Load A tile: [BM, BK]
        a_ptrs = A_ptr + b * A_stride_b + offs_m * A_stride_m + offs_k[None, :] * A_stride_k
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)

        # Load B^T tile: [BK, BN]
        bt_ptrs = Bt_ptr + offs_k[:, None] * Bt_stride_k + offs_n[None, :] * Bt_stride_n
        bt = tl.load(bt_ptrs, mask=mask_bt, other=0.0)

        # Accumulate
        acc += tl.dot(a, bt)

        k += BLOCK_K

    # Write back
    c_ptrs = C_ptr + b * C_stride_b + offs_m * C_stride_m + offs_n * C_stride_n
    tl.store(c_ptrs, acc, mask=mask_c)


class ModelNew(torch.nn.Module):
    def forward(self, encoder_hidden_states: torch.Tensor,
                hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized forward:
        - Concatenate encoder_hidden_states and hidden_states along sequence dimension.
        - Apply linear projection via Triton GEMM.
        - Split back into encoder and hidden streams.
        All heavy work is done in Triton; host code only prepares and returns tensors.
        """
        # Shapes
        B, T, H = encoder_hidden_states.shape
        _, I, _ = hidden_states.shape
        assert I == H, "hidden_states last dim must match hidden_dim"
        assert process_weight.shape[0] == H and process_weight.shape[1] == H, "process_weight must be [H, H]"

        # Make inputs contiguous and cast to float32 for robust Triton computation
        enc32 = encoder_hidden_states.contiguous().to(torch.float32)
        hid32 = hidden_states.contiguous().to(torch.float32)
        w32 = process_weight.contiguous().to(torch.float32)

        # 1) Triton concatenation: out_cat [B, L, H], L = T + I
        L = T + I
        out_cat = torch.empty((B, L, H), device=enc32.device, dtype=torch.float32)

        grid_cat = (B, L, H)
        cat_seq_kernel[grid_cat](
            enc32, hid32, out_cat,
            B, T, I, H,
            enc32.stride(0), enc32.stride(1), enc32.stride(2),
            hid32.stride(0), hid32.stride(1), hid32.stride(2),
            out_cat.stride(0), out_cat.stride(1), out_cat.stride(2),
            num_warps=1, num_stages=2,
        )

        # 2) Triton GEMM: C = out_cat @ process_weight.T
        # Prepare W^T [K, N] = [H, H], float32
        W_T = w32.transpose(0, 1).contiguous()  # [H, H], float32

        C = torch.empty((B, L, H), device=enc32.device, dtype=torch.float32)

        # Tile sizes — robust defaults; can be tuned for performance
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 64

        grid_mm = (B, triton.cdiv(L, BLOCK_M), triton.cdiv(H, BLOCK_N))
        batched_matmul_kernel[grid_mm](
            out_cat, W_T, C,
            B, L, H, H,
            out_cat.stride(0), out_cat.stride(1), out_cat.stride(2),
            W_T.stride(0), W_T.stride(1),
            C.stride(0), C.stride(1), C.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )

        # 3) Split streams
        processed_encoder = C[:, :T, :]
        processed_hidden = C[:, T:, :]

        # Return outputs in float32 (evaluation typically expects float32)
        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
