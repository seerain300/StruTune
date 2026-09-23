import torch
import triton
import triton.language as tl


@triton.jit
def cat_rows_kernel(
    e_ptr,         # *f32, [B, T, H]
    h_ptr,         # *f32, [B, I, H]
    x_ptr,         # *f32, [B, T+I, H]
    B: tl.constexpr, T: tl.constexpr, I: tl.constexpr, H: tl.constexpr,
    stride_e_b, stride_e_t, stride_e_h,
    stride_h_b, stride_h_i, stride_h_h,
    stride_x_b, stride_x_m, stride_x_n,
):
    # program ids
    b = tl.program_id(0)  # batch
    p = tl.program_id(1)  # row index in concatenated matrix [0, T+I)

    # compute source tensor and row offset
    # if p < T: row comes from encoder_hidden_states[b, p, :]
    # else:     row comes from hidden_states[b, p - T, :]
    is_encoder = p < T

    # pointers for source
    # For encoder row p:
    #   e_row_ptr = e_ptr + b*stride_e_b + p*stride_e_t
    # For hidden row (p - T):
    #   h_row_ptr = h_ptr + b*stride_h_b + (p - T)*stride_h_i
    # Load H elements for that row into a vector
    # We'll load with mask over H dimension; create arange for columns
    cols = tl.arange(0, H)
    if is_encoder:
        e_row_ptr = e_ptr + b * stride_e_b + p * stride_e_t
        vals = tl.load(e_row_ptr + cols * stride_e_h, mask=cols < H, other=0.0)
    else:
        h_row_ptr = h_ptr + b * stride_h_b + (p - T) * stride_h_i
        vals = tl.load(h_row_ptr + cols * stride_h_h, mask=cols < H, other=0.0)

    # store into X_cat[b, p, :]
    x_row_ptr = x_ptr + b * stride_x_b + p * stride_x_m
    tl.store(x_row_ptr + cols * stride_x_n, vals, mask=cols < H)


@triton.jit
def batched_matmul_kernel(
    x_ptr,         # *f32, [B, M, K]  where M=T+I, K=H
    w_ptr,         # *f32, [K, N]     where K=H, N=H (process_weight)
    y_ptr,         # *f32, [B, M, N]  where M=T+I, N=H
    B: tl.constexpr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_x_b, stride_x_m, stride_x_k,
    stride_w_k, stride_w_n,
    stride_y_b, stride_y_m, stride_y_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # One program per batch; tile over M and N
    b = tl.program_id(0)

    # Tiling
    m_tiles = tl.cdiv(M, BLOCK_M)
    n_tiles = tl.cdiv(N, BLOCK_N)

    # Loop over tiles
    for tm in range(0, m_tiles):
        for tn in range(0, n_tiles):
            # Compute tile offsets
            offs_m = tm * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
            offs_n = tn * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
            # Accumulator
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            # Reduction over K in steps of BLOCK_K
            for kk in range(0, K, BLOCK_K):
                offs_k = kk + tl.arange(0, BLOCK_K)  # [BLOCK_K]

                # Load A tile: x[b, offs_m, offs_k] -> [BLOCK_M, BLOCK_K]
                a_ptrs = x_ptr + b * stride_x_b + offs_m[:, None] * stride_x_m + offs_k[None, :] * stride_x_k
                a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
                a = tl.load(a_ptrs, mask=a_mask, other=0.0)

                # Load B tile: w[offs_k, offs_n] -> [BLOCK_K, BLOCK_N]
                w_ptrs = w_ptr + offs_k[:, None] * stride_w_k + offs_n[None, :] * stride_w_n
                w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
                w = tl.load(w_ptrs, mask=w_mask, other=0.0)

                # acc += a @ w
                acc += tl.dot(a, w)

            # Store acc to y[b, offs_m, offs_n]
            y_ptrs = y_ptr + b * stride_y_b + offs_m[:, None] * stride_y_m + offs_n[None, :] * stride_y_n
            y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
            tl.store(y_ptrs, acc, mask=y_mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-only implementation of the original run function:
        1) Concatenate along sequence dimension (avoid torch.cat)
        2) Apply linear projection (avoid torch.matmul)
        3) Split into encoder and hidden streams

        Returns:
            Tuple of (processed_encoder_hidden_states, processed_hidden_states)
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All tensors must be on CUDA"
        assert hidden_states.dtype in (torch.float32, torch.float16, torch.bfloat16), "Unsupported dtype"
        assert encoder_hidden_states.dtype == hidden_states.dtype == process_weight.dtype, "All tensors must have same dtype"

        # Shapes
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]

        # Ensure inputs are contiguous (strides are used, but contiguous is safer for Triton)
        e = encoder_hidden_states.contiguous()
        h = hidden_states.contiguous()
        w = process_weight.contiguous()

        # 1) Concatenation: build X_cat[b, m, :] for m in [0, T+I)
        M = T + I

        # Allocate X_cat as float32 for compute; will cast to original dtype on return
        x_cat = torch.empty((B, M, H), device=e.device, dtype=torch.float32)

        # Strides
        stride_e_b, stride_e_t, stride_e_h = e.stride(0), e.stride(1), e.stride(2)
        stride_h_b, stride_h_i, stride_h_h = h.stride(0), h.stride(1), h.stride(2)
        stride_x_b, stride_x_m, stride_x_n = x_cat.stride(0), x_cat.stride(1), x_cat.stride(2)

        # Launch cat_rows_kernel: grid = (B, M)
        grid_cat = (B, M)
        cat_rows_kernel[grid_cat](
            e, h, x_cat,
            B, T, I, H,
            stride_e_b, stride_e_t, stride_e_h,
            stride_h_b, stride_h_i, stride_h_h,
            stride_x_b, stride_x_m, stride_x_n,
            num_warps=4, num_stages=2,
        )

        # 2) Batched matmul: Y = X_cat @ w (w is [H, H])
        y = torch.empty((B, M, H), device=e.device, dtype=torch.float32)

        # Strides for matmul
        stride_x_b, stride_x_m, stride_x_k = x_cat.stride(0), x_cat.stride(1), x_cat.stride(2)
        stride_w_k, stride_w_n = w.stride(0), w.stride(1)
        stride_y_b, stride_y_m, stride_y_n = y.stride(0), y.stride(1), y.stride(2)

        # Choose tiling based on H (simple heuristic)
        # For small H, smaller tiles; for larger H, larger tiles
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32
        grid_mm = (B,)

        batched_matmul_kernel[grid_mm](
            x_cat, w, y,
            B, M, H, H,  # M=T+I, K=H, N=H
            stride_x_b, stride_x_m, stride_x_k,
            stride_w_k, stride_w_n,
            stride_y_b, stride_y_m, stride_y_n,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )

        # 3) Split (host-side metadata ops)
        processed_encoder = y[:, :T, :]
        processed_hidden = y[:, T:, :]

        # Cast back to original dtype to match original behavior
        processed_encoder = processed_encoder.to(hidden_states.dtype)
        processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
