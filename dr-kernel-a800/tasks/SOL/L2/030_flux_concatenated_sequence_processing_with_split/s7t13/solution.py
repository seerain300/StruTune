import torch
import triton
import triton.language as tl


@triton.jit
def copy_row_to_cat_kernel(
    e_ptr,  # *float32, encoder_hidden_states[b, p, :]
    h_ptr,  # *float32, hidden_states[b, p-T, :]
    out_ptr,  # *float32, X_cat[b, p, :]
    T: tl.int32,  # text_seq_len
    I: tl.int32,  # img_seq_len
    H: tl.int32,  # hidden_dim
    stride_eb: tl.int32, stride_et: tl.int32, stride_eh: tl.int32,
    stride_hb: tl.int32, stride_hi: tl.int32, stride_hh: tl.int32,
    stride_ob: tl.int32, stride_om: tl.int32, stride_on: tl.int32,
):
    # 2D grid: (B, M) where M = T + I
    b = tl.program_id(0)
    p = tl.program_id(1)

    is_encoder = p < T
    src_idx = p if is_encoder else (p - T)

    # vector of column indices [0, H)
    n = tl.arange(0, H)

    # base pointers
    e_base = e_ptr + b * stride_eb
    h_base = h_ptr + b * stride_hb
    out_base = out_ptr + b * stride_ob

    # source row pointers
    e_row_ptr = e_base + src_idx * stride_et + n * stride_eh
    h_row_ptr = h_base + src_idx * stride_hi + n * stride_hh
    out_row_ptr = out_base + p * stride_om + n * stride_on

    # load and store
    val = tl.zeros([H], dtype=tl.float32)
    if is_encoder:
        val = tl.load(e_row_ptr)
    else:
        val = tl.load(h_row_ptr)
    tl.store(out_row_ptr, val)


@triton.jit
def per_row_matmul_kernel(
    a_row_ptr,   # *float32, input row pointer (X_cat[b, p, :])
    w_ptr,       # *float32, process_weight [H, H]
    out_row_ptr, # *float32, output row pointer (Y[b, p, :])
    H: tl.int32,          # hidden_dim (K and N)
    stride_ab: tl.int32,  # stride for A row (which is H)
    stride_wk: tl.int32,  # stride for W along K (columns)
    stride_wn: tl.int32,  # stride for W along N (rows)
    stride_ob: tl.int32,  # stride for output row (which is H)
    BLOCK_N: tl.constexpr,  # tile over N
    BLOCK_K: tl.constexpr,  # tile over K
):
    # N vector for this tile
    n0 = tl.program_id(1) * BLOCK_N
    n = n0 + tl.arange(0, BLOCK_N)

    # accumulator for this row across N tile
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, H, BLOCK_K):
        k = k0 + tl.arange(0, BLOCK_K)

        # Load A row slice (length BLOCK_K) and W tile [BLOCK_K, BLOCK_N]
        a_vals = tl.load(a_row_ptr + k * stride_ab, mask=k < H, other=0.0)  # [BLOCK_K]
        w_ptrs = w_ptr + k[:, None] * stride_wk + n[None, :] * stride_wn  # [BLOCK_K, BLOCK_N]
        w_mask = (k[:, None] < H) & (n[None, :] < H)
        w_vals = tl.load(w_ptrs, mask=w_mask, other=0.0)  # [BLOCK_K, BLOCK_N]

        # Accumulate dot product: [BLOCK_K] @ [BLOCK_K, BLOCK_N] -> [BLOCK_N]
        acc += tl.sum(w_vals * a_vals[:, None], axis=0)

    # Store the result row slice
    out_ptrs = out_row_ptr + n * stride_ob
    store_mask = n < H
    tl.store(out_ptrs, acc, mask=store_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Ensure CUDA tensors and contiguous
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be CUDA tensors"
        hidden_states = hidden_states.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        process_weight = process_weight.contiguous()

        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        M = T + I

        # 1) Build concatenated matrix X_cat [B, M, H] using Triton kernel (per-row copy), avoiding torch.cat
        X_cat = torch.empty((B, M, H), device=hidden_states.device, dtype=torch.float32)

        e_strides = encoder_hidden_states.stride()
        h_strides = hidden_states.stride()
        x_strides = X_cat.stride()

        grid_cat = (B, M)
        copy_row_to_cat_kernel[grid_cat](
            encoder_hidden_states, hidden_states, X_cat,
            T, I, H,
            e_strides[0], e_strides[1], e_strides[2],
            h_strides[0], h_strides[1], h_strides[2],
            x_strides[0], x_strides[1], x_strides[2],
            num_warps=2, num_stages=2,
        )

        # 2) Compute Y[b, p, :] = X_cat[b, p, :] @ process_weight for all rows p in [0, M)
        #    Output Y [B, M, H]
        Y = torch.empty((B, M, H), device=hidden_states.device, dtype=torch.float32)

        # Strides
        a_row_stride = X_cat.stride()[2]  # since we access a row as a vector of length H: stride along N dim
        w_strides = process_weight.stride()  # [H, H]
        y_row_stride = Y.stride()[2]

        # Choose tile sizes; for robustness, keep moderate tiles
        BLOCK_N = 64
        BLOCK_K = 32

        # grid: (B, ceil(H/BLOCK_N))
        grid_mm = (B, triton.cdiv(H, BLOCK_N))
        for b in range(B):
            # launch per-row kernels for each p
            for p in range(M):
                out_row_ptr = Y + b * Y.stride(0) + p * Y.stride(1)
                a_row_ptr = X_cat + b * X_cat.stride(0) + p * X_cat.stride(1)
                per_row_matmul_kernel[(1,)](  # one program per (b, p)
                    a_row_ptr, process_weight, out_row_ptr,
                    H,
                    a_row_stride, w_strides[0], w_strides[1], y_row_stride,
                    BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                    num_warps=2, num_stages=2,
                )

        # 3) Split into encoder and hidden streams and return torch.Tensor outputs directly
        processed_encoder = Y[:, :T, :]
        processed_hidden = Y[:, T:, :]
        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
