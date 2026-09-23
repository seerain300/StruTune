import torch
import triton
import triton.language as tl


@triton.jit
def matmul_cat_kernel(
    A_ptr,        # *f32, shape [M, K], M = B * S, K = H
    B_ptr,        # *f32, shape [K, N], B = process_weight.T, K=N=H
    C_ptr,        # *f32, shape [M, N]
    M, N, K,      # int32 sizes
    A_stride0, A_stride1,  # int64 strides for A (row, col)
    B_stride0, B_stride1,  # int64 strides for B (row, col)
    C_stride0, C_stride1,  # int64 strides for C (row, col)
    BLOCK_N: tl.constexpr, # tile size along N
    BLOCK_K: tl.constexpr, # tile size along K
):
    # Each program handles one row m and a tile along N
    m = tl.program_id(0)
    n_start = tl.program_id(1) * BLOCK_N

    n_offsets = n_start + tl.arange(0, BLOCK_N)
    # Initialize accumulator
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Loop over K dimension
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)

        # Compute pointers for A[m, k] and B[k, n]
        # A is [M, K], with strides (A_stride0, A_stride1)
        a_ptrs = A_ptr + m * A_stride0 + k_offsets * A_stride1
        # B is [K, N], with strides (B_stride0, B_stride1)
        b_ptrs = B_ptr + k_offsets[:, None] * B_stride0 + n_offsets[None, :] * B_stride1

        # Mask for valid n
        n_mask = n_offsets < N
        # Mask for valid k
        k_mask = k_offsets < K

        # Load A row segment and B tile, using masks
        a_vals = tl.load(a_ptrs, mask=k_mask, other=0.0)  # shape [BLOCK_K]
        b_vals = tl.load(b_ptrs, mask=(k_mask[:, None] & n_mask[None, :]), other=0.0)  # shape [BLOCK_K, BLOCK_N]

        # Accumulate
        acc += tl.sum(a_vals[:, None] * b_vals, axis=0)

    # Store results to C[m, n]
    c_ptrs = C_ptr + m * C_stride0 + n_offsets * C_stride1
    c_mask = n_offsets < N
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, encoder_hidden_states: torch.Tensor, hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized forward:
        - Concatenate along sequence dim in PyTorch (torch.cat).
        - Compute matmul in Triton (A @ B, where A is concatenated [B, S, H], B = process_weight.T [H, H]).
        - Split outputs back into encoder and hidden streams using slicing.
        """
        # Ensure CUDA and contiguous
        assert encoder_hidden_states.is_cuda and hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA"
        enc = encoder_hidden_states.contiguous()
        hid = hidden_states.contiguous()
        Bw = process_weight.contiguous()

        # Shapes
        B, T, H = enc.shape
        _, I, _ = hid.shape
        S = T + I

        # 1) Concatenate along sequence dim using PyTorch
        concatenated = torch.cat([enc, hid], dim=1)  # [B, S, H]
        assert concatenated.shape == (B, S, H), "Concatenation failed"

        # 2) Prepare B = process_weight.T [H, H] and output C [B*S, H]
        Bw_T = Bw.transpose(0, 1).contiguous()  # [H, H]

        M = B * S  # number of rows in concatenated viewed as [M, H]
        # Create C as [M, H] to store the full result and then slice
        C = torch.empty((M, H), device=enc.device, dtype=enc.dtype)

        # 3) Launch Triton matmul kernel: C[M, H] = concatenated[M, H] @ Bw_T[H, H]
        # Treat concatenated as [M, H] with strides: row stride = H, col stride = 1
        A_stride0 = H
        A_stride1 = 1
        B_stride0 = H
        B_stride1 = 1
        C_stride0 = H
        C_stride1 = 1

        # Grid: (M rows, tiles over N=H). Use BLOCK_N=H to avoid partial tiles.
        BLOCK_N = H
        BLOCK_K = 128  # H=1024 -> 8 iterations
        grid = (M, 1)

        matmul_cat_kernel[grid](
            concatenated, Bw_T, C,
            M, H, H,  # M rows, N=H cols, K=H
            A_stride0, A_stride1,
            B_stride0, B_stride1,
            C_stride0, C_stride1,
            BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )

        # 4) Fill outputs by slicing C back to [B, T, H] and [B, I, H]
        # For each batch b, rows [b*S : b*S + T] go to encoder, remaining to hidden.
        processed_encoder = torch.empty((B, T, H), device=enc.device, dtype=enc.dtype)
        processed_hidden = torch.empty((B, I, H), device=enc.device, dtype=enc.dtype)

        for b in range(B):
            start = b * S
            end_e = start + T
            # Slice C for this batch and copy into outputs
            processed_encoder[b] = C[start:start + T].clone()  # in-place fill
            processed_hidden[b] = C[end_e:].clone()            # remaining rows are the hidden part

        return processed_encoder, processed_hidden