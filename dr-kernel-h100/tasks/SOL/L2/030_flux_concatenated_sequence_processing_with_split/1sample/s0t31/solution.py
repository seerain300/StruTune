import torch
import triton
import triton.language as tl


@triton.jit
def _concatenate_kernel(
    encoder_ptr,       # *const T [B, T, H]
    hidden_ptr,        # *const T [B, I, H]
    out_ptr,           # *T [B, T+I, H]
    B, T, I, H,        # int32
):
    b = tl.program_id(0)
    # Guard: if b >= B, return (grid size equals B, so no OOB)
    if b >= B:
        return

    total = T + I
    # Loop over concatenated sequence length with mask
    for l in range(0, total):
        row_is_encoder = l < T
        # Compute source index for hidden based on row_is_encoder
        # If row_is_encoder: src = encoder[b, l, :]
        # Else: src = hidden[b, l - T, :]
        # Note: Triton doesn't support dynamic branching here on a scalar,
        # but we can express it via masks without branching.
        # However, Triton needs static shapes; so we use a simple approach:
        # We'll write via pointer arithmetic for each case.
        # We can compute source row index j as l if row_is_encoder else l - T.
        j = tl.where(row_is_encoder, l, l - T)
        # Row base pointers
        enc_row_ptr = encoder_ptr + b * H * T + j * H
        hid_row_ptr = hidden_ptr + b * H * I + j * H
        out_row_ptr = out_ptr + b * H * (T + I) + l * H

        # Load from source and store to out
        # Since j is valid when row_is_encoder or l >= T, and l < total,
        # j is in [0, T) or [0, I), so pointers are valid.
        val = tl.load(enc_row_ptr)
        # If not encoder row, val is already from enc_row_ptr; we don't need to override.
        # Now store: out[l, :] = val
        # We don't have vectorized tl.arange for this simple loop, so we rely on pointer arithmetic.
        # For each column h in [0, H), load/store element. Triton allows scalar indexing via + h.
        # But Triton prefers vectorized loads; implement column vectorized load/store:
        h_offsets = tl.arange(0, H)
        # We cannot use h_offsets directly in pointer arithmetic in this loop,
        # so we implement element-wise load/store using scalar loop across H.
        # However, Triton kernels prefer vectorized ops; we'll instead write with vectorized loads
        # by constructing pointers with h_offsets and tl.load/tl.store.
        # To do that, we need to construct the full column vector. Triton doesn't support Python for-loops
        # over H; instead, we rely on Triton's element-wise masked vectorized operations by broadcasting.
        # Here, since we need to copy H elements, we do a vectorized approach for load/store:
        # Create a column vector of indices for H
        h = h_offsets  # shape [H]
        # Build pointers for source and destination for this row
        enc_row_vec_ptr = enc_row_ptr + h
        hid_row_vec_ptr = hid_row_ptr + h
        out_row_vec_ptr = out_row_ptr + h
        # Decide which source to use: if row_is_encoder, use enc_row_vec_ptr, else use hid_row_vec_ptr.
        # Triton allows scalar branching; we can create a mask and select
        use_enc = row_is_encoder
        src_vec_ptr = tl.where(use_enc, enc_row_vec_ptr, hid_row_vec_ptr)
        vals = tl.load(src_vec_ptr)
        tl.store(out_row_ptr + h, vals)


@triton.jit
def _triton_gemm_cat_wT_kernel(
    A_ptr,     # *const T [M, K], M = B*(T+I), K = H
    W_ptr,     # *const T [K, K], W is process_weight
    C_ptr,     # *T [M, K]
    M, K,      # int32
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D launch grid over output tiles
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K in chunks
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)

        # Load A_tile: shape [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + (offs_m[:, None] * K + offs_k[None, :])
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        A_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load W_tile as [BLOCK_K, BLOCK_N]: W[offs_k, offs_n]
        w_ptrs = W_ptr + (offs_k[:, None] * K + offs_n[None, :])
        w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < K)
        W_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Accumulate
        acc += tl.dot(A_tile, W_tile)

    # Store results to C with masks
    c_ptrs = C_ptr + (offs_m[:, None] * K + offs_n[None, :])
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < K)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def _split_streams_kernel(
    C_ptr,              # *const T [M, H], M = B*(T+I), H=hidden_dim
    encoder_out_ptr,    # *T [B, T, H]
    hidden_out_ptr,     # *T [B, I, H]
    B, T, I, H,         # int32
):
    b = tl.program_id(0)
    if b >= B:
        return

    # First, copy encoder rows: rows [0 .. B*T)
    start = 0
    end = T
    # We'll iterate over T with a loop; Triton supports Python 'for' loops with compile-time bounds.
    for t in range(0, T):
        row = b * (T + I) + t
        dst_row_ptr = encoder_out_ptr + b * H * T + t * H
        src_row_ptr = C_ptr + row * H
        h = tl.arange(0, H)
        vals = tl.load(src_row_ptr + h)
        tl.store(dst_row_ptr + h, vals)

    # Then, copy hidden rows: rows [T .. T+I)
    for i in range(0, I):
        row = b * (T + I) + t + i  # but t is final after loop; instead compute base then loop i after
        # To correctly compute, we can recompute base and loop i independently. Since Triton needs static loops,
        # we restructure: two separate loops over T and I. The above first loop did T; now do I.
        # Instead, we can use nested loops structure by having two kernels; here we inline the second loop.
        # But Triton kernels don't support nested runtime loops well in this form; better to have two launches.
        # However, we can keep a single kernel by reusing t after the first loop; but Triton requires static loop bounds.
        # So we'll implement a second loop with range(I):
        # Note: Triton allows Python 'for' with range when bounds are known at JIT time. We pass I as runtime arg,
        # but Triton can handle runtime loop bounds as long as we structure the kernel accordingly.
        # To keep it simple and robust, we'll use a second launch; but here we demonstrate inline loop.
        # Since Triton requires static bounds, we restructure to two launches; but to keep in one kernel, we use
        # a while loop. Triton supports while loops.
        i = 0
        while i < I:
            row = b * (T + I) + T + i
            dst_row_ptr = hidden_out_ptr + b * H * I + i * H
            src_row_ptr = C_ptr + row * H
            h = tl.arange(0, H)
            vals = tl.load(src_row_ptr + h)
            tl.store(dst_row_ptr + h, vals)
            i += 1

# Note: The above split kernel uses while for the I loop to be robust. Triton supports while loops.


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        hidden_states: [B, I, H]
        encoder_hidden_states: [B, T, H]
        process_weight: [H, H] (no bias)
        returns: (processed_encoder: [B, T, H], processed_hidden: [B, I, H])
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA device for Triton"
        B, I, H = hidden_states.shape
        B2, T, H2 = encoder_hidden_states.shape
        assert B == B2 and H == H2, "Batch and hidden_dim must match"
        dtype = hidden_states.dtype
        device = hidden_states.device

        # 1) Concatenate along sequence dim using Triton
        out_cat = torch.empty((B, T + I, H), dtype=dtype, device=device)
        _concatenate_kernel[(B,)](
            encoder_hidden_states, hidden_states, out_cat,
            B, T, I, H,
            num_warps=1, num_stages=1
        )

        # 2) GEMM: C = out_cat @ process_weight.T using Triton
        M = B * (T + I)
        # Ensure process_weight is [H, H], we want W^T which is [H, H]
        # Allocate output C [M, H]
        C = torch.empty((M, H), dtype=torch.float32, device=device)  # accumulate in fp32 for stability
        # Triton expects strides; since we made out_cat contiguous, A is contiguous. W should be contiguous.
        # Choose reasonable blocks. For H up to a few hundred, 64x64x32 is fine. We’ll use 128x128x32 for larger.
        BLOCK_M = 128
        BLOCK_N = 128
        BLOCK_K = 32
        grid_m = triton.cdiv(M, BLOCK_M)
        grid_n = triton.cdiv(H, BLOCK_N)
        _triton_gemm_cat_wT_kernel[(grid_m, grid_n)](
            out_cat, process_weight, C,
            M, H,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # 3) Split streams using Triton
        processed_encoder = torch.empty((B, T, H), dtype=dtype, device=device)
        processed_hidden = torch.empty((B, I, H), dtype=dtype, device=device)
        _split_streams_kernel[(B,)](
            C, processed_encoder, processed_hidden,
            B, T, I, H,
            num_warps=1, num_stages=1
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
