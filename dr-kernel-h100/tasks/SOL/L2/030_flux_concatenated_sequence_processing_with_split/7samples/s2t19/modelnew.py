import torch
import triton
import triton.language as tl


@triton.jit
def concatenate_seqs_kernel(
    enc_ptr,           # *const T: [B, T, H]
    hid_ptr,           # *const T: [B, I, H]
    out_ptr,           # *T: [B, S, H], S = T + I
    B: tl.int32, T: tl.int32, I: tl.int32, H: tl.int32,
    enc_stride_b: tl.int32, enc_stride_t: tl.int32, enc_stride_h: tl.int32,
    hid_stride_b: tl.int32, hid_stride_i: tl.int32, hid_stride_h: tl.int32,
    out_stride_b: tl.int32, out_stride_s: tl.int32, out_stride_h: tl.int32,
    BLOCK_H: tl.constexpr,
):
    # Each program handles one (b, j) pair where j is either < T (encoder) or < I (hidden)
    b = tl.program_id(0)
    j = tl.program_id(1)  # j ranges over T + I

    # Determine whether this j corresponds to encoder or hidden
    is_encoder = j < T
    # Compute source row index
    src_row = tl.where(is_encoder, j, j - T)
    # Compute destination row index in out: s = j (since we insert encoder first, then hidden)
    s = j

    # Load the row and store into out[b, s, :]
    # Loop over H dimension in chunks of BLOCK_H
    offs_h = tl.arange(0, BLOCK_H)
    # We'll load from either enc_ptr or hid_ptr depending on is_encoder
    # Load
    enc_row_ptr = enc_ptr + b * enc_stride_b + src_row * enc_stride_t
    hid_row_ptr = hid_ptr + b * hid_stride_b + src_row * hid_stride_i
    out_row_ptr = out_ptr + b * out_stride_b + s * out_stride_s

    # Since we don't know at compile time, we implement guarded load/store using masks.
    # We'll use a mask to load only when is_encoder is True or False, and then store all valid offs_h.
    # For store, we always store for offs_h < H.
    for h_start in range(0, H, BLOCK_H):
        mask_h = (h_start + offs_h) < H
        # Load from encoder or hidden based on is_encoder
        # We construct pointers and use mask accordingly.
        # Note: Triton supports elementwise operations; we can use tl.load with scalar is_encoder.
        # However, Triton doesn't support scalar branching in this way directly, so we do two loads and select.
        # To ensure correctness, we do a conditional load using masks.
        # First, assume encoder load; mask will prevent reading when not encoder.
        vals = tl.load(
            enc_row_ptr + (h_start + offs_h) * enc_stride_h,
            mask=mask_h & is_encoder,
            other=0.0
        )
        # If not encoder, we need to override vals with load from hid_ptr. Triton doesn't allow
        # "else load" here, so we perform a second load and overwrite vals where not encoder.
        # We can't branch per element, but we can just load hidden and write selectively by masking.
        # Since we can't do per-element override, we instead load hidden into a separate tensor
        # and then write out: we'll compute which rows are hidden by checking s >= T.
        # The concatenation out is laid out such that s < T comes from encoder, s >= T from hidden.
        # So for store, we can write vals into out for both cases; the vals for hidden rows will be 0 for enc positions.
        # This approach doesn't work; therefore, we simplify by assuming that the out_row_ptr is used
        # only when we know the source. To avoid complexity, we implement a 2D grid launch that
        # only covers encoder rows and hidden rows separately. However, Triton requires a 1D grid here.
        # To keep correctness, we instead perform two separate kernels for encoder and hidden copies.
        # But since we need a single kernel, we exit: Triton doesn't support per-element conditional loads
        # without knowing the branch at compile time. Hence, we need two kernels after all.
    # The above approach is not correct due to Triton's lack of per-branch elementwise override here.
    # Therefore, we revert to two kernels for concatenation: copy_encoder_rows and copy_hidden_rows.
    # This ensures no ambiguity and removes the need for per-element branching in the kernel.
    # Note: We'll not reach this code; see the alternative kernels below.


# We actually implement concatenation via two dedicated kernels to avoid ambiguous branching.


@triton.jit
def copy_encoder_rows_kernel(
    enc_ptr, out_ptr,
    B: tl.int32, T: tl.int32, H: tl.int32,
    enc_stride_b: tl.int32, enc_stride_t: tl.int32, enc_stride_h: tl.int32,
    out_stride_b: tl.int32, out_stride_s: tl.int32, out_stride_h: tl.int32,
    BLOCK_H: tl.constexpr,
):
    b = tl.program_id(0)
    t = tl.program_id(1)
    # Guard t < T
    if t < T:
        src_row_ptr = enc_ptr + b * enc_stride_b + t * enc_stride_t
        dst_row_ptr = out_ptr + b * out_stride_b + t * out_stride_s
        for h_start in range(0, H, BLOCK_H):
            offs_h = h_start + tl.arange(0, BLOCK_H)
            mask = offs_h < H
            vals = tl.load(src_row_ptr + offs_h * enc_stride_h, mask=mask, other=0.0)
            tl.store(dst_row_ptr + offs_h * out_stride_h, vals, mask=mask)


@triton.jit
def copy_hidden_rows_kernel(
    hid_ptr, out_ptr,
    B: tl.int32, I: tl.int32, T: tl.int32, H: tl.int32,
    hid_stride_b: tl.int32, hid_stride_i: tl.int32, hid_stride_h: tl.int32,
    out_stride_b: tl.int32, out_stride_s: tl.int32, out_stride_h: tl.int32,
    BLOCK_H: tl.constexpr,
):
    b = tl.program_id(0)
    i = tl.program_id(1)  # i ranges over I
    s = T + i  # destination row in out for hidden
    if i < I:
        src_row_ptr = hid_ptr + b * hid_stride_b + i * hid_stride_i
        dst_row_ptr = out_ptr + b * out_stride_b + s * out_stride_s
        for h_start in range(0, H, BLOCK_H):
            offs_h = h_start + tl.arange(0, BLOCK_H)
            mask = offs_h < H
            vals = tl.load(src_row_ptr + offs_h * hid_stride_h, mask=mask, other=0.0)
            tl.store(dst_row_ptr + offs_h * out_stride_h, vals, mask=mask)


@triton.jit
def matmul_rowwise_kernel(
    A_ptr,      # *const T: [S, H], contiguous along H
    B_ptr,      # *const T: [H, H], process_weight.T
    C_ptr,      # *T: [S, H]
    S: tl.int32, H: tl.int32,
    A_stride_s: tl.int32, A_stride_h: tl.int32,
    B_stride_k: tl.int32, B_stride_n: tl.int32,
    C_stride_s: tl.int32, C_stride_h: tl.int32,
    BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # 2D grid: (row_id in [0..S), tile over N in steps of BLOCK_N)
    row_id = tl.program_id(0)
    tile_n = tl.program_id(1)

    # Accumulator for this row tile
    n_start = tile_n * BLOCK_N
    offs_n = n_start + tl.arange(0, BLOCK_N)
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Loop over K (H) in chunks of BLOCK_K
    for k_start in range(0, H, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        # Load A[row_id, offs_k]
        a = tl.load(
            A_ptr + row_id * A_stride_s + offs_k * A_stride_h,
            mask=offs_k < H,
            other=0.0,
        )
        # Load B[offs_k, offs_n]
        b = tl.load(
            B_ptr + offs_k[:, None] * B_stride_k + offs_n[None, :] * B_stride_n,
            mask=(offs_k[:, None] < H) & (offs_n[None, :] < H),
            other=0.0,
        )
        # FMA: acc += sum over k of a[k] * b[k, :]
        acc += tl.sum(b * a[:, None], axis=0)

    # Store acc to C[row_id, n_start:n_start+BLOCK_N]
    tl.store(
        C_ptr + row_id * C_stride_s + offs_n * C_stride_h,
        acc,
        mask=offs_n < H,
    )


@triton.jit
def split_kernel(
    C_ptr, enc_out_ptr, hid_out_ptr,
    B: tl.int32, T: tl.int32, I: tl.int32, H: tl.int32, S: tl.int32,
    C_stride_b: tl.int32, C_stride_s: tl.int32, C_stride_h: tl.int32,
    enc_stride_b: tl.int32, enc_stride_t: tl.int32, enc_stride_h: tl.int32,
    hid_stride_b: tl.int32, hid_stride_i: tl.int32, hid_stride_h: tl.int32,
    BLOCK_H: tl.constexpr,
):
    b = tl.program_id(0)
    for t in range(0, T):
        src_ptr = C_ptr + b * C_stride_b + t * C_stride_s
        dst_ptr = enc_out_ptr + b * enc_stride_b + t * enc_stride_t
        for h_start in range(0, H, BLOCK_H):
            offs_h = h_start + tl.arange(0, BLOCK_H)
            mask = offs_h < H
            vals = tl.load(src_ptr + offs_h * C_stride_h, mask=mask, other=0.0)
            tl.store(dst_ptr + offs_h * enc_stride_h, vals, mask=mask)
    for i in range(0, I):
        src_ptr = C_ptr + b * C_stride_b + (T + i) * C_stride_s
        dst_ptr = hid_out_ptr + b * hid_stride_b + i * hid_stride_i
        for h_start in range(0, H, BLOCK_H):
            offs_h = h_start + tl.arange(0, BLOCK_H)
            mask = offs_h < H
            vals = tl.load(src_ptr + offs_h * C_stride_h, mask=mask, other=0.0)
            tl.store(dst_ptr + offs_h * hid_stride_h, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, encoder_hidden_states: torch.Tensor,
                hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run:
        - Concatenate [B, T, H] and [B, I, H] along sequence dim.
        - Apply linear projection via matmul with process_weight.T.
        - Split back into [B, T, H] and [B, I, H].
        All computation is done inside Triton kernels; forward only allocates and launches.
        """
        assert encoder_hidden_states.is_cuda and hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA"
        assert encoder_hidden_states.dtype == torch.float32 and hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, "Use float32 for correctness"
        B, T, H = encoder_hidden_states.shape
        _, I, _ = hidden_states.shape

        # 1) Concatenate into out [B, S, H], S = T + I
        out = torch.empty((B, T + I, H), device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype)

        BLOCK_H = 128
        grid_encoder = (B, T)
        copy_encoder_rows_kernel[grid_encoder](
            encoder_hidden_states, out,
            B, T, H,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            BLOCK_H=BLOCK_H,
            num_warps=4, num_stages=2,
        )

        grid_hidden = (B, I)
        copy_hidden_rows_kernel[grid_hidden](
            hidden_states, out,
            B, I, T, H,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            BLOCK_H=BLOCK_H,
            num_warps=4, num_stages=2,
        )

        # 2) Compute C = out @ process_weight.T  -> [B, S, H]
        # Treat out as [S, H] by using correct strides and grid over S
        # We'll launch one program per row of out and per N tile. Since we want a single N tile to cover H,
        # we set BLOCK_N = H and tile_n = 0. We still use a 2D grid with tile_n for safety, but here S is small.
        S = T + I
        C = torch.empty((B, S, H), device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype)

        Bw_T = process_weight.t().contiguous()  # [H, H]

        # Use a single N tile covering H
        BLOCK_N = H  # set to H for simplicity; masking still works, but no partial tiles now
        grid_m = (S,)  # one program per row
        grid_n = (1,)  # single tile over N
        grid = (S, 1)

        matmul_rowwise_kernel[grid](
            out, Bw_T, C,
            S, H,
            out.stride(0), out.stride(2),        # A strides: s, h
            Bw_T.stride(0), Bw_T.stride(1),      # B strides: k, n
            C.stride(0), C.stride(2),            # C strides: s, h
            BLOCK_K=64, BLOCK_N=H,
            num_warps=4, num_stages=2,
        )

        # 3) Split C back into encoder and hidden parts
        processed_encoder = torch.empty((B, T, H), device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype)
        processed_hidden = torch.empty((B, I, H), device=encoder_hidden_states.device, dtype=encoder_hidden_states.dtype)

        BLOCK_H_split = 128
        split_kernel[(B,)](
            C, processed_encoder, processed_hidden,
            B, T, I, H, S,
            C.stride(0), C.stride(1), C.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_H=BLOCK_H_split,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden