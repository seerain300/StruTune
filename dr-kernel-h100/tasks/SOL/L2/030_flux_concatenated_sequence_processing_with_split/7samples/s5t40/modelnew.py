import torch
import triton
import triton.language as tl


@triton.jit
def concat_encoder_image_kernel(
    e_ptr,                # *const float: [B, T, H]
    i_ptr,                # *const float: [B, I, H]
    out_ptr,              # *float: [B, T+I, H]
    B: tl.int32, T: tl.int32, I: tl.int32, H: tl.int32,
    e_s0, e_s1, e_s2,     # strides for e_ptr
    i_s0, i_s1, i_s2,     # strides for i_ptr
    o_s0, o_s1, o_s2,     # strides for out_ptr
    BLOCK_l: tl.constexpr, BLOCK_h: tl.constexpr,
):
    # program ids for tiling
    pid_b = tl.program_id(0)  # batch
    pid_l = tl.program_id(1)  # tiles along sequence
    pid_h = tl.program_id(2)  # tiles along hidden

    # indices within tiles
    l = pid_l * BLOCK_l + tl.arange(0, BLOCK_l)  # sequence indices in [0, T+I)
    h = pid_h * BLOCK_h + tl.arange(0, BLOCK_h)  # hidden indices in [0, H)

    # masks for bounds
    mask_l = l < (T + I)
    mask_h = h < H
    mask = mask_l[:, None] & mask_h[None, :]  # broadcast to [BLOCK_l, BLOCK_h]

    # Determine source: first T rows from encoder, remaining from image
    from_encoder = l < T  # [BLOCK_l] boolean
    # Base pointers for the current batch
    e_base = e_ptr + pid_b * e_s0
    i_base = i_ptr + pid_b * i_s0
    o_base = out_ptr + pid_b * o_s0

    # Compute addresses for load
    # e_addr: [BLOCK_l, BLOCK_h]
    e_addr = e_base + (l[:, None] * e_s1) + (h[None, :] * e_s2)
    # i_addr: [BLOCK_l, BLOCK_h]
    i_addr = i_base + ((l[:, None] - T) * i_s1) + (h[None, :] * i_s2)

    # Load values; use 0.0 for masked elements
    e_vals = tl.load(e_addr, mask=(mask & from_encoder[:, None]), other=0.0)
    i_vals = tl.load(i_addr, mask=(mask & (~from_encoder)[:, None]), other=0.0)

    # Select appropriate source
    # from_encoder is a 1D predicate; broadcast to 2D with [:, None]
    select = from_encoder[:, None]
    val = tl.where(select, e_vals, i_vals)

    # Compute output addresses and store
    o_addr = o_base + (l[:, None] * o_s1) + (h[None, :] * o_s2)
    tl.store(o_addr, val, mask=mask)


@triton.jit
def matmul_linear_kernel(
    A_ptr,                # *const float: [B, L, H], where L = T+I
    W_ptr,                # *const float: [H, H]
    C_ptr,                # *float: [B, L, H]
    B: tl.int32, L: tl.int32, H: tl.int32,
    A_s0, A_s1, A_s2,     # strides for A_ptr
    W_s0, W_s1,           # strides for W_ptr (row-major [H, H])
    C_s0, C_s1, C_s2,     # strides for C_ptr
    BLOCK_m: tl.constexpr, BLOCK_n: tl.constexpr, BLOCK_k: tl.constexpr,
):
    # Each program handles a tile of output: rows over B*L, columns over H.
    pid_m = tl.program_id(0)  # over rows (M = B*L)
    pid_n = tl.program_id(1)  # over columns (N = H)

    # Compute indices for this tile
    m = pid_m * BLOCK_m + tl.arange(0, BLOCK_m)  # shape [BLOCK_m], in [0, B*L)
    n = pid_n * BLOCK_n + tl.arange(0, BLOCK_n)  # shape [BLOCK_n], in [0, H)

    mask_m = m < (B * L)
    mask_n = n < H
    mask_out = mask_m[:, None] & mask_n[None, :]  # [BLOCK_m, BLOCK_n]

    # Map m to (b, l) where l = m % L, b = m // L
    b_idx = m // L
    l_idx = m % L

    # Initialize accumulator
    acc = tl.zeros((BLOCK_m, BLOCK_n), dtype=tl.float32)

    # Loop over K (hidden dimension) in chunks
    for k0 in range(0, H, BLOCK_k):
        k = k0 + tl.arange(0, BLOCK_k)  # [BLOCK_k]
        mask_k = k < H

        # Load A tile: [BLOCK_m, BLOCK_k]
        # Address: A[b, l, k]
        a_addr = A_ptr + (b_idx[:, None] * A_s0) + (l_idx[:, None] * A_s1) + (k[None, :] * A_s2)
        a_tile = tl.load(a_addr, mask=(mask_m[:, None] & mask_k[None, :]), other=0.0)

        # Load W^T tile: we want W[k, n] as we reduce over k
        # Address: W[k, n]
        w_addr = W_ptr + (k[:, None] * W_s0) + (n[None, :] * W_s1)
        w_tile = tl.load(w_addr, mask=(mask_k[:, None] & mask_n[None, :]), other=0.0)

        # Accumulate
        acc += tl.dot(a_tile, w_tile)

    # Store results to C
    c_addr = C_ptr + (b_idx[:, None] * C_s0) + (l_idx[:, None] * C_s1) + (n[None, :] * C_s2)
    tl.store(c_addr, acc, mask=mask_out)


@triton.jit
def copy_rows_kernel(
    src_ptr, dest_ptr,
    B: tl.int32, NUM_ROWS: tl.int32, H: tl.int32,
    src_s0, src_s1, src_s2,
    dest_s0, dest_s1, dest_s2,
    ROW_START: tl.int32,
    BLOCK_l: tl.constexpr, BLOCK_h: tl.constexpr,
):
    # Tiles across rows (NUM_ROWS) and hidden (H)
    pid_l = tl.program_id(0)  # tile along rows
    pid_h = tl.program_id(1)  # tile along hidden

    rows = pid_l * BLOCK_l + tl.arange(0, BLOCK_l)  # [BLOCK_l], in [0, NUM_ROWS)
    cols = pid_h * BLOCK_h + tl.arange(0, BLOCK_h)  # [BLOCK_h], in [0, H)

    mask_rows = rows < NUM_ROWS
    mask_cols = cols < H
    mask = mask_rows[:, None] & mask_cols[None, :]

    # Effective source row indices
    src_row = ROW_START + rows  # [BLOCK_l]
    mask_row = src_row < (B * 0 + (B * 0))  # placeholder; will be masked by src_row < (B * L) in host? This kernel assumes rows are valid. For exactness, rely on host to pass NUM_ROWS.

    # Compute src addresses: [B, L, H]
    # We cannot directly index B here, so we assume src_ptr is addressed as [L_total, H] flattened? To copy rows, we need row and column strides.
    # Given we will launch this kernel after matmul, we can access strides from processed (which has [B, L, H]).
    # Derive base pointer for each b by splitting rows into batches. However, Triton kernels don't have access to b per element, so we rely on host to pass total rows per batch and use src_ptr as [B, L, H] with row index src_row.

    # We cannot compute src_ptr + src_row * src_s0 correctly without per-program b. To fix, restructure: launch per-batch copy kernels in host, but here we keep it simple by assuming src_ptr layout [B, L, H].

    # Placeholder addressing: treat src_ptr as [B, L, H] and src_row is valid for each b. Triton doesn't support dynamic b here, so we redesign forward to use per-batch copy.

    # NOTE: This kernel is a placeholder. In practice, we will avoid this and instead perform slicing using two specialized Triton copy kernels per output stream. However, to keep it single, we implement per-batch copy by launching ModelNew.forward with separate calls for encoder and image rows. To avoid complexity, we will instead implement two specialized copy kernels below.

# The above placeholder kernel is kept only for structure. We will not use it; instead we implement per-batch row copying via specialized kernels below.


@triton.jit
def copy_rows_for_encoder_kernel(
    src_ptr, dest_ptr,
    B: tl.int32, T: tl.int32, H: tl.int32,
    src_s0, src_s1, src_s2,
    dest_s0, dest_s1, dest_s2,
    BLOCK_l: tl.constexpr, BLOCK_h: tl.constexpr,
):
    # Each program copies a tile of rows for encoder part: rows in [0, T)
    pid = tl.program_id(0)  # single grid along rows and hidden
    l = pid * BLOCK_l + tl.arange(0, BLOCK_l)  # [BLOCK_l], rows to copy (0..T-1)
    h = tl.program_id(1) * BLOCK_h + tl.arange(0, BLOCK_h)  # [BLOCK_h], columns (0..H-1)
    mask_l = l < T
    mask_h = h < H
    mask = mask_l[:, None] & mask_h[None, :]

    # b index: we launch per-batch here; get b from program_id? Triton allows only 3D; for simplicity, we use grid (B, tiles over T, tiles over H) and get b via division of a combined id? Simpler: launch per-batch by calling the kernel B times in host.

    # Placeholder: We will instead implement per-batch copying below by launching with grid (B, ...). To avoid complexity, we define a per-batch variant kernel.

# Implement per-batch copying directly instead of this. We will define a kernel that takes b via a combined grid and compute b. Triton supports up to 3 program_id dims; we can use one dimension for b and 2 for tiling.

@triton.jit
def copy_rows_per_batch_kernel(
    src_ptr, dest_ptr,
    B: tl.int32, T: tl.int32, H: tl.int32,
    src_s0, src_s1, src_s2,
    dest_s0, dest_s1, dest_s2,
    b: tl.int32,
    BLOCK_l: tl.constexpr, BLOCK_h: tl.constexpr,
):
    # Tiles across rows (T) and hidden (H) for a given batch b
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    rows = pid_t * BLOCK_l + tl.arange(0, BLOCK_l)  # [BLOCK_l], in [0, T)
    cols = pid_h * BLOCK_h + tl.arange(0, BLOCK_h)  # [BLOCK_h], in [0, H)

    mask_rows = rows < T
    mask_cols = cols < H
    mask = mask_rows[:, None] & mask_cols[None, :]

    # Addresses: src[b, rows, cols], dest[b, rows, cols]
    src_base = src_ptr + b * src_s0
    dest_base = dest_ptr + b * dest_s0

    src_addr = src_base + (rows[:, None] * src_s1) + (cols[None, :] * src_s2)
    dest_addr = dest_base + (rows[:, None] * dest_s1) + (cols[None, :] * dest_s2)

    vals = tl.load(src_addr, mask=mask, other=0.0)
    tl.store(dest_addr, vals, mask=mask)


# Implement two specialized kernels for copying encoder and hidden parts:
# 1) Copy first T rows (encoder)
# 2) Copy next I rows (image)
@triton.jit
def copy_encoder_rows_kernel(
    src_ptr, dest_ptr,
    B: tl.int32, T: tl.int32, H: tl.int32,
    src_s0, src_s1, src_s2,
    dest_s0, dest_s1, dest_s2,
    BLOCK_l: tl.constexpr, BLOCK_h: tl.constexpr,
):
    # Tiles across rows (T) and hidden (H)
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    rows = pid_t * BLOCK_l + tl.arange(0, BLOCK_l)  # [BLOCK_l], in [0, T)
    cols = pid_h * BLOCK_h + tl.arange(0, BLOCK_h)  # [BLOCK_h], in [0, H)

    mask_rows = rows < T
    mask_cols = cols < H
    mask = mask_rows[:, None] & mask_cols[None, :]

    # Launch per-batch in host. For generality, we include batch in grid using (B, ...) and compute b. Triton supports only 3 program_id dims. We'll instead launch this kernel with grid (B, tiles over T, tiles over H) by calling it B times? Simpler: compute b via program_id(2) if allowed; Triton has 3. So we'll launch per-batch by constructing separate calls in host. Here we keep it single kernel and rely on host to run it per-batch.

# Instead of trying to integrate batch in kernel, we will call these kernels per-batch in host, as shown below in ModelNew.forward.


# Implement a per-batch variant for hidden rows copy:
@triton.jit
def copy_hidden_rows_kernel(
    src_ptr, dest_ptr,
    B: tl.int32, I: tl.int32, H: tl.int32,
    src_s0, src_s1, src_s2,
    dest_s0, dest_s1, dest_s2,
    ROW_START: tl.int32,  # start row in src = T
    BLOCK_l: tl.constexpr, BLOCK_h: tl.constexpr,
):
    # Tiles across rows (I) and hidden (H), starting at ROW_START
    pid_i = tl.program_id(0)
    pid_h = tl.program_id(1)

    rows = pid_i * BLOCK_l + tl.arange(0, BLOCK_l)  # [BLOCK_l], in [0, I)
    cols = pid_h * BLOCK_h + tl.arange(0, BLOCK_h)  # [BLOCK_h], in [0, H)

    src_row = ROW_START + rows  # [BLOCK_l]
    mask_rows = src_row < (B * 0 + I)  # generic; in practice host ensures NUM_ROWS passed for each batch. Simpler: we pass per-batch pointers and rely on src_row < (T+I). For hidden, src_row < T+I, but dest has only I. We will launch per-batch with correct ranges. Triton kernels don't take per-call B; we fix by launching kernels per-batch via host calls.

# Note: Triton kernels cannot take dynamic batch size in their signature other than program_id dims. Therefore, we will launch copy kernels per batch by constructing calls outside this code. To keep code self-contained, we define per-batch versions below.

# We'll now implement the forward that actually launches these kernels, one per batch. Since this environment may not allow per-batch launches in a single ModelNew, we provide a simplified approach: we will keep a single per-batch kernel signature by launching ModelNew.forward with per-batch kernels. In practice, we'll call the kernels B times in host, but since this is a single file, we will emulate per-batch by using the batch dimension in grid and compute b via program_id(0) using Triton's 3D limit. Simpler: we will define per-batch kernels below and call them from ModelNew.forward.

# Implement per-batch copy for encoder rows:
@triton.jit
def copy_rows_per_batch_encoder(
    src_ptr, dest_ptr,
    T: tl.int32, H: tl.int32,
    src_s0, src_s1, src_s2,
    dest_s0, dest_s1, dest_s2,
    b: tl.int32,
    BLOCK_l: tl.constexpr, BLOCK_h: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    rows = pid_t * BLOCK_l + tl.arange(0, BLOCK_l)  # [BLOCK_l], in [0, T)
    cols = pid_h * BLOCK_h + tl.arange(0, BLOCK_h)  # [BLOCK_h], in [0, H)

    mask_rows = rows < T
    mask_cols = cols < H
    mask = mask_rows[:, None] & mask_cols[None, :]

    src_base = src_ptr + b * src_s0
    dest_base = dest_ptr + b * dest_s0

    src_addr = src_base + (rows[:, None] * src_s1) + (cols[None, :] * src_s2)
    dest_addr = dest_base + (rows[:, None] * dest_s1) + (cols[None, :] * dest_s2)

    vals = tl.load(src_addr, mask=mask, other=0.0)
    tl.store(dest_addr, vals, mask=mask)


@triton.jit
def copy_rows_per_batch_hidden(
    src_ptr, dest_ptr,
    I: tl.int32, H: tl.int32,
    src_s0, src_s1, src_s2,
    dest_s0, dest_s1, dest_s2,
    b: tl.int32,
    ROW_START: tl.int32,  # start row = T
    BLOCK_l: tl.constexpr, BLOCK_h: tl.constexpr,
):
    pid_i = tl.program_id(0)
    pid_h = tl.program_id(1)

    rows = pid_i * BLOCK_l + tl.arange(0, BLOCK_l)  # [BLOCK_l], in [0, I)
    cols = pid_h * BLOCK_h + tl.arange(0, BLOCK_h)  # [BLOCK_h], in [0, H)

    src_row = ROW_START + rows  # [BLOCK_l]
    mask_rows = src_row < (I)
    mask_cols = cols < H
    mask = mask_rows[:, None] & mask_cols[None, :]

    src_base = src_ptr + b * src_s0
    dest_base = dest_ptr + b * dest_s0

    src_addr = src_base + (src_row[:, None] * src_s1) + (cols[None, :] * src_s2)
    dest_addr = dest_base + (rows[:, None] * dest_s1) + (cols[None, :] * dest_s2)

    vals = tl.load(src_addr, mask=mask, other=0.0)
    tl.store(dest_addr, vals, mask=mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized version:
        - Concatenate encoder_hidden_states and hidden_states along sequence dimension.
        - Apply linear projection: concatenated @ process_weight.T.
        - Split back into processed_encoder and processed_hidden.
        """
        # Ensure tensors are on CUDA and contiguous
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All inputs must be CUDA tensors."
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        assert encoder_hidden_states.shape[2] == H and process_weight.shape[0] == H and process_weight.shape[1] == H, "Hidden dims must match."

        # 1) Concatenate using Triton
        L = T + I
        concatenated = torch.empty((B, L, H), device=hidden_states.device, dtype=hidden_states.dtype)

        grid_concat = (B, triton.cdiv(L, 64), triton.cdiv(H, 64))
        concat_encoder_image_kernel[grid_concat](
            encoder_hidden_states, hidden_states, concatenated,
            B, T, I, H,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            BLOCK_l=64, BLOCK_h=64,
            num_warps=4, num_stages=2,
        )

        # 2) Matmul: processed = concatenated @ process_weight.T (no bias)
        # Create output tensor
        processed = torch.empty((B, L, H), device=concatenated.device, dtype=torch.float32)

        # Launch matmul kernel
        # Note: We compute in float32 for stability. If inputs are fp32, this is fine.
        grid_matmul = (triton.cdiv(B * L, 32), triton.cdiv(H, 64), triton.cdiv(H, 32))
        matmul_linear_kernel[grid_matmul](
            concatenated, process_weight, processed,
            B, L, H,
            concatenated.stride(0), concatenated.stride(1), concatenated.stride(2),
            process_weight.stride(0), process_weight.stride(1),
            processed.stride(0), processed.stride(1), processed.stride(2),
            BLOCK_m=32, BLOCK_n=64, BLOCK_k=32,
            num_warps=4, num_stages=2,
        )

        # 3) Copy encoder rows: processed_encoder = processed[:, :T, :]
        processed_encoder = torch.empty((B, T, H), device=processed.device, dtype=processed.dtype)
        # Launch per-batch encoder copy
        grid_encoder = (triton.cdiv(T, 64), triton.cdiv(H, 64))
        for b in range(B):
            copy_rows_per_batch_encoder[grid_encoder](
                processed, processed_encoder,
                T, H,
                processed.stride(0), processed.stride(1), processed.stride(2),
                processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
                b,
                BLOCK_l=64, BLOCK_h=64,
                num_warps=4, num_stages=2,
            )

        # 4) Copy hidden rows: processed_hidden = processed[:, T:, :]
        processed_hidden = torch.empty((B, I, H), device=processed.device, dtype=processed.dtype)
        grid_hidden = (triton.cdiv(I, 64), triton.cdiv(H, 64))
        for b in range(B):
            copy_rows_per_batch_hidden[grid_hidden](
                processed, processed_hidden,
                I, H,
                processed.stride(0), processed.stride(1), processed.stride(2),
                processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
                b, T,
                BLOCK_l=64, BLOCK_h=64,
                num_warps=4, num_stages=2,
            )

        return processed_encoder, processed_hidden