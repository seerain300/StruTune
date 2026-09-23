import torch
import triton
import triton.language as tl


@triton.jit
def concat_seqs_kernel(
    enc_ptr,            # *float32, [B, T, H]
    hid_ptr,            # *float32, [B, I, H]
    out_ptr,            # *float32, [B, S, H], S = T + I
    B, T, I, H, S,
    stride_e_b, stride_e_t, stride_e_h,
    stride_h_b, stride_h_i, stride_h_h,
    stride_o_b, stride_o_s, stride_o_h,
    BLOCK_S: tl.constexpr,
):
    # Grid: (B, cdiv(S, BLOCK_S))
    b = tl.program_id(0)
    tile = tl.program_id(1)
    s_start = tile * BLOCK_S
    s_offsets = s_start + tl.arange(0, BLOCK_S)
    mask_s = s_offsets < S

    # Each s in [0, S) maps to either encoder (s < T) or hidden (s >= T)
    # We derive i and t for masked loads/stores.
    # For masked elements, we use neutral values (0.0), which won't be written.
    # Note: Triton supports dynamic masking per element.
    for s in s_offsets:
        if s < T:
            i = s
            # Load encoder row for batch b
            enc_row_ptr = enc_ptr + b * stride_e_b + i * stride_e_t
            out_row_ptr = out_ptr + b * stride_o_b + s * stride_o_s
            x = tl.load(enc_row_ptr + tl.arange(0, H) * stride_e_h, mask=True, other=0.0)
            tl.store(out_row_ptr + tl.arange(0, H) * stride_o_h, x, mask=mask_s[s])
        else:
            i = s - T
            # Load hidden row for batch b
            hid_row_ptr = hid_ptr + b * stride_h_b + i * stride_h_i
            out_row_ptr = out_ptr + b * stride_o_b + s * stride_o_s
            x = tl.load(hid_row_ptr + tl.arange(0, H) * stride_h_h, mask=True, other=0.0)
            tl.store(out_row_ptr + tl.arange(0, H) * stride_o_h, x, mask=mask_s[s])

    # Note: The above loop uses Python-level if-else; Triton allows such control flow per element.
    # If needed, we can compute mask_t = (s_offsets < T) and use masked load/store on vectors.
    # Implementing vectorized masked load/store is cleaner:
    # mask_t = s_offsets < T
    # mask_i = s_offsets >= T
    # t_offsets = s_offsets where mask_t, else 0 (unused)
    # i_offsets = s_offsets - T where mask_i, else 0
    # Then do vectorized masked loads and stores. Below we provide that vectorized version.
    # However, Triton doesn't allow per-element branching inside a vectorized expression in simple way;
    # So we fallback to per-element handling above, which Triton supports through scalar conditionals.

    # To keep it Triton-friendly and vectorized, we re-implement with vectorized masks:
    # Vectorized masked implementation:
    # t_mask = s_offsets < T
    # i_mask = s_offsets >= T
    # Compute t_offsets and i_offsets safely:
    # t_offsets = tl.where(t_mask, s_offsets, 0)
    # i_offsets = tl.where(i_mask, s_offsets - T, 0)
    # Then perform masked loads/stores for each source.
    # Triton does not support vector conditionals directly on pointers; hence we use per-element scalar if as above.

    # The per-element approach is correct but less vectorized. For robustness, we keep it simple.


@triton.jit
def matmul_seqs_kernel(
    A_ptr,              # *float32, [B*S, H] row-wise pointer to concatenated tensor
    BT_ptr,             # *float32, [H, H] = process_weight.T
    C_ptr,              # *float32, [B*S, H] output
    B, S, H,
    stride_A_m, stride_A_n,
    stride_BT_k, stride_BT_n,
    stride_C_m, stride_C_n,
    BLOCK_M: tl.constexpr,    # tile size over rows (M = B*S)
    BLOCK_N: tl.constexpr,    # tile size over N (H)
    BLOCK_K: tl.constexpr,    # reduction tile (K dimension)
):
    # Grid: (cdiv(B*S, BLOCK_M), cdiv(H, BLOCK_N))
    pid_m = tl.program_id(0)  # tile over M
    pid_n = tl.program_id(1)  # tile over N
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = m_offsets < (B * S)
    mask_n = n_offsets < H

    # Accumulator for the tile [BLOCK_M, BLOCK_N]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Reduction over K
    for k_start in range(0, H, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < H

        # Load A[m, k] as vector of length BLOCK_M, and BT[k, n] as BLOCK_K x BLOCK_N
        # For each m in BLOCK_M, load one vector A[m, k_offsets], then reduce with BT chunk.
        # Load BT tile BT[k_offsets][:, n_offsets] -> shape [BLOCK_K, BLOCK_N]
        BT_tile = tl.load(
            BT_ptr + k_offsets[:, None] * stride_BT_k + n_offsets[None, :] * stride_BT_n,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0
        )

        # For each m in this tile, load A[m, k_offsets]
        # We will load a vector of length BLOCK_M and broadcast it across BLOCK_N during multiplication.
        # However, Triton supports loading a vector across BLOCK_M by constructing pointer per m.
        # Better: iterate over m within the tile and accumulate.
        for mm in range(BLOCK_M):
            m = m_offsets[mm]
            a_vec = tl.load(
                A_ptr + m * stride_A_m + k_offsets * stride_A_n,
                mask=mask_m[mm] & mask_k,
                other=0.0
            )
            # Outer product: a_vec[:, None] * BT_tile[None, :, :]
            # Shapes: a_vec [BLOCK_K], BT_tile [BLOCK_K, BLOCK_N] -> result [BLOCK_K, BLOCK_N]
            # Then we need to add to acc[mm, :]
            # Instead of broadcasting, we can do:
            # We want: for each k in BLOCK_K, acc[mm, n] += a_vec[k] * BT_tile[k, n]
            # We can compute row-wise contributions:
            # Compute sum over k: sum_k a_vec[k] * BT_tile[k, :]
            # Use tl.sum over axis=0
            contrib = tl.sum(a_vec[:, None] * BT_tile, axis=0)  # shape [BLOCK_N]
            # Add to accumulator row mm across all columns in this tile
            # Create a vector of length BLOCK_N and set acc[mm, :] += contrib
            # Triton allows elementwise assignment for vectors:
            acc[mm, :] += contrib

    # Store the accumulated tile to C
    # For elements outside m range, acc[:, :] already zeroed; for n outside, mask_n guards.
    # Store acc into C[m_offsets, n_offsets].
    # We use broadcasting in store: C[m, n] = acc[m, n]
    # Triton supports direct store of acc to a 2D pointer.
    C_sub_ptr = C_ptr + m_offsets[:, None] * stride_C_m + n_offsets[None, :] * stride_C_n
    tl.store(C_sub_ptr, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def split_seqs_kernel(
    C_ptr,              # *float32, [B, S, H]
    out_e_ptr,          # *float32, [B, T, H]
    out_i_ptr,          # *float32, [B, I, H]
    B, T, I, H, S,
    stride_C_b, stride_C_s, stride_C_h,
    stride_e_b, stride_e_s, stride_e_h,
    stride_i_b, stride_i_i, stride_i_h,
    BLOCK_S: tl.constexpr,
):
    # Grid: (B, cdiv(T, BLOCK_S)) for encoder; and (B, cdiv(I, BLOCK_S)) for hidden
    b = tl.program_id(0)
    pid_t = tl.program_id(1) if tl.num_program_id() >= 2 else 0  # not used for hidden; placeholder

    # Copy encoder rows [0, T) into out_e
    for t_start in range(0, T, BLOCK_S):
        t_offsets = t_start + tl.arange(0, BLOCK_S)
        mask_t = t_offsets < T
        # For each t, copy C[b, t, :]
        for tt in t_offsets:
            if mask_t[tt]:
                src_ptr = C_ptr + b * stride_C_b + tt * stride_C_s
                dst_ptr = out_e_ptr + b * stride_e_b + tt * stride_e_s
                x = tl.load(src_ptr + tl.arange(0, H) * stride_C_h, mask=True, other=0.0)
                tl.store(dst_ptr + tl.arange(0, H) * stride_e_h, x, mask=mask_t[tt])

    # Copy hidden rows [T, T+I) into out_i
    for i_start in range(0, I, BLOCK_S):
        i_offsets = i_start + tl.arange(0, BLOCK_S)
        mask_i = i_offsets < I
        for ii in i_offsets:
            if mask_i[ii]:
                src_ptr = C_ptr + b * stride_C_b + (T + ii) * stride_C_s
                dst_ptr = out_i_ptr + b * stride_i_b + ii * stride_i_i
                x = tl.load(src_ptr + tl.arange(0, H) * stride_C_h, mask=True, other=0.0)
                tl.store(dst_ptr + tl.arange(0, H) * stride_i_h, x, mask=mask_i[ii])

# Note: Using nested loops and scalar conditionals here for simplicity and robustness.
# Triton supports these control flows. For higher performance, vectorized masked copies can be used,
# but correctness is the priority.

class ModelNew(torch.nn.Module):
    def forward(self, encoder_hidden_states: torch.Tensor, hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-only implementation:
        - Concatenate along sequence dimension in Triton.
        - Apply linear projection (matmul with process_weight.T) in Triton.
        - Split back into encoder and hidden streams in Triton.
        """
        assert encoder_hidden_states.is_cuda and hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA for Triton."
        assert encoder_hidden_states.dtype == hidden_states.dtype == process_weight.dtype, "All tensors must have the same dtype."
        assert encoder_hidden_states.dim() == 3 and hidden_states.dim() == 3 and process_weight.dim() == 2, "Invalid tensor shapes."
        B, T, H = encoder_hidden_states.shape
        Bi, I, Hi = hidden_states.shape
        assert Bi == B, "Batch size must match for encoder and hidden states."
        assert H == Hi, "Hidden dim must match."
        # process_weight should be [H, H], used as B^T
        assert process_weight.shape[0] == H and process_weight.shape[1] == H, "process_weight must be [H, H]."

        # Ensure contiguous
        enc = encoder_hidden_states.contiguous()
        hid = hidden_states.contiguous()
        Bw = process_weight.contiguous()  # [H, H]
        BT = Bw.transpose(0, 1).contiguous()  # [H, H], A @ BT

        # 1) Concatenate into out_concat [B, S, H], S = T + I
        S = T + I
        out_concat = torch.empty((B, S, H), device=enc.device, dtype=enc.dtype)

        grid_concat = (B, triton.cdiv(S, 128))  # tiles over S; 128 works well for H=1024
        concat_seqs_kernel[grid_concat](
            enc, hid, out_concat,
            B, T, I, H, S,
            enc.stride(0), enc.stride(1), enc.stride(2),
            hid.stride(0), hid.stride(1), hid.stride(2),
            out_concat.stride(0), out_concat.stride(1), out_concat.stride(2),
            BLOCK_S=128,
            num_warps=4, num_stages=2,
        )

        # 2) Compute processed = out_concat @ BT using Triton GEMM
        # Treat out_concat as [M, N] with M=B*S, N=H
        M = B * S
        C = torch.empty((M, H), device=enc.device, dtype=enc.dtype)

        # We need to pass A as a row-wise pointer to [M, N]. Triton can read from out_concat.
        # Define strides for A: out_concat [B, S, H], to view as [M, H], use:
        # For any m in [0, M), we map to (b, s) via b = m // S, s = m % S.
        # Then A[m, :] = out_concat[b, s, :].
        # We'll construct pointers accordingly in the kernel launch by providing C and BT, and letting kernel read from out_concat via pointer math.
        # However, Triton kernels read from pointer arrays; we cannot directly index out_concat by m.
        # So we make A as a separate tensor or compute via pointer arithmetic inside kernel? Triton doesn't support direct dynamic indexing.
        # To keep it simple, we load per-row vectors inside the kernel from out_concat.
        # Define grid over (M tiles, N tiles). Let BLOCK_M=128, BLOCK_N=128, BLOCK_K=64.

        grid_m = triton.cdiv(M, 128)
        grid_n = triton.cdiv(H, 128)
        matmul_seqs_kernel[(grid_m, grid_n)](
            out_concat, BT, C,
            B, S, H,
            out_concat.stride(0), out_concat.stride(1), out_concat.stride(2),  # stride_A_m = stride(0), stride_A_n = stride(2)
            BT.stride(0), BT.stride(1),                                     # [H, H]
            C.stride(0), C.stride(1),
            BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,
            num_warps=4, num_stages=2,
        )

        # 3) Split C [M, H] back into encoder [B, T, H] and hidden [B, I, H]
        # C is [B*S, H]. We need to map m -> (b, s). Use split kernel with B and S.
        # But Triton kernel expects 3D shapes; we'll reconstruct out_e and out_h via copy kernels.
        # Create outputs
        processed_encoder = torch.empty((B, T, H), device=enc.device, dtype=enc.dtype)
        processed_hidden = torch.empty((B, I, H), device=enc.device, dtype=enc.dtype)

        grid_split_e = (B, triton.cdiv(T, 128))
        grid_split_h = (B, triton.cdiv(I, 128))

        # We need to pass C[:, :T, :] and C[:, T:, :] to the split kernel. Triton doesn't support slicing in kernel args,
        # so we perform copy in kernels as above.

        # Copy encoder part: for each batch b, copy rows 0..T-1 from C[m] where m = b*S + t
        # Implement using the split_seqs_kernel specialized for copying rows from C to out_e/out_h.

        # First, copy encoder rows:
        for b in range(B):
            # We need to launch per-batch kernels with t grid. Triton supports loop, but better to use grid over (b, tiles).
            # We'll use a wrapper pattern: launch grid with (B, tiles) and compute b inside.
            # Triton kernels are launched with fixed grid; we can call the kernel once per b by setting grid's first dim as b.
            # However, Triton grid is static; we can compute b via program_id(0) and iterate t via program_id(1).
            # Let's set grid (B, cdiv(T, 128)) and inside the kernel use b = program_id(0).
            pass  # Not needed; handled below with explicit calls

        # Instead, we call split_seqs_kernel twice: once for encoder rows, once for hidden rows.
        # We need to pass C as a pointer and write to out_e/out_h. To do that, we reimplement copying logic here.

        # Implement encoder copy with Triton:
        # For each b, copy rows 0..T-1 from C to out_e. We can launch a grid over (B, tiles over T).
        # Triton doesn't support nested loops over program_id like that; we can launch once per b, but Triton expects static grid.
        # Therefore, we use a simple torch slicing for correctness in this environment.
        # However, we must use Triton kernel as requested. We'll implement a minimal copy kernel per batch.

        # We'll define a simple copy kernel: copy rows from C [B, S, H] to out_e [B, T, H], mapping row t -> C[b, t, :].
        # But Triton kernel split_seqs_kernel expects C [B, S, H], so we reuse it to copy rows.

        # To avoid confusion, we will implement encoder copy and hidden copy explicitly in Triton fashion:
        # Since Triton requires static grid, we'll use a kernel that copies rows given b and t_offsets.
        # Define a simple copy_rows_kernel that copies C[b, t_offsets, :] to out_e[b, t_offsets, :].

        # However, we don't have a copy_rows defined; we'll create it here.

        @triton.jit
        def copy_rows_kernel(
            src_ptr,          # *float32, [B, S, H], but we pass C
            dst_ptr,          # *float32, [B, T, H] for encoder or [B, I, H] for hidden
            B, T, I, H, S,    # shapes
            stride_src_b, stride_src_s, stride_src_h,
            stride_dst_b, stride_dst_s, stride_dst_h,
            BLOCK_S: tl.constexpr,
        ):
            b = tl.program_id(0)
            tile = tl.program_id(1)
            s_start = tile * BLOCK_S
            s_offsets = s_start + tl.arange(0, BLOCK_S)
            mask_s = s_offsets < T  # for encoder copy we use s_offsets as t
            # Load and store per row
            for tt in s_offsets:
                if mask_s[tt]:
                    src_row_ptr = src_ptr + b * stride_src_b + tt * stride_src_s
                    dst_row_ptr = dst_ptr + b * stride_dst_b + tt * stride_dst_s
                    x = tl.load(src_row_ptr + tl.arange(0, H) * stride_src_h, mask=True, other=0.0)
                    tl.store(dst_row_ptr + tl.arange(0, H) * stride_dst_h, x, mask=mask_s[tt])

        # Now, we will use this kernel to copy encoder rows and hidden rows.
        # For encoder: src is C [B, S, H], dst is processed_encoder [B, T, H], mapping row t to C[b, t, :]
        # We need a way to pass C as src. Triton requires pointers; we can pass C directly.

        # Launch encoder copy
        grid_e = (B, triton.cdiv(T, 128))
        copy_rows_kernel[grid_e](
            C, processed_encoder,
            B, T, I, H, S,
            C.stride(0), C.stride(1), C.stride(2),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            BLOCK_S=128,
            num_warps=4, num_stages=2,
        )

        # Launch hidden copy from C rows [T, T+I)
        @triton.jit
        def copy_rows_kernel_hidden(
            src_ptr,          # *float32, [B, S, H], pass C
            dst_ptr,          # *float32, [B, I, H]
            B, T, I, H, S,
            stride_src_b, stride_src_s, stride_src_h,
            stride_dst_b, stride_dst_i, stride_dst_h,
            BLOCK_S: tl.constexpr,
        ):
            b = tl.program_id(0)
            tile = tl.program_id(1)
            i_start = tile * BLOCK_S
            i_offsets = i_start + tl.arange(0, BLOCK_S)
            mask_i = i_offsets < I
            for ii in i_offsets:
                if mask_i[ii]:
                    src_row_ptr = src_ptr + b * stride_src_b + (T + ii) * stride_src_s
                    dst_row_ptr = dst_ptr + b * stride_dst_b + ii * stride_dst_i
                    x = tl.load(src_row_ptr + tl.arange(0, H) * stride_src_h, mask=True, other=0.0)
                    tl.store(dst_row_ptr + tl.arange(0, H) * stride_dst_h, x, mask=mask_i[ii])

        grid_h = (B, triton.cdiv(I, 128))
        copy_rows_kernel_hidden[grid_h](
            C, processed_hidden,
            B, T, I, H, S,
            C.stride(0), C.stride(1), C.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_S=128,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden