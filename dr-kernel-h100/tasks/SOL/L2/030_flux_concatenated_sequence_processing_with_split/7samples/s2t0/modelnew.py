import torch
import triton
import triton.language as tl

# Triton kernel: batched GEMM for A[M, K] x B[K, N] -> C[M, N]
# We will launch it twice: once for the encoder part (first seq_len rows) and once for the image part (remaining rows).
@triton.jit
def _matmul_seq_hidden_kernel(
    A_ptr,  # *ptr to [M, K]
    B_ptr,  # *ptr to [K, N] (process_weight.T)
    C_ptr,  # *ptr to [M, N]
    M,      # int: number of sequence positions in this stream
    K,      # int: hidden_dim
    N,      # int: hidden_dim
    stride_am,  # int: stride for A along M
    stride_ak,  # int: stride for A along K
    stride_bk,  # int: stride for B along K (rows of B)
    stride_bn,  # int: stride for B along N (cols of B)
    stride_cm,  # int: stride for C along M
    stride_cn,  # int: stride for C along N
    BLOCK_M: tl.constexpr,  # tile size along M
    BLOCK_N: tl.constexpr,  # tile size along N
    BLOCK_K: tl.constexpr,  # tile size along K
):
    # program ids
    pid_m = tl.program_id(0)  # along batch*seq groups
    pid_n = tl.program_id(1)  # along N tiles

    # compute indices handled by this program
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # loop over K dimension
    for k in range(0, K, BLOCK_K):
        k_offsets = k + tl.arange(0, BLOCK_K)

        # pointers for A and B tiles
        # A: [m, k] -> A_ptr + m*stride_am + k*stride_ak
        a_ptrs = A_ptr + (m_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak)
        # B: [k, n] -> B_ptr + k*stride_bk + n*stride_bn
        b_ptrs = B_ptr + (k_offsets[:, None] * stride_bk + n_offsets[None, :] * stride_bn)

        # masks for boundaries
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        b_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)

        # load tiles (promote to fp32 if needed)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # accumulate
        acc += tl.dot(a, b)

    # write back
    c_ptrs = C_ptr + (m_offsets[:, None] * stride_cm + n_offsets[None, :] * stride_cn)
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    # we keep acc as fp32; if inputs are fp32, this matches original
    tl.store(c_ptrs, acc, mask=c_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function:
        - Concatenation is logically handled by launching the kernel twice.
        - No torch computation inside forward; all matmul done by Triton.
        """
        # Input checks and shapes
        assert hidden_states.dim() == 3 and encoder_hidden_states.dim() == 3, "Inputs must be 3D tensors [batch, seq, hidden_dim]"
        batch = hidden_states.shape[0]
        hidden_dim = hidden_states.shape[2]
        assert encoder_hidden_states.shape[2] == hidden_dim, "hidden_dim must match for both inputs"
        assert process_weight.shape == (hidden_dim, hidden_dim), "process_weight must be square [hidden_dim, hidden_dim]"
        assert process_weight.is_cuda, "process_weight must be on CUDA for Triton"
        device = hidden_states.device
        assert device.type == "cuda", "This Triton implementation requires CUDA tensors"

        text_seq_len = encoder_hidden_states.shape[1]
        img_seq_len = hidden_states.shape[1]
        seq_total = text_seq_len + img_seq_len

        # Make sure inputs are contiguous and on CUDA
        # We do not need to physically concatenate; we'll feed slices of A to the kernel.
        # We'll allocate outputs directly.
        out_encoder = torch.empty((batch, text_seq_len, hidden_dim), device=device, dtype=torch.float32)
        out_img = torch.empty((batch, img_seq_len, hidden_dim), device=device, dtype=torch.float32)

        # B = process_weight.T is [hidden_dim, hidden_dim]
        B = process_weight.transpose(0, 1).contiguous()  # [hidden_dim, hidden_dim]

        # Choose tiling. These are robust defaults; Triton handles masks for non-multiples.
        BLOCK_M = 64
        BLOCK_N = 128
        BLOCK_K = 32

        # Launch kernel for encoder part: M = text_seq_len
        if text_seq_len > 0:
            grid_m = (text_seq_len + BLOCK_M - 1) // BLOCK_M
            grid_n = (hidden_dim + BLOCK_N - 1) // BLOCK_N
            _matmul_seq_hidden_kernel[(grid_m, grid_n)](
                encoder_hidden_states,  # A for encoder
                B,                       # [K, N] = [hidden_dim, hidden_dim]
                out_encoder,             # C [M, N] = [text_seq_len, hidden_dim]
                text_seq_len, hidden_dim, hidden_dim,
                encoder_hidden_states.stride(0), encoder_hidden_states.stride(2),
                B.stride(0), B.stride(1),
                out_encoder.stride(0), out_encoder.stride(2),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=4, num_stages=3,
            )

        # Launch kernel for image part: M = img_seq_len, starting from row index text_seq_len
        # We pass a view of hidden_states as A2; Triton treats pointers, and we adjust strides appropriately.
        # Important: we must ensure the "row base" offset for A is correct. Triton uses pointers; we can construct a new tensor for A2 if needed.
        # To avoid confusion, we'll create A2 as a view by slicing hidden_states along batch and sequence, but since batch is shared, we just pass pointer and correct strides.
        # However, Triton kernels don't take start row; we can launch the same kernel for hidden part. We need a pointer to the appropriate rows; we can make a contiguous view for simplicity.
        # Create A2_view: slice hidden_states to get only the image part, then we can pass it directly. But Triton kernel expects a contiguous [M,K] tensor. So we make a contiguous copy.

        # Make contiguous A2: [img_seq_len, hidden_dim]
        # We can extract the image part rows directly by indexing hidden_states along dim=1.
        # Create A2 by copying relevant rows from hidden_states into a new [img_seq_len, hidden_dim] tensor.
        # Initialize A2 as zeros, then fill rows.
        A2 = torch.empty((img_seq_len, hidden_dim), device=device, dtype=torch.float32)
        # Fill A2 rows using hidden_states: hidden_states[:, :img_seq_len, :].reshape(...).contiguous()
        # First ensure hidden_states is contiguous; if not, make a contiguous view.
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()
        # hidden_states: [batch, seq, hidden_dim]; we want to extract rows 0..img_seq_len-1 for all batch items
        # But Triton kernel expects a single [M, K] matrix. We'll build that by copying rows from hidden_states.
        # We'll iterate batch and copy rows into A2. However, since Triton launch is per forward, and batch is typically small, we can simply copy:
        # For each batch, copy rows to A2. Since batch dimension is separate in original, we need A2 to represent all rows of image stream across batch.
        # The original run() returns per-batch split; our approach is per-batch. We'll compute per batch by launching multiple times per batch? No, Triton kernel expects pointers; we can launch per-batch loop in host.
        # Simplify: We'll assume batch=1 for kernel launches. But the original Model.forward(self, *args) passes batch-size correctly. We need to compute per batch.
        # The original run function expects batched inputs, and we need to produce batched outputs. Triton kernels operate on tensors; we can loop over batch and launch for each batch item.

        # We'll loop over batch to compute per-batch outputs. This is acceptable; it keeps everything in Triton and avoids torch matmul.
        # For clarity, we implement per-batch computation here.
        for b in range(batch):
            # encoder part for batch b
            A1 = encoder_hidden_states[b]  # [text_seq_len, hidden_dim]
            A1 = A1.contiguous()           # ensure contiguous
            out_encoder_b = out_encoder[b] # [text_seq_len, hidden_dim]
            # Launch kernel for batch b
            grid_m = (text_seq_len + BLOCK_M - 1) // BLOCK_M
            grid_n = (hidden_dim + BLOCK_N - 1) // BLOCK_N
            _matmul_seq_hidden_kernel[(grid_m, grid_n)](
                A1, B, out_encoder_b,
                text_seq_len, hidden_dim, hidden_dim,
                A1.stride(0), A1.stride(1),
                B.stride(0), B.stride(1),
                out_encoder_b.stride(0), out_encoder_b.stride(1),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=4, num_stages=3,
            )

            # image part for batch b
            # Build A2 for batch b: rows [b, 0..img_seq_len-1, :]
            # hidden_states[b] has shape [img_seq_len, hidden_dim]; make contiguous 2D [img_seq_len, hidden_dim]
            A2 = hidden_states[b].contiguous()  # [img_seq_len, hidden_dim]
            out_img_b = out_img[b]              # [img_seq_len, hidden_dim]
            grid_m = (img_seq_len + BLOCK_M - 1) // BLOCK_M
            grid_n = (hidden_dim + BLOCK_N - 1) // BLOCK_N
            _matmul_seq_hidden_kernel[(grid_m, grid_n)](
                A2, B, out_img_b,
                img_seq_len, hidden_dim, hidden_dim,
                A2.stride(0), A2.stride(1),
                B.stride(0), B.stride(1),
                out_img_b.stride(0), out_img_b.stride(1),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
                num_warps=4, num_stages=3,
            )

        # Return per-batch outputs stacked along batch dim
        processed_encoder = out_encoder
        processed_hidden = out_img
        return processed_encoder, processed_hidden