import torch
import triton
import triton.language as tl


@triton.jit
def concat_seqs_kernel(
    enc_ptr,        # *ptr to encoder_hidden_states [B, T, H]
    hid_ptr,        # *ptr to hidden_states [B, I, H]
    out_ptr,        # *ptr to concatenated [B, S, H]
    B: tl.constexpr,
    T: tl.constexpr,
    I: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    # 2D launch: program_id(0) over B, program_id(1) over tiles of m in [0, S)
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)

    # Compute m indices for this tile
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = m_offsets < S

    # For each m, find corresponding source pointer: first T rows come from encoder, rest from hidden
    # Row index within original tensor is m
    # Pointer arithmetic:
    # out_ptr[b, m, :] = enc_ptr[b, m, :] if m < T else hid_ptr[b, m - T, :]
    # We use masked loads/stores to avoid illegal memory access.
    # Note: We need to load/store entire H for each m; but the kernel is structured so that
    # we will call it per-row; here we implement per-tile over m with a loop over H inside,
    # but Triton doesn't support loops over dynamic H. So we instead use a 3D grid over (B, m, H).
    # To avoid that complexity, we instead call a simpler per-row kernel. For now, we keep this
    # as a sketch and note that the practical approach is per-row kernel launch.
    # The following is a placeholder; in practice we'll use per-row kernels with 2D grid over (B, m).
    pass


# We will implement per-row kernels instead of the above. For clarity, here is the per-row concat kernel.


@triton.jit
def concat_per_row_kernel(
    enc_ptr,        # *ptr to encoder_hidden_states [B, T, H]
    hid_ptr,        # *ptr to hidden_states [B, I, H]
    out_ptr,        # *ptr to concatenated [B, S, H]
    b: tl.constexpr,
    T: tl.constexpr,
    I: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
):
    # Each program handles one row m in [0, S) for a given batch b
    m = tl.program_id(0)
    if m < T:
        src_ptr = enc_ptr + b * (T * H) + m * H
    else:
        src_ptr = hid_ptr + b * (I * H) + (m - T) * H
    dst_ptr = out_ptr + b * (S * H) + m * H
    # Copy H elements
    for h in range(0, H):
        val = tl.load(src_ptr + h)
        tl.store(dst_ptr + h, val)


@triton.jit
def matmul_row_kernel(
    A_ptr,          # *ptr to A [S, H], where A is the concatenated tensor
    B_ptr,          # *ptr to B^T [H, H]
    C_ptr,          # *ptr to output [S, H]
    S: tl.constexpr,  # number of rows in A (S = T + I)
    H: tl.constexpr,  # dimension
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # 2D grid: (m row in [0, S), tile over N=H)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m = pid_m  # each program handles one row
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = n_offsets < H

    # Accumulator vector for the N tile
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Loop over K dimension in chunks
    for k_start in range(0, H, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < H

        # Load A[m, k] vector for this row
        a_row_ptr = A_ptr + m * H + k_offsets  # A is row-major [S, H] with strides (H, 1)
        a_vec = tl.load(a_row_ptr, mask=mask_k, other=0.0)

        # Load B^T[k, n] as a BLOCK_N vector
        b_ptr = B_ptr + k_offsets[:, None] * H + n_offsets[None, :]  # shape [BLOCK_K, BLOCK_N]
        b_tile = tl.load(b_ptr, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # Multiply-accumulate: sum over K chunk
        # b_tile shape [BLOCK_K, BLOCK_N], a_vec shape [BLOCK_K]
        acc += tl.sum(b_tile * a_vec[:, None], axis=0)

    # Store the accumulated result to C[m, n]
    c_row_ptr = C_ptr + m * H + n_offsets
    tl.store(c_row_ptr, acc, mask=mask_n)


@triton.jit
def split_seqs_kernel(
    C_ptr,          # *ptr to C [B, S, H]
    out_enc_ptr,    # *ptr to processed_encoder [B, T, H]
    out_hid_ptr,    # *ptr to processed_hidden [B, I, H]
    B: tl.constexpr,
    T: tl.constexpr,
    I: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = m_offsets < S

    # For rows m in [0, T): write to encoder
    # For rows m in [T, T+I): write to hidden
    for m in range(0, BLOCK_M):
        idx = m_offsets[m]
        if idx < T:
            src_ptr = C_ptr + pid_b * (S * H) + idx * H
            dst_ptr = out_enc_ptr + pid_b * (T * H) + m * H
        else:
            src_ptr = C_ptr + pid_b * (S * H) + idx * H
            dst_ptr = out_hid_ptr + pid_b * (I * H) + (idx - T) * H
        # Copy H elements
        for h in range(0, H):
            val = tl.load(src_ptr + h)
            if idx < T:
                tl.store(dst_ptr + h, val)
            else:
                tl.store(dst_ptr + h, val)


class ModelNew(torch.nn.Module):
    def forward(self, encoder_hidden_states: torch.Tensor, hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function:
        - Concatenate encoder_hidden_states and hidden_states along sequence dimension
        - Apply linear projection via Triton GEMM: concatenated @ process_weight.T
        - Split back into processed_encoder and processed_hidden
        """
        assert encoder_hidden_states.is_cuda and hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be on CUDA device"
        assert encoder_hidden_states.dtype == torch.float32 and hidden_states.dtype == torch.float32 and process_weight.dtype == torch.float32, "Use float32 tensors"
        enc = encoder_hidden_states.contiguous()
        hid = hidden_states.contiguous()
        B, T, H = enc.shape
        _, I, _ = hid.shape
        assert enc.shape[2] == H and hid.shape[2] == H and process_weight.shape[1] == H, "Hidden dimension must match"
        S = T + I

        # 1) Concatenate into [B, S, H]
        concatenated = torch.empty((B, S, H), device=enc.device, dtype=enc.dtype)
        # Launch per-row concat kernel: grid over (B, S)
        grid_concat = (B, S)
        concat_per_row_kernel[grid_concat](enc, hid, concatenated, B, T, I, H, S, num_warps=1, num_stages=1)

        # 2) Matmul: C = concatenated @ process_weight.T
        # Ensure process_weight.T is [H, H]
        Bw_T = process_weight.t().contiguous()
        C = torch.empty((S, H), device=enc.device, dtype=enc.dtype)
        # 2D grid over (rows, tiles over N=H). For H=1024, BLOCK_N=1024 gives a single tile.
        grid_matmul = (S, 1)
        matmul_row_kernel[grid_matmul](
            concatenated, Bw_T, C,
            S, H,
            BLOCK_K=64, BLOCK_N=H,
            num_warps=4, num_stages=2,
        )

        # Reshape C to [B, S, H]
        C = C.view(B, S, H)

        # 3) Split into encoder and hidden streams
        processed_encoder = torch.empty((B, T, H), device=enc.device, dtype=enc.dtype)
        processed_hidden = torch.empty((B, I, H), device=enc.device, dtype=enc.dtype)
        grid_split = (B, 1)  # S is small in typical workloads; we can set BLOCK_M=S
        split_seqs_kernel[grid_split](
            C, processed_encoder, processed_hidden,
            B, T, I, H, S,
            BLOCK_M=S,
            num_warps=1, num_stages=1,
        )

        return processed_encoder, processed_hidden