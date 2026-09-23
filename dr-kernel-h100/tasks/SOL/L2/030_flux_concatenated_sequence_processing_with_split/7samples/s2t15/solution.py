import torch
import triton
import triton.language as tl

# 1) Triton kernel: concatenate encoder_hidden_states and hidden_states along sequence dim
@triton.jit
def concatenate_seqs_kernel(
    enc_ptr,            # *const float32, [B, T, H]
    hid_ptr,            # *const float32, [B, I, H]
    concat_ptr,         # *float32,       [B, S, H], S = T + I
    B: tl.constexpr,    # int
    T: tl.constexpr,    # int
    I: tl.constexpr,    # int
    H: tl.constexpr,    # int
    enc_stride_b, enc_stride_t, enc_stride_h,
    hid_stride_b, hid_stride_i, hid_stride_h,
    concat_stride_b, concat_stride_s, concat_stride_h,
    BLOCK_T: tl.constexpr,  # tile along T
    BLOCK_I: tl.constexpr,  # tile along I
):
    # We process one (b, s) pair per program
    # s ranges over 0..S-1, where S = T + I
    # We use masks to copy from enc/hid into concat for valid s in [0, T) or [T, T+I)
    pid = tl.program_id(0)  # single 1D launch with size B*S
    s = pid  # linear index for seq positions
    S = T + I

    # Compute b and local s
    b = s // S  # s >= 0 always, b in [0, B)
    local_s = s % S

    # Determine source: encoder or hidden
    is_encoder = local_s < T

    if is_encoder:
        # Copy from encoder_hidden_states[b, local_s, :]
        src_t = local_s
        # Row pointer in encoder for batch b
        enc_row_ptr = enc_ptr + b * enc_stride_b + src_t * enc_stride_t
        dst_row_ptr = concat_ptr + b * concat_stride_b + local_s * concat_stride_s
        # Copy H elements
        for n in range(0, H):
            val = tl.load(enc_row_ptr + n * enc_stride_h)
            tl.store(dst_row_ptr + n * concat_stride_h, val)
    else:
        # Copy from hidden_states[b, local_s - T, :]
        src_i = local_s - T
        # Row pointer in hidden for batch b
        hid_row_ptr = hid_ptr + b * hid_stride_b + src_i * hid_stride_i
        dst_row_ptr = concat_ptr + b * concat_stride_b + local_s * concat_stride_s
        # Copy H elements
        for n in range(0, H):
            val = tl.load(hid_row_ptr + n * hid_stride_h)
            tl.store(dst_row_ptr + n * concat_stride_h, val)

# 2) Triton kernel: matmul over concatenated [S, H] and process_weight.T [H, H]
# Output is processed_flat [B*S, H]
@triton.jit
def matmul_seqs_kernel(
    A_ptr,       # *const float32, [B*S, H] where A[b*S + m, k] = concatenated[b, m, k]
    B_ptr,       # *const float32, [H, H]   (process_weight.T)
    C_ptr,       # *float32,       [B*S, H] output
    B_size: tl.constexpr,  # S = T + I
    H: tl.constexpr,
    A_stride_m, A_stride_k,
    B_stride_k, B_stride_n,
    C_stride_m, C_stride_n,
    BLOCK_M: tl.constexpr,  # tile over M = B*S
    BLOCK_N: tl.constexpr,  # tile over N = H
    BLOCK_K: tl.constexpr,  # tile over K = H
):
    # 2D grid over (tiles of M, tiles of N)
    pid_m = tl.program_id(0)  # tile over M
    pid_n = tl.program_id(1)  # tile over N

    # Compute row/col offsets
    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # over B*S
    n_offsets = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # over H

    # Mask for valid rows and cols
    mask_m = m_offsets < (B_size * H)
    mask_n = n_offsets < H

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K
    for k_start in range(0, H, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < H

        # Load A tile: [BLOCK_M, BLOCK_K]
        # A[m, k] with m in m_offsets, k in k_offsets
        a_ptrs = A_ptr + m_offsets[:, None] * A_stride_m + k_offsets[None, :] * A_stride_k
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # Load B tile: [BLOCK_K, BLOCK_N]
        # B[k, n] with k in k_offsets, n in n_offsets
        b_ptrs = B_ptr + k_offsets[:, None] * B_stride_k + n_offsets[None, :] * B_stride_n
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        # Accumulate
        acc += tl.dot(a, b)

    # Store result: C[m, n] with m in m_offsets, n in n_offsets
    c_ptrs = C_ptr + m_offsets[:, None] * C_stride_m + n_offsets[None, :] * C_stride_n
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])

# 3) Triton kernel: split processed_flat [B*S, H] back into encoder and hidden outputs
@triton.jit
def split_seqs_kernel(
    processed_flat_ptr,  # *const float32, [B*S, H]
    out_enc_ptr,         # *float32,       [B, T, H]
    out_hid_ptr,         # *float32,       [B, I, H]
    B: tl.constexpr,     # int
    T: tl.constexpr,     # int
    I: tl.constexpr,     # int
    H: tl.constexpr,     # int
    S: tl.constexpr,     # int = T + I
    processed_stride_b, processed_stride_s, processed_stride_h,
    out_enc_stride_b, out_enc_stride_t, out_enc_stride_h,
    out_hid_stride_b, out_hid_stride_i, out_hid_stride_h,
    BLOCK_T: tl.constexpr,  # tile along T
    BLOCK_I: tl.constexpr,  # tile along I
):
    # For each batch b, copy processed_flat[b*T:T*S, :] into out_enc and out_hid
    # We process all b in a single 1D grid; b can be derived from pid as b = pid // num_tiles
    # But since we launch with grid=(B,), pid is directly the batch index.
    pid = tl.program_id(0)
    b = pid

    # Copy encoder slice: rows [b*T : b*T + T)
    t_offsets = tl.arange(0, BLOCK_T)
    for t in range(0, T, BLOCK_T):
        t_idx = t + t_offsets
        mask_t = t_idx < T
        # src indices in processed_flat: m = b*S + t
        src_m = b * S + t_idx
        src_ptrs = processed_flat_ptr + src_m[:, None] * processed_stride_s + tl.arange(0, H)[None, :] * processed_stride_h
        vals = tl.load(src_ptrs, mask=mask_t[:, None], other=0.0)
        dst_ptrs = out_enc_ptr + b * out_enc_stride_b + t_idx[:, None] * out_enc_stride_t + tl.arange(0, H)[None, :] * out_enc_stride_h
        tl.store(dst_ptrs, vals, mask=mask_t[:, None])

    # Copy hidden slice: rows [b*T + T : b*T + T + I)
    i_offsets = tl.arange(0, BLOCK_I)
    for i in range(0, I, BLOCK_I):
        i_idx = i + i_offsets
        mask_i = i_idx < I
        # src indices in processed_flat: m = b*S + T + i
        src_m = b * S + T + i_idx
        src_ptrs = processed_flat_ptr + src_m[:, None] * processed_stride_s + tl.arange(0, H)[None, :] * processed_stride_h
        vals = tl.load(src_ptrs, mask=mask_i[:, None], other=0.0)
        dst_ptrs = out_hid_ptr + b * out_hid_stride_b + i_idx[:, None] * out_hid_stride_i + tl.arange(0, H)[None, :] * out_hid_stride_h
        tl.store(dst_ptrs, vals, mask=mask_i[:, None])

# ModelNew: forward uses only Triton kernels
class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function.
        - Concatenates encoder_hidden_states and hidden_states along sequence dimension in Triton.
        - Applies linear projection via Triton matmul with process_weight.T.
        - Splits outputs back into encoder and image streams in Triton.
        Returns (processed_encoder_hidden_states, processed_hidden_states).
        """
        # Ensure device and dtype
        device = hidden_states.device
        if encoder_hidden_states.device != device or hidden_states.device != device or process_weight.device != device:
            raise RuntimeError("All tensors must be on the same device.")
        # We will compute in float32; if inputs are not float32, cast for computation
        # (The original code uses default float32; we mirror that.)
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]

        # Ensure contiguous
        encoder_hidden_states = encoder_hidden_states.contiguous()
        hidden_states = hidden_states.contiguous()
        process_weight_T = process_weight.t().contiguous()  # [H, H]

        # 1) Concatenate in Triton: concatenated [B, S, H], S = T + I
        S = T + I
        concatenated = torch.empty((B, S, H), device=device, dtype=torch.float32)

        # Launch concatenate kernel: we need to pass strides
        enc_stride_b, enc_stride_t, enc_stride_h = encoder_hidden_states.stride()
        hid_stride_b, hid_stride_i, hid_stride_h = hidden_states.stride()
        concat_stride_b, concat_stride_s, concat_stride_h = concatenated.stride()

        # Use tiles along T and I; we pick small tiles to handle arbitrary sizes
        BLOCK_T = 64
        BLOCK_I = 64
        grid_concat = (B * S,)  # one program per element (b, s)
        concatenate_seqs_kernel[grid_concat](
            encoder_hidden_states, hidden_states, concatenated,
            B, T, I, H,
            enc_stride_b, enc_stride_t, enc_stride_h,
            hid_stride_b, hid_stride_i, hid_stride_h,
            concat_stride_b, concat_stride_s, concat_stride_h,
            BLOCK_T=BLOCK_T, BLOCK_I=BLOCK_I,
        )

        # 2) Matmul: processed_flat [B*S, H] = concatenated_flat [B*S, H] @ process_weight_T [H, H]
        processed_flat = torch.empty((B * S, H), device=device, dtype=torch.float32)

        # Compute strides for A (concatenated flattened) and B (process_weight.T)
        # A[m, k] => m in [0, B*S), k in [0, H)
        A_stride_m = concatenated.stride(1)  # stride along S
        A_stride_k = concatenated.stride(2)  # stride along H
        B_stride_k = process_weight_T.stride(0)  # along H
        B_stride_n = process_weight_T.stride(1)  # along H
        C_stride_m = processed_flat.stride(0)    # along B*S
        C_stride_n = processed_flat.stride(1)    # along H

        # Grid for 2D tiling over M and N
        BLOCK_M = 64    # tile along M = B*S
        BLOCK_N = 128   # tile along N = H
        BLOCK_K = 64    # tile along K = H
        grid_matmul = (triton.cdiv(B * S, BLOCK_M), triton.cdiv(H, BLOCK_N))
        matmul_seqs_kernel[grid_matmul](
            concatenated, process_weight_T, processed_flat,
            B * S, H,
            A_stride_m, A_stride_k,
            B_stride_k, B_stride_n,
            C_stride_m, C_stride_n,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=3,
        )

        # 3) Split in Triton: processed_flat -> (processed_encoder [B, T, H], processed_hidden [B, I, H])
        processed_encoder = torch.empty((B, T, H), device=device, dtype=torch.float32)
        processed_hidden = torch.empty((B, I, H), device=device, dtype=torch.float32)

        processed_stride_b, processed_stride_s, processed_stride_h = processed_flat.stride()
        out_enc_stride_b, out_enc_stride_t, out_enc_stride_h = processed_encoder.stride()
        out_hid_stride_b, out_hid_stride_i, out_hid_stride_h = processed_hidden.stride()

        # Launch split kernel with grid size equal to batch count
        BLOCK_T_split = 128
        BLOCK_I_split = 128
        grid_split = (B,)
        split_seqs_kernel[grid_split](
            processed_flat, processed_encoder, processed_hidden,
            B, T, I, H, S,
            processed_stride_b, processed_stride_s, processed_stride_h,
            out_enc_stride_b, out_enc_stride_t, out_enc_stride_h,
            out_hid_stride_b, out_hid_stride_i, out_hid_stride_h,
            BLOCK_T=BLOCK_T_split, BLOCK_I=BLOCK_I_split,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
