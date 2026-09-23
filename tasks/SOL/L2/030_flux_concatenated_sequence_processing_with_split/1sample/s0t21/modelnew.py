import torch
import triton
import triton.language as tl


@triton.jit
def _concatenate_sequences_kernel(
    encoder_ptr,  # *const float, [B, T, H]
    hidden_ptr,   # *const float, [B, I, H]
    out_ptr,      # *float, [B, T+I, H]
    B: tl.constexpr, T: tl.constexpr, I: tl.constexpr, H: tl.constexpr,
    e_stride_b, e_stride_t, e_stride_h,
    h_stride_b, h_stride_i, h_stride_h,
    o_stride_b, o_stride_l, o_stride_h,
    BLOCK_L: tl.constexpr,
):
    # One program per batch
    b = tl.program_id(0)

    # Loop over sequence positions L = T + I
    for l in range(0, T + I, BLOCK_L):
        idx = l + tl.arange(0, BLOCK_L)
        mask = idx < (T + I)

        # Compute pointers for out[b, idx, :]
        out_ptrs = out_ptr + b * o_stride_b + idx * o_stride_l + tl.arange(0, H) * o_stride_h

        # Determine which rows come from encoder vs hidden
        enc_mask = idx < T
        hid_mask = ~enc_mask  # idx >= T

        # Load from encoder for idx < T
        enc_ptrs = encoder_ptr + b * e_stride_b + tl.where(enc_mask, idx, 0) * e_stride_t + tl.arange(0, H) * e_stride_h
        val = tl.zeros([BLOCK_L, H], dtype=tl.float32)
        # Only load where idx < T
        val = tl.load(enc_ptrs, mask=mask & enc_mask, other=0.0)

        # Load from hidden for idx >= T
        hid_ptrs = hidden_ptr + b * h_stride_b + (idx - T) * h_stride_i + tl.arange(0, H) * h_stride_h
        val = tl.load(hid_ptrs, mask=mask & hid_mask, other=0.0)

        # Store to out
        tl.store(out_ptrs, val, mask=mask)


@triton.jit
def _batched_gemm_right_kernel(
    a_ptr,          # *const float, [M, H] where M = B*(T+I)
    w_ptr,          # *const float, [H, H] (right-multiply by W^T)
    c_ptr,          # *float, [M, H]
    M: tl.constexpr,  # total rows = B*(T+I)
    H: tl.constexpr,  # hidden_dim
    a_stride_m, a_stride_k,
    w_stride_k, w_stride_n,
    c_stride_m, c_stride_k,
    BLOCK_M: tl.constexpr,  # tile rows in M
    BLOCK_N: tl.constexpr,  # tile columns in N (H)
    BLOCK_K: tl.constexpr,  # reduction tile in K
):
    # 2D grid: (tiles along M, tiles along N)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Row and column ranges for this tile
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_idx = m_start + tl.arange(0, BLOCK_M)
    n_idx = n_start + tl.arange(0, BLOCK_N)

    # Accumulator
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # Reduction over K = H
    for k_start in range(0, H, BLOCK_K):
        k_idx = k_start + tl.arange(0, BLOCK_K)

        # Load A tile: [BLOCK_M, BLOCK_K]
        a_ptrs = a_ptr + m_idx[:, None] * a_stride_m + k_idx[None, :] * a_stride_k
        a_mask = (m_idx[:, None] < M) & (k_idx[None, :] < H)
        a_tile = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load W^T tile: we need W[k, n] -> W[k, n], shape [BLOCK_K, BLOCK_N]
        w_ptrs = w_ptr + k_idx[:, None] * w_stride_k + n_idx[None, :] * w_stride_n
        w_mask = (k_idx[:, None] < H) & (n_idx[None, :] < H)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Accumulate
        acc += tl.dot(a_tile, w_tile)

    # Store C tile
    c_ptrs = c_ptr + m_idx[:, None] * c_stride_m + n_idx[None, :] * c_stride_k
    c_mask = (m_idx[:, None] < M) & (n_idx[None, :] < H)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def _split_streams_kernel(
    c_ptr,          # *const float, [M, H] flattened as [M, H]
    out_e_ptr,      # *float, [B, T, H]
    out_i_ptr,      # *float, [B, I, H]
    M: tl.constexpr,  # B*(T+I)
    T: tl.constexpr, I: tl.constexpr, H: tl.constexpr,
    c_stride_m, c_stride_n,
    oes_stride_b, oes_stride_t, oes_stride_h,
    ois_stride_b, ois_stride_i, ois_stride_h,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
):
    # One program per batch
    b = tl.program_id(0)

    # Row base in A = concatenated: rows [0..T-1] go to out_e, [T..T+I-1] go to out_i
    # We iterate seq indices and hidden columns
    for t in range(0, T):
        row = b * (T + I) + t
        for h in range(0, H):
            val = tl.load(c_ptr + row * c_stride_m + h * c_stride_n)
            tl.store(out_e_ptr + b * oes_stride_b + t * oes_stride_t + h * oes_stride_h, val)

    for i in range(0, I):
        row = b * (T + I) + T + i
        for h in range(0, H):
            val = tl.load(c_ptr + row * c_stride_m + h * c_stride_n)
            tl.store(out_i_ptr + b * ois_stride_b + i * ois_stride_i + h * ois_stride_h, val)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        # Ensure inputs are on CUDA and have expected dtype
        device = hidden_states.device
        assert device.type == "cuda", "ModelNew requires CUDA device"

        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        H = hidden_states.shape[2]
        assert encoder_hidden_states.shape[2] == H, "hidden_dim must match between inputs"
        assert process_weight.shape[0] == H and process_weight.shape[1] == H, "process_weight must be [H, H]"

        # 1) Concatenate sequences along sequence dimension using Triton
        out_cat = torch.empty((B, T + I, H), dtype=torch.float32, device=device)
        _concatenate_sequences_kernel[(B,)](
            encoder_hidden_states, hidden_states, out_cat,
            B, T, I, H,
            *encoder_hidden_states.stride(), *hidden_states.stride(),
            *out_cat.stride(),
            BLOCK_L=256,
            num_warps=1,
            num_stages=1,
        )
        # out_cat now holds [B, T+I, H]

        # 2) GEMM in Triton: C = out_cat @ process_weight.T
        # out_cat shape [M, K] where M = B*(T+I), K = H
        M = B * (T + I)
        A = out_cat  # [M, H], contiguous in last dim
        W = process_weight  # [H, H]
        C = torch.empty((M, H), dtype=torch.float32, device=device)

        # Launch 2D grid over output tiles
        BLOCK_M = 128
        BLOCK_N = 64
        BLOCK_K = 32
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(H, BLOCK_N))
        _batched_gemm_right_kernel[grid](
            A, W, C,
            M, H,
            *A.stride(), *W.stride(), *C.stride(),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # 3) Split into two outputs using Triton
        processed_encoder = torch.empty((B, T, H), dtype=torch.float32, device=device)
        processed_hidden = torch.empty((B, I, H), dtype=torch.float32, device=device)
        _split_streams_kernel[(B,)](
            C,
            processed_encoder, processed_hidden,
            M, T, I, H,
            *C.stride(),
            *processed_encoder.stride(), *processed_hidden.stride(),
            num_warps=1, num_stages=1,
        )

        return processed_encoder, processed_hidden