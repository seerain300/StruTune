import torch
import triton
import triton.language as tl

@triton.jit
def _concatenate_streams_kernel(
    E_ptr,        # encoder_hidden_states: [B, T, D]
    H_ptr,        # hidden_states: [B, I, D]
    C_ptr,        # concatenated output: [B, T+I, D]
    B, T, I, D,
    E_b_stride, E_t_stride, E_d_stride,
    H_b_stride, H_i_stride, H_d_stride,
    C_b_stride, C_seqlen_stride, C_d_stride,
):
    pid_b = tl.program_id(0)  # batch dimension
    pid_pos = tl.program_id(1)  # position in [T+I]
    # Compute which stream and index this row corresponds to
    t_total = T + I
    stream = pid_pos // t_total  # 0 => encoder, 1 => hidden
    pos = pid_pos % t_total
    # If beyond T, subtract T to map to hidden seq
    if stream == 1:
        pos = pos - T

    # Pointers
    if stream == 0:
        src_ptr = E_ptr + pid_b * E_b_stride + pos * E_t_stride
    else:
        src_ptr = H_ptr + pid_b * H_b_stride + pos * H_i_stride
    dst_ptr = C_ptr + pid_b * C_b_stride + pid_pos * C_seqlen_stride

    # Copy vector of length D
    # Vectorized loop over D dimension
    for d in range(0, D):
        val = tl.load(src_ptr + d * E_d_stride)  # E_d_stride == H_d_stride == C_d_stride == 1 for contiguous
        tl.store(dst_ptr + d * C_d_stride, val)


@triton.jit
def _batched_gemm_two_outputs_kernel(
    E_ptr,        # encoder_hidden_states: [B, T, D]
    H_ptr,        # hidden_states: [B, I, D]
    W_ptr,        # process_weight: [D, D]
    Y0_ptr,       # output for encoder stream: [B, T, D]
    Y1_ptr,       # output for image stream: [B, I, D]
    B, T, I, D,
    E_b_stride, E_t_stride, E_d_stride,
    H_b_stride, H_i_stride, H_d_stride,
    W0_stride, W1_stride,
    Y0_b_stride, Y0_t_stride, Y0_d_stride,
    Y1_b_stride, Y1_i_stride, Y1_d_stride,
    BLOCK_M: tl.constexpr,  # tile over sequence (rows) for yA
    BLOCK_N: tl.constexpr,  # tile over output columns (D)
    BLOCK_K: tl.constexpr,  # tile over input columns (D)
):
    # This kernel computes both outputs yA = E @ W and yB = H @ W in parallel for each batch.
    # We use per-row (batch, pos) programs and iterate over K tiles to accumulate.
    for b in range(0, B):
        # Output for encoder stream: [T, D]
        for t_start in range(0, T, BLOCK_M):
            t_offsets = t_start + tl.arange(0, BLOCK_M)  # [BLOCK_M]
            t_mask = t_offsets < T
            # Initialize accumulator for this tile
            acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            # Loop over K (hidden_dim) in tiles
            for k_start in range(0, D, BLOCK_K):
                k_offsets = k_start + tl.arange(0, BLOCK_K)  # [BLOCK_K]
                k_mask = k_offsets < D
                # Load weight tile W_sub of shape [BLOCK_K, BLOCK_N]
                w_ptrs = W_ptr + k_offsets[:, None] * W0_stride + tl.arange(0, BLOCK_N)[None, :] * W1_stride
                w_vals = tl.load(w_ptrs, mask=k_mask[:, None] & (tl.arange(0, BLOCK_N)[None, :] < D), other=0.0)
                # Load input vector E_vec for rows t_offsets
                e_ptrs = E_ptr + b * E_b_stride + t_offsets[:, None] * E_t_stride + k_offsets[None, :] * E_d_stride
                e_vals = tl.load(e_ptrs, mask=t_mask[:, None] & (k_offsets[None, :] < D), other=0.0)
                # Accumulate: acc += sum_k e_vals[k] * w_vals[k, :]
                # e_vals: [BLOCK_M, BLOCK_K], w_vals: [BLOCK_K, BLOCK_N]
                # Sum over K dimension
                for kk in range(0, BLOCK_K):
                    e_k = e_vals[:, kk]  # [BLOCK_M]
                    w_k = w_vals[kk, :]  # [BLOCK_N]
                    acc += e_k[:, None] * w_k[None, :]
            # Store results to Y0[b, t_offsets, :]
            y0_ptrs = Y0_ptr + b * Y0_b_stride + t_offsets[:, None] * Y0_t_stride + tl.arange(0, BLOCK_N)[None, :] * Y0_d_stride
            tl.store(y0_ptrs, acc, mask=t_mask[:, None])

        # Output for hidden stream: [I, D]
        for i_start in range(0, I, BLOCK_M):
            i_offsets = i_start + tl.arange(0, BLOCK_M)  # [BLOCK_M]
            i_mask = i_offsets < I
            acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
            for k_start in range(0, D, BLOCK_K):
                k_offsets = k_start + tl.arange(0, BLOCK_K)
                k_mask = k_offsets < D
                w_ptrs = W_ptr + k_offsets[:, None] * W0_stride + tl.arange(0, BLOCK_N)[None, :] * W1_stride
                w_vals = tl.load(w_ptrs, mask=k_mask[:, None] & (tl.arange(0, BLOCK_N)[None, :] < D), other=0.0)
                h_ptrs = H_ptr + b * H_b_stride + i_offsets[:, None] * H_i_stride + k_offsets[None, :] * H_d_stride
                h_vals = tl.load(h_ptrs, mask=i_mask[:, None] & (k_offsets[None, :] < D), other=0.0)
                for kk in range(0, BLOCK_K):
                    h_k = h_vals[:, kk]
                    w_k = w_vals[kk, :]
                    acc += h_k[:, None] * w_k[None, :]
            y1_ptrs = Y1_ptr + b * Y1_b_stride + i_offsets[:, None] * Y1_i_stride + tl.arange(0, BLOCK_N)[None, :] * Y1_d_stride
            tl.store(y1_ptrs, acc, mask=i_mask[:, None])


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor):
        """
        Triton-optimized forward:
        - Concatenates encoder_hidden_states and hidden_states along the sequence dimension in Triton.
        - Computes processed_encoder and processed_hidden via Triton batched GEMMs without torch.matmul.
        Returns:
          processed_encoder: [B, T, D]
          processed_hidden: [B, I, D]
        """
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "All inputs must be CUDA tensors for Triton kernels"
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        D = hidden_states.shape[2]
        assert encoder_hidden_states.shape[0] == B and encoder_hidden_states.shape[2] == D
        assert process_weight.shape[0] == D and process_weight.shape[1] == D

        # Ensure contiguous tensors for simpler stride handling
        E = encoder_hidden_states.contiguous()
        H = hidden_states.contiguous()
        W = process_weight.contiguous()

        # 1) Concatenate streams in Triton: [B, T+I, D]
        C_total = T + I
        C = torch.empty((B, C_total, D), device=E.device, dtype=torch.float32)

        grid_concat = (B, C_total)
        _concatenate_streams_kernel[grid_concat](
            E, H, C,
            B, T, I, D,
            E.stride(0), E.stride(1), E.stride(2),
            H.stride(0), H.stride(1), H.stride(2),
            C.stride(0), C.stride(1), C.stride(2),
            num_warps=4, num_stages=2,
        )

        # 2) Compute processed = C @ W^T in Triton (batched GEMM)
        # Output tensors
        processed_encoder = torch.empty((B, T, D), device=E.device, dtype=torch.float32)
        processed_hidden = torch.empty((B, I, D), device=E.device, dtype=torch.float32)

        # Launch Triton kernel to compute both outputs yA (encoder) and yB (hidden) without torch.matmul
        # Choose tile sizes robust for typical D up to a few thousands
        BLOCK_M = 64  # tile along sequence dimension
        BLOCK_N = 64  # tile along output columns (D)
        BLOCK_K = 64  # tile along input columns (D)
        grid = (B,)
        _batched_gemm_two_outputs_kernel[grid](
            C, E, H, W, processed_encoder, processed_hidden,
            B, T, I, D,
            C.stride(0), C.stride(1), C.stride(2),
            E.stride(0), E.stride(1), E.stride(2),
            H.stride(0), H.stride(1), H.stride(2),
            W.stride(0), W.stride(1),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
