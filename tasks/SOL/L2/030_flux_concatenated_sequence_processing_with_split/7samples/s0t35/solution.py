import torch
import triton
import triton.language as tl


@triton.jit
def compute_encoder_kernel(
    out_ptr,       # *fp32, processed_encoder: [B, T, K]
    x1_ptr,        # *fp32, encoder_hidden_states: [B, T, K]
    w_ptr,         # *fp32, process_weight.T: [K, K]
    B: tl.constexpr, # batch size
    T: tl.constexpr, # text_seq_len
    K: tl.constexpr, # hidden_dim
    X1_s0, X1_s1, X1_s2,
    W_s0, W_s1,
    O_s0, O_s1, O_s2,
    BLOCK_T: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 3D grid: (B, tiles over T, tiles over K)
    b = tl.program_id(0)
    t_block = tl.program_id(1)
    k_block = tl.program_id(2)

    t_offsets = t_block * BLOCK_T + tl.arange(0, BLOCK_T)
    k_offsets = k_block * BLOCK_K + tl.arange(0, BLOCK_K)

    mask_t = t_offsets < T
    mask_k = k_offsets < K

    # Accumulator per t
    acc = tl.zeros((BLOCK_T,), dtype=tl.float32)

    # Loop over k_offsets and accumulate dot products
    for kk in range(0, BLOCK_K):
        k_idx = k_offsets[kk]
        if not mask_k[kk]:
            continue
        # Load w[:, k_idx] (i.e., each row k of W)
        w_row_base = k_idx * W_s0
        w_row = tl.load(w_ptr + w_row_base + tl.arange(0, K) * W_s1, mask=(tl.arange(0, K) < K), other=0.0)  # load entire row
        # For each t in this tile, compute x1[b, t, k_idx] and accumulate
        x1_row_base = b * X1_s0 + t_offsets * X1_s1 + k_idx * X1_s2
        x1_vals = tl.load(x1_ptr + x1_row_base, mask=mask_t, other=0.0)  # shape [BLOCK_T]
        # acc += x1_vals * w_row
        # Since w_row is [K], we need per-t element-wise multiply with corresponding row of W at k_idx.
        # To multiply vector x1_vals with w_row for each t, we broadcast:
        # Build a [BLOCK_T, K] matrix where each row is x1_vals[:, None] and multiply elementwise with w_row[None, :]
        # However, tl.dot requires compatible shapes; here we'll do elementwise multiply and reduce across K:
        # Create a mask for K: we need to multiply x1_vals per t with w_row across all K (but x1_vals only depends on k_idx).
        # Better approach: perform per-k scalar accumulation:
        # acc += x1_vals for each t at this k_idx multiplied by corresponding W[k, k_idx]
        # We can do this by looping j in K:
        for j in range(0, K):
            # w_j = w_ptr[k_idx, j] = w_ptr + k_idx*W_s0 + j*W_s1
            w_j = tl.load(w_ptr + k_idx * W_s0 + j * W_s1)
            # x1_scalar = x1_ptr[b, t, k_idx] = b*X1_s0 + t*X1_s1 + k_idx*X1_s2
            # Load x1_scalar per t
            x1_scalar = tl.load(x1_ptr + b * X1_s0 + t_offsets * X1_s1 + k_idx * X1_s2, mask=mask_t, other=0.0)
            # acc += x1_scalar * w_j
            acc += x1_scalar * w_j

    # Store results: out[b, t, k] = acc[t]
    for tt in range(0, BLOCK_T):
        t = t_offsets[tt]
        if mask_t[tt]:
            out_base = b * O_s0 + t * O_s1
            for kk in range(0, BLOCK_K):
                k = k_offsets[kk]
                if mask_k[kk]:
                    tl.store(out_ptr + out_base + k * O_s2, acc[tt])


@triton.jit
def compute_hidden_kernel(
    out_ptr,       # *fp32, processed_hidden: [B, I, K]
    x2_ptr,        # *fp32, hidden_states: [B, I, K]
    w_ptr,         # *fp32, process_weight.T: [K, K]
    B: tl.constexpr, # batch size
    I: tl.constexpr, # img_seq_len
    K: tl.constexpr, # hidden_dim
    X2_s0, X2_s1, X2_s2,
    W_s0, W_s1,
    O_s0, O_s1, O_s2,
    BLOCK_I: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # 3D grid: (B, tiles over I, tiles over K)
    b = tl.program_id(0)
    i_block = tl.program_id(1)
    k_block = tl.program_id(2)

    i_offsets = i_block * BLOCK_I + tl.arange(0, BLOCK_I)
    k_offsets = k_block * BLOCK_K + tl.arange(0, BLOCK_K)

    mask_i = i_offsets < I
    mask_k = k_offsets < K

    # Accumulator per i
    acc = tl.zeros((BLOCK_I,), dtype=tl.float32)

    # Loop over k_offsets and accumulate dot products
    for kk in range(0, BLOCK_K):
        k_idx = k_offsets[kk]
        if not mask_k[kk]:
            continue
        # Load w[:, k_idx] (rows of W)
        w_row_base = k_idx * W_s0
        w_row = tl.load(w_ptr + w_row_base + tl.arange(0, K) * W_s1, mask=(tl.arange(0, K) < K), other=0.0)
        # For each i in this tile, compute x2[b, i, k_idx] and accumulate
        x2_row_base = b * X2_s0 + i_offsets * X2_s1 + k_idx * X2_s2
        x2_vals = tl.load(x2_ptr + x2_row_base, mask=mask_i, other=0.0)  # shape [BLOCK_I]
        # acc += x2_vals * w_row
        for j in range(0, K):
            w_j = tl.load(w_ptr + k_idx * W_s0 + j * W_s1)
            x2_scalar = tl.load(x2_ptr + b * X2_s0 + i_offsets * X2_s1 + k_idx * X2_s2, mask=mask_i, other=0.0)
            acc += x2_scalar * w_j

    # Store results: out[b, i, k] = acc[i]
    for ii in range(0, BLOCK_I):
        i = i_offsets[ii]
        if mask_i[ii]:
            out_base = b * O_s0 + i * O_s1
            for kk in range(0, BLOCK_K):
                k = k_offsets[kk]
                if mask_k[kk]:
                    tl.store(out_ptr + out_base + k * O_s2, acc[ii])


class ModelNew(torch.nn.Module):
    def forward(self, encoder_hidden_states: torch.Tensor,
                hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run function:
        - Concatenate along sequence dim is not needed; we compute linear directly.
        - We perform the heavy computation in Triton via two kernels.
        Returns (processed_encoder, processed_hidden).
        """
        # Ensure inputs are contiguous and on CUDA
        assert encoder_hidden_states.is_cuda and hidden_states.is_cuda and process_weight.is_cuda, \
            "All tensors must be on CUDA device for Triton kernels."
        B = encoder_hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        K = encoder_hidden_states.shape[2]
        assert hidden_states.shape == (B, I, K), "hidden_states must have shape [batch, img_seq_len, hidden_dim]"
        assert process_weight.shape == (K, K), "process_weight must have shape [hidden_dim, hidden_dim]"

        # Prepare inputs: ensure dtype float32 for kernels
        x1 = encoder_hidden_states.contiguous().to(torch.float32)
        x2 = hidden_states.contiguous().to(torch.float32)
        w = process_weight.contiguous().to(torch.float32)

        # Allocate outputs (float32)
        processed_encoder = torch.empty((B, T, K), device=x1.device, dtype=torch.float32)
        processed_hidden = torch.empty((B, I, K), device=x2.device, dtype=torch.float32)

        # Launch kernels
        BLOCK_T = 64
        BLOCK_I = 64
        BLOCK_K = 64

        # Kernel 1: compute processed_encoder
        grid1 = (B, triton.cdiv(T, BLOCK_T), triton.cdiv(K, BLOCK_K))
        compute_encoder_kernel[grid1](
            processed_encoder, x1, w,
            B, T, K,
            x1.stride(0), x1.stride(1), x1.stride(2),
            w.stride(0), w.stride(1),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            BLOCK_T=BLOCK_T, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # Kernel 2: compute processed_hidden
        grid2 = (B, triton.cdiv(I, BLOCK_I), triton.cdiv(K, BLOCK_K))
        compute_hidden_kernel[grid2](
            processed_hidden, x2, w,
            B, I, K,
            x2.stride(0), x2.stride(1), x2.stride(2),
            w.stride(0), w.stride(1),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_I=BLOCK_I, BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2,
        )

        # Cast outputs back to original input dtypes to match original behavior
        processed_encoder = processed_encoder.to(encoder_hidden_states.dtype)
        processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
