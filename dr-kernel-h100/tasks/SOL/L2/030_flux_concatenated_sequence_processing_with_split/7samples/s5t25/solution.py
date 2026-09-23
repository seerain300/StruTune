import torch
import triton
import triton.language as tl


@triton.jit
def matmul_flattened_kernel(
    A_ptr,  # *const T: [M, K]
    B_ptr,  # *const T: [K, N]
    C_ptr,  # *T: [M, N]
    M: tl.int32, K: tl.int32, N: tl.int32,
    A_s0, A_s1,
    B_s0, B_s1,
    C_s0, C_s1,
    BLOCK_m: tl.constexpr, BLOCK_n: tl.constexpr, BLOCK_k: tl.constexpr,
):
    # Grid is (ceil_div(M, BLOCK_m), ceil_div(N, BLOCK_n))
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offsets = pid_m * BLOCK_m + tl.arange(0, BLOCK_m)  # rows in A (and C)
    n_offsets = pid_n * BLOCK_n + tl.arange(0, BLOCK_n)  # columns in B (and C)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_m, BLOCK_n), dtype=tl.float32)

    # Reduction over K in chunks
    for k0 in range(0, K, BLOCK_k):
        k_offsets = k0 + tl.arange(0, BLOCK_k)

        # Load A tile: [BLOCK_m, BLOCK_k]
        a_ptrs = A_ptr + m_offsets[:, None] * A_s0 + k_offsets[None, :] * A_s1
        a_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0).to(tl.float32)

        # Load B tile: [BLOCK_k, BLOCK_n]
        b_ptrs = B_ptr + k_offsets[:, None] * B_s0 + n_offsets[None, :] * B_s1
        b_mask = (k_offsets[:, None] < K) & (n_offsets[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0).to(tl.float32)

        # Accumulate
        acc += tl.dot(a, b)

    # Store C tile
    c_ptrs = C_ptr + m_offsets[:, None] * C_s0 + n_offsets[None, :] * C_s1
    c_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    # We store in original dtype of C_ptr; cast to float32 if needed (assumes float32 output)
    tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def compute_encoder_stream_kernel(
    hidden_ptr,         # *const T: [B, I, H]
    weight_ptr,         # *const T: [H, H]
    out_ptr,            # *T: [B, T, H]
    B: tl.int32, I: tl.int32, T: tl.int32, H: tl.int32,
    hidden_s0, hidden_s1, hidden_s2,
    weight_s0, weight_s1,
    out_s0, out_s1, out_s2,
    BLOCK_t: tl.constexpr, BLOCK_h: tl.constexpr, BLOCK_k: tl.constexpr,
):
    # Grid over (B, T, H) tiles
    pid_b = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_h = tl.program_id(2)

    t_offsets = pid_t * BLOCK_t + tl.arange(0, BLOCK_t)   # [BLOCK_t]
    h_offsets = pid_h * BLOCK_h + tl.arange(0, BLOCK_h)   # [BLOCK_h]

    # For each (b, t), we sum over K=H: out[b, t, h] = sum_k hidden[b, t, k] * weight[k, h]
    acc = tl.zeros((BLOCK_t, BLOCK_h), dtype=tl.float32)

    # Reduction over k
    for k0 in range(0, H, BLOCK_k):
        k_offsets = k0 + tl.arange(0, BLOCK_k)

        # Load hidden tile: [BLOCK_t, BLOCK_k] -> hidden[b, t, k]
        # m = b, n = t, k = k
        # Note: b is scalar here; we'll broadcast later
        # We build pointers for each (b, t, k)
        # hidden[b, t, k] => b*hidden_s0 + t*hidden_s1 + k*hidden_s2
        # We need to compute b index: b = pid_b
        b_idx = pid_b  # scalar
        # Create meshgrid for (t, k)
        t_grid = t_offsets[:, None]  # [BLOCK_t, 1]
        k_grid = k_offsets[None, :]  # [1, BLOCK_k]
        hidden_ptrs = hidden_ptr + b_idx * hidden_s0 + t_grid * hidden_s1 + k_grid * hidden_s2
        hidden_mask = (t_offsets[:, None] < T) & (k_offsets[None, :] < H)
        hidden_vals = tl.load(hidden_ptrs, mask=hidden_mask, other=0.0).to(tl.float32)  # [BLOCK_t, BLOCK_k]

        # Load weight tile: [BLOCK_k, BLOCK_h]
        weight_ptrs = weight_ptr + k_offsets[:, None] * weight_s0 + h_offsets[None, :] * weight_s1
        weight_mask = (k_offsets[:, None] < H) & (h_offsets[None, :] < H)
        weight_vals = tl.load(weight_ptrs, mask=weight_mask, other=0.0).to(tl.float32)  # [BLOCK_k, BLOCK_h]

        # Accumulate: (BLOCK_t, BLOCK_k) x (BLOCK_k, BLOCK_h) -> (BLOCK_t, BLOCK_h)
        acc += tl.dot(hidden_vals, weight_vals)

    # Store to output out[b, t, h]
    out_ptrs = out_ptr + b_idx * out_s0 + t_offsets[:, None] * out_s1 + h_offsets[None, :] * out_s2
    out_mask = (t_offsets[:, None] < T) & (h_offsets[None, :] < H)
    tl.store(out_ptrs, acc, mask=out_mask)


@triton.jit
def compute_image_stream_kernel(
    hidden_ptr,         # *const T: [B, I, H]
    weight_ptr,         # *const T: [H, H]
    out_ptr,            # *T: [B, I, H]
    B: tl.int32, I: tl.int32, T: tl.int32, H: tl.int32,
    hidden_s0, hidden_s1, hidden_s2,
    weight_s0, weight_s1,
    out_s0, out_s1, out_s2,
    BLOCK_i: tl.constexpr, BLOCK_h: tl.constexpr, BLOCK_k: tl.constexpr,
):
    # Grid over (B, I, H) tiles
    pid_b = tl.program_id(0)
    pid_i = tl.program_id(1)
    pid_h = tl.program_id(2)

    i_offsets = pid_i * BLOCK_i + tl.arange(0, BLOCK_i)   # [BLOCK_i]
    h_offsets = pid_h * BLOCK_h + tl.arange(0, BLOCK_h)   # [BLOCK_h]

    acc = tl.zeros((BLOCK_i, BLOCK_h), dtype=tl.float32)

    # Reduction over k
    for k0 in range(0, H, BLOCK_k):
        k_offsets = k0 + tl.arange(0, BLOCK_k)

        # Load hidden tile: hidden[b, i, k] -> [BLOCK_i, BLOCK_k]
        b_idx = pid_b
        i_grid = i_offsets[:, None]  # [BLOCK_i, 1]
        k_grid = k_offsets[None, :]  # [1, BLOCK_k]
        hidden_ptrs = hidden_ptr + b_idx * hidden_s0 + i_grid * hidden_s1 + k_grid * hidden_s2
        hidden_mask = (i_offsets[:, None] < I) & (k_offsets[None, :] < H)
        hidden_vals = tl.load(hidden_ptrs, mask=hidden_mask, other=0.0).to(tl.float32)  # [BLOCK_i, BLOCK_k]

        # Load weight tile: [BLOCK_k, BLOCK_h]
        weight_ptrs = weight_ptr + k_offsets[:, None] * weight_s0 + h_offsets[None, :] * weight_s1
        weight_mask = (k_offsets[:, None] < H) & (h_offsets[None, :] < H)
        weight_vals = tl.load(weight_ptrs, mask=weight_mask, other=0.0).to(tl.float32)  # [BLOCK_k, BLOCK_h]

        acc += tl.dot(hidden_vals, weight_vals)

    # Store to output out[b, i, h]
    out_ptrs = out_ptr + b_idx * out_s0 + i_offsets[:, None] * out_s1 + h_offsets[None, :] * out_s2
    out_mask = (i_offsets[:, None] < I) & (h_offsets[None, :] < H)
    tl.store(out_ptrs, acc, mask=out_mask)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,    # [B, I, H]
        encoder_hidden_states: torch.Tensor,  # [B, T, H]
        process_weight: torch.Tensor,  # [H, H]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized forward that avoids torch operations (no torch.cat, no torch.matmul, no slicing).
        Computes:
          concatenated = cat([encoder_hidden_states, hidden_states], dim=1)  # [B, T+I, H]
          processed = concatenated @ process_weight.T  # [B, T+I, H]
          processed_encoder = processed[:, :T, :]      # [B, T, H]
          processed_hidden = processed[:, T:, :]       # [B, I, H]
        But all are computed via Triton kernels.
        """
        # Ensure tensors are on CUDA and contiguous
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Tensors must be on CUDA."
        B = hidden_states.shape[0]
        I = hidden_states.shape[1]
        T = encoder_hidden_states.shape[1]
        H = hidden_states.shape[2]
        assert process_weight.shape[0] == H and process_weight.shape[1] == H, "process_weight must be [H, H]."

        # Prepare outputs (float32 for numerical stability; match input dtype if you prefer)
        processed_encoder = torch.empty((B, T, H), device=hidden_states.device, dtype=torch.float32)
        processed_hidden = torch.empty((B, I, H), device=hidden_states.device, dtype=torch.float32)

        # Launch Triton kernel for processed_encoder: out[b, t, h] = sum_k hidden[b, t, k] * weight[k, h]
        # We treat each (b, t) row and reduce over k=H. Use grid over (B, T, H) tiles.
        BLOCK_t = 32
        BLOCK_h = 64
        BLOCK_k = 32
        grid_encoder = (B, triton.cdiv(T, BLOCK_t), triton.cdiv(H, BLOCK_h))
        compute_encoder_stream_kernel[grid_encoder](
            hidden_states, process_weight,
            processed_encoder,
            B, I, T, H,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            process_weight.stride(0), process_weight.stride(1),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            BLOCK_t=BLOCK_t, BLOCK_h=BLOCK_h, BLOCK_k=BLOCK_k,
            num_warps=4, num_stages=2,
        )

        # Launch Triton kernel for processed_hidden: out[b, i, h] = sum_k hidden[b, i, k] * weight[k, h]
        grid_image = (B, triton.cdiv(I, BLOCK_t), triton.cdiv(H, BLOCK_h))
        compute_image_stream_kernel[grid_image](
            hidden_states, process_weight,
            processed_hidden,
            B, I, T, H,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            process_weight.stride(0), process_weight.stride(1),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_i=BLOCK_t, BLOCK_h=BLOCK_h, BLOCK_k=BLOCK_k,  # BLOCK_i replaced by i tile
            num_warps=4, num_stages=2,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
