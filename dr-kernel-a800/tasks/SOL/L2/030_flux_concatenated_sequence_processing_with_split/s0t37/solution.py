import torch
import triton
import triton.language as tl


@triton.jit
def matmul_bmn_kernel(
    A_ptr, W_ptr, C_ptr,
    B, L_out, D,
    A_stride_b, A_stride_m, A_stride_d,
    W_stride_k, W_stride_n,
    C_stride_b, C_stride_m, C_stride_d,
    BLOCK_K: tl.constexpr,
):
    # One program computes one output element: c[b, m, n]
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)  # we iterate n with a while loop, pid_n controls the initial block
    b = pid_b
    m = pid_m

    # n starts from pid_n * BLOCK_N, we keep a running n and use masks
    n = pid_n * BLOCK_K  # actually, we'll loop; pid_n sets starting block
    # We loop over n in tiles, but for simplicity and robustness, we process one n per program by iterating k:
    # However, Triton requires a static range; so we implement a single-n program and loop over k inside.
    # For better performance, one could use a 2D tile; here we keep it simple to ensure correctness.

    # Accumulator for a single n
    acc = tl.zeros((), dtype=tl.float32)

    k = 0
    while k < D:
        # Load A[b, m, k]
        a_val = tl.load(
            A_ptr + b * A_stride_b + m * A_stride_m + k * A_stride_d,
            mask=(b < B) & (m < L_out),
            other=0.0
        )
        # Load W[k, n]
        w_val = tl.load(
            W_ptr + k * W_stride_k + n * W_stride_n,
            mask=(k < D) & (n < D),
            other=0.0
        )
        acc += a_val * w_val
        k += 1

    # Store result to C[b, m, n]
    tl.store(
        C_ptr + b * C_stride_b + m * C_stride_m + n * C_stride_d,
        acc,
        mask=(b < B) & (m < L_out) & (n < D)
    )


# Triton kernel for processed_encoder = encoder_hidden_states @ process_weight.T
@triton.jit
def matmul_encoder_kernel(
    E_ptr, W_ptr, C_encoder_ptr,
    B, L_txt, D,
    E_stride_b, E_stride_s, E_stride_d,
    W_stride_k, W_stride_n,
    C_encoder_stride_b, C_encoder_stride_s, C_encoder_stride_d,
    BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)  # m in [0, L_txt)
    pid_n = tl.program_id(2)

    b = pid_b
    m = pid_m
    n = pid_n * BLOCK_K

    acc = tl.zeros((), dtype=tl.float32)
    k = 0
    while k < D:
        e_val = tl.load(
            E_ptr + b * E_stride_b + m * E_stride_s + k * E_stride_d,
            mask=(b < B) & (m < L_txt),
            other=0.0
        )
        w_val = tl.load(
            W_ptr + k * W_stride_k + n * W_stride_n,
            mask=(k < D) & (n < D),
            other=0.0
        )
        acc += e_val * w_val
        k += 1

    tl.store(
        C_encoder_ptr + b * C_encoder_stride_b + m * C_encoder_stride_s + n * C_encoder_stride_d,
        acc,
        mask=(b < B) & (m < L_txt) & (n < D)
    )


# Triton kernel for processed_hidden = hidden_states @ process_weight.T
@triton.jit
def matmul_hidden_kernel(
    H_ptr, W_ptr, C_hidden_ptr,
    B, L_img, D,
    H_stride_b, H_stride_s, H_stride_d,
    W_stride_k, W_stride_n,
    C_hidden_stride_b, C_hidden_stride_s, C_hidden_stride_d,
    BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)  # m in [0, L_img)
    pid_n = tl.program_id(2)

    b = pid_b
    m = pid_m
    n = pid_n * BLOCK_K

    acc = tl.zeros((), dtype=tl.float32)
    k = 0
    while k < D:
        h_val = tl.load(
            H_ptr + b * H_stride_b + m * H_stride_s + k * H_stride_d,
            mask=(b < B) & (m < L_img),
            other=0.0
        )
        w_val = tl.load(
            W_ptr + k * W_stride_k + n * W_stride_n,
            mask=(k < D) & (n < D),
            other=0.0
        )
        acc += h_val * w_val
        k += 1

    tl.store(
        C_hidden_ptr + b * C_hidden_stride_b + m * C_hidden_stride_s + n * C_hidden_stride_d,
        acc,
        mask=(b < B) & (m < L_img) & (n < D)
    )


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        process_weight: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version that computes:
          processed_encoder = encoder_hidden_states @ process_weight.T
          processed_hidden = hidden_states @ process_weight.T
        without using any torch ops on tensors in forward.
        """
        # Shapes
        B = hidden_states.shape[0]
        L_img = hidden_states.shape[1]
        L_txt = encoder_hidden_states.shape[1]
        D = hidden_states.shape[2]
        assert process_weight.shape == (D, D), "process_weight must be [D, D]"
        assert encoder_hidden_states.shape[2] == D, "encoder_hidden_states and hidden_states must have same hidden_dim"

        # Allocate outputs
        processed_encoder = torch.empty((B, L_txt, D), dtype=torch.float32, device=hidden_states.device)
        processed_hidden = torch.empty((B, L_img, D), dtype=torch.float32, device=hidden_states.device)

        # Launch Triton kernels

        # For processed_encoder = E @ W^T
        grid_encoder = (B, L_txt, D)
        matmul_encoder_kernel[grid_encoder](
            encoder_hidden_states, process_weight,
            processed_encoder,
            B, L_txt, D,
            encoder_hidden_states.stride(0), encoder_hidden_states.stride(1), encoder_hidden_states.stride(2),
            process_weight.stride(0), process_weight.stride(1),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            BLOCK_K=64, num_warps=2, num_stages=2
        )

        # For processed_hidden = H @ W^T
        grid_hidden = (B, L_img, D)
        matmul_hidden_kernel[grid_hidden](
            hidden_states, process_weight,
            processed_hidden,
            B, L_img, D,
            hidden_states.stride(0), hidden_states.stride(1), hidden_states.stride(2),
            process_weight.stride(0), process_weight.stride(1),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            BLOCK_K=64, num_warps=2, num_stages=2
        )

        # Return with original dtype (the original code uses float32 by default)
        processed_encoder = processed_encoder.to(hidden_states.dtype)
        processed_hidden = processed_hidden.to(hidden_states.dtype)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
