import torch
import triton
import triton.language as tl


@triton.jit
def _gemm_elementwise_A_shape(B: tl.constexpr, M: tl.constexpr, N: tl.constexpr,
                               A_ptr, WT_ptr, Out_ptr,
                               stride_A_b, stride_A_m, stride_A_n,
                               stride_W_k, stride_W_n,
                               stride_O_b, stride_O_m, stride_O_n):
    # Each program computes one output element: out[b, m, n]
    b = tl.program_id(0)
    m = tl.program_id(1)
    n = tl.program_id(2)

    # Bounds guard (in case grid slightly overshoots)
    if b >= B or m >= M or n >= N:
        return

    # Reduction over k from 0 to N-1
    acc = tl.zeros((), dtype=tl.float32)
    for k in range(0, N):
        # Load A[b, m, k] and WT[k, n]
        a_val = tl.load(A_ptr + b * stride_A_b + m * stride_A_m + k * stride_A_n)
        w_val = tl.load(WT_ptr + k * stride_W_k + n * stride_W_n)
        # Accumulate in fp32
        acc += (a_val.to(tl.float32) * w_val.to(tl.float32))

    # Store result
    tl.store(Out_ptr + b * stride_O_b + m * stride_O_m + n * stride_O_n, acc)


@triton.jit
def _gemm_elementwise_B_shape(B: tl.constexpr, M: tl.constexpr, N: tl.constexpr,
                               A_ptr, WT_ptr, Out_ptr,
                               stride_A_b, stride_A_m, stride_A_n,
                               stride_W_k, stride_W_n,
                               stride_O_b, stride_O_m, stride_O_n):
    # Each program computes one output element: out[b, m, n] for A with shape [B, M, N]
    b = tl.program_id(0)
    m = tl.program_id(1)
    n = tl.program_id(2)

    if b >= B or m >= M or n >= N:
        return

    acc = tl.zeros((), dtype=tl.float32)
    for k in range(0, N):
        a_val = tl.load(A_ptr + b * stride_A_b + m * stride_A_m + k * stride_A_n)
        w_val = tl.load(WT_ptr + k * stride_W_k + n * stride_W_n)
        acc += (a_val.to(tl.float32) * w_val.to(tl.float32))

    tl.store(Out_ptr + b * stride_O_b + m * stride_O_m + n * stride_O_n, acc)


class ModelNew(torch.nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,      # [B, I, D]
        encoder_hidden_states: torch.Tensor,  # [B, T, D]
        process_weight: torch.Tensor        # [D, D]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Triton requires CUDA tensors
        assert hidden_states.is_cuda and encoder_hidden_states.is_cuda and process_weight.is_cuda, "Inputs must be CUDA tensors for Triton."
        B = hidden_states.shape[0]
        T = encoder_hidden_states.shape[1]
        I = hidden_states.shape[1]
        D = hidden_states.shape[2]

        # Ensure contiguity for simple stride arithmetic
        enc = encoder_hidden_states.contiguous()      # [B, T, D]
        hst = hidden_states.contiguous()             # [B, I, D]
        WT = process_weight.t().contiguous()         # [D, D]

        # Allocate outputs
        processed_encoder = torch.empty((B, T, D), dtype=enc.dtype, device=enc.device)
        processed_hidden = torch.empty((B, I, D), dtype=hst.dtype, device=hst.device)

        # Launch Triton kernels: elementwise reduction over K=D
        grid_enc = (B, T, D)
        _gemm_elementwise_A_shape[grid_enc](
            B, T, D,
            enc, WT, processed_encoder,
            enc.stride(0), enc.stride(1), enc.stride(2),
            WT.stride(0), WT.stride(1),
            processed_encoder.stride(0), processed_encoder.stride(1), processed_encoder.stride(2),
            num_warps=1, num_stages=1,
        )

        grid_hid = (B, I, D)
        _gemm_elementwise_B_shape[grid_hid](
            B, I, D,
            hst, WT, processed_hidden,
            hst.stride(0), hst.stride(1), hst.stride(2),
            WT.stride(0), WT.stride(1),
            processed_hidden.stride(0), processed_hidden.stride(1), processed_hidden.stride(2),
            num_warps=1, num_stages=1,
        )

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
