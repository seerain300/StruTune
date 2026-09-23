import torch
import triton
import triton.language as tl


# Elementwise-unrolled GEMM kernel: out = A @ WT
# A: [B, M, D] (stream-specific)
# WT: [D, D] (process_weight transposed)
# out: [B, M, D]
@triton.jit
def _matmul_elementwise_unrolled8_kernel(
    A_ptr, WT_ptr, Out_ptr,
    B, M, D,
    A_s0, A_s1, A_s2,
    WT_s0, WT_s1,
    Out_s0, Out_s1, Out_s2,
    UNROLL: tl.constexpr,
):
    # Grid: (B, M, D) -> each program handles one (b, m, d) output element
    b = tl.program_id(0)
    m = tl.program_id(1)
    d = tl.program_id(2)

    # Initialize accumulator in fp32
    acc = tl.zeros((), dtype=tl.float32)

    # Base pointer for this row in A
    base_A = A_ptr + b * A_s0 + m * A_s1

    # Unrolled reduction over K with UNROLL=8 to reduce loop overhead
    for k in range(0, D, UNROLL):
        if (k + 0) < D:
            a = tl.load(base_A + (k + 0) * A_s2, mask=True, other=0.0)
            w = tl.load(WT_ptr + (k + 0) * WT_s0 + d * WT_s1, mask=True, other=0.0)
            acc += a.to(tl.float32) * w.to(tl.float32)
        if (k + 1) < D:
            a = tl.load(base_A + (k + 1) * A_s2, mask=True, other=0.0)
            w = tl.load(WT_ptr + (k + 1) * WT_s0 + d * WT_s1, mask=True, other=0.0)
            acc += a.to(tl.float32) * w.to(tl.float32)
        if (k + 2) < D:
            a = tl.load(base_A + (k + 2) * A_s2, mask=True, other=0.0)
            w = tl.load(WT_ptr + (k + 2) * WT_s0 + d * WT_s1, mask=True, other=0.0)
            acc += a.to(tl.float32) * w.to(tl.float32)
        if (k + 3) < D:
            a = tl.load(base_A + (k + 3) * A_s2, mask=True, other=0.0)
            w = tl.load(WT_ptr + (k + 3) * WT_s0 + d * WT_s1, mask=True, other=0.0)
            acc += a.to(tl.float32) * w.to(tl.float32)
        if (k + 4) < D:
            a = tl.load(base_A + (k + 4) * A_s2, mask=True, other=0.0)
            w = tl.load(WT_ptr + (k + 4) * WT_s0 + d * WT_s1, mask=True, other=0.0)
            acc += a.to(tl.float32) * w.to(tl.float32)
        if (k + 5) < D:
            a = tl.load(base_A + (k + 5) * A_s2, mask=True, other=0.0)
            w = tl.load(WT_ptr + (k + 5) * WT_s0 + d * WT_s1, mask=True, other=0.0)
            acc += a.to(tl.float32) * w.to(tl.float32)
        if (k + 6) < D:
            a = tl.load(base_A + (k + 6) * A_s2, mask=True, other=0.0)
            w = tl.load(WT_ptr + (k + 6) * WT_s0 + d * WT_s1, mask=True, other=0.0)
            acc += a.to(tl.float32) * w.to(tl.float32)
        if (k + 7) < D:
            a = tl.load(base_A + (k + 7) * A_s2, mask=True, other=0.0)
            w = tl.load(WT_ptr + (k + 7) * WT_s0 + d * WT_s1, mask=True, other=0.0)
            acc += a.to(tl.float32) * w.to(tl.float32)

    # Store result to Out[b, m, d]
    out_ptr = Out_ptr + b * Out_s0 + m * Out_s1 + d * Out_s2
    tl.store(out_ptr, acc)


def _run_triton_elementwise(B, M, D, A: torch.Tensor, WT: torch.Tensor, out: torch.Tensor):
    """
    Launch the elementwise-unrolled GEMM kernel. Assumes A is [B, M, D], WT is [D, D], out is [B, M, D].
    All tensors must be contiguous. We accumulate in fp32; out is created as fp32.
    """
    A = A.contiguous()
    WT = WT.contiguous()
    # Ensure output is fp32 for numerical fidelity
    if out.dtype != torch.float32:
        out = out.to(torch.float32)

    # Grid covers all (b, m, d) outputs
    grid = (B, M, D)
    _matmul_elementwise_unrolled8_kernel[grid](
        A, WT, out,
        B, M, D,
        A.stride(0), A.stride(1), A.stride(2),
        WT.stride(0), WT.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        UNROLL=8,
        num_warps=1, num_stages=1,
    )
    return out


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized version of the original run:
        1) Compute processed_encoder = encoder_hidden_states @ process_weight.T
        2) Compute processed_hidden = hidden_states @ process_weight.T
        All computation is done by Triton kernels; no torch.cat or torch.matmul is used.
        """
        # Transpose process_weight to [D, D] and make contiguous
        WT = process_weight.t().contiguous()  # [D, D]

        # Compute encoder stream: [B, T, D] @ [D, D] -> [B, T, D]
        B, T, D = encoder_hidden_states.shape
        processed_encoder = torch.empty((B, T, D), device=encoder_hidden_states.device, dtype=torch.float32)
        _run_triton_elementwise(B, T, D, encoder_hidden_states, WT, processed_encoder)

        # Compute hidden stream: [B, I, D] @ [D, D] -> [B, I, D]
        B2, I, D2 = hidden_states.shape
        assert B2 == B and D2 == D, "Batch and hidden_dim must match across inputs."
        processed_hidden = torch.empty((B, I, D), device=hidden_states.device, dtype=torch.float32)
        _run_triton_elementwise(B, I, D, hidden_states, WT, processed_hidden)

        return processed_encoder, processed_hidden


def run(*args):
    return ModelNew()(*args)
