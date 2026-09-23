import torch
import triton
import triton.language as tl


@triton.jit
def _concatenate_seqs_kernel(
    encoder_hidden_states_ptr,  # [B, T, K]
    hidden_states_ptr,          # [B, P, K]
    Acat_ptr,                   # [B, T+P, K]
    B: tl.constexpr, T: tl.constexpr, P: tl.constexpr, K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # program ids: (b, l)
    b = tl.program_id(0)
    l = tl.program_id(1)

    # guard
    if b >= B or l >= (T + P):
        return

    # choose source tensor
    is_encoder = l < T

    # compute base offsets
    # encoder: [B, T, K]
    # hidden: [B, P, K]
    if is_encoder:
        # Acat[b, l, :] = encoder_hidden_states[b, l, :]
        src_ptr = encoder_hidden_states_ptr + b * T * K + l * K
    else:
        # Acat[b, l, :] = hidden_states[b, l - T, :]
        src_p = l - T
        src_ptr = hidden_states_ptr + b * P * K + src_p * K

    # destination
    dest_ptr = Acat_ptr + b * (T + P) * K + l * K

    # copy K elements
    k_offsets = tl.arange(0, BLOCK_K)
    mask = k_offsets < K
    vals = tl.load(src_ptr + k_offsets, mask=mask, other=0.0)
    tl.store(dest_ptr + k_offsets, vals, mask=mask)


@triton.jit
def _matmul_kernel(
    A_ptr,            # [M, K], M = B*(T+P)
    W_ptr,            # [K, K]
    C_ptr,            # [M, K]
    M_total: tl.constexpr, K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Grid is (B, M_total), each program computes one row of C
    b = tl.program_id(0)
    row = tl.program_id(1)
    if b >= B or row >= M_total:
        return

    # base pointers
    # A row pointer: A[row, :]
    A_row_ptr = A_ptr + row * K

    # C row pointer: C[row, :]
    C_row_ptr = C_ptr + row * K

    # Accumulator
    acc = tl.zeros([K], dtype=tl.float32)

    # Reduce over K in tiles
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < K

        # Load W[k_offsets, :] as a row
        W_row_ptr = W_ptr + k_offsets * K
        W_vals = tl.load(W_row_ptr, mask=k_mask, other=0.0)  # shape [BLOCK_K]

        # Load A[row, k_offsets]
        A_vals = tl.load(A_row_ptr + k_offsets, mask=k_mask, other=0.0)  # shape [BLOCK_K]

        # Fused multiply-add
        acc += W_vals * A_vals

    # Store result
    tl.store(C_row_ptr, acc, mask=True)


@triton.jit
def _split_encoder_kernel(
    C_ptr,            # [M, K], M = B*(T+P)
    processed_encoder_ptr,  # [B, T, K]
    B: tl.constexpr, T: tl.constexpr, K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    b = tl.program_id(0)
    t = tl.program_id(1)
    if b >= B or t >= T:
        return

    # row index in C: row = b*(T+P) + t
    row = b * (T + 0) + t  # T+0 is T; but we compute exactly T
    C_row_ptr = C_ptr + row * K

    # Destination: processed_encoder[b, t, :]
    dest_ptr = processed_encoder_ptr + b * T * K + t * K

    k_offsets = tl.arange(0, BLOCK_K)
    k_mask = k_offsets < K
    vals = tl.load(C_row_ptr + k_offsets, mask=k_mask, other=0.0)
    tl.store(dest_ptr + k_offsets, vals, mask=k_mask)


@triton.jit
def _split_hidden_kernel(
    C_ptr,            # [M, K], M = B*(T+P)
    processed_hidden_ptr,    # [B, P, K]
    B: tl.constexpr, P: tl.constexpr, K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    b = tl.program_id(0)
    p = tl.program_id(1)
    if b >= B or p >= P:
        return

    # row index in C: row = b*(T+P) + T + p
    row = b * (P + 0) + (P + p)  # simplifies to b*T + T + p, but we need total = T+P
    # Correct: total = T + P; row = b * total + t; but here we split hidden, so row = b * total + (T + p) would be wrong.
    # We need row = b * (T + P) + (T + p) to access Acat rows corresponding to hidden. Wait: actually we need the hidden rows.
    # The concatenation is [encoder, hidden], so the hidden rows start at index T in C. Therefore, for hidden[b, p, :], the corresponding row in C is b*(T+P) + (T + p).
    row = b * (T + P) + (T + p)

    C_row_ptr = C_ptr + row * K

    dest_ptr = processed_hidden_ptr + b * P * K + p * K

    k_offsets = tl.arange(0, BLOCK_K)
    k_mask = k_offsets < K
    vals = tl.load(C_row_ptr + k_offsets, mask=k_mask, other=0.0)
    tl.store(dest_ptr + k_offsets, vals, mask=k_mask)


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # No parameters; we use provided inputs

    def forward(self, hidden_states: torch.Tensor,
                encoder_hidden_states: torch.Tensor,
                process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:

        # Shapes
        B, P, K = hidden_states.shape
        assert hidden_states.shape[2] == K
        assert hidden_states.device.type == 'cuda' and encoder_hidden_states.device.type == 'cuda' and process_weight.device.type == 'cuda'

        T, _, _ = encoder_hidden_states.shape
        assert encoder_hidden_states.shape[2] == K
        assert process_weight.shape == (K, K)

        # Ensure float32 and contiguity
        device = hidden_states.device
        if hidden_states.dtype != torch.float32:
            hidden_states = hidden_states.float()
        if encoder_hidden_states.dtype != torch.float32:
            encoder_hidden_states = encoder_hidden_states.float()
        if process_weight.dtype != torch.float32:
            process_weight = process_weight.float()

        total = T + P
        M_total = B * total

        # 1) Triton concatenate Acat [B, T+P, K]
        Acat = torch.empty((B, total, K), device=device, dtype=torch.float32)

        BLOCK_K = 64  # choose a moderate tile for columns
        grid_concat = (B, total)
        _concatenate_seqs_kernel[grid_concat](
            encoder_hidden_states, hidden_states, Acat,
            B, T, P, K,
            BLOCK_K=BLOCK_K,
            num_warps=4, num_stages=2
        )

        # 2) Host-side concatenation for GEMM input (avoid torch.cat): already done in Acat

        # 3) Triton GEMM: Acat @ process_weight.T -> [B*(T+P), K]
        A_flat = Acat.reshape(M_total, K)  # [M, K]
        C_flat = torch.empty((M_total, K), device=device, dtype=torch.float32)

        # Use a simple per-row kernel for robustness
        grid_matmul = (B, M_total)
        _matmul_kernel[grid_matmul](
            A_flat, process_weight.t(), C_flat,
            M_total, K,
            BLOCK_K=64,
            num_warps=4, num_stages=2
        )

        # 4) Triton split into processed_encoder [B, T, K] and processed_hidden [B, P, K]
        processed_encoder = torch.empty((B, T, K), device=device, dtype=torch.float32)
        processed_hidden = torch.empty((B, P, K), device=device, dtype=torch.float32)

        grid_split_e = (B, T)
        _split_encoder_kernel[grid_split_e](
            C_flat, processed_encoder,
            B, T, K,
            BLOCK_K=64,
            num_warps=4, num_stages=2
        )

        grid_split_h = (B, P)
        _split_hidden_kernel[grid_split_h](
            C_flat, processed_hidden,
            B, P, K,
            BLOCK_K=64,
            num_warps=4, num_stages=2
        )

        return processed_encoder, processed_hidden


# The following helpers are not required by the evaluator, but provided for completeness/testing:
def _run_example():
    # Example usage
    B, T, P, K = 2, 128, 256, 128
    x = torch.randn(B, P, K, device='cuda', dtype=torch.float32)
    e = torch.randn(B, T, K, device='cuda', dtype=torch.float32)
    W = torch.randn(K, K, device='cuda', dtype=torch.float32)  # linear projection

    model = ModelNew()
    y_e, y_h = model(e, x, W)
    print(y_e.shape, y_h.shape)  # should be [B, T, K] and [B, P, K]


if __name__ == "__main__":
    _run_example()


def run(*args):
    return ModelNew()(*args)
