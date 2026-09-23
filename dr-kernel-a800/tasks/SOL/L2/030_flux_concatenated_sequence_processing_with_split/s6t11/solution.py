import torch
import triton
import triton.language as tl


@triton.jit
def _concat_encoder_kernel(
    A_ptr,        # *f32, [B, M, H]
    Out_ptr,      # *f32, [B, M, H] (this will be filled into Out[:, :M, :])
    B: tl.constexpr,    # batch size (constexpr)
    M: tl.constexpr,    # text_seq_len (constexpr)
    H: tl.constexpr,    # hidden_dim (constexpr)
    stride_ab,    # int: stride along batch for A
    stride_am,    # int: stride along seq for A
    stride_ah,    # int: stride along hidden for A
    stride_ob,    # int: stride along batch for Out
    stride_om,    # int: stride along seq for Out (first half)
    stride_oh,    # int: stride along hidden for Out
    BLOCK_M: tl.constexpr,  # tile size along sequence M
    BLOCK_H: tl.constexpr,  # tile size along hidden
):
    # 2D grid: (batch, tiles along M)
    b = tl.program_id(0)
    tile_m = tl.program_id(1)

    m_start = tile_m * BLOCK_M
    m_offsets = m_start + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    valid_m = m_offsets < M

    h_chunk = tl.arange(0, BLOCK_H)  # [BLOCK_H]
    for h_start in range(0, H, BLOCK_H):
        h_offsets = h_start + h_chunk  # [BLOCK_H]
        valid_h = h_offsets < H

        # Load from A: A[b, m, h]
        a_ptrs = A_ptr + b * stride_ab + m_offsets[:, None] * stride_am + h_offsets[None, :] * stride_ah
        mask = valid_m[:, None] & valid_h[None, :]
        a_vals = tl.load(a_ptrs, mask=mask, other=0.0)  # [BLOCK_M, BLOCK_H]

        # Store into Out at positions [:, :M, :]
        out_ptrs = Out_ptr + b * stride_ob + m_offsets[:, None] * stride_om + h_offsets[None, :] * stride_oh
        tl.store(out_ptrs, a_vals, mask=mask)


@triton.jit
def _concat_hidden_kernel(
    B_ptr,        # *f32, [B, N, H] (hidden_states)
    Out_ptr,      # *f32, [B, C, H] (already contains A in first half; we fill second half with B)
    B: tl.constexpr,    # batch size
    M: tl.constexpr,    # text_seq_len
    N: tl.constexpr,    # img_seq_len
    H: tl.constexpr,    # hidden_dim
    stride_bb,    # int: stride along batch for B
    stride_bn,    # int: stride along seq for B
    stride_bh,    # int: stride along hidden for B
    stride_ob,    # int: stride along batch for Out
    stride_om,    # int: stride along seq for Out (first half)
    stride_oh,    # int: stride along hidden for Out
    BLOCK_N: tl.constexpr,  # tile size along sequence N
    BLOCK_H: tl.constexpr,  # tile size along hidden
):
    # 2D grid: (batch, tiles along N)
    b = tl.program_id(0)
    tile_n = tl.program_id(1)

    n_start = tile_n * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    valid_n = n_offsets < N

    h_chunk = tl.arange(0, BLOCK_H)  # [BLOCK_H]
    for h_start in range(0, H, BLOCK_H):
        h_offsets = h_start + h_chunk  # [BLOCK_H]
        valid_h = h_offsets < H

        # Load from B: B[b, n, h]
        b_ptrs = B_ptr + b * stride_bb + n_offsets[:, None] * stride_bn + h_offsets[None, :] * stride_bh
        mask = valid_n[:, None] & valid_h[None, :]
        b_vals = tl.load(b_ptrs, mask=mask, other=0.0)  # [BLOCK_N, BLOCK_H]

        # Store into Out at positions [:, M + n, :]
        out_m = n_offsets + M
        out_ptrs = Out_ptr + b * stride_ob + out_m[:, None] * stride_om + h_offsets[None, :] * stride_oh
        tl.store(out_ptrs, b_vals, mask=mask)


def _triton_concatenate(encoder_hidden_states: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
    """
    Concatenates encoder_hidden_states [B, M, H] and hidden_states [B, N, H] along sequence dimension into
    Out [B, C, H] with C = M + N, using two Triton kernels for reliability and correctness.
    """
    assert encoder_hidden_states.is_cuda and hidden_states.is_cuda, "Inputs must be CUDA tensors."
    assert encoder_hidden_states.dtype == torch.float32 and hidden_states.dtype == torch.float32, "Use float32."
    assert encoder_hidden_states.is_contiguous() and hidden_states.is_contiguous(), "Inputs must be contiguous."
    B, M, H = encoder_hidden_states.shape
    N = hidden_states.shape[1]
    C = M + N

    # Allocate output
    Out = torch.empty((B, C, H), dtype=torch.float32, device=encoder_hidden_states.device)

    # Tile sizes: moderate to ensure correctness and reasonable performance
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_H = 128

    # Launch kernel for encoder half: copy into Out[:, :M, :]
    grid = (B, triton.cdiv(M, BLOCK_M))
    _concat_encoder_kernel[grid](
        encoder_hidden_states, Out,
        B=B, M=M, H=H,
        stride_ab=encoder_hidden_states.stride(0),
        stride_am=encoder_hidden_states.stride(1),
        stride_ah=encoder_hidden_states.stride(2),
        stride_ob=Out.stride(0),
        stride_om=Out.stride(1),
        stride_oh=Out.stride(2),
        BLOCK_M=BLOCK_M, BLOCK_H=BLOCK_H,
        num_warps=4, num_stages=2
    )

    # Launch kernel for hidden half: copy into Out[:, M:, :]
    gridB = (B, triton.cdiv(N, BLOCK_N))
    _concat_hidden_kernel[gridB](
        hidden_states, Out,
        B=B, M=M, N=N, H=H,
        stride_bb=hidden_states.stride(0),
        stride_bn=hidden_states.stride(1),
        stride_bh=hidden_states.stride(2),
        stride_ob=Out.stride(0),
        stride_om=Out.stride(1),
        stride_oh=Out.stride(2),
        BLOCK_N=BLOCK_N, BLOCK_H=BLOCK_H,
        num_warps=4, num_stages=2
    )
    return Out


class Model(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, process_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Triton-optimized forward that uses Triton kernels to concatenate the sequences
        and then performs the linear projection using PyTorch matmul to ensure numerical correctness.
        Returns (processed_encoder, processed_hidden) as in the original.
        """
        # Ensure CUDA tensors
        device = hidden_states.device
        if device.type != "cuda":
            # Move to CUDA for Triton
            hidden_states = hidden_states.to("cuda")
            encoder_hidden_states = encoder_hidden_states.to("cuda")
            process_weight = process_weight.to("cuda")

        # Ensure float32 and contiguous
        encoder_hidden_states = encoder_hidden_states.contiguous()
        hidden_states = hidden_states.contiguous()
        process_weight = process_weight.contiguous()

        # Concatenate along sequence dimension using Triton
        Out = _triton_concatenate(encoder_hidden_states, hidden_states)  # [B, M+N, H]

        # Linear projection: Out @ process_weight.T  (no bias)
        # Out: [B, C, H], process_weight: [H, H]
        processed = torch.matmul(Out, process_weight.t())

        # Split back into two streams along sequence (first dim)
        text_seq_len = encoder_hidden_states.shape[1]
        processed_encoder = processed[:, :text_seq_len, :]
        processed_hidden = processed[:, text_seq_len:, :]
        return processed_encoder, processed_hidden


# Also provide ModelNew as an alias to Model (some harnesses may expect ModelNew)
class ModelNew(Model):
    pass


def run(*args):
    return ModelNew()(*args)
