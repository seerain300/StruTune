import torch
import triton
import triton.language as tl


# 1) Triton kernel: F.pad on the last dimension (seq_len). Input: [B, L]; Output: [B, L+pad]. Pad is added to the end.
@triton.jit
def pad_last_dim_kernel(in_ptr, out_ptr,
                         B, L, pad,
                         in_stride_b, in_stride_l,
                         out_stride_b, out_stride_outl,
                         BLOCK_B: tl.constexpr):
    pid = tl.program_id(axis=0)
    b = pid // BLOCK_B
    if b >= B:
        return
    in_b_addr = in_ptr + b * in_stride_b
    out_b_addr = out_ptr + b * out_stride_b
    # copy first L elements
    i = 0
    while i < L:
        val = tl.load(in_b_addr + i * in_stride_l)
        tl.store(out_b_addr + i * out_stride_outl, val)
        i += 1
    # write pad zeros
    while i < L + pad:
        tl.store(out_b_addr + i * out_stride_outl, 0.0)
        i += 1


# 2) Triton kernel: Inclusive cumsum along last axis for tensor of shape [B, NH, NC, CS].
# One program handles one row (b, nh, nc), scanning across chunk_size.
@triton.jit
def cumsum_last_axis_kernel(in_ptr, out_ptr,
                             B, NH, NC, CS,
                             in_stride_b, in_stride_nh, in_stride_nc, in_stride_cs,
                             out_stride_b, out_stride_nh, out_stride_nc, out_stride_cs,
                             BLOCK_CS: tl.constexpr):
    pid = tl.program_id(axis=0)
    b = pid // (NH * NC)
    nh = (pid // NC) % NH
    nc = pid % NC
    in_row_addr = in_ptr + b * in_stride_b + nh * in_stride_nh + nc * in_stride_nc
    out_row_addr = out_ptr + b * out_stride_b + nh * out_stride_nh + nc * out_stride_nc

    running = 0.0
    t = 0
    while t < CS:
        val = tl.load(in_row_addr + t * in_stride_cs)
        running += val
        tl.store(out_row_addr + t * out_stride_cs, running)
        t += 1


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.chunk_size = 256  # matches the original
        self.num_heads = 16    # default used in original; we’ll read from input as well
        self.state_size = 256  # default used in original

    def forward(self, hidden_states: torch.Tensor,
                A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: torch.Tensor,
                initial_states: torch.Tensor):
        # Convert to float32 for numerical stability
        hidden_f32 = hidden_states.to(torch.float32)
        A_f32 = A.to(torch.float32)
        B_f32 = B.to(torch.float32)
        C_f32 = C.to(torch.float32)
        D_f32 = D.to(torch.float32)
        initial_f32 = initial_states.to(torch.float32)

        batch_size, seq_len, num_heads, head_dim = hidden_f32.shape
        NH = num_heads
        HD = head_dim
        S = self.state_size

        # 1) Pad last dimension to make seq_len a multiple of chunk_size
        pad_size = (self.chunk_size - (seq_len % self.chunk_size)) % self.chunk_size
        hidden_padded = torch.empty((batch_size, seq_len + pad_size), dtype=torch.float32, device=hidden_f32.device)

        # Launch Triton pad kernel
        grid_pad = (batch_size,)
        pad_last_dim_kernel[grid_pad](
            hidden_f32, hidden_padded,
            batch_size, seq_len, pad_size,
            hidden_f32.stride(0), hidden_f32.stride(1),
            hidden_padded.stride(0), hidden_padded.stride(1),
            BLOCK_B=1,
            num_warps=1
        )

        # 2) Permute A to [B, NH, L], then view as [B, NH, N, T]
        A_perm = A_f32.transpose(1, 2)  # [B, NH, L]
        L = A_perm.shape[-1]
        N = (seq_len + pad_size) // self.chunk_size  # number of chunks after padding
        A_perm_view = A_perm.view(batch_size, NH, N, self.chunk_size)  # [B, NH, N, T]
        A_cumsum_out = torch.empty_like(A_perm_view)  # [B, NH, N, T]

        # Launch Triton cumsum along last axis
        grid_cumsum = (batch_size * NH * N,)
        cumsum_last_axis_kernel[grid_cumsum](
            A_perm_view, A_cumsum_out,
            batch_size, NH, N, self.chunk_size,
            A_perm_view.stride(0), A_perm_view.stride(1), A_perm_view.stride(2), A_perm_view.stride(3),
            A_cumsum_out.stride(0), A_cumsum_out.stride(1), A_cumsum_out.stride(2), A_cumsum_out.stride(3),
            BLOCK_CS=self.chunk_size,
            num_warps=1
        )

        # 3) Now perform the rest of the original logic in PyTorch to ensure correctness
        # Reshape padded hidden states into chunks
        seq_len_padded = seq_len + pad_size
        hidden_chunked = hidden_padded.reshape(batch_size, N, self.chunk_size, NH, HD)  # [B, N, T, NH, HD]

        # Expand B and C to match num_heads and state_size
        # Note: The original code expands B and C to [B, N, T, NH, S]. We'll do the same via broadcasting.
        B_expanded = B_f32.unsqueeze(0).unsqueeze(1).unsqueeze(2).expand(batch_size, N, self.chunk_size, NH, S)  # [B, N, T, NH, S]
        C_expanded = C_f32.unsqueeze(0).unsqueeze(1).unsqueeze(2).expand(batch_size, N, self.chunk_size, NH, S)  # [B, N, T, NH, S]
        D_expanded = D_f32.unsqueeze(0).unsqueeze(1).unsqueeze(2).expand(batch_size, N, self.chunk_size, NH, S)  # [B, N, T, NH, S]

        # Compute D residual: D_f32[None, None, :, None] * hidden_padded (we'll use hidden_chunked as the chunked view)
        # D_residual: [B, N, T, NH, HD] (not S). The original uses D expanded to S; to keep semantics, we compute D residual
        # by expanding D to [B, N, T, NH, HD] and multiplying by hidden_chunked. For simplicity, we use hidden_padded
        # and compute per-chunk as hidden_chunked. Since D_expanded is [B, N, T, NH, S], we need to align dims.
        # The original code applies D to hidden_states (shape [B, L, NH, HD]), and then pads. We'll compute D residual
        # per chunk using D expanded to HD. We can construct D per chunk by repeating D_f32 across N, T:
        D_per_chunk = D_f32.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(batch_size, N, self.chunk_size, NH, HD)  # [B, N, T, NH, HD]
        D_residual = D_per_chunk * hidden_chunked  # [B, N, T, NH, HD]

        # We need D_residual in S dimension to multiply with B/C (S=256). Since original uses S from A, and A is [B, NH, L]
        # with S


def run(*args):
    return ModelNew()(*args)
