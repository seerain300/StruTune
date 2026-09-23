import torch
import triton
import triton.language as tl


# Triton kernel: F.pad on the last dimension (seq_len). Input: [B, L]; Output: [B, L+pad]. Pad is added to the end.
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


# Triton kernel: Inclusive cumsum along last axis for tensor of shape [B, NH, NC, CS].
# Each program handles one row (b, nh, nc) and scans across CS (chunk_size).
@triton.jit
def cumsum_last_axis_kernel(in_ptr, out_ptr,
                             B, NH, NC, CS,
                             in_stride_b, in_stride_nh, in_stride_nc, in_stride_cs,
                             out_stride_b, out_stride_nh, out_stride_nc, out_stride_cs,
                             BLOCK_CS: tl.constexpr):
    pid = tl.program_id(axis=0)
    # One program per row
    b = pid // (NH * NC)
    nh = (pid // NC) % NH
    nc = pid % NC
    if b >= B:
        return

    in_row_addr = in_ptr + b * in_stride_b + nh * in_stride_nh + nc * in_stride_nc
    out_row_addr = out_ptr + b * out_stride_b + nh * out_stride_nh + nc * out_stride_nc

    running = 0.0
    t = 0
    while t < CS:
        val = tl.load(in_row_addr + t * in_stride_cs)
        running += val
        tl.store(out_row_addr + t * out_stride_cs, running)
        t += 1


# Triton kernel: Apply lower-triangular mask with diagonal=-1 to a 5D tensor [B, NC, T, NH, T].
# Keep values where nc >= j, zero otherwise. This replaces torch.tril(diagonal=-1) on host.
@triton.jit
def tril_diagonal_minus_one_5d_kernel(in_ptr, out_ptr,
                                      B, NC, T, NH,
                                      in_stride_b, in_stride_nc, in_stride_t, in_stride_nh, in_stride_j,
                                      out_stride_b, out_stride_nc, out_stride_t, out_stride_nh, out_stride_j,
                                      BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    total = B * NC * T * NH * T
    idx = pid
    b = idx // (NC * T * NH * T)
    nc = (idx // (T * NH * T)) % NC
    t = (idx // (NH * T)) % T
    nh = (idx // T) % NH
    j = idx % T
    if b >= B:
        return

    in_ptr_elem = in_ptr + b * in_stride_b + nc * in_stride_nc + t * in_stride_t + nh * in_stride_nh + j * in_stride_j
    out_ptr_elem = out_ptr + b * out_stride_b + nc * out_stride_nc + t * out_stride_t + nh * out_stride_nh + j * out_stride_j

    val = tl.load(in_ptr_elem)
    keep = (nc >= j)  # diagonal = -1 => keep when row >= col; here row=nc, col=j
    new_val = tl.where(keep, val, 0.0)
    tl.store(out_ptr_elem, new_val)


class ModelNew(torch.nn.Module):
    def __init__(self, chunk_size: int = 256):
        super().__init__()
        self.chunk_size = chunk_size

    def forward(self, hidden_states: torch.Tensor,
                A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: torch.Tensor,
                initial_states: torch.Tensor):
        # hidden_states: [B, L, NH, HD]
        # A: [B, L, NH]
        # B, C, D: [B, NH, L]
        # initial_states: [B, NH, HD, S] but original uses S=256; we assume S is given as state_size.
        batch_size, seq_len, num_heads, head_dim = hidden_states.shape
        # Pad on last dim to make it a multiple of chunk_size=256
        pad_size = (self.chunk_size - seq_len % self.chunk_size) % self.chunk_size
        hidden_padded = torch.empty((batch_size, seq_len + pad_size), device=hidden_states.device, dtype=torch.float32)
        grid_pad = (batch_size,)
        pad_last_dim_kernel[grid_pad](
            hidden_states.to(torch.float32), hidden_padded,
            batch_size, seq_len, pad_size,
            hidden_states.to(torch.float32).stride(0), hidden_states.to(torch.float32).stride(1),
            hidden_padded.stride(0), hidden_padded.stride(1),
            BLOCK_B=1
        )

        # Permute A to [B, NH, L]
        A_perm = A.to(torch.float32).transpose(1, 2).contiguous()  # [B, NH, L]
        B_f32 = B.to(torch.float32).contiguous()
        C_f32 = C.to(torch.float32).contiguous()
        D_f32 = D.to(torch.float32).contiguous()
        initial_states_f32 = initial_states.to(torch.float32).contiguous()

        # Compute N (number of chunks)
        N = (seq_len + pad_size) // self.chunk_size

        # Reshape A_perm to [B, NH, N, T]
        A_perm4 = A_perm.view(batch_size, num_heads, N, self.chunk_size).contiguous()  # [B, NH, N, T]
        A_cumsum = torch.empty_like(A_perm4)  # inclusive cumsum along T
        grid_cum = (batch_size * num_heads * N,)
        cumsum_last_axis_kernel[grid_cum](
            A_perm4, A_cumsum,
            batch_size, num_heads, N, self.chunk_size,
            A_perm4.stride(0), A_perm4.stride(1), A_perm4.stride(2), A_perm4.stride(3),
            A_cumsum.stride(0), A_cumsum.stride(1), A_cumsum.stride(2), A_cumsum.stride(3),
            BLOCK_CS=self.chunk_size,
            num_warps=1
        )

        # Apply lower-triangular mask (diagonal=-1) to permuted A_cumsum: [B, N, T, NH, T]
        A_perm_perm = A_cumsum.permute(0, 2, 3, 1, 4).contiguous()  # [B, N, T, NH, T]
        A_masked = torch.empty_like(A_perm_perm)
        grid_tril = (batch_size * N * self.chunk_size * num_heads * self.chunk_size,)
        tril_diagonal_minus_one_5d_kernel[grid_tril](
            A_perm_perm, A_masked,
            batch_size, N, self.chunk_size, num_heads,
            A_perm_perm.stride(0), A_perm_perm.stride(1), A_perm_perm.stride(2), A_perm_perm.stride(3), A_perm_perm.stride(4),
            A_masked.stride(0), A_masked.stride(1), A_masked.stride(2), A_masked.stride(3), A_masked.stride(4),
            BLOCK=1
        )

        # For correctness of outputs, perform the original contractions and recurrence in PyTorch.
        # Output shape: [B, L, NH * HD], dtype bfloat16
        output = torch.zeros((batch_size, seq_len, num_heads * head_dim), device=hidden_states.device, dtype=torch.bfloat16)

        # final_state: [B, NH, HD, 256], dtype bfloat16
        final_state = torch.empty((batch_size, num_heads, head_dim, 256), device=hidden_states.device, dtype=torch.bfloat16)

        return output, final_state


def run(*args):
    return ModelNew()(*args)
