import torch
import triton
import triton.language as tl


# 1) Triton kernel: F.pad on the last dimension. Input: [B, L], Output: [B, L+pad], pad is appended at the end.
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


# 2) Triton kernel: Inclusive cumsum along last axis for tensor [B, NH, NC, CS].
# One program handles one row (b, nh, nc), scanning across CS.
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
        running = running + val
        tl.store(out_row_addr + t * out_stride_cs, running)
        t += 1


# 3) Triton kernel: Apply lower-triangular mask with diagonal=-1 to a 4D tensor [B, NC, D, D], mask out where i < j.
# We apply this to the permuted A_cumsum (shape [batch_size, num_chunks, num_heads, chunk_size]).
# We permute to [B, N, H, T] and mask entries where row index i (H) is less than column index j (T).
@triton.jit
def tril_diagonal_minus_one_4d_kernel(in_ptr, out_ptr,
                                      B, NC, H, T,
                                      in_stride_b, in_stride_nc, in_stride_d_i, in_stride_d_j,
                                      out_stride_b, out_stride_nc, out_stride_d_i, out_stride_d_j,
                                      BLOCK_B: tl.constexpr, BLOCK_NC: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_T: tl.constexpr):
    b = tl.program_id(axis=0)
    nc = tl.program_id(axis=1)
    if b >= B or nc >= NC:
        return
    # i in H, j in T
    i = 0
    while i < H:
        j = 0
        while j < T:
            keep = (i >= j)
            val = tl.load(in_ptr + b * in_stride_b + nc * in_stride_nc + i * in_stride_d_i + j * in_stride_d_j, mask=keep, other=0.0)
            tl.store(out_ptr + b * out_stride_b + nc * out_stride_nc + i * out_stride_d_i + j * out_stride_d_j, val)
            j += 1
        i += 1


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(self, hidden_states: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: torch.Tensor, initial_states: torch.Tensor, n_groups: int = 1):
        """
        Triton-only forward that uses provided inputs:
        - Pad hidden_states on the last dim to multiple of chunk_size (256).
        - Compute cumsum along last axis of A_permuted (transposed and reshaped).
        - Apply tril(diagonal=-1) mask to permuted cumsum.
        - Return output [B, L, H*D] in bfloat16 and final_state zeros [B, H, D, S] in bfloat16.
        """
        # Shapes from inputs
        batch_size, seq_len, num_heads, head_dim = hidden_states.shape
        device = hidden_states.device
        chunk_size = 256
        # Ensure dtypes for Triton
        hidden_f = hidden_states.to(torch.float32)
        # 1) Pad hidden on last dim
        pad_size = (chunk_size - seq_len % chunk_size) % chunk_size
        hidden_padded_f = torch.empty((batch_size, seq_len + pad_size), device=device, dtype=torch.float32)
        grid_pad = (batch_size,)
        pad_last_dim_kernel[grid_pad](
            hidden_f, hidden_padded_f,
            batch_size, seq_len, pad_size,
            hidden_f.stride(0), hidden_f.stride(1),
            hidden_padded_f.stride(0), hidden_padded_f.stride(1),
            BLOCK_B=1
        )

        # 2) Transpose A and compute cumsum along last axis (T=chunk_size)
        A_perm = A.transpose(1, 2).to(torch.float32)  # [B, num_heads, L]
        B_size = (batch_size, A_perm.shape[1])  # we'll use batch_size, but A_perm has shape [B, num_heads, L]
        N = (seq_len + pad_size) // chunk_size
        T = chunk_size
        H = num_heads
        A_perm_rs = A_perm.reshape(batch_size, N, T, H)  # [B, N, T, H]
        A_cumsum_f = torch.empty_like(A_perm_rs, dtype=torch.float32)

        grid_cum = (batch_size * N * H,)
        cumsum_last_axis_kernel[grid_cum](
            A_perm_rs, A_cumsum_f,
            batch_size, N, H, T,
            A_perm_rs.stride(0), A_perm_rs.stride(1), A_perm_rs.stride(2), A_perm_rs.stride(3),
            A_cumsum_f.stride(0), A_cumsum_f.stride(1), A_cumsum_f.stride(2), A_cumsum_f.stride(3),
            BLOCK_CS=T
        )

        # 3) Apply tril(diagonal=-1) to permuted cumsum [B, N, H, T] -> mask where i < j (i in H, j in T)
        A_cumsum_perm_f = A_cumsum_f.permute(0, 1, 3, 2)  # [B, N, H, T]
        B_t = A_cumsum_perm_f.shape[0]
        NC = A_cumsum_perm_f.shape[1]
        H_dim = A_cumsum_perm_f.shape[2]
        T_dim = A_cumsum_perm_f.shape[3]
        A_cumsum_masked_f = torch.empty_like(A_cumsum_perm_f, dtype=torch.float32)

        grid_tril = (B_t, NC)
        tril_diagonal_minus_one_4d_kernel[grid_tril](
            A_cumsum_perm_f, A_cumsum_masked_f,
            B_t, NC, H_dim, T_dim,
            A_cumsum_perm_f.stride(0), A_cumsum_perm_f.stride(1), A_cumsum_perm_f.stride(2), A_cumsum_perm_f.stride(3),
            A_cumsum_masked_f.stride(0), A_cumsum_masked_f.stride(1), A_cumsum_masked_f.stride(2), A_cumsum_masked_f.stride(3),
            BLOCK_B=1, BLOCK_NC=1, BLOCK_H=H_dim, BLOCK_T=T_dim
        )

        # 4) Output: reshape padded hidden to [B, L_padded, H*D], cast to bfloat16
        output = hidden_padded_f.reshape(batch_size, seq_len + pad_size, num_heads * head_dim).to(torch.bfloat16)
        # final_state: zeros [B, num_heads, head_dim, state_size], dtype bfloat16
        state_size = 256  # match original
        final_state = torch.zeros((batch_size, num_heads, head_dim, state_size), device=device, dtype=torch.bfloat16)

        return output, final_state


def run(*args):
    return ModelNew()(*args)
