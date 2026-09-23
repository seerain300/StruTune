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
    pid_b = tl.program_id(axis=0)
    if pid_b >= B:
        return
    in_b_addr = in_ptr + pid_b * in_stride_b
    out_b_addr = out_ptr + pid_b * out_stride_b
    i = 0
    while i < L:
        val = tl.load(in_b_addr + i * in_stride_l)
        tl.store(out_b_addr + i * out_stride_outl, val)
        i += 1
    while i < L + pad:
        tl.store(out_b_addr + i * out_stride_outl, 0.0)
        i += 1


# 2) Triton kernel: Inclusive cumsum along the last axis (seq_len) for tensor [B, NH, L].
# Launch with grid = (B * NH, L). Each program handles one row for (b, nh) and scans across L.
@triton.jit
def cumsum_last_axis_2d_kernel(in_ptr, out_ptr,
                               B, NH, L,
                               in_stride_b, in_stride_nh, in_stride_l,
                               out_stride_b, out_stride_nh, out_stride_l,
                               BLOCK_B: tl.constexpr):
    pid_row = tl.program_id(axis=0)
    pid_col = tl.program_id(axis=1)
    if pid_row >= B * NH:
        return
    b = pid_row // NH
    nh = pid_row % NH
    in_row_addr = in_ptr + b * in_stride_b + nh * in_stride_nh
    out_row_addr = out_ptr + b * out_stride_b + nh * out_stride_nh
    running = 0.0
    t = 0
    while t <= pid_col:
        val = tl.load(in_row_addr + t * in_stride_l)
        running += val
        tl.store(out_row_addr + t * out_stride_l, running)
        t += 1


# 3) Triton kernel: Apply lower-triangular mask (diagonal=-1) to a 5D tensor [B, NC, T, H, S].
# For each (b, nc, t, h, s), if t < h, set to 0. We use a 3D launch grid (B, NC, T) and loop over H and S.
@triton.jit
def tril_diagonal_minus_one_5d_kernel(in_ptr, out_ptr,
                                      B, NC, T, H, S,
                                      in_stride_b, in_stride_nc, in_stride_t, in_stride_h, in_stride_s,
                                      out_stride_b, out_stride_nc, out_stride_t, out_stride_h, out_stride_s,
                                      BLOCK_B: tl.constexpr, BLOCK_NC: tl.constexpr, BLOCK_T: tl.constexpr):
    pid = tl.program_id(axis=0)
    # The grid is set to B * NC * T; decode pid into (b, nc, t)
    b = pid // (NC * T)
    rem = pid % (NC * T)
    nc = rem // T
    t = rem % T
    # Loop over H and S; these are runtime, but Triton supports loops
    h = 0
    while h < H:
        s = 0
        while s < S:
            in_addr = in_ptr + b * in_stride_b + nc * in_stride_nc + t * in_stride_t + h * in_stride_h + s * in_stride_s
            val = tl.load(in_addr)
            # tril(-1): keep if t >= h, else set to 0
            if t >= h:
                tl.store(out_ptr + b * out_stride_b + nc * out_stride_nc + t * out_stride_t + h * out_stride_h + s * out_stride_s, val)
            else:
                tl.store(out_ptr + b * out_stride_b + nc * out_stride_nc + t * out_stride_t + h * out_stride_h + s * out_stride_s, 0.0)
            s += 1
        h += 1


class ModelNew(torch.nn.Module):
    def __init__(self, num_heads: int, head_dim: int, state_size: int, chunk_size: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.state_size = state_size
        self.chunk_size = chunk_size

    def forward(self,
                hidden_states: torch.Tensor,  # [B, L, num_heads, head_dim]
                A: torch.Tensor,              # [B, L, num_heads]
                B: torch.Tensor,              # [B, L, num_heads, state_size]
                C: torch.Tensor,              # [B, L, num_heads, state_size]
                D: torch.Tensor,              # [num_heads, state_size]
                initial_states: torch.Tensor  # [B, num_heads, head_dim, state_size]
                ):
        batch_size, seq_len, num_heads, head_dim = hidden_states.shape
        state_size = self.state_size
        chunk_size = self.chunk_size

        # 1) Pad hidden states on the last dim using Triton
        pad_size = (chunk_size - seq_len % chunk_size) % chunk_size
        seq_len_padded = seq_len + pad_size
        hidden_states_padded = torch.empty((batch_size, seq_len_padded, num_heads, head_dim),
                                           device=hidden_states.device, dtype=hidden_states.dtype)

        grid_pad = (batch_size,)
        pad_last_dim_kernel[grid_pad](
            hidden_states, hidden_states_padded,
            batch_size, seq_len, pad_size,
            hidden_states.stride(0), hidden_states.stride(1),
            hidden_states_padded.stride(0), hidden_states_padded.stride(1),
            BLOCK_B=1
        )

        # 2) Compute A_perm and its cumsum along last axis using Triton
        # A_perm = A.transpose(1, 2) -> [B, num_heads, L]
        A_perm = A.transpose(1, 2)  # [B, num_heads, L]

        # Inclusive cumsum along last axis (seq_len) for each (b, nh): output [B, num_heads, L]
        A_cumsum = torch.empty_like(A_perm)

        grid_cumsum = (batch_size * num_heads, seq_len)
        cumsum_last_axis_2d_kernel[grid_cumsum](
            A_perm, A_cumsum,
            batch_size, num_heads, seq_len,
            A_perm.stride(0), A_perm.stride(1), A_perm.stride(2),
            A_cumsum.stride(0), A_cumsum.stride(1), A_cumsum.stride(2),
            BLOCK_B=1
        )

        # 3) Apply lower-triangular mask (diagonal=-1) to A_cumsum [B, num_heads, L] via Triton
        # We need a 5D representation [B, NC, T, H, S]. Since original segment_sum operates on expanded hidden states,
        # we emulate the mask on A_cumsum by introducing NC, T, H, S. We set NC=1, T=L, H=num_heads, S=head_dim.
        # Then apply tril(-1): zero where t < h.
        B_dummy = batch_size
        NC = 1
        T = seq_len  # last axis of A_cumsum
        H = num_heads
        S = head_dim
        A_cumsum_5d = A_cumsum.unsqueeze(1)  # [B, 1, L, num_heads]
        # Expand to [B, 1, L, num_heads, head_dim]
        A_cumsum_5d = A_cumsum_5d.expand(B_dummy, NC, T, H, S).contiguous()

        grid_tril = (B_dummy * NC * T,)
        tril_diagonal_minus_one


def run(*args):
    return ModelNew()(*args)
