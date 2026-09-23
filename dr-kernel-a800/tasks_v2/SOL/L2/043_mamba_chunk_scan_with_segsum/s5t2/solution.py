import torch
import torch.nn.functional as F

import triton
import triton.language as tl


@triton.jit
def pad_kernel(out_ptr, in_ptr, pad_last, Bsz: tl.constexpr, S: tl.constexpr, H: tl.constexpr, D_in: tl.constexpr, D_out: tl.constexpr):
    """
    Pad tensor along last dimension. Input shape [Bsz, S, H, D_in], Output shape [Bsz, S, H, D_out].
    For each (b, s, h), copy in_ptr[b, s, h, d] to out_ptr[b, s, h, d + pad_last].
    """
    b = tl.program_id(0)
    s = tl.program_id(1)
    h = tl.program_id(2)

    d_offsets = tl.arange(0, 64)
    for d_start in range(0, D_in, 64):
        d_idx = d_start + d_offsets
        d_mask = d_idx < D_in
        d_out_idx = d_idx + pad_last
        in_base = b * (S * H * D_in) + s * (H * D_in) + h * D_in
        out_base = b * (S * H * D_out) + s * (H * D_out) + h * D_out
        tl.store(out_ptr + out_base + d_out_idx, tl.load(in_ptr + in_base + d_idx, mask=d_mask), mask=d_mask)


@triton.jit
def cumsum_last_axis_kernel(out_ptr, in_ptr, Bsz: tl.constexpr, H: tl.constexpr, NC: tl.constexpr, K: tl.constexpr):
    """
    Compute inclusive cumsum along last axis K for each (b, h, nc).
    Input: [Bsz, H, NC, K], Output: [Bsz, H, NC, K]
    """
    b = tl.program_id(0)
    h = tl.program_id(1)
    nc = tl.program_id(2)

    base = b * (H * NC * K) + h * (NC * K) + nc * K
    acc = 0.0
    for k in range(0, K):
        val = tl.load(in_ptr + base + k)
        acc += val
        tl.store(out_ptr + base + k, acc)


@triton.jit
def states_contract_kernel(out_ptr, B_ptr, hidden_ptr, Bsz: tl.constexpr, NC: tl.constexpr, K: tl.constexpr, H: tl.constexpr, D: tl.constexpr, S: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_S: tl.constexpr):
    """
    Compute states[b, nc, h, d, s] = sum_{i=0..K-1} B_ptr[b, nc, i, h, s] * hidden_ptr[b, nc, i, h, d]
    Input pointers:
      - B_ptr: [Bsz, NC, K, H, S]
      - hidden_ptr: [Bsz, NC, K, H, D]
    Output pointer:
      - out_ptr: [Bsz, NC, H, D, S]
    """
    b = tl.program_id(0)
    nc = tl.program_id(1)
    h = tl.program_id(2)

    d_offsets = tl.arange(0, BLOCK_D)
    s_offsets = tl.arange(0, BLOCK_S)

    for d_start in range(0, D, BLOCK_D):
        d_idx = d_start + d_offsets
        d_mask = d_idx < D
        for s_start in range(0, S, BLOCK_S):
            s_idx = s_start + s_offsets
            s_mask = s_idx < S

            acc = tl.zeros([BLOCK_D, BLOCK_S], dtype=tl.float32)

            for i in range(0, K):
                # load B[b, nc, i, h, s] -> [S]
                B_base = b * (NC * K * H * S) + nc * (K * H * S) + i * (H * S) + h * S
                B_vals = tl.load(B_ptr + B_base + s_idx, mask=s_mask, other=0.0)  # [BLOCK_S]

                # load hidden[b, nc, i, h, d] -> [D]
                hid_base = b * (NC * K * H * D) + nc * (K * H * D) + i * (H * D) + h * D
                hid_vals = tl.load(hidden_ptr + hid_base + d_idx, mask=d_mask, other=0.0)  # [BLOCK_D]

                # accumulate outer product
                acc += hid_vals[:, None] * B_vals[None, :]

            # store to out[b, nc, h, d, s]
            out_base = b * (NC * H * D * S) + nc * (H * D * S) + h * (D * S)
            tl.store(out_ptr + out_base + d_idx[:, None] * S + s_idx[None, :], acc, mask=d_mask[:, None] & s_mask[None, :])


@triton.jit
def c_times_states_kernel(out_ptr, C_ptr, states_ptr, Bsz: tl.constexpr, NC: tl.constexpr, T: tl.constexpr, H: tl.constexpr, D: tl.constexpr, S: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_S: tl.constexpr):
    """
    Compute out[b, nc, t, h, d] = sum_s C_ptr[b, nc, t, h, s] * states_ptr[b, nc, h, d, s]
    Input pointers:
      - C_ptr: [Bsz, NC, T, H, S]
      - states_ptr: [Bsz, NC, H, D, S]
    Output pointer:
      - out_ptr: [Bsz, NC, T, H, D]
    """
    b = tl.program_id(0)
    nc = tl.program_id(1)
    h = tl.program_id(2)

    d_offsets = tl.arange(0, BLOCK_D)
    s_offsets = tl.arange(0, BLOCK_S)

    for t in range(0, T):
        for d_start in range(0, D, BLOCK_D):
            d_idx = d_start + d_offsets
            d_mask = d_idx < D
            for s_start in range(0, S, BLOCK_S):
                s_idx = s_start + s_offsets
                s_mask = s_idx < S

                acc = tl.zeros([BLOCK_D, BLOCK_S], dtype=tl.float32)

                # loop over s
                for ss in range(0, S):
                    # load C[b, nc, t, h, ss]
                    C_base = b * (NC * T * H * S) + nc * (T * H * S) + t * (H * S) + h * S
                    C_val = tl.load(C_ptr + C_base + ss)
                    # load states[b, nc, h, d, ss]
                    st_base = b * (NC * H * D * S) + nc * (H * D * S) + h * (D * S)
                    st_ptr = states_ptr + st_base
                    st_tile = tl.load(
                        st_ptr + d_idx[:, None] * S + (s_start + ss),
                        mask=d_mask[:, None],
                        other=0.0
                    )
                    acc += st_tile * C_val

                # store out[b, nc, t, h, d]
                out_base = b * (NC * T * H * D) + nc * (T * H * D) + t * (H * D)
                tl.store(out_ptr + out_base + h * D + d_idx, acc[:, 0], mask=d_mask)


class ModelNew(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: torch.Tensor,
                initial_states: torch.Tensor):
        # Shapes
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]
        num_heads = hidden_states.shape[2]
        head_dim = hidden_states.shape[3]

        state_size = 256
        chunk_size = 256

        # Compute padding size to make seq_len multiple of chunk_size
        pad_last = (chunk_size - seq_len % chunk_size) % chunk_size
        seq_len_padded = seq_len + pad_last
        num_chunks = (seq_len_padded + chunk_size - 1) // chunk_size

        # Convert to float32 for numerical stability
        hidden_states_f = hidden_states.to(torch.float32)
        A_f = A.to(torch.float32)
        B_f = B.to(torch.float32)
        C_f = C.to(torch.float32)
        D_f = D.to(torch.float32)
        initial_states_f = initial_states.to(torch.float32)

        # 1) Pad hidden states along last dimension (head_dim) using Triton
        hidden_padded = torch.zeros((batch_size, seq_len, num_heads, head_dim + pad_last), dtype=torch.float32, device=hidden_states.device)
        def pad_tensor_triton(out_tensor: torch.Tensor, in_tensor: torch.Tensor, pad_last: int):
            Bsz = in_tensor.shape[0]
            S = in_tensor.shape[1]
            H = in_tensor.shape[2]
            D_in = in_tensor.shape[3]
            D_out = out_tensor.shape[3]
            grid = (Bsz, S, H)
            pad_kernel[grid](out_tensor, in_tensor, pad_last, Bsz=Bsz, S=S, H=H, D_in=D_in, D_out=D_out)
        pad_tensor_triton(hidden_padded, hidden_states_f, pad_last)

        # Reshape hidden padded into chunks: [B, NC, K, H, D]
        hidden_chunked = hidden_padded.reshape(batch_size, num_chunks, chunk_size, num_heads, head_dim)

        # 2) Compute A_cumsum along K axis for [B, H, NC, K]


def run(*args):
    return ModelNew()(*args)
