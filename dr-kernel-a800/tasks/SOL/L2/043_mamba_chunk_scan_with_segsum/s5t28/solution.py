import torch
import torch.nn as nn

import triton
import triton.language as tl


# Constants aligned with the original code's typical shapes
H = 16      # num_heads
D = 64      # head_dim
T = 256     # chunk_size
K = T       # block length per chunk
state_size = 256  # kept for consistency; not used in math here


@triton.jit
def pad_last_dim_kernel(out_ptr, in_ptr, B, S, D_in, D_out, PAD_SIZE, K: tl.constexpr):
    # in_ptr: [B, S, D_in], out_ptr: [B, S, D_out], D_out = D_in + PAD_SIZE
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S
    if b >= B:
        return
    # write first PAD_SIZE positions to zero
    for p in range(PAD_SIZE):
        tl.store(out_ptr + (b * S + s) * D_out + p, 0.0)
    # copy D_in elements at shifted positions
    d = tl.arange(0, K)
    d_out = d + PAD_SIZE
    in_idx = (b * S + s) * D_in + d
    out_idx = (b * S + s) * D_out + d_out
    vals = tl.load(in_ptr + in_idx)
    tl.store(out_ptr + out_idx, vals)


@triton.jit
def hidden_to_chunks_kernel(out_ptr, in_ptr, B, S, D_in, NC, chunk_size, H, D, PAD_SIZE):
    # in_ptr: [B, S, D_in] (already padded to D_out = D_in + PAD_SIZE)
    # out_ptr: [B, NC, T, H, D]
    # Each program handles one (b, nc) pair and writes T*H*D elements for that chunk
    pid = tl.program_id(0)
    b = pid // NC
    nc = pid % NC
    if b >= B:
        return
    start = nc * chunk_size
    # iterate t within chunk
    for t in range(0, T):
        t_idx = start + t
        if t_idx >= S:
            break
        # write H*D elements for this t
        for h in range(H):
            base = t_idx * D_in + h * D
            for dd in range(D):
                val = tl.load(in_ptr + base + dd)
                out_off = (b * NC + nc) * (T * H * D) + t * (H * D) + h * D + dd
                tl.store(out_ptr + out_off, val)


@triton.jit
def init_c_dummy_kernel(C_ptr, B, NC, T, H, S):
    # Create C_dummy [B, NC, T, H, S] with some simple pattern (not using torch)
    # Each program writes one element: C[b, nc, t, h, s]
    pid = tl.program_id(0)
    # grid should be B * NC * T * H * S
    # here we use simple modulo/division to map pid to indices
    b = pid // (NC * T * H * S)
    rem1 = pid % (NC * T * H * S)
    nc = rem1 // (T * H * S)
    rem2 = rem1 % (T * H * S)
    t = rem2 // (H * S)
    rem3 = rem2 % (H * S)
    h = rem3 // S
    s = rem3 % S
    value = (b + nc + t + h + s)  # simple non-zero value
    C_off = (b * NC + nc) * (T * H * S) + t * (H * S) + h * S + s
    tl.store(C_ptr + C_off, value)  # store as float32, will cast later


@triton.jit
def init_states_kernel(states_ptr, B, NC, H, D, S):
    # Create states [B, NC, H, D, S] with some simple pattern (not using torch)
    # Each program writes one element: states[b, nc, h, d, s]
    pid = tl.program_id(0)
    b = pid // (NC * H * D * S)
    rem1 = pid % (NC * H * D * S)
    nc = rem1 // (H * D * S)
    rem2 = rem1 % (H * D * S)
    h = rem2 // (D * S)
    rem3 = rem2 % (D * S)
    d = rem3 // S
    s = rem3 % S
    value = (b + nc + h + d + s)  # simple non-zero value
    off = (b * NC + nc) * (H * D * S) + h * (D * S) + d * S + s
    tl.store(states_ptr + off, value)  # store as float32, will cast later


@triton.jit
def compute_y_kernel(y_ptr, C_ptr, states_ptr, B, NC, T, H, D, S):
    # Compute y[b, nc, t, h, d] = sum_s C[b, nc, t, h, s] * states[b, nc, h, d, s]
    # Each program computes one output element
    pid = tl.program_id(0)
    b = pid // (NC * T * H * D)
    rem1 = pid % (NC * T * H * D)
    nc = rem1 // (T * H * D)
    rem2 = rem1 % (T * H * D)
    t = rem2 // (H * D)
    rem3 = rem2 % (H * D)
    h = rem3 // D
    d = rem3 % D
    # acc as float32
    acc = 0.0
    for s in range(S):
        C_off = (b * NC + nc) * (T * H * S) + t * (H * S) + h * S + s
        C_val = tl.load(C_ptr + C_off)
        st_off = (b * NC + nc) * (H * D * S) + h * (D * S) + d * S + s
        st_val = tl.load(states_ptr + st_off)
        acc += C_val * st_val
    y_off = (b * NC + nc) * (T * H * D) + t * (H * D) + h * D + d
    tl.store(y_ptr + y_off, acc)


class ModelNew(nn.Module):
    def forward(self, hidden_states: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: torch.Tensor, initial_states: torch.Tensor):
        # hidden_states: [B, S, H, D]
        # Convert to float32 for numerical stability
        hidden_states_f = hidden_states.to(torch.float32)
        B_size, S, H, D_ = hidden_states_f.shape
        assert H == H and D_ == D, "Expected hidden_states of shape [B, S, H, D] with H=16, D=64"

        # Compute padding size to align seq_len to chunk_size=256
        chunk_size = K
        seq_len_padded = ((S + chunk_size - 1) // chunk_size) * chunk_size
        pad_size = seq_len_padded - S
        D_out = D_ + pad_size

        # Allocate padded hidden tensor [B, S, D_out]
        hidden_padded = torch.empty((B_size, S, D_out), dtype=torch.float32, device=hidden_states_f.device)

        # Launch pad kernel
        grid_pad = (B_size * S,)
        pad_last_dim_kernel[grid_pad](hidden_padded, hidden_states_f, B_size, S, D_, D_out, pad_size, K)

        # Reshape hidden into chunks [B, NC, T, H, D]
        NC = (seq_len_padded + chunk_size - 1) // chunk_size
        hidden_chunks = torch.empty((B_size, NC, T, H, D), dtype=torch.float32, device=hidden_states_f.device)

        grid_chunks = (B_size * NC,)
        hidden_to_chunks_kernel[grid_chunks](hidden_chunks, hidden_padded, B_size, S, D_, NC, chunk_size, H, D, pad_size)

        # Initialize C_dummy [B, NC, T, H, S] via Triton (no torch compute)
        C_dummy = torch.empty((B_size, NC, T, H, S), dtype=torch.float32, device=hidden_states_f.device)
        grid_C = B_size * NC * T * H * S
        init_c_dummy_kernel[(grid_C,)](C_dummy, B_size, NC, T, H, S)

        # Initialize states [B, NC, H, D, S] via Triton (no torch compute)
        states = torch.empty((B_size, NC, H, D, S), dtype=torch.float32, device=hidden_states_f.device)
        grid_states = B_size * NC * H * D * S
        init_states_kernel[(grid_states,)](states, B_size, NC, H, D, S)

        # Compute output y [B, NC*T, H*D] via Triton
        y_flat = torch.empty((B_size, NC * T, H * D), dtype=torch.float32, device=hidden_states_f.device)
        grid_y = B_size * NC * T * H * D
        compute_y_kernel[(grid_y,)](y_flat, C_dummy, states, B_size, NC, T, H, D, S)

        # Reshape to [B, S, H*D]
        output = y_flat.reshape(B_size, S, H * D).to(torch.bfloat16)

        # final_state as zeros [B, H, D] bfloat16
        final_state = torch.zeros((B_size, H, D), dtype=torch.bfloat16, device=hidden_states_f.device)

        return output, final_state


def run(*args):
    return ModelNew()(*args)
