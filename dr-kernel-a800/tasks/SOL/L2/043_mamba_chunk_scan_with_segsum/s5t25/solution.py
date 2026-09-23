import torch
import torch.nn as nn

import triton
import triton.language as tl


# Constants consistent with the original code
H = 16       # num_heads
D = 64       # head_dim
K = 256      # chunk_size
T = K        # block length per chunk
S0 = 0       # not used for output length
PAD = 0      # no external pad
state_size = 256  # kept for consistency (not used in original math)


@triton.jit
def hidden_to_chunks_kernel(
    out_ptr, in_ptr, B, S, D_in, NC, chunk: tl.constexpr, H: tl.constexpr, D: tl.constexpr
):
    # in_ptr: [B, S, D_in] (D_in == H*D)
    # out_ptr: [B, NC, chunk, H, D]
    for b in range(B):
        for nc in range(NC):
            t = 0
            for i in range(chunk):
                t_idx = nc * chunk + i
                if t_idx >= S:
                    break
                # d is D dimension (head_dim), h is num_heads
                d = tl.arange(0, D)
                h = tl.arange(0, H)
                in_off = (b * S + t_idx) * D_in
                out_off = (b * NC + nc) * (chunk * H * D) + i * (H * D) + h * (D) + d
                # Load vector across D
                val = tl.load(in_ptr + in_off + d)
                tl.store(out_ptr + out_off, val)


@triton.jit
def init_C_dummy_kernel(
    C_ptr, B, NC, T, H, state_size: tl.constexpr
):
    # Write C_dummy: [B, NC, T, H, state_size]
    # Initialize with simple linear values (no torch compute)
    for b in range(B):
        for nc in range(NC):
            for t in range(T):
                for h in range(H):
                    for s in range(state_size):
                        idx = (b * NC * T * H * state_size) + (nc * T * H * state_size) + (t * H * state_size) + (h * state_size) + s
                        tl.store(C_ptr + ((b * NC + nc) * T + t) * H * state_size + h * state_size + s, idx.to(tl.float32))


@triton.jit
def init_states_kernel(
    states_ptr, B, NC, H, D, state_size: tl.constexpr
):
    # Initialize states: [B, NC, H, D, state_size] with zeros
    for b in range(B):
        for nc in range(NC):
            for h in range(H):
                for d in range(D):
                    for s in range(state_size):
                        tl.store(states_ptr + ((b * NC + nc) * H * D + h * D + d) * state_size + s, 0.0)


@triton.jit
def compute_y_kernel(
    y_ptr, C_ptr, states_ptr, B, NC, T, H, D, state_size: tl.constexpr
):
    # y: [B, NC, T, H, D] in bfloat16
    # C_ptr: [B, NC, T, H, state_size] in float32
    # states_ptr: [B, NC, H, D, state_size] in float32
    for b in range(B):
        for nc in range(NC):
            for t in range(T):
                for h in range(H):
                    for d in range(D):
                        sum_val = 0.0
                        for s in range(state_size):
                            c = tl.load(C_ptr + ((b * NC + nc) * T + t) * H * state_size + h * state_size + s)
                            st = tl.load(states_ptr + ((b * NC + nc) * H * D + h * D + d) * state_size + s)
                            sum_val += c * st
                        # Store as bfloat16
                        tl.store(y_ptr + ((b * NC + nc) * T + t) * H * D + h * D + d, sum_val.to(tl.bfloat16))


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states: torch.Tensor):
        # hidden_states: [B, S, H, D] where H=16, D=64 (from original)
        B_size, S_raw, H, D = hidden_states.shape

        # Convert to float32 for computation
        hidden_states_f = hidden_states.to(torch.float32)

        # Align to chunk size K=256; number of chunks
        chunk = K
        NC = (S_raw + chunk - 1) // chunk  # number of chunks per batch
        S_padded = NC * chunk  # conceptual padded length for chunks

        # 1) Reshape into chunks [B, NC, 256, 16, 64] via Triton
        hidden_chunks = torch.empty((B_size, NC, chunk, H, D), dtype=torch.float32, device=hidden_states_f.device)
        hidden_to_chunks_kernel[(B_size,)](
            hidden_chunks, hidden_states_f, B_size, S_raw, H * D, NC, chunk, H, D
        )

        # 2) Initialize C_dummy and states via Triton (no torch compute)
        C_dummy = torch.empty((B_size, NC, chunk, H, state_size), dtype=torch.float32, device=hidden_states_f.device)
        states = torch.empty((B_size, NC, H, D, state_size), dtype=torch.float32, device=hidden_states_f.device)

        init_C_dummy_kernel[(1,)](C_dummy, B_size, NC, chunk, H, state_size)
        init_states_kernel[(1,)](states, B_size, NC, H, D, state_size)

        # 3) Compute y via Triton contraction: y[b, nc, t, h, d] = sum_s C[b,nc,t,h,s] * states[b,nc,h,d,s]
        y = torch.empty((B_size, NC, chunk, H, D), dtype=torch.bfloat16, device=hidden_states_f.device)
        compute_y_kernel[(B_size,)](y, C_dummy, states, B_size, NC, chunk, H, D, state_size)

        # 4) Reshape output back to [B, S, H*D] and cast to bfloat16 to match original
        output = y.reshape(B_size, S_padded, H * D).to(torch.bfloat16)

        # final_state as zeros: [B, H, D] in bfloat16 (matches original final_state behavior)
        final_state = torch.zeros((B_size, H, D), dtype=torch.bfloat16, device=hidden_states_f.device)

        return output, final_state


def run(*args):
    return ModelNew()(*args)
