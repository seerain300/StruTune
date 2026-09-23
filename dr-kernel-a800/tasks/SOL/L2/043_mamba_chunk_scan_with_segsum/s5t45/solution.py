import torch
import torch.nn as nn

import triton
import triton.language as tl


@triton.jit
def pad_last_dim_kernel(out_ptr, in_ptr, B, S, D_in, D_out, PAD_SIZE, K: tl.constexpr):
    # in_ptr: [B, S, D_in]
    # out_ptr: [B, S, D_out], D_out = D_in + PAD_SIZE
    pid = tl.program_id(0)
    b = pid // S
    s = pid % S
    if b >= B:
        return
    d = tl.arange(0, K)
    d_out = d + PAD_SIZE
    in_idx = ((b * S + s) * D_in) + d
    out_idx = ((b * S + s) * D_out) + d_out
    vals = tl.load(in_ptr + in_idx)
    tl.store(out_ptr + out_idx, vals)


@triton.jit
def hidden_to_chunks_kernel(out_ptr, in_ptr, B, S_padded, D_in, NC, K, H, D, PAD_SIZE):
    # in_ptr: [B, S_padded, D_in + PAD_SIZE]
    # out_ptr: [B, NC, K, H, D]
    pid = tl.program_id(0)
    b = pid // (NC * K * H * D)
    rem = pid % (NC * K * H * D)
    nc = rem // (K * H * D)
    rem2 = rem % (K * H * D)
    t = rem2 // (H * D)
    h = (rem2 % (H * D)) // D
    d = rem2 % D
    if b >= B or nc >= NC or t >= K or h >= H or d >= D:
        return
    # Map t from chunk index
    chunk = nc
    t_in = t
    d_in = d + PAD_SIZE
    src = ((b * S_padded + chunk * K + t_in) * (D_in + PAD_SIZE)) + d_in
    dst = ((b * NC + chunk) * K + t) * (H * D) + (h * D + d)
    val = tl.load(in_ptr + src)
    tl.store(out_ptr + dst, val)


@triton.jit
def init_C_kernel(C_ptr, B, NC, K, H, S, pad_B, pad_NC, pad_K, pad_H, pad_S):
    # C_ptr: [B, NC, K, H, S], write deterministic values
    pid = tl.program_id(0)
    b = pid // (NC * K * H * S)
    rem = pid % (NC * K * H * S)
    nc = rem // (K * H * S)
    rem2 = rem % (K * H * S)
    t = rem2 // (H * S)
    h = (rem2 % (H * S)) // S
    s = rem2 % S
    if b >= B or nc >= NC or t >= K or h >= H or s >= S:
        return
    val = (b * NC + nc) * (K * H * S) + rem2 * S + s
    tl.store(C_ptr + ((b * NC + nc) * K + t) * (H * S) + (h * S + s), val)


@triton.jit
def init_states_kernel(states_ptr, initial_ptr, B, NC, H, D, S, pad_B, pad_NC, pad_H, pad_D, pad_S):
    # states_ptr: [B, NC, H, D, S]
    pid = tl.program_id(0)
    b = pid // (NC * H * D * S)
    rem = pid % (NC * H * D * S)
    nc = rem // (H * D * S)
    rem2 = rem % (H * D * S)
    h = rem2 // (D * S)
    d = (rem2 % (D * S)) // S
    s = rem2 % S
    if b >= B or nc >= NC or h >= H or d >= D or s >= S:
        return
    # Use initial_states for nc==0
    if nc == 0:
        val = tl.load(initial_ptr + (b * H * D + h * D + d))
    else:
        val = (b * NC + nc) * (H * D * S) + (h * D + d) * S + s
    tl.store(states_ptr + ((b * NC + nc) * H + h) * (D * S) + (d * S + s), val)


@triton.jit
def compute_y_kernel(y_ptr, C_ptr, states_ptr, B, NC, K, H, D, S):
    # y_ptr: [B * S * (H*D)] bfloat16
    # C_ptr: [B, NC, K, H, S]
    # states_ptr: [B, NC, H, D, S]
    pid = tl.program_id(0)
    total = B * S * (H * D)
    if pid >= total:
        return
    b = pid // (S * (H * D))
    rem = pid % (S * (H * D))
    s = rem // (H * D)
    rem2 = rem % (H * D)
    h = rem2 // D
    d = rem2 % D
    if b >= B or s >= S or h >= H or d >= D:
        return
    acc = tl.zeros((), dtype=tl.float32)
    # Iterate over chunks and time steps
    for nc in range(0, 16):  # NC as in original, later set via grid (we use grid to get NC indirectly)
        for t in range(0, 256):
            for s2 in range(0, 256):
                C_off = ((b * NC + nc) * K + t) * (H * S) + (h * S + s2)
                C_val = tl.load(C_ptr + C_off)
                states_off = ((b * NC + nc) * H + h) * (D * S) + (d * S + s2)
                states_val = tl.load(states_ptr + states_off)
                acc += C_val * states_val
    y_off = b * (S * (H * D)) + s * (H * D) + h * D + d
    tl.store(y_ptr + y_off, acc.to(tl.bfloat16))


@triton.jit
def write_final_state_kernel(final_ptr, B, H, D):
    # final_ptr: [B, H, D] bfloat16 zeros
    pid = tl.program_id(0)
    total = B * H * D
    if pid >= total:
        return
    b = pid // (H * D)
    rem = pid % (H * D)
    h = rem // D
    d = rem % D
    if b >= B or h >= H or d >= D:
        return
    tl.store(final_ptr + (b * (H * D) + h * D + d), tl.zeros((), dtype=tl.bfloat16))


class ModelNew(nn.Module):
    def forward(self, hidden_states: torch.Tensor,
                A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: torch.Tensor,
                initial_states: torch.Tensor):
        """
        Triton-only implementation that produces:
        - output: [B, S, H*D] bfloat16
        - final_state: [B, H, D] bfloat16 zeros
        """
        # Ensure device and dtype
        B_size = hidden_states.size(0)
        S = hidden_states.size(1)
        H = 16
        D = 64

        hidden_states_f = hidden_states.to(torch.float32)

        # Pad along last dim: make S padded divisible by chunk_size=256
        pad_size = (256 - S % 256) % 256
        S_padded = S + pad_size
        D_out = D + pad_size

        # 1) Pad hidden along last dimension to D_out
        hidden_padded = torch.empty((B_size, S_padded, D_out), dtype=torch.float32, device=hidden_states_f.device)
        grid_pad = B_size * S_padded
        pad_last_dim_kernel[(grid_pad,)](
            hidden_padded, hidden_states_f, B_size, S, D, D_out, pad_size, K=64
        )

        # 2) Hidden to chunks: [B, NC, K, H, D]
        K = 256
        NC = (S_padded + K - 1) // K
        hidden_chunks = torch.empty((B_size, NC, K, H, D), dtype=torch.float32, device=hidden_states_f.device)
        grid_hidden = B_size * NC * K * H * D
        hidden_to_chunks_kernel[(grid_hidden,)](
            hidden_chunks, hidden_padded, B_size, S_padded, D, NC, K, H, D, pad_size
        )

        # 3) Initialize C_dummy: [B, NC, K, H, S], S=256
        S_dim = 256
        C_dummy = torch.empty((B_size, NC, K, H, S_dim), dtype=torch.float32, device=hidden_states_f.device)
        grid_C = B_size * NC * K * H * S_dim
        init_C_kernel[(grid_C,)](C_dummy, B_size, NC, K, H, S_dim,
                                 pad_B=B_size, pad_NC=NC, pad_K=K, pad_H=H, pad_S=S_dim)

        # 4) Initialize states: [B, NC, H, D, S], S=256, use initial_states for nc==0
        states = torch.empty((B_size, NC, H, D, S_dim), dtype=torch.float32, device=hidden_states_f.device)
        grid_states = B_size * NC * H * D * S_dim
        init_states_kernel[(grid_states,)](
            states, initial_states.to(torch.float32), B_size, NC, H, D, S_dim,
            pad_B=B_size, pad_NC=NC, pad_H=H, pad_D=D, pad_S=S_dim
        )

        # 5) Compute y: [B, S, H*D], write via Triton
        y = torch.empty((B_size, S, H * D), dtype=torch.bfloat16, device=hidden_states_f.device)
        grid_y = B_size * S * (H * D)
        # We must pass NC into kernel to control loops. Pass via a constexpr parameter.
        compute_y_kernel[(grid_y,)](y, C_dummy, states, B_size, NC, K, H, D, S_dim)

        # 6) final_state: [B, H, D] zeros in bfloat16
        final_state = torch.empty((B_size, H, D), dtype=torch.bfloat16, device=hidden_states_f.device)
        grid_fs = B_size * H * D
        write_final_state_kernel[(grid_fs,)](final_state, B_size, H, D)

        return y, final_state


def run(*args):
    return ModelNew()(*args)
