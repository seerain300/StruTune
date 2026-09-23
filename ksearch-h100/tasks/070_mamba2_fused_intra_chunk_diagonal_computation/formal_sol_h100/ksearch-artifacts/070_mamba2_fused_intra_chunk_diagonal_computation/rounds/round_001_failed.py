# solution=GPT-5.6-Sol_070_mamba2_fused_intra_chunk_diagonal_computation_triton_optimized_r1 score=-1.0 passed=False
import torch
import triton
import triton.language as tl


CHUNK_SIZE = 128
NUM_HEADS = 32
N_GROUPS = 8
HEAD_DIM = 128
STATE_SIZE = 128


@triton.jit
def _compute_shared_g_kernel(
    B,
    C,
    G,
    num_chunks: tl.constexpr,
    stride_bb: tl.constexpr,
    stride_bc: tl.constexpr,
    stride_bt: tl.constexpr,
    stride_bg: tl.constexpr,
    stride_bs: tl.constexpr,
    stride_cb: tl.constexpr,
    stride_cc: tl.constexpr,
    stride_ct: tl.constexpr,
    stride_cg: tl.constexpr,
    stride_cs: tl.constexpr,
):
    unit = tl.program_id(0)
    group = tl.program_id(1)

    batch = unit // num_chunks
    chunk = unit - batch * num_chunks

    rows = tl.arange(0, CHUNK_SIZE)
    states = tl.arange(0, STATE_SIZE)
    cols = tl.arange(0, CHUNK_SIZE)

    c_ptrs = (
        C
        + batch * stride_cb
        + chunk * stride_cc
        + rows[:, None] * stride_ct
        + group * stride_cg
        + states[None, :] * stride_cs
    )
    b_ptrs = (
        B
        + batch * stride_bb
        + chunk * stride_bc
        + cols[None, :] * stride_bt
        + group * stride_bg
        + states[:, None] * stride_bs
    )

    c = tl.load(c_ptrs)
    b = tl.load(b_ptrs)
    g = tl.dot(c, b, out_dtype=tl.float32)

    g_ptrs = (
        G
        + ((unit * N_GROUPS + group) * CHUNK_SIZE + rows[:, None])
        * CHUNK_SIZE
        + cols[None, :]
    )
    tl.store(g_ptrs, g)


@triton.jit
def _apply_shared_g_kernel(
    hidden_states,
    A_cumsum,
    G,
    Y,
    num_chunks: tl.constexpr,
    stride_xb: tl.constexpr,
    stride_xc: tl.constexpr,
    stride_xt: tl.constexpr,
    stride_xh: tl.constexpr,
    stride_xd: tl.constexpr,
    stride_ab: tl.constexpr,
    stride_ah: tl.constexpr,
    stride_ac: tl.constexpr,
    stride_at: tl.constexpr,
    stride_yb: tl.constexpr,
    stride_yc: tl.constexpr,
    stride_yt: tl.constexpr,
    stride_yh: tl.constexpr,
    stride_yd: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    unit = tl.program_id(0)
    head = tl.program_id(1)
    tile = tl.program_id(2)

    batch = unit // num_chunks
    chunk = unit - batch * num_chunks
    group = head // (NUM_HEADS // N_GROUPS)

    num_d_tiles: tl.constexpr = HEAD_DIM // BLOCK_D
    m_tile = tile // num_d_tiles
    d_tile = tile - m_tile * num_d_tiles

    offs_m = m_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_j = tl.arange(0, CHUNK_SIZE)
    offs_d = d_tile * BLOCK_D + tl.arange(0, BLOCK_D)

    a_ptrs = (
        A_cumsum
        + batch * stride_ab
        + head * stride_ah
        + chunk * stride_ac
        + offs_j * stride_at
    )
    a = tl.load(a_ptrs).to(tl.float32)

    prefix_i = tl.sum(
        tl.where(
            offs_j[None, :] <= offs_m[:, None],
            a[None, :],
            0.0,
        ),
        axis=1,
    )
    prefix_j = tl.cumsum(a, axis=0)

    g_ptrs = (
        G
        + ((unit * N_GROUPS + group) * CHUNK_SIZE + offs_m[:, None])
        * CHUNK_SIZE
        + offs_j[None, :]
    )
    g = tl.load(g_ptrs)

    causal = offs_j[None, :] <= offs_m[:, None]
    decay = tl.exp(prefix_i[:, None] - prefix_j[None, :])
    weights = tl.where(causal, g * decay, 0.0)

    x_ptrs = (
        hidden_states
        + batch * stride_xb
        + chunk * stride_xc
        + offs_j[:, None] * stride_xt
        + head * stride_xh
        + offs_d[None, :] * stride_xd
    )
    x = tl.load(x_ptrs).to(tl.float32)

    output = tl.dot(
        weights,
        x,
        input_precision="tf32",
        out_dtype=tl.float32,
    )

    y_ptrs = (
        Y
        + batch * stride_yb
        + chunk * stride_yc
        + offs_m[:, None] * stride_yt
        + head * stride_yh
        + offs_d[None, :] * stride_yd
    )
    tl.store(y_ptrs, output)


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    A_cumsum: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
) -> torch.Tensor:
    batch_size, num_chunks, _, _, _ = hidden_states.shape
    units = batch_size * num_chunks

    shared_g = torch.empty(
        (units, N_GROUPS, CHUNK_SIZE, CHUNK_SIZE),
        device=hidden_states.device,
        dtype=torch.float32,
    )
    output = torch.empty_like(hidden_states)

    _compute_shared_g_kernel[(units, N_GROUPS)](
        B,
        C,
        shared_g,
        num_chunks,
        B.stride(0),
        B.stride(1),
        B.stride(2),
        B.stride(3),
        B.stride(4),
        C.stride(0),
        C.stride(1),
        C.stride(2),
        C.stride(3),
        C.stride(4),
        num_warps=8,
        num_stages=3,
    )

    block_m = 32
    block_d = 64
    tiles = (CHUNK_SIZE // block_m) * (HEAD_DIM // block_d)

    _apply_shared_g_kernel[(units, NUM_HEADS, tiles)](
        hidden_states,
        A_cumsum,
        shared_g,
        output,
        num_chunks,
        hidden_states.stride(0),
        hidden_states.stride(1),
        hidden_states.stride(2),
        hidden_states.stride(3),
        hidden_states.stride(4),
        A_cumsum.stride(0),
        A_cumsum.stride(1),
        A_cumsum.stride(2),
        A_cumsum.stride(3),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        output.stride(4),
        BLOCK_M=block_m,
        BLOCK_D=block_d,
        num_warps=8,
        num_stages=3,
    )

    return output