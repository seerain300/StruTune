# solution=GPT-5.6-Sol_070_mamba2_fused_intra_chunk_diagonal_computation_triton_optimized_r1 score=183.299067035816 passed=True
import torch
import triton
import triton.language as tl


CHUNK_SIZE = 128
NUM_HEADS = 32
N_GROUPS = 8
HEAD_DIM = 128
STATE_SIZE = 128


@triton.jit
def _fused_group_output_kernel(
    hidden_states,
    A_cumsum,
    B,
    C,
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
    stride_yb: tl.constexpr,
    stride_yc: tl.constexpr,
    stride_yt: tl.constexpr,
    stride_yh: tl.constexpr,
    stride_yd: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    m_tile = tl.program_id(0)
    group = tl.program_id(1)
    unit = tl.program_id(2)

    batch = unit // num_chunks
    chunk = unit - batch * num_chunks

    offs_j = tl.arange(0, 128)
    offs_s = tl.arange(0, 128)
    offs_d = tl.arange(0, 128)

    b_ptrs = (
        B
        + batch * stride_bb
        + chunk * stride_bc
        + offs_s[:, None] * stride_bs
        + group * stride_bg
        + offs_j[None, :] * stride_bt
    )
    b = tl.load(b_ptrs)

    for subtile in tl.static_range(0, 2):
        offs_m = (
            m_tile * BLOCK_M
            + subtile * (BLOCK_M // 2)
            + tl.arange(0, BLOCK_M // 2)
        )

        c_ptrs = (
            C
            + batch * stride_cb
            + chunk * stride_cc
            + offs_m[:, None] * stride_ct
            + group * stride_cg
            + offs_s[None, :] * stride_cs
        )
        c = tl.load(c_ptrs)
        g = tl.dot(c, b, out_dtype=tl.float32)

        causal = offs_j[None, :] <= offs_m[:, None]

        for head_offset in tl.static_range(0, 4):
            head = group * 4 + head_offset

            a_ptrs = (
                A_cumsum
                + batch * stride_ab
                + head * stride_ah
                + chunk * stride_ac
                + offs_j * stride_at
            )
            a = tl.load(a_ptrs).to(tl.float32)
            prefix = tl.cumsum(a, axis=0)
            prefix_m = tl.gather(prefix, offs_m, axis=0)

            decay = tl.exp(prefix_m[:, None] - prefix[None, :])
            weights = tl.where(causal, g * decay, 0.0).to(tl.bfloat16)

            x_ptrs = (
                hidden_states
                + batch * stride_xb
                + chunk * stride_xc
                + offs_j[:, None] * stride_xt
                + head * stride_xh
                + offs_d[None, :] * stride_xd
            )
            x = tl.load(x_ptrs)
            output = tl.dot(weights, x, out_dtype=tl.float32)

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
    output = torch.empty_like(hidden_states)

    block_m = 32
    grid = (CHUNK_SIZE // block_m, N_GROUPS, batch_size * num_chunks)

    _fused_group_output_kernel[grid](
        hidden_states,
        A_cumsum,
        B,
        C,
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
        output.stride(0),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        output.stride(4),
        BLOCK_M=block_m,
        num_warps=4,
        num_stages=1,
    )

    return output