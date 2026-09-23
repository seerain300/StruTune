# task: 070_mamba2_fused_intra_chunk_diagonal_computation
# bench: SOL-L1 | batch: formal_20260914
# final eval (official evaluator, full workloads): valid=True pass=14/14 geomean=192.994x
# feedback best (5-workload sample during search): 375.506x
# torch fallback audit: 干净 (-)
# tokens: 1,779,550

import torch
import triton
import triton.language as tl


@triton.jit
def _mamba2_group_gram_prefix_kernel(
    B,
    C,
    A_cumsum,
    prefix_output,
    gram_output,
    stride_b_b: tl.constexpr,
    stride_b_c: tl.constexpr,
    stride_b_s: tl.constexpr,
    stride_b_g: tl.constexpr,
    stride_b_n: tl.constexpr,
    stride_c_b: tl.constexpr,
    stride_c_c: tl.constexpr,
    stride_c_s: tl.constexpr,
    stride_c_g: tl.constexpr,
    stride_c_n: tl.constexpr,
    stride_a_b: tl.constexpr,
    stride_a_h: tl.constexpr,
    stride_a_c: tl.constexpr,
    stride_a_s: tl.constexpr,
    num_chunks: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_J: tl.constexpr,
    STATE_SIZE: tl.constexpr,
):
    pid_tile = tl.program_id(0)
    group = tl.program_id(1)
    batch_chunk = tl.program_id(2)

    batch = batch_chunk // num_chunks
    chunk = batch_chunk - batch * num_chunks

    tile_i = tl.where(
        pid_tile == 0,
        0,
        tl.where(pid_tile == 1, 1, tl.where(pid_tile < 4, 2, 3)),
    )
    tile_j = tl.where(pid_tile < 2, 0, (pid_tile - 2) & 1)

    offs_i = tile_i * BLOCK_I + tl.arange(0, BLOCK_I)
    offs_j = tile_j * BLOCK_J + tl.arange(0, BLOCK_J)
    offs_n = tl.arange(0, STATE_SIZE)

    c_ptrs = (
        C
        + batch * stride_c_b
        + chunk * stride_c_c
        + offs_i[:, None] * stride_c_s
        + group * stride_c_g
        + offs_n[None, :] * stride_c_n
    )
    b_ptrs = (
        B
        + batch * stride_b_b
        + chunk * stride_b_c
        + offs_j[:, None] * stride_b_s
        + group * stride_b_g
        + offs_n[None, :] * stride_b_n
    )

    c_values = tl.load(c_ptrs)
    b_values = tl.load(b_ptrs)
    gram = tl.dot(c_values, tl.trans(b_values))

    gram_ptrs = (
        gram_output
        + batch_chunk * (8 * 128 * 128)
        + group * (128 * 128)
        + offs_i[:, None] * 128
        + offs_j[None, :]
    )
    tl.store(gram_ptrs, gram.to(tl.bfloat16))

    if pid_tile == 0:
        prefix_offs = tl.arange(0, 128)
        for head_offset in tl.static_range(0, 4):
            head = group * 4 + head_offset
            a_ptrs = (
                A_cumsum
                + batch * stride_a_b
                + head * stride_a_h
                + chunk * stride_a_c
                + prefix_offs * stride_a_s
            )
            a_values = tl.load(a_ptrs).to(tl.float32)
            prefix_values = tl.cumsum(a_values, axis=0)

            prefix_ptrs = (
                prefix_output
                + (batch_chunk * 32 + head) * 128
                + prefix_offs
            )
            tl.store(prefix_ptrs, prefix_values)


@triton.jit
def _mamba2_causal_weight_tiles_kernel(
    hidden_states,
    prefix,
    gram,
    output,
    num_chunks: tl.constexpr,
    stride_hs_b: tl.constexpr,
    stride_hs_c: tl.constexpr,
    stride_hs_s: tl.constexpr,
    stride_hs_h: tl.constexpr,
    stride_hs_d: tl.constexpr,
    stride_o_b: tl.constexpr,
    stride_o_c: tl.constexpr,
    stride_o_s: tl.constexpr,
    stride_o_h: tl.constexpr,
    stride_o_d: tl.constexpr,
    BLOCK_I: tl.constexpr,
    BLOCK_J: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_tile = tl.program_id(0)
    head = tl.program_id(1)
    batch_chunk = tl.program_id(2)

    num_d_tiles: tl.constexpr = 128 // BLOCK_D
    num_source_tiles: tl.constexpr = 128 // BLOCK_J
    target_tile = pid_tile // num_d_tiles
    dim_tile = pid_tile - target_tile * num_d_tiles

    batch = batch_chunk // num_chunks
    chunk = batch_chunk - batch * num_chunks
    group = head // 4

    offs_i = target_tile * BLOCK_I + tl.arange(0, BLOCK_I)
    offs_d = dim_tile * BLOCK_D + tl.arange(0, BLOCK_D)

    prefix_base = prefix + (batch_chunk * 32 + head) * 128
    prefix_i = tl.load(prefix_base + offs_i)

    accumulator = tl.zeros((BLOCK_I, BLOCK_D), dtype=tl.float32)

    for source_tile in tl.static_range(0, num_source_tiles):
        if source_tile <= target_tile:
            offs_j = source_tile * BLOCK_J + tl.arange(0, BLOCK_J)

            gram_ptrs = (
                gram
                + batch_chunk * (8 * 128 * 128)
                + group * (128 * 128)
                + offs_i[:, None] * 128
                + offs_j[None, :]
            )
            gram_values = tl.load(gram_ptrs).to(tl.float32)
            prefix_j = tl.load(prefix_base + offs_j)

            causal = offs_j[None, :] <= offs_i[:, None]
            decay = tl.exp(prefix_i[:, None] - prefix_j[None, :])
            weights = tl.where(causal, gram_values * decay, 0.0).to(tl.float16)

            hidden_ptrs = (
                hidden_states
                + batch * stride_hs_b
                + chunk * stride_hs_c
                + offs_j[:, None] * stride_hs_s
                + head * stride_hs_h
                + offs_d[None, :] * stride_hs_d
            )
            hidden_values = tl.load(hidden_ptrs).to(tl.float16)

            accumulator += tl.dot(weights, hidden_values)

    output_ptrs = (
        output
        + batch * stride_o_b
        + chunk * stride_o_c
        + offs_i[:, None] * stride_o_s
        + head * stride_o_h
        + offs_d[None, :] * stride_o_d
    )
    tl.store(output_ptrs, accumulator.to(tl.bfloat16))


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    A_cumsum: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
) -> torch.Tensor:
    batch_size, num_chunks, chunk_size, num_heads, head_dim = hidden_states.shape

    assert chunk_size == 128
    assert num_heads == 32
    assert head_dim == 128
    assert A_cumsum.shape == (
        batch_size,
        num_heads,
        num_chunks,
        chunk_size,
    )
    assert B.shape == (
        batch_size,
        num_chunks,
        chunk_size,
        8,
        128,
    )
    assert C.shape == (
        batch_size,
        num_chunks,
        chunk_size,
        8,
        128,
    )

    batch_chunks = batch_size * num_chunks

    gram = torch.empty(
        (batch_chunks, 8, 128, 128),
        device=hidden_states.device,
        dtype=torch.bfloat16,
    )
    prefix = torch.empty(
        (batch_chunks, 32, 128),
        device=hidden_states.device,
        dtype=torch.float32,
    )
    output = torch.empty_like(hidden_states)

    _mamba2_group_gram_prefix_kernel[(6, 8, batch_chunks)](
        B,
        C,
        A_cumsum,
        prefix,
        gram,
        *B.stride(),
        *C.stride(),
        *A_cumsum.stride(),
        num_chunks,
        BLOCK_I=32,
        BLOCK_J=64,
        STATE_SIZE=128,
        num_warps=4,
        num_stages=2,
    )

    block_i = 32
    block_d = 64
    grid = (
        (128 // block_i) * (128 // block_d),
        32,
        batch_chunks,
    )

    _mamba2_causal_weight_tiles_kernel[grid](
        hidden_states,
        prefix,
        gram,
        output,
        num_chunks,
        *hidden_states.stride(),
        *output.stride(),
        BLOCK_I=block_i,
        BLOCK_J=32,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=1,
    )

    return output