# task: 043_mamba_chunk_scan_with_segsum
# bench: SOL-L2 | batch: formal2_solL2_20260916
# final eval (official evaluator, full workloads): valid=True pass=16/16 geomean=9.169x
# feedback best (5-workload sample during search): 8.523x
# torch fallback audit: 干净 (-)
# tokens: 2,425,533

import torch
import triton
import triton.language as tl


_CHUNK_SIZE = 256
_NUM_HEADS = 16
_HEAD_DIM = 64
_STATE_SIZE = 256


@triton.jit
def _chunk_prefix_kernel(
    A_ptr,
    prefix_ptr,
    seq_len,
    num_chunks: tl.constexpr,
    stride_ab,
    stride_ah,
    stride_at,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    chunk = pid % num_chunks
    bh = pid // num_chunks
    head = bh % 16
    batch = bh // 16

    offsets = tl.arange(0, BLOCK)
    positions = chunk * BLOCK + offsets
    a = tl.load(
        A_ptr
        + batch * stride_ab
        + head * stride_ah
        + positions * stride_at,
        mask=positions < seq_len,
        other=0.0,
    ).to(tl.float32)

    prefix = tl.cumsum(a, axis=0)
    out_offsets = (
        ((batch * 16 + head) * num_chunks + chunk) * BLOCK
        + offsets
    )
    tl.store(prefix_ptr + out_offsets, prefix)


@triton.jit
def _group_gram_kernel(
    B_ptr,
    C_ptr,
    gram_ptr,
    seq_len,
    num_chunks: tl.constexpr,
    launch_chunks: tl.constexpr,
    chunk_offset: tl.constexpr,
    stride_bb,
    stride_bt,
    stride_bs,
    stride_cb,
    stride_ct,
    stride_cs,
    BI: tl.constexpr,
    BJ: tl.constexpr,
    BK: tl.constexpr,
):
    bc = tl.program_id(0)
    local_chunk = bc % launch_chunks
    batch = bc // launch_chunks
    chunk = local_chunk + chunk_offset
    tile = tl.program_id(1)

    block_i = tl.where(
        tile < 1,
        0,
        tl.where(
            tile < 3,
            1,
            tl.where(
                tile < 6,
                2,
                tl.where(
                    tile < 10,
                    3,
                    tl.where(
                        tile < 15,
                        4,
                        tl.where(
                            tile < 21,
                            5,
                            tl.where(tile < 28, 6, 7),
                        ),
                    ),
                ),
            ),
        ),
    )
    block_j = tile - (block_i * (block_i + 1)) // 2

    rows = block_i * BI + tl.arange(0, BI)
    cols = block_j * BJ + tl.arange(0, BJ)
    state_offsets = tl.arange(0, BK)

    chunk_start = chunk * 256
    row_positions = chunk_start + rows
    col_positions = chunk_start + cols
    row_valid = row_positions < seq_len
    col_valid = col_positions < seq_len

    acc = tl.zeros((BI, BJ), dtype=tl.float32)

    for state_start in range(0, 256, BK):
        states = state_start + state_offsets
        c = tl.load(
            C_ptr
            + batch * stride_cb
            + row_positions[:, None] * stride_ct
            + states[None, :] * stride_cs,
            mask=row_valid[:, None],
            other=0.0,
        )
        b = tl.load(
            B_ptr
            + batch * stride_bb
            + col_positions[:, None] * stride_bt
            + states[None, :] * stride_bs,
            mask=col_valid[:, None],
            other=0.0,
        )
        acc += tl.dot(c, tl.trans(b))

    tl.store(
        gram_ptr
        + ((batch * num_chunks + chunk) * 256 * 256)
        + rows[:, None] * 256
        + cols[None, :],
        acc,
    )


@triton.jit
def _chunk_output_kernel(
    hidden_ptr,
    C_ptr,
    D_ptr,
    state_ptr,
    prefix_ptr,
    gram_ptr,
    output_ptr,
    seq_len,
    chunk: tl.constexpr,
    num_chunks: tl.constexpr,
    stride_xb,
    stride_xt,
    stride_xh,
    stride_xd,
    stride_cb,
    stride_ct,
    stride_cs,
    stride_state_b,
    stride_state_h,
    stride_state_d,
    stride_state_s,
    stride_ob,
    stride_ot,
    stride_oh,
    stride_od,
    BI: tl.constexpr,
    BJ: tl.constexpr,
    BS: tl.constexpr,
    BD: tl.constexpr,
    TMAX: tl.constexpr,
):
    bh = tl.program_id(0)
    row_block = tl.program_id(1)
    batch = bh // 16
    head = bh % 16

    rows = row_block * BI + tl.arange(0, BI)
    dims = tl.arange(0, BD)
    positions = chunk * 256 + rows
    row_valid = positions < seq_len

    prefix_base = ((batch * 16 + head) * num_chunks + chunk) * 256
    gram_base = (batch * num_chunks + chunk) * 256 * 256

    prefix_i = tl.load(
        prefix_ptr + prefix_base + rows,
        mask=rows < 256,
        other=0.0,
    ).to(tl.float32)

    prefix_anchor = tl.load(
        prefix_ptr + prefix_base + row_block * BI
    ).to(tl.float32)
    decay_rows = tl.exp(prefix_i - prefix_anchor)

    acc = tl.zeros((BI, BD), dtype=tl.float32)

    for state_start in range(0, 256, BS):
        states = state_start + tl.arange(0, BS)
        c = tl.load(
            C_ptr
            + batch * stride_cb
            + positions[:, None] * stride_ct
            + states[None, :] * stride_cs,
            mask=row_valid[:, None],
            other=0.0,
        )
        entering_state = tl.load(
            state_ptr
            + batch * stride_state_b
            + head * stride_state_h
            + dims[:, None] * stride_state_d
            + states[None, :] * stride_state_s,
        ).to(tl.bfloat16)
        acc += tl.dot(c, tl.trans(entering_state))

    acc *= tl.exp(prefix_i)[:, None]

    for col_start in range(0, TMAX, BJ):
        if col_start <= (row_block + 1) * BI - 1:
            cols = col_start + tl.arange(0, BJ)
            col_positions = chunk * 256 + cols
            col_valid = col_positions < seq_len

            gram = tl.load(
                gram_ptr
                + gram_base
                + rows[:, None] * 256
                + cols[None, :],
                mask=(rows[:, None] < 256)
                & (cols[None, :] < 256),
                other=0.0,
            ).to(tl.float32)

            prefix_j = tl.load(
                prefix_ptr + prefix_base + cols,
                mask=cols < 256,
                other=0.0,
            ).to(tl.float32)

            decay_cols = tl.exp(prefix_anchor - prefix_j)

            causal = (
                row_valid[:, None]
                & col_valid[None, :]
                & (cols[None, :] <= rows[:, None])
            )
            weights = tl.where(
                causal,
                gram
                * decay_rows[:, None]
                * decay_cols[None, :],
                0.0,
            ).to(tl.bfloat16)

            x = tl.load(
                hidden_ptr
                + batch * stride_xb
                + col_positions[:, None] * stride_xt
                + head * stride_xh
                + dims[None, :] * stride_xd,
                mask=col_valid[:, None],
                other=0.0,
            )
            acc += tl.dot(weights, x)

    current_x = tl.load(
        hidden_ptr
        + batch * stride_xb
        + positions[:, None] * stride_xt
        + head * stride_xh
        + dims[None, :] * stride_xd,
        mask=row_valid[:, None],
        other=0.0,
    ).to(tl.float32)

    acc += tl.load(D_ptr + head).to(tl.float32) * current_x

    tl.store(
        output_ptr
        + batch * stride_ob
        + positions[:, None] * stride_ot
        + head * stride_oh
        + dims[None, :] * stride_od,
        acc,
        mask=row_valid[:, None],
    )


@triton.jit
def _chunk_state_kernel(
    hidden_ptr,
    B_ptr,
    state_in_ptr,
    state_out_ptr,
    prefix_ptr,
    seq_len,
    chunk: tl.constexpr,
    num_chunks: tl.constexpr,
    stride_xb,
    stride_xt,
    stride_xh,
    stride_xd,
    stride_bb,
    stride_bt,
    stride_bs,
    stride_state_b,
    stride_state_h,
    stride_state_d,
    stride_state_s,
    BD: tl.constexpr,
    BS: tl.constexpr,
    BT: tl.constexpr,
    TMAX: tl.constexpr,
):
    bh = tl.program_id(0)
    d_block = tl.program_id(1)
    s_block = tl.program_id(2)
    batch = bh // 16
    head = bh % 16

    dims = d_block * BD + tl.arange(0, BD)
    states = s_block * BS + tl.arange(0, BS)

    prefix_base = ((batch * 16 + head) * num_chunks + chunk) * 256
    prefix_last = tl.load(
        prefix_ptr + prefix_base + 255
    ).to(tl.float32)

    acc = tl.zeros((BD, BS), dtype=tl.float32)

    for time_start in range(0, TMAX, BT):
        times = time_start + tl.arange(0, BT)
        positions = chunk * 256 + times
        time_valid = positions < seq_len

        prefix_t = tl.load(
            prefix_ptr + prefix_base + times,
            mask=times < 256,
            other=0.0,
        ).to(tl.float32)
        decay = tl.exp(prefix_last - prefix_t)

        x = tl.load(
            hidden_ptr
            + batch * stride_xb
            + positions[:, None] * stride_xt
            + head * stride_xh
            + dims[None, :] * stride_xd,
            mask=time_valid[:, None],
            other=0.0,
        )
        b = tl.load(
            B_ptr
            + batch * stride_bb
            + positions[:, None] * stride_bt
            + states[None, :] * stride_bs,
            mask=time_valid[:, None],
            other=0.0,
        )
        b = (b.to(tl.float32) * decay[:, None]).to(tl.bfloat16)
        acc += tl.dot(tl.trans(x), b)

    old_state = tl.load(
        state_in_ptr
        + batch * stride_state_b
        + head * stride_state_h
        + dims[:, None] * stride_state_d
        + states[None, :] * stride_state_s,
    ).to(tl.float32)

    acc += old_state * tl.exp(prefix_last)

    tl.store(
        state_out_ptr
        + batch * stride_state_b
        + head * stride_state_h
        + dims[:, None] * stride_state_d
        + states[None, :] * stride_state_s,
        acc,
    )


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    initial_states: torch.Tensor,
):
    batch_size, seq_len, num_heads, head_dim = hidden_states.shape
    if num_heads != _NUM_HEADS or head_dim != _HEAD_DIM:
        raise ValueError("Expected hidden_states with 16 heads and head_dim 64")

    num_chunks = triton.cdiv(seq_len, _CHUNK_SIZE)

    output = torch.empty(
        (batch_size, seq_len, _NUM_HEADS, _HEAD_DIM),
        device=hidden_states.device,
        dtype=torch.bfloat16,
    )
    final_state = torch.empty_like(initial_states)

    prefix = torch.empty(
        (batch_size, _NUM_HEADS, num_chunks, _CHUNK_SIZE),
        device=hidden_states.device,
        dtype=torch.float32,
    )
    gram = torch.empty(
        (batch_size, num_chunks, _CHUNK_SIZE, _CHUNK_SIZE),
        device=hidden_states.device,
        dtype=torch.bfloat16,
    )

    _chunk_prefix_kernel[(batch_size * _NUM_HEADS * num_chunks,)](
        A,
        prefix,
        seq_len,
        num_chunks,
        A.stride(0),
        A.stride(1),
        A.stride(2),
        BLOCK=_CHUNK_SIZE,
        num_warps=8,
    )

    last_chunk_len = seq_len - (num_chunks - 1) * _CHUNK_SIZE
    full_chunks = (
        num_chunks
        if last_chunk_len == _CHUNK_SIZE
        else num_chunks - 1
    )

    if full_chunks > 0:
        _group_gram_kernel[(batch_size * full_chunks, 36)](
            B,
            C,
            gram,
            seq_len,
            num_chunks,
            full_chunks,
            0,
            B.stride(0),
            B.stride(1),
            B.stride(3),
            C.stride(0),
            C.stride(1),
            C.stride(3),
            BI=32,
            BJ=32,
            BK=64,
            num_warps=4,
        )

    if full_chunks < num_chunks:
        if last_chunk_len <= 64:
            partial_tile = 16
            partial_gram_num_warps = 4
        else:
            partial_tile = 32
            partial_gram_num_warps = 8

        partial_blocks = triton.cdiv(last_chunk_len, partial_tile)
        partial_tiles = partial_blocks * (partial_blocks + 1) // 2

        _group_gram_kernel[(batch_size, partial_tiles)](
            B,
            C,
            gram,
            seq_len,
            num_chunks,
            1,
            full_chunks,
            B.stride(0),
            B.stride(1),
            B.stride(3),
            C.stride(0),
            C.stride(1),
            C.stride(3),
            BI=partial_tile,
            BJ=partial_tile,
            BK=64,
            num_warps=partial_gram_num_warps,
        )

    state_buffer = None
    if num_chunks > 1:
        state_buffer = torch.empty(
            initial_states.shape,
            device=initial_states.device,
            dtype=torch.float32,
        )

    state = initial_states
    output_bi = 32 if batch_size == 1 else 64

    for chunk in range(num_chunks):
        chunk_start = chunk * _CHUNK_SIZE
        chunk_len = min(_CHUNK_SIZE, seq_len - chunk_start)

        chunk_output_bi = output_bi
        chunk_output_bj = 64
        output_num_warps = 8

        if chunk == num_chunks - 1 and chunk_len < output_bi:
            if chunk_len <= 16:
                chunk_output_bi = 16
                chunk_output_bj = 16
                output_num_warps = 2
            else:
                chunk_output_bi = 32
                chunk_output_bj = 32
                output_num_warps = 4

        if (
            chunk == num_chunks - 1
            and chunk_len >= 33
            and chunk_len <= 64
            and chunk_output_bi == 32
        ):
            output_num_warps = 4

        _chunk_output_kernel[
            (
                batch_size * _NUM_HEADS,
                triton.cdiv(chunk_len, chunk_output_bi),
            )
        ](
            hidden_states,
            C,
            D,
            state,
            prefix,
            gram,
            output,
            seq_len,
            chunk,
            num_chunks,
            hidden_states.stride(0),
            hidden_states.stride(1),
            hidden_states.stride(2),
            hidden_states.stride(3),
            C.stride(0),
            C.stride(1),
            C.stride(3),
            state.stride(0),
            state.stride(1),
            state.stride(2),
            state.stride(3),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            output.stride(3),
            BI=chunk_output_bi,
            BJ=chunk_output_bj,
            BS=64,
            BD=64,
            TMAX=chunk_len,
            num_warps=output_num_warps,
        )

        next_state = final_state if chunk == num_chunks - 1 else state_buffer

        if chunk_len <= 16:
            state_bt = 16
            state_num_warps = 2
        elif chunk_len <= 32:
            state_bt = 32
            state_num_warps = 4
        elif chunk_len <= 64:
            state_bt = 64
            state_num_warps = 4
        else:
            state_bt = 64
            state_num_warps = 8

        _chunk_state_kernel[
            (
                batch_size * _NUM_HEADS,
                1,
                _STATE_SIZE // 64,
            )
        ](
            hidden_states,
            B,
            state,
            next_state,
            prefix,
            seq_len,
            chunk,
            num_chunks,
            hidden_states.stride(0),
            hidden_states.stride(1),
            hidden_states.stride(2),
            hidden_states.stride(3),
            B.stride(0),
            B.stride(1),
            B.stride(3),
            state.stride(0),
            state.stride(1),
            state.stride(2),
            state.stride(3),
            BD=64,
            BS=64,
            BT=state_bt,
            TMAX=chunk_len,
            num_warps=state_num_warps,
        )
        state = next_state

    return (
        output.reshape(
            batch_size,
            seq_len,
            _NUM_HEADS * _HEAD_DIM,
        ),
        final_state,
    )