# solution=GPT-5.6-Sol_043_mamba_chunk_scan_with_segsum_triton_optimized_r2 score=-1.0 passed=False
I’m keeping the validated affine-scan structure and targeting launch/runtime overhead in the multi-chunk path. The main low-risk change is to specialize the full-chunk case so it does not carry tail-related compile-time plumbing or tail checks through the boundary kernel.import torch
import triton
import triton.language as tl


_CHUNK_SIZE = 256
_STATE_SIZE = 256
_NUM_HEADS = 16
_HEAD_DIM = 64
_SINGLE_BLOCK_D = 16
_BLOCK_D = 32


@triton.jit
def _single_chunk_kernel(
    hidden_ptr,
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
    initial_ptr,
    output_ptr,
    final_ptr,
    seq_len,
    stride_hidden_t,
    stride_hidden_h,
    stride_hidden_d,
    stride_a_h,
    stride_a_t,
    stride_b_t,
    stride_b_s,
    stride_c_t,
    stride_c_s,
    stride_initial_h,
    stride_initial_d,
    stride_initial_s,
    stride_output_t,
    stride_output_h,
    stride_output_d,
    stride_final_h,
    stride_final_d,
    stride_final_s,
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid = tl.program_id(0)
    dim_blocks: tl.constexpr = HEAD_DIM // BLOCK_D
    per_batch: tl.constexpr = NUM_HEADS * dim_blocks
    batch = pid // per_batch
    rem = pid % per_batch
    head = rem // dim_blocks
    dim_block = rem % dim_blocks

    dims = dim_block * BLOCK_D + tl.arange(0, BLOCK_D)
    states = tl.arange(0, BLOCK_S)

    initial_offsets = (
        batch * NUM_HEADS * stride_initial_h
        + head * stride_initial_h
        + dims[:, None] * stride_initial_d
        + states[None, :] * stride_initial_s
    )
    state = tl.load(initial_ptr + initial_offsets).to(tl.float32)
    skip = tl.load(d_ptr + head).to(tl.float32)

    batch_hidden_offset = batch * seq_len * stride_hidden_t
    batch_a_offset = batch * NUM_HEADS * seq_len
    batch_b_offset = batch * seq_len * stride_b_t
    batch_c_offset = batch * seq_len * stride_c_t
    batch_output_offset = batch * seq_len * stride_output_t

    for index in range(CHUNK_SIZE):
        token_mask = index < seq_len

        a_value = tl.load(
            a_ptr
            + batch_a_offset
            + head * stride_a_h
            + index * stride_a_t,
            mask=token_mask,
            other=0.0,
        ).to(tl.float32)

        hidden_value = tl.load(
            hidden_ptr
            + batch_hidden_offset
            + index * stride_hidden_t
            + head * stride_hidden_h
            + dims * stride_hidden_d,
            mask=token_mask,
            other=0.0,
        ).to(tl.float32)

        b_value = tl.load(
            b_ptr
            + batch_b_offset
            + index * stride_b_t
            + states * stride_b_s,
            mask=token_mask,
            other=0.0,
        ).to(tl.float32)

        state = state * tl.exp(a_value) + hidden_value[:, None] * b_value[None, :]

        c_value = tl.load(
            c_ptr
            + batch_c_offset
            + index * stride_c_t
            + states * stride_c_s,
            mask=token_mask,
            other=0.0,
        ).to(tl.float32)

        output_value = tl.sum(state * c_value[None, :], axis=1)
        output_value += skip * hidden_value

        output_offsets = (
            batch_output_offset
            + index * stride_output_t
            + head * stride_output_h
            + dims * stride_output_d
        )
        tl.store(output_ptr + output_offsets, output_value, mask=token_mask)

    final_offsets = (
        batch * NUM_HEADS * stride_final_h
        + head * stride_final_h
        + dims[:, None] * stride_final_d
        + states[None, :] * stride_final_s
    )
    tl.store(final_ptr + final_offsets, state)


@triton.jit
def _chunk_summary_kernel(
    hidden_ptr,
    a_ptr,
    b_ptr,
    local_states_ptr,
    chunk_decay_ptr,
    seq_len,
    num_chunks,
    grid_chunks,
    chunk_offset,
    stride_hidden_t,
    stride_hidden_h,
    stride_hidden_d,
    stride_a_h,
    stride_a_t,
    stride_b_t,
    stride_b_s,
    stride_local_c,
    stride_local_h,
    stride_local_d,
    stride_local_s,
    stride_decay_c,
    stride_decay_h,
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid = tl.program_id(0)
    dim_blocks: tl.constexpr = HEAD_DIM // BLOCK_D
    head_dim_blocks: tl.constexpr = NUM_HEADS * dim_blocks
    per_batch = grid_chunks * head_dim_blocks

    batch = pid // per_batch
    rem = pid % per_batch
    local_chunk = rem // head_dim_blocks
    rem = rem % head_dim_blocks
    head = rem // dim_blocks
    dim_block = rem % dim_blocks
    chunk = chunk_offset + local_chunk

    dims = dim_block * BLOCK_D + tl.arange(0, BLOCK_D)
    states = tl.arange(0, BLOCK_S)
    state = tl.zeros((BLOCK_D, BLOCK_S), dtype=tl.float32)
    a_sum = tl.zeros((), dtype=tl.float32)

    batch_hidden_offset = batch * seq_len * stride_hidden_t
    batch_a_offset = batch * NUM_HEADS * seq_len
    batch_b_offset = batch * seq_len * stride_b_t
    chunk_token = chunk * CHUNK_SIZE

    for index in range(CHUNK_SIZE):
        token = chunk_token + index

        a_value = tl.load(
            a_ptr
            + batch_a_offset
            + head * stride_a_h
            + token * stride_a_t
        ).to(tl.float32)

        hidden_value = tl.load(
            hidden_ptr
            + batch_hidden_offset
            + token * stride_hidden_t
            + head * stride_hidden_h
            + dims * stride_hidden_d
        ).to(tl.float32)

        b_value = tl.load(
            b_ptr
            + batch_b_offset
            + token * stride_b_t
            + states * stride_b_s
        ).to(tl.float32)

        decay = tl.exp(a_value)
        state = state * decay + hidden_value[:, None] * b_value[None, :]
        a_sum += a_value

    local_offsets = (
        batch * num_chunks * stride_local_c
        + chunk * stride_local_c
        + head * stride_local_h
        + dims[:, None] * stride_local_d
        + states[None, :] * stride_local_s
    )
    tl.store(local_states_ptr + local_offsets, state)

    if dim_block == 0:
        decay_offset = (
            batch * num_chunks * stride_decay_c
            + chunk * stride_decay_c
            + head * stride_decay_h
        )
        tl.store(chunk_decay_ptr + decay_offset, tl.exp(a_sum))


@triton.jit
def _chunk_start_kernel(
    hidden_ptr,
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
    initial_ptr,
    local_states_ptr,
    chunk_decay_ptr,
    output_ptr,
    final_ptr,
    seq_len,
    num_chunks,
    scan_chunks,
    stride_hidden_t,
    stride_hidden_h,
    stride_hidden_d,
    stride_a_h,
    stride_a_t,
    stride_b_t,
    stride_b_s,
    stride_c_t,
    stride_c_s,
    stride_initial_h,
    stride_initial_d,
    stride_initial_s,
    stride_local_c,
    stride_local_h,
    stride_local_d,
    stride_local_s,
    stride_decay_c,
    stride_decay_h,
    stride_output_t,
    stride_output_h,
    stride_output_d,
    stride_final_h,
    stride_final_d,
    stride_final_s,
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    TAIL_SIZE: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HAS_TAIL: tl.constexpr,
):
    pid = tl.program_id(0)
    dim_blocks: tl.constexpr = HEAD_DIM // BLOCK_D
    per_batch: tl.constexpr = NUM_HEADS * dim_blocks
    batch = pid // per_batch
    rem = pid % per_batch
    head = rem // dim_blocks
    dim_block = rem % dim_blocks

    dims = dim_block * BLOCK_D + tl.arange(0, BLOCK_D)
    states = tl.arange(0, BLOCK_S)

    initial_offsets = (
        batch * NUM_HEADS * stride_initial_h
        + head * stride_initial_h
        + dims[:, None] * stride_initial_d
        + states[None, :] * stride_initial_s
    )
    state = tl.load(initial_ptr + initial_offsets).to(tl.float32)

    batch_local_offset = batch * num_chunks * stride_local_c
    batch_decay_offset = batch * num_chunks * stride_decay_c

    for chunk in range(scan_chunks):
        local_offsets = (
            batch_local_offset
            + chunk * stride_local_c
            + head * stride_local_h
            + dims[:, None] * stride_local_d
            + states[None, :] * stride_local_s
        )
        local_state = tl.load(local_states_ptr + local_offsets).to(tl.float32)
        tl.store(local_states_ptr + local_offsets, state)

        decay = tl.load(
            chunk_decay_ptr
            + batch_decay_offset
            + chunk * stride_decay_c
            + head * stride_decay_h
        ).to(tl.float32)
        state = state * decay + local_state

    if HAS_TAIL:
        skip = tl.load(d_ptr + head).to(tl.float32)
        batch_hidden_offset = batch * seq_len * stride_hidden_t
        batch_a_offset = batch * NUM_HEADS * seq_len
        batch_b_offset = batch * seq_len * stride_b_t
        batch_c_offset = batch * seq_len * stride_c_t
        batch_output_offset = batch * seq_len * stride_output_t
        tail_token = scan_chunks * CHUNK_SIZE

        for index in range(TAIL_SIZE):
            token = tail_token + index

            a_value = tl.load(
                a_ptr
                + batch_a_offset
                + head * stride_a_h
                + token * stride_a_t
            ).to(tl.float32)

            hidden_value = tl.load(
                hidden_ptr
                + batch_hidden_offset
                + token * stride_hidden_t
                + head * stride_hidden_h
                + dims * stride_hidden_d
            ).to(tl.float32)

            b_value = tl.load(
                b_ptr
                + batch_b_offset
                + token * stride_b_t
                + states * stride_b_s
            ).to(tl.float32)

            state = state * tl.exp(a_value) + hidden_value[:, None] * b_value[None, :]

            c_value = tl.load(
                c_ptr
                + batch_c_offset
                + token * stride_c_t
                + states * stride_c_s
            ).to(tl.float32)

            output_value = tl.sum(state * c_value[None, :], axis=1)
            output_value += skip * hidden_value

            output_offsets = (
                batch_output_offset
                + token * stride_output_t
                + head * stride_output_h
                + dims * stride_output_d
            )
            tl.store(output_ptr + output_offsets, output_value)

    final_offsets = (
        batch * NUM_HEADS * stride_final_h
        + head * stride_final_h
        + dims[:, None] * stride_final_d
        + states[None, :] * stride_final_s
    )
    tl.store(final_ptr + final_offsets, state)


@triton.jit
def _chunk_output_kernel(
    hidden_ptr,
    a_ptr,
    b_ptr,
    c_ptr,
    d_ptr,
    starts_ptr,
    output_ptr,
    seq_len,
    num_chunks,
    grid_chunks,
    chunk_offset,
    stride_hidden_t,
    stride_hidden_h,
    stride_hidden_d,
    stride_a_h,
    stride_a_t,
    stride_b_t,
    stride_b_s,
    stride_c_t,
    stride_c_s,
    stride_starts_c,
    stride_starts_h,
    stride_starts_d,
    stride_starts_s,
    stride_output_t,
    stride_output_h,
    stride_output_d,
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    ITER_SIZE: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid = tl.program_id(0)
    dim_blocks: tl.constexpr = HEAD_DIM // BLOCK_D
    head_dim_blocks: tl.constexpr = NUM_HEADS * dim_blocks
    per_batch = grid_chunks * head_dim_blocks

    batch = pid // per_batch
    rem = pid % per_batch
    local_chunk = rem // head_dim_blocks
    rem = rem % head_dim_blocks
    head = rem // dim_blocks
    dim_block = rem % dim_blocks
    chunk = chunk_offset + local_chunk

    dims = dim_block * BLOCK_D + tl.arange(0, BLOCK_D)
    states = tl.arange(0, BLOCK_S)

    start_offsets = (
        batch * num_chunks * stride_starts_c
        + chunk * stride_starts_c
        + head * stride_starts_h
        + dims[:, None] * stride_starts_d
        + states[None, :] * stride_starts_s
    )
    state = tl.load(starts_ptr + start_offsets).to(tl.float32)
    skip = tl.load(d_ptr + head).to(tl.float32)

    batch_hidden_offset = batch * seq_len * stride_hidden_t
    batch_a_offset = batch * NUM_HEADS * seq_len
    batch_b_offset = batch * seq_len * stride_b_t
    batch_c_offset = batch * seq_len * stride_c_t
    batch_output_offset = batch * seq_len * stride_output_t
    chunk_token = chunk * CHUNK_SIZE

    for index in range(ITER_SIZE):
        token = chunk_token + index

        a_value = tl.load(
            a_ptr
            + batch_a_offset
            + head * stride_a_h
            + token * stride_a_t
        ).to(tl.float32)

        hidden_value = tl.load(
            hidden_ptr
            + batch_hidden_offset
            + token * stride_hidden_t
            + head * stride_hidden_h
            + dims * stride_hidden_d
        ).to(tl.float32)

        b_value = tl.load(
            b_ptr
            + batch_b_offset
            + token * stride_b_t
            + states * stride_b_s
        ).to(tl.float32)

        state = state * tl.exp(a_value) + hidden_value[:, None] * b_value[None, :]

        c_value = tl.load(
            c_ptr
            + batch_c_offset
            + token * stride_c_t
            + states * stride_c_s
        ).to(tl.float32)

        output_value = tl.sum(state * c_value[None, :], axis=1)
        output_value += skip * hidden_value

        output_offsets = (
            batch_output_offset
            + token * stride_output_t
            + head * stride_output_h
            + dims * stride_output_d
        )
        tl.store(output_ptr + output_offsets, output_value)


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
    num_chunks = (seq_len + _CHUNK_SIZE - 1) // _CHUNK_SIZE

    hidden_states_f = hidden_states.contiguous()
    A_f = A.contiguous()
    B_f = B.contiguous()
    C_f = C.contiguous()
    D_f = D.contiguous()
    initial_states_f = initial_states.contiguous()

    final_state = torch.empty(
        (batch_size, num_heads, head_dim, _STATE_SIZE),
        device=hidden_states.device,
        dtype=torch.bfloat16,
    )
    output = torch.empty(
        (batch_size, seq_len, num_heads, head_dim),
        device=hidden_states.device,
        dtype=torch.bfloat16,
    )

    if num_chunks == 1:
        single_dim_blocks = head_dim // _SINGLE_BLOCK_D
        _single_chunk_kernel[
            (batch_size * num_heads * single_dim_blocks,)
        ](
            hidden_states_f,
            A_f,
            B_f,
            C_f,
            D_f,
            initial_states_f,
            output,
            final_state,
            seq_len,
            hidden_states_f.stride(1),
            hidden_states_f.stride(2),
            hidden_states_f.stride(3),
            A_f.stride(1),
            A_f.stride(2),
            B_f.stride(1),
            B_f.stride(3),
            C_f.stride(1),
            C_f.stride(3),
            initial_states_f.stride(1),
            initial_states_f.stride(2),
            initial_states_f.stride(3),
            output.stride(1),
            output.stride(2),
            output.stride(3),
            final_state.stride(1),
            final_state.stride(2),
            final_state.stride(3),
            BLOCK_D=_SINGLE_BLOCK_D,
            BLOCK_S=_STATE_SIZE,
            CHUNK_SIZE=_CHUNK_SIZE,
            NUM_HEADS=_NUM_HEADS,
            HEAD_DIM=_HEAD_DIM,
            num_warps=8,
        )
        return output.reshape(batch_size, seq_len, num_heads * head_dim), final_state

    dim_blocks = head_dim // _BLOCK_D
    tail_size = seq_len % _CHUNK_SIZE
    full_chunks = seq_len // _CHUNK_SIZE
    has_tail = tail_size != 0
    local_chunk_count = full_chunks if has_tail else num_chunks

    local_states = torch.empty(
        (batch_size, local_chunk_count, num_heads, head_dim, _STATE_SIZE),
        device=hidden_states.device,
        dtype=torch.bfloat16,
    )
    chunk_decay = torch.empty(
        (batch_size, local_chunk_count, num_heads),
        device=hidden_states.device,
        dtype=torch.float32,
    )

    _chunk_summary_kernel[
        (batch_size * full_chunks * num_heads * dim_blocks,)
    ](
        hidden_states_f,
        A_f,
        B_f,
        local_states,
        chunk_decay,
        seq_len,
        local_chunk_count,
        full_chunks,
        0,
        hidden_states_f.stride(1),
        hidden_states_f.stride(2),
        hidden_states_f.stride(3),
        A_f.stride(1),
        A_f.stride(2),
        B_f.stride(1),
        B_f.stride(3),
        local_states.stride(1),
        local_states.stride(2),
        local_states.stride(3),
        local_states.stride(4),
        chunk_decay.stride(1),
        chunk_decay.stride(2),
        BLOCK_D=_BLOCK_D,
        BLOCK_S=_STATE_SIZE,
        CHUNK_SIZE=_CHUNK_SIZE,
        NUM_HEADS=_NUM_HEADS,
        HEAD_DIM=_HEAD_DIM,
        num_warps=4,
    )

    _chunk_start_kernel[
        (batch_size * num_heads * dim_blocks,)
    ](
        hidden_states_f,
        A_f,
        B_f,
        C_f,
        D_f,
        initial_states_f,
        local_states,
        chunk_decay,
        output,
        final_state,
        seq_len,
        local_chunk_count,
        full_chunks,
        hidden_states_f.stride(1),
        hidden_states_f.stride(2),
        hidden_states_f.stride(3),
        A_f.stride(1),
        A_f.stride(2),
        B_f.stride(1),
        B_f.stride(3),
        C_f.stride(1),
        C_f.stride(3),
        initial_states_f.stride(1),
        initial_states_f.stride(2),
        initial_states_f.stride(3),
        local_states.stride(1),
        local_states.stride(2),
        local_states.stride(3),
        local_states.stride(4),
        chunk_decay.stride(1),
        chunk_decay.stride(2),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        final_state.stride(1),
        final_state.stride(2),
        final_state.stride(3),
        BLOCK_D=_BLOCK_D,
        BLOCK_S=_STATE_SIZE,
        CHUNK_SIZE=_CHUNK_SIZE,
        TAIL_SIZE=tail_size,
        NUM_HEADS=_NUM_HEADS,
        HEAD_DIM=_HEAD_DIM,
        HAS_TAIL=has_tail,
        num_warps=8,
    )

    _chunk_output_kernel[
        (batch_size * full_chunks * num_heads * dim_blocks,)
    ](
        hidden_states_f,
        A_f,
        B_f,
        C_f,
        D_f,
        local_states,
        output,
        seq_len,
        local_chunk_count,
        full_chunks,
        0,
        hidden_states_f.stride(1),
        hidden_states_f.stride(2),
        hidden_states_f.stride(3),
        A_f.stride(1),
        A_f.stride(2),
        B_f.stride(1),
        B_f.stride(3),
        C_f.stride(1),
        C_f.stride(3),
        local_states.stride(1),
        local_states.stride(2),
        local_states.stride(3),
        local_states.stride(4),
        output.stride(1),
        output.stride(2),
        output.stride(3),
        BLOCK_D=_BLOCK_D,
        BLOCK_S=_STATE_SIZE,
        CHUNK_SIZE=_CHUNK_SIZE,
        ITER_SIZE=_CHUNK_SIZE,
        NUM_HEADS=_NUM_HEADS,
        HEAD_DIM=_HEAD_DIM,
        num_warps=8,
    )

    return output.reshape(batch_size, seq_len, num_heads * head_dim), final_state