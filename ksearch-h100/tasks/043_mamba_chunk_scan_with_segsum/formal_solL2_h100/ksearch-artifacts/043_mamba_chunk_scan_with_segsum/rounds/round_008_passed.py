# solution=GPT-5.6-Sol_043_mamba_chunk_scan_with_segsum_triton_optimized_r8 score=2.8212671026368383 passed=True
import torch
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
        (((batch * NUM_HEADS + head) * HEAD_DIM + dims[:, None]) * BLOCK_S)
        + states[None, :]
    )
    state = tl.load(initial_ptr + initial_offsets).to(tl.float32)
    skip = tl.load(d_ptr + head).to(tl.float32)

    for index in range(CHUNK_SIZE):
        token_mask = index < seq_len

        a_offset = (batch * NUM_HEADS + head) * seq_len + index
        a_value = tl.load(
            a_ptr + a_offset,
            mask=token_mask,
            other=0.0,
        ).to(tl.float32)

        hidden_offsets = (
            ((batch * seq_len + index) * NUM_HEADS + head) * HEAD_DIM + dims
        )
        hidden_value = tl.load(
            hidden_ptr + hidden_offsets,
            mask=token_mask,
            other=0.0,
        ).to(tl.float32)

        b_offsets = (batch * seq_len + index) * BLOCK_S + states
        b_value = tl.load(
            b_ptr + b_offsets,
            mask=token_mask,
            other=0.0,
        ).to(tl.float32)

        state = state * tl.exp(a_value) + hidden_value[:, None] * b_value[None, :]

        c_value = tl.load(
            c_ptr + b_offsets,
            mask=token_mask,
            other=0.0,
        ).to(tl.float32)

        output_value = tl.sum(state * c_value[None, :], axis=1)
        output_value += skip * hidden_value

        output_offsets = (
            ((batch * seq_len + index) * NUM_HEADS + head) * HEAD_DIM + dims
        )
        tl.store(
            output_ptr + output_offsets,
            output_value,
            mask=token_mask,
        )

    final_offsets = (
        (((batch * NUM_HEADS + head) * HEAD_DIM + dims[:, None]) * BLOCK_S)
        + states[None, :]
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
    chunk_decay = tl.full((), 1.0, dtype=tl.float32)

    chunk_token = chunk * CHUNK_SIZE

    for index in range(CHUNK_SIZE):
        token = chunk_token + index

        a_offset = (batch * NUM_HEADS + head) * seq_len + token
        a_value = tl.load(a_ptr + a_offset).to(tl.float32)

        hidden_offsets = (
            ((batch * seq_len + token) * NUM_HEADS + head) * HEAD_DIM + dims
        )
        hidden_value = tl.load(hidden_ptr + hidden_offsets).to(tl.float32)

        b_offsets = (batch * seq_len + token) * BLOCK_S + states
        b_value = tl.load(b_ptr + b_offsets).to(tl.float32)

        decay = tl.exp(a_value)
        state = state * decay + hidden_value[:, None] * b_value[None, :]

        if dim_block == 0:
            chunk_decay *= decay

    local_offsets = (
        ((((batch * num_chunks + chunk) * NUM_HEADS + head) * HEAD_DIM)
         + dims[:, None]) * BLOCK_S
        + states[None, :]
    )
    tl.store(local_states_ptr + local_offsets, state)

    if dim_block == 0:
        decay_offset = (batch * num_chunks + chunk) * NUM_HEADS + head
        tl.store(chunk_decay_ptr + decay_offset, chunk_decay)


@triton.jit
def _chunk_start_kernel(
    initial_ptr,
    local_states_ptr,
    chunk_decay_ptr,
    final_ptr,
    num_chunks,
    scan_chunks,
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
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
        (((batch * NUM_HEADS + head) * HEAD_DIM + dims[:, None]) * BLOCK_S)
        + states[None, :]
    )
    state = tl.load(initial_ptr + initial_offsets).to(tl.float32)

    for chunk in range(scan_chunks):
        local_offsets = (
            ((((batch * num_chunks + chunk) * NUM_HEADS + head) * HEAD_DIM)
             + dims[:, None]) * BLOCK_S
            + states[None, :]
        )
        local_state = tl.load(local_states_ptr + local_offsets).to(tl.float32)
        tl.store(local_states_ptr + local_offsets, state)

        decay_offset = (batch * num_chunks + chunk) * NUM_HEADS + head
        decay = tl.load(chunk_decay_ptr + decay_offset).to(tl.float32)
        state = state * decay + local_state

    if HAS_TAIL:
        tail_offsets = (
            ((((batch * num_chunks + scan_chunks) * NUM_HEADS + head) * HEAD_DIM)
             + dims[:, None]) * BLOCK_S
            + states[None, :]
        )
        tl.store(local_states_ptr + tail_offsets, state)
    else:
        final_offsets = (
            (((batch * NUM_HEADS + head) * HEAD_DIM + dims[:, None]) * BLOCK_S)
            + states[None, :]
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
    final_ptr,
    seq_len,
    num_chunks,
    grid_chunks,
    chunk_offset,
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    ITER_SIZE: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STORE_FINAL: tl.constexpr,
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
        ((((batch * num_chunks + chunk) * NUM_HEADS + head) * HEAD_DIM)
         + dims[:, None]) * BLOCK_S
        + states[None, :]
    )
    state = tl.load(starts_ptr + start_offsets).to(tl.float32)
    skip = tl.load(d_ptr + head).to(tl.float32)

    chunk_token = chunk * CHUNK_SIZE

    for index in range(ITER_SIZE):
        token = chunk_token + index

        a_offset = (batch * NUM_HEADS + head) * seq_len + token
        a_value = tl.load(a_ptr + a_offset).to(tl.float32)

        hidden_offsets = (
            ((batch * seq_len + token) * NUM_HEADS + head) * HEAD_DIM + dims
        )
        hidden_value = tl.load(hidden_ptr + hidden_offsets).to(tl.float32)

        b_offsets = (batch * seq_len + token) * BLOCK_S + states
        b_value = tl.load(b_ptr + b_offsets).to(tl.float32)

        state = state * tl.exp(a_value) + hidden_value[:, None] * b_value[None, :]

        c_value = tl.load(c_ptr + b_offsets).to(tl.float32)
        output_value = tl.sum(state * c_value[None, :], axis=1)
        output_value += skip * hidden_value

        output_offsets = (
            ((batch * seq_len + token) * NUM_HEADS + head) * HEAD_DIM + dims
        )
        tl.store(output_ptr + output_offsets, output_value)

    if STORE_FINAL:
        final_offsets = (
            (((batch * NUM_HEADS + head) * HEAD_DIM + dims[:, None]) * BLOCK_S)
            + states[None, :]
        )
        tl.store(final_ptr + final_offsets, state)


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

    output = torch.empty(
        (batch_size, seq_len, num_heads, head_dim),
        device=hidden_states.device,
        dtype=torch.bfloat16,
    )
    final_state = torch.empty(
        (batch_size, num_heads, head_dim, _STATE_SIZE),
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
            BLOCK_D=_SINGLE_BLOCK_D,
            BLOCK_S=_STATE_SIZE,
            CHUNK_SIZE=_CHUNK_SIZE,
            NUM_HEADS=_NUM_HEADS,
            HEAD_DIM=_HEAD_DIM,
            num_warps=8,
        )
        return output.reshape(batch_size, seq_len, num_heads * head_dim), final_state

    dim_blocks = head_dim // _BLOCK_D
    local_states = torch.empty(
        (batch_size, num_chunks, num_heads, head_dim, _STATE_SIZE),
        device=hidden_states.device,
        dtype=torch.bfloat16,
    )
    chunk_decay = torch.empty(
        (batch_size, num_chunks, num_heads),
        device=hidden_states.device,
        dtype=torch.float32,
    )

    tail_size = seq_len % _CHUNK_SIZE
    full_chunks = seq_len // _CHUNK_SIZE
    has_tail = tail_size != 0

    _chunk_summary_kernel[
        (batch_size * full_chunks * num_heads * dim_blocks,)
    ](
        hidden_states_f,
        A_f,
        B_f,
        local_states,
        chunk_decay,
        seq_len,
        num_chunks,
        full_chunks,
        0,
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
        initial_states_f,
        local_states,
        chunk_decay,
        final_state,
        num_chunks,
        full_chunks,
        BLOCK_D=_BLOCK_D,
        BLOCK_S=_STATE_SIZE,
        NUM_HEADS=_NUM_HEADS,
        HEAD_DIM=_HEAD_DIM,
        HAS_TAIL=has_tail,
        num_warps=4,
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
        final_state,
        seq_len,
        num_chunks,
        full_chunks,
        0,
        BLOCK_D=_BLOCK_D,
        BLOCK_S=_STATE_SIZE,
        CHUNK_SIZE=_CHUNK_SIZE,
        ITER_SIZE=_CHUNK_SIZE,
        NUM_HEADS=_NUM_HEADS,
        HEAD_DIM=_HEAD_DIM,
        STORE_FINAL=False,
        num_warps=8,
    )

    if has_tail:
        _chunk_output_kernel[
            (batch_size * num_heads * dim_blocks,)
        ](
            hidden_states_f,
            A_f,
            B_f,
            C_f,
            D_f,
            local_states,
            output,
            final_state,
            seq_len,
            num_chunks,
            1,
            full_chunks,
            BLOCK_D=_BLOCK_D,
            BLOCK_S=_STATE_SIZE,
            CHUNK_SIZE=_CHUNK_SIZE,
            ITER_SIZE=tail_size,
            NUM_HEADS=_NUM_HEADS,
            HEAD_DIM=_HEAD_DIM,
            STORE_FINAL=True,
            num_warps=8,
        )

    return output.reshape(batch_size, seq_len, num_heads * head_dim), final_state