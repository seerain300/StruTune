# solution=GPT-5.6-Sol_030_flux_concatenated_sequence_processing_with_split_triton_optimized_r36 score=1.4386555537051136 passed=True
import torch
import triton
import triton.language as tl


@triton.jit
def _grouped_projection_kernel(
    hidden_states,
    encoder_hidden_states,
    process_weight,
    processed_hidden,
    processed_encoder,
    TEXT_SEQ_LEN: tl.constexpr,
    IMG_SEQ_LEN: tl.constexpr,
    HIDDEN_DIM: tl.constexpr,
    TEXT_TILES: tl.constexpr,
    TILES_PER_BATCH: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tile_group = tl.program_id(0)
    tile_n = tl.program_id(1)

    batch_idx = tile_group // TILES_PER_BATCH
    stream_tile = tile_group - batch_idx * TILES_PER_BATCH
    is_text = stream_tile < TEXT_TILES
    tile_m = tl.where(is_text, stream_tile, stream_tile - TEXT_TILES)

    rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)

    stream_len = tl.where(is_text, TEXT_SEQ_LEN, IMG_SEQ_LEN)
    input_ptr = tl.where(
        is_text,
        encoder_hidden_states,
        hidden_states,
    )
    output_ptr = tl.where(
        is_text,
        processed_encoder,
        processed_hidden,
    )

    row_mask = rows < stream_len
    row_offsets = (batch_idx * stream_len + rows) * HIDDEN_DIM

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, HIDDEN_DIM, BLOCK_K):
        k = k_start + k_offsets
        values = tl.load(
            input_ptr + row_offsets[:, None] + k[None, :],
            mask=row_mask[:, None],
            other=0.0,
            eviction_policy="evict_first",
        )
        weights = tl.load(
            process_weight + cols[None, :] * HIDDEN_DIM + k[:, None],
            eviction_policy="evict_last",
        )
        accumulator = tl.dot(
            values,
            weights,
            accumulator,
            input_precision="tf32x3",
        )

    tl.store(
        output_ptr + row_offsets[:, None] + cols[None, :],
        accumulator,
        mask=row_mask[:, None],
    )


@triton.jit
def _flattened_projection_kernel(
    hidden_states,
    encoder_hidden_states,
    process_weight,
    processed_hidden,
    processed_encoder,
    TEXT_SEQ_LEN: tl.constexpr,
    IMG_SEQ_LEN: tl.constexpr,
    TOTAL_ROWS: tl.constexpr,
    HIDDEN_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tile_m = tl.program_id(0)
    tile_n = tl.program_id(1)

    rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)

    sequence_len = TEXT_SEQ_LEN + IMG_SEQ_LEN
    batch_idx = rows // sequence_len
    sequence_row = rows - batch_idx * sequence_len
    is_text = sequence_row < TEXT_SEQ_LEN

    stream_row = tl.where(
        is_text,
        batch_idx * TEXT_SEQ_LEN + sequence_row,
        batch_idx * IMG_SEQ_LEN + sequence_row - TEXT_SEQ_LEN,
    )
    row_offsets = stream_row * HIDDEN_DIM

    input_rows = tl.where(
        is_text,
        encoder_hidden_states + row_offsets,
        hidden_states + row_offsets,
    )
    output_rows = tl.where(
        is_text,
        processed_encoder + row_offsets,
        processed_hidden + row_offsets,
    )

    row_mask = rows < TOTAL_ROWS
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, HIDDEN_DIM, BLOCK_K):
        k = k_start + k_offsets
        values = tl.load(
            input_rows[:, None] + k[None, :],
            mask=row_mask[:, None],
            other=0.0,
            eviction_policy="evict_first",
        )
        weights = tl.load(
            process_weight + cols[None, :] * HIDDEN_DIM + k[:, None],
            eviction_policy="evict_last",
        )
        accumulator = tl.dot(
            values,
            weights,
            accumulator,
            input_precision="tf32x3",
        )

    tl.store(
        output_rows[:, None] + cols[None, :],
        accumulator,
        mask=row_mask[:, None],
    )


@torch.no_grad()
def run(
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    process_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, img_seq_len, hidden_dim = hidden_states.shape
    text_seq_len = encoder_hidden_states.shape[1]

    processed_encoder = torch.empty_like(encoder_hidden_states)
    processed_hidden = torch.empty_like(hidden_states)

    total_rows = batch_size * (text_seq_len + img_seq_len)

    if total_rows <= 512:
        block_m = 64
        block_n = 64
        num_warps = 4
    elif total_rows <= 1024:
        block_m = 64
        block_n = 128
        num_warps = 4
    else:
        block_m = 128
        block_n = 128
        num_warps = 8

    block_k = 64
    num_stages = 2 if block_m == 128 and block_n == 128 else 3

    text_tiles = triton.cdiv(text_seq_len, block_m)
    img_tiles = triton.cdiv(img_seq_len, block_m)
    tiles_per_batch = text_tiles + img_tiles
    grouped_tiles = batch_size * tiles_per_batch
    flattened_tiles = triton.cdiv(total_rows, block_m)
    n_tiles = triton.cdiv(hidden_dim, block_n)

    if flattened_tiles * 10 <= grouped_tiles * 9:
        _flattened_projection_kernel[(flattened_tiles, n_tiles)](
            hidden_states,
            encoder_hidden_states,
            process_weight,
            processed_hidden,
            processed_encoder,
            TEXT_SEQ_LEN=text_seq_len,
            IMG_SEQ_LEN=img_seq_len,
            TOTAL_ROWS=total_rows,
            HIDDEN_DIM=hidden_dim,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=num_warps,
            num_stages=num_stages,
        )
    else:
        _grouped_projection_kernel[(grouped_tiles, n_tiles)](
            hidden_states,
            encoder_hidden_states,
            process_weight,
            processed_hidden,
            processed_encoder,
            TEXT_SEQ_LEN=text_seq_len,
            IMG_SEQ_LEN=img_seq_len,
            HIDDEN_DIM=hidden_dim,
            TEXT_TILES=text_tiles,
            TILES_PER_BATCH=tiles_per_batch,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    return processed_encoder, processed_hidden