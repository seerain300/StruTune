# solution=GPT-5.6-Sol_030_flux_concatenated_sequence_processing_with_split_triton_optimized_r8 score=1.5376528204005326 passed=True
import torch
import triton
import triton.language as tl


@triton.jit
def _stream_projection_kernel(
    hidden_states,
    encoder_hidden_states,
    process_weight,
    processed_hidden,
    processed_encoder,
    TEXT_ROWS: tl.constexpr,
    IMG_ROWS: tl.constexpr,
    TEXT_TILES: tl.constexpr,
    HIDDEN_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    stream_tile = tl.program_id(0)
    tile_n = tl.program_id(1)

    is_text = stream_tile < TEXT_TILES
    tile_m = tl.where(is_text, stream_tile, stream_tile - TEXT_TILES)

    rows = tile_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tile_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_offsets = tl.arange(0, BLOCK_K)

    stream_rows = tl.where(is_text, TEXT_ROWS, IMG_ROWS)
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

    row_mask = rows < stream_rows
    row_offsets = rows * HIDDEN_DIM

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

    text_rows = batch_size * text_seq_len
    img_rows = batch_size * img_seq_len
    total_rows = text_rows + img_rows

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

    text_tiles = triton.cdiv(text_rows, block_m)
    img_tiles = triton.cdiv(img_rows, block_m)
    stream_tiles = text_tiles + img_tiles
    n_tiles = triton.cdiv(hidden_dim, block_n)

    num_stages = 2 if block_m == 128 and stream_tiles >= 64 else 3

    _stream_projection_kernel[(stream_tiles, n_tiles)](
        hidden_states,
        encoder_hidden_states,
        process_weight,
        processed_hidden,
        processed_encoder,
        TEXT_ROWS=text_rows,
        IMG_ROWS=img_rows,
        TEXT_TILES=text_tiles,
        HIDDEN_DIM=hidden_dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return processed_encoder, processed_hidden