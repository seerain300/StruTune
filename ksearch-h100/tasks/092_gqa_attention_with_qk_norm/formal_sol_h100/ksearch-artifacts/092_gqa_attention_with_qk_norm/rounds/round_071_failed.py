# solution=GPT-5.6-Sol_092_gqa_attention_with_qk_norm_triton_optimized_r10 score=-1.0 passed=False
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _norm_rope_kernel(
    X,
    W,
    COS,
    SIN,
    OUT,
    S: tl.constexpr,
    H: tl.constexpr,
    EPS: tl.constexpr,
    ROWS: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    d = tl.arange(0, 128)

    b = tl.program_id(1)
    token = rows // H
    head = rows % H
    valid = token < S

    x = tl.load(
        X
        + b * S * H * 128
        + token[:, None] * H * 128
        + head[:, None] * 128
        + d[None, :],
        valid[:, None],
        0,
    ).to(tl.float32)

    weight = tl.load(W + d).to(tl.float32)
    variance = tl.sum(x * x, 1) * (1.0 / 128.0)
    x = (
        x
        * tl.rsqrt(variance[:, None] + EPS)
        * weight[None, :]
    ).to(tl.bfloat16)

    other_d = tl.where(d < 64, d + 64, d - 64)
    other = tl.gather(
        x,
        tl.broadcast_to(other_d[None, :], (ROWS, 128)),
        1,
    )
    rotated = tl.where(d[None, :] < 64, -other, other)

    if H == 96:
        rope_token = (tl.program_id(0) * ROWS) // H
        rope_valid = rope_token < S

        cos_row = tl.load(
            COS + b * S * 128 + rope_token * 128 + d,
            rope_valid,
            0,
        ).to(tl.float32)
        sin_row = tl.load(
            SIN + b * S * 128 + rope_token * 128 + d,
            rope_valid,
            0,
        ).to(tl.float32)

        cos = tl.broadcast_to(cos_row[None, :], (ROWS, 128))
        sin = tl.broadcast_to(sin_row[None, :], (ROWS, 128))
    elif H == 8:
        rope_token0 = (tl.program_id(0) * ROWS) // H
        rope_token1 = rope_token0 + 1

        cos_row0 = tl.load(
            COS + b * S * 128 + rope_token0 * 128 + d,
            rope_token0 < S,
            0,
        ).to(tl.float32)
        cos_row1 = tl.load(
            COS + b * S * 128 + rope_token1 * 128 + d,
            rope_token1 < S,
            0,
        ).to(tl.float32)
        sin_row0 = tl.load(
            SIN + b * S * 128 + rope_token0 * 128 + d,
            rope_token0 < S,
            0,
        ).to(tl.float32)
        sin_row1 = tl.load(
            SIN + b * S * 128 + rope_token1 * 128 + d,
            rope_token1 < S,
            0,
        ).to(tl.float32)

        first_token = token[:, None] == rope_token0
        cos = tl.where(
            first_token,
            tl.broadcast_to(cos_row0[None, :], (ROWS, 128)),
            tl.broadcast_to(cos_row1[None, :], (ROWS, 128)),
        )
        sin = tl.where(
            first_token,
            tl.broadcast_to(sin_row0[None, :], (ROWS, 128)),
            tl.broadcast_to(sin_row1[None, :], (ROWS, 128)),
        )
    else:
        cos = tl.load(
            COS
            + b * S * 128
            + token[:, None] * 128
            + d[None, :],
            valid[:, None],
            0,
        ).to(tl.float32)
        sin = tl.load(
            SIN
            + b * S * 128
            + token[:, None] * 128
            + d[None, :],
            valid[:, None],
            0,
        ).to(tl.float