# solution=GPT-5.6-Sol_043_mamba_chunk_scan_with_segsum_triton_optimized_r3 score=5.865770915875288 passed=True
import torch
import torch.nn.functional as F


@torch.no_grad()
@torch.compile(fullgraph=True)
def run(
    hidden_states: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    initial_states: torch.Tensor,
):
    batch_size, seq_len, num_heads, head_dim = hidden_states.shape
    chunk_size = 256
    state_size = 256

    pad_size = (-seq_len) % chunk_size
    padded_seq_len = seq_len + pad_size
    num_chunks = padded_seq_len // chunk_size

    B = B.squeeze(2)
    C = C.squeeze(2)

    if pad_size:
        hidden_states_padded = F.pad(
            hidden_states,
            (0, 0, 0, 0, 0, pad_size),
        )
        A_padded = F.pad(A, (0, pad_size))
        B_padded = F.pad(B, (0, 0, 0, pad_size))
        C_padded = F.pad(C, (0, 0, 0, pad_size))
    else:
        hidden_states_padded = hidden_states
        A_padded = A
        B_padded = B
        C_padded = C

    hidden_states_chunked = hidden_states_padded.reshape(
        batch_size,
        num_chunks,
        chunk_size,
        num_heads,
        head_dim,
    )
    A_chunked = A_padded.transpose(1, 2).reshape(
        batch_size,
        num_chunks,
        chunk_size,
        num_heads,
    )
    B_chunked = B_padded.reshape(
        batch_size,
        num_chunks,
        chunk_size,
        state_size,
    )
    C_chunked = C_padded.reshape(
        batch_size,
        num_chunks,
        chunk_size,
        state_size,
    )

    A_cumsum = torch.cumsum(
        A_chunked.permute(0, 3, 1, 2),
        dim=-1,
        dtype=torch.float32,
    )

    decay_intra = torch.exp(
        A_cumsum[..., :, None] - A_cumsum[..., None, :]
    )
    decay_intra.tril_()
    decay_intra = decay_intra.to(hidden_states.dtype)

    gram = torch.bmm(
        C_chunked.reshape(
            batch_size * num_chunks,
            chunk_size,
            state_size,
        ),
        B_chunked.reshape(
            batch_size * num_chunks,
            chunk_size,
            state_size,
        ).transpose(1, 2),
    ).reshape(
        batch_size,
        num_chunks,
        chunk_size,
        chunk_size,
    )
    decay_intra.mul_(gram.unsqueeze(1))

    hidden_by_head = hidden_states_chunked.permute(
        0, 3, 1, 2, 4
    )
    y_diag = torch.bmm(
        decay_intra.reshape(
            batch_size * num_heads * num_chunks,
            chunk_size,
            chunk_size,
        ),
        hidden_by_head.reshape(
            batch_size * num_heads * num_chunks,
            chunk_size,
            head_dim,
        ),
    ).reshape(
        batch_size,
        num_heads,
        num_chunks,
        chunk_size,
        head_dim,
    ).permute(0, 2, 3, 1, 4)

    decay_to_chunk_end = torch.exp(
        A_cumsum[..., -1:] - A_cumsum
    ).permute(0, 2, 3, 1).to(hidden_states.dtype)

    weighted_hidden = (
        hidden_states_chunked * decay_to_chunk_end[..., None]
    )
    states = torch.matmul(
        weighted_hidden.permute(0, 1, 3, 4, 2),
        B_chunked.unsqueeze(2),
    )

    if num_chunks == 1:
        states_out = initial_states[:, None]
        final_state = (
            initial_states
            * torch.exp(A_cumsum[:, :, 0, -1])[:, :, None, None].to(
                initial_states.dtype
            )
            + states[:, 0]
        )
    else:
        chunk_end_values = F.pad(
            A_cumsum[..., -1],
            (1, 0),
        )
        chunk_prefix = torch.cumsum(chunk_end_values, dim=-1)

        decay_chunks = torch.exp(
            chunk_prefix[..., :, None] - chunk_prefix[..., None, :]
        )
        decay_chunks.tril_()
        decay_chunks = decay_chunks.to(hidden_states.dtype)

        states_with_init = torch.cat(
            (initial_states[:, None], states),
            dim=1,
        )
        recurrence_size = num_chunks + 1
        new_states = torch.bmm(
            decay_chunks.reshape(
                batch_size * num_heads,
                recurrence_size,
                recurrence_size,
            ),
            states_with_init.permute(
                0, 2, 1, 3, 4
            ).reshape(
                batch_size * num_heads,
                recurrence_size,
                head_dim * state_size,
            ),
        ).reshape(
            batch_size,
            num_heads,
            recurrence_size,
            head_dim,
            state_size,
        ).permute(0, 2, 1, 3, 4)

        states_out = new_states[:, :-1]
        final_state = new_states[:, -1]

    state_decay_out = torch.exp(A_cumsum).permute(
        0, 2, 3, 1
    ).to(hidden_states.dtype)

    c_times_states = torch.bmm(
        C_chunked.reshape(
            batch_size * num_chunks,
            chunk_size,
            state_size,
        ),
        states_out.reshape(
            batch_size * num_chunks,
            num_heads * head_dim,
            state_size,
        ).transpose(1, 2),
    ).reshape(
        batch_size,
        num_chunks,
        chunk_size,
        num_heads,
        head_dim,
    )

    y = y_diag + c_times_states * state_decay_out[..., None]

    y = y.reshape(
        batch_size,
        padded_seq_len,
        num_heads,
        head_dim,
    )
    y.add_(
        hidden_states_padded * D[None, None, :, None]
    )

    if pad_size:
        y = y[:, :seq_len]

    output = y.reshape(
        batch_size,
        seq_len,
        num_heads * head_dim,
    )

    return output, final_state.to(torch.bfloat16)