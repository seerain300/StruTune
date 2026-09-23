# solution=GPT-5.6-Sol_043_mamba_chunk_scan_with_segsum_triton_optimized_r11 score=-1.0 passed=False
import torch
import torch.nn.functional as F


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
    state_size = 256
    chunk_size = 256

    pad_size = (chunk_size - seq_len % chunk_size) % chunk_size
    padded_seq_len = seq_len + pad_size
    num_chunks = padded_seq_len // chunk_size

    hidden_states_f = hidden_states.float()
    A_f = A.float()
    B_f = B.float().squeeze(2)
    C_f = C.float().squeeze(2)
    D_f = D.float()
    initial_states_f = initial_states.float()

    if pad_size:
        hidden_states_padded = F.pad(
            hidden_states_f, (0, 0, 0, 0, 0, pad_size)
        )
        A_padded = F.pad(A_f, (0, pad_size))
        B_padded = F.pad(B_f, (0, 0, 0, pad_size))
        C_padded = F.pad(C_f, (0, 0, 0, pad_size))
    else:
        hidden_states_padded = hidden_states_f
        A_padded = A_f
        B_padded = B_f
        C_padded = C_f

    D_residual = D_f[None, None, :, None] * hidden_states_padded

    hidden_states_chunked = hidden_states_padded.reshape(
        batch_size, num_chunks, chunk_size, num_heads, head_dim
    )

    A_chunked = A_padded.transpose(1, 2).reshape(
        batch_size, num_chunks, chunk_size, num_heads
    )

    B_chunked = B_padded.reshape(
        batch_size, num_chunks, chunk_size, state_size
    )
    C_chunked = C_padded.reshape(
        batch_size, num_chunks, chunk_size, state_size
    )

    A_chunked_perm = A_chunked.permute(0, 3, 1, 2)
    A_cumsum = torch.cumsum(A_chunked_perm, dim=-1)

    triangular_mask = torch.tril(
        torch.ones(
            chunk_size,
            chunk_size,
            device=hidden_states.device,
            dtype=torch.bool,
        )
    )

    decay_intra = torch.exp(
        A_cumsum[..., :, None] - A_cumsum[..., None, :]
    )
    decay_intra = decay_intra.masked_fill(~triangular_mask, 0.0)

    # B and C are shared by all heads because n_groups == 1. Compute the
    # chunk Gram matrix once and reuse it for every head's decay pattern.
    gram = torch.einsum(
        "bcis,bcjs->bcij",
        C_chunked,
        B_chunked,
    )

    y_diag = torch.einsum(
        "bhcij,bcij,bcjhd->bcihd",
        decay_intra,
        gram,
        hidden_states_chunked,
    )

    decay_to_chunk_end = torch.exp(
        A_cumsum[..., -1:, :] - A_cumsum
    ).permute(0, 2, 3, 1)

    B_decay = B_chunked[..., None, :] * decay_to_chunk_end[..., None, :, None]
    B_decay = B_chunked * decay_to_chunk_end[..., None]

    states = torch.einsum(
        "bct s,bcthd->bchds".replace(" ", ""),
        B_decay,
        hidden_states_chunked,
    )

    initial_states_expanded = initial_states_f[:, None, :, :, :]
    states_with_init = torch.cat(
        [initial_states_expanded, states],
        dim=1,
    )

    chunk_end_values = A_cumsum[..., -1]
    chunk_end_values = F.pad(chunk_end_values, (1, 0))

    chunk_prefix = torch.cumsum(chunk_end_values, dim=-1)
    decay_chunks = torch.exp(
        chunk_prefix[..., :, None] - chunk_prefix[..., None, :]
    )

    chunk_mask = torch.tril(
        torch.ones(
            num_chunks + 1,
            num_chunks + 1,
            device=hidden_states.device,
            dtype=torch.bool,
        )
    )
    decay_chunks = decay_chunks.masked_fill(~chunk_mask, 0.0)

    new_states = torch.einsum(
        "bhij,bjhds->bihds",
        decay_chunks,
        states_with_init,
    )

    states_out = new_states[:, :-1]
    final_state = new_states[:, -1]

    state_decay_out = torch.exp(A_cumsum).permute(0, 2, 3, 1)

    c_times_states = torch.einsum(
        "bct s,bchds->bcthd".replace(" ", ""),
        C_chunked,
        states_out,
    )
    y_off = c_times_states * state_decay_out[..., None]

    y = y_diag + y_off
    y = y.reshape(batch_size, padded_seq_len, num_heads, head_dim)
    y = y + D_residual

    if pad_size:
        y = y[:, :seq_len]

    output = y.reshape(
        batch_size,
        seq_len,
        num_heads * head_dim,
    ).to(torch.bfloat16)

    return output, final_state.to(torch.bfloat16)