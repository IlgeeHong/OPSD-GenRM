# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fused Triton kernel for extracting top-k log-probabilities from logits.

Computes log_softmax(logits)[indices] without materializing the full
log_softmax tensor, making additional memory O(k) instead of O(vocab).

Both forward and backward use Triton kernels with float32 internal
computation and streaming access — no vocab-sized intermediate tensors.

Based on the flash-attn cross-entropy kernel pattern:
    flash_attn/ops/triton/cross_entropy.py
"""

import torch
import triton
import triton.language as tl


@triton.jit
def topk_logprobs_fwd_kernel(
    output_ptr,       # (n_rows, k) — output top-k log-probs (float32)
    lse_ptr,          # (n_rows,) — logsumexp per row (float32)
    logits_ptr,       # (n_rows, n_cols) — input logits (any dtype)
    indices_ptr,      # (n_rows, k) — which vocab indices to gather
    n_cols,           # vocab_size
    k,                # number of top-k indices
    logits_row_stride,
    indices_row_stride,
    output_row_stride,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute log_softmax(logits)[indices] per row without materializing log_softmax.

    For each row:
      1. Stream through logits in blocks to compute logsumexp (online softmax trick)
      2. Gather logits at k indices
      3. Output: gathered_logits - logsumexp = log_softmax at those indices

    Internal computation in float32. Additional memory per row: O(k).
    """
    row_idx = tl.program_id(0)
    logits_ptr = logits_ptr + row_idx * logits_row_stride.to(tl.int64)
    indices_ptr = indices_ptr + row_idx * indices_row_stride.to(tl.int64)
    output_ptr = output_ptr + row_idx * output_row_stride.to(tl.int64)

    # Step 1: Compute logsumexp via online softmax (streaming, O(1) memory)
    m_i = -float("inf")
    l_i = 0.0
    for col_offset in range(0, n_cols, BLOCK_SIZE):
        cols = col_offset + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(logits_ptr + cols, mask=cols < n_cols, other=-float("inf")).to(tl.float32)
        m_i_new = tl.maximum(m_i, tl.max(logits))
        l_i = tl.exp(m_i - m_i_new) * l_i + tl.sum(tl.exp(logits - m_i_new))
        m_i = m_i_new
    lse = tl.log(l_i) + m_i
    tl.store(lse_ptr + row_idx, lse)

    # Step 2: Gather logits at top-k indices and subtract logsumexp
    for k_offset in range(0, k, BLOCK_SIZE):
        k_cols = k_offset + tl.arange(0, BLOCK_SIZE)
        mask = k_cols < k
        idx = tl.load(indices_ptr + k_cols, mask=mask, other=0)
        gathered = tl.load(logits_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        result = gathered - lse
        tl.store(output_ptr + k_cols, result, mask=mask)


@triton.jit
def topk_logprobs_bwd_kernel(
    dlogits_ptr,      # (n_rows, n_cols) — gradient w.r.t. logits (same dtype as logits)
    doutput_ptr,      # (n_rows, k) — gradient w.r.t. output log-probs (float32)
    logits_ptr,       # (n_rows, n_cols) — original logits (any dtype)
    lse_ptr,          # (n_rows,) — precomputed logsumexp (float32)
    indices_ptr,      # (n_rows, k) — top-k indices
    dout_sum_ptr,     # (n_rows,) — precomputed sum of doutput per row (float32)
    n_cols,
    logits_row_stride,
    dlogits_row_stride,
    indices_row_stride,
    doutput_row_stride,
    K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Backward pass for topk_logprobs — Triton streaming kernel.

    For each position j in the vocab:
        dlogits[j] = -dout_sum * softmax(j) + sum_i(dout_i * (j == idx_i))

    Streams through the vocab in blocks (like flash_attn cross_entropy backward):
    - Loads logits in bf16, computes softmax in float32
    - Writes dlogits back in original dtype (bf16)
    - Never materializes full float32 vocab tensor

    The scatter term (dout_i for j == idx_i) is handled by checking each
    of the k indices against the current block. Since k is small (typically 20),
    this is efficient.
    """
    row_idx = tl.program_id(0)
    col_block_idx = tl.program_id(1)

    logits_row_ptr = logits_ptr + row_idx * logits_row_stride.to(tl.int64)
    dlogits_row_ptr = dlogits_ptr + row_idx * dlogits_row_stride.to(tl.int64)
    indices_row_ptr = indices_ptr + row_idx * indices_row_stride.to(tl.int64)
    doutput_row_ptr = doutput_ptr + row_idx * doutput_row_stride.to(tl.int64)

    dout_sum = tl.load(dout_sum_ptr + row_idx)  # float32
    lse = tl.load(lse_ptr + row_idx)  # float32

    # Process this block of the vocab dimension
    col_offsets = col_block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols
    logits = tl.load(logits_row_ptr + col_offsets, mask=mask, other=-float("inf")).to(tl.float32)
    probs = tl.exp(logits - lse)  # softmax in float32

    # Gradient from logsumexp: -dout_sum * softmax(j)
    dlogits = -dout_sum * probs

    # Gradient from gather: +dout_i where j == idx_i
    # Iterate over k indices (K is constexpr, compile-time unrolled)
    for ki in range(K):
        idx_i = tl.load(indices_row_ptr + ki)
        dout_i = tl.load(doutput_row_ptr + ki).to(tl.float32)
        dlogits += tl.where(col_offsets == idx_i, dout_i, 0.0)

    # Write back in original dtype (bf16/fp16) — streaming, no full float32 tensor
    tl.store(dlogits_row_ptr + col_offsets, dlogits, mask=mask)


class TopkLogprobs(torch.autograd.Function):
    """Fused top-k log-probability extraction.

    Computes log_softmax(logits)[indices] without materializing the full
    log_softmax tensor. Both forward and backward use Triton kernels with
    float32 internal computation.

    Memory: O(k) additional instead of O(vocab) in both directions.
    """

    @staticmethod
    def forward(ctx, logits, indices):
        """
        Args:
            logits: (n_rows, vocab_size) — model output logits
            indices: (n_rows, k) — which vocab indices to extract

        Returns:
            topk_logprobs: (n_rows, k) — log_softmax(logits) at the given indices (float32)
        """
        n_rows, n_cols = logits.shape
        k = indices.shape[1]

        if logits.stride(-1) != 1:
            logits = logits.contiguous()
        if indices.stride(-1) != 1:
            indices = indices.contiguous()

        topk_logprobs = torch.empty(n_rows, k, dtype=torch.float32, device=logits.device)
        lse = torch.empty(n_rows, dtype=torch.float32, device=logits.device)

        MAX_BLOCK_SIZE = 16 * 1024
        BLOCK_SIZE = min(triton.next_power_of_2(max(n_cols, k)), MAX_BLOCK_SIZE)
        num_warps = (
            4
            if BLOCK_SIZE < 2048
            else (8 if BLOCK_SIZE < 8192 else (16 if BLOCK_SIZE < 128 * 1024 else 32))
        )

        with torch.cuda.device(logits.device.index):
            topk_logprobs_fwd_kernel[(n_rows,)](
                topk_logprobs,
                lse,
                logits,
                indices,
                n_cols,
                k,
                logits.stride(0),
                indices.stride(0),
                topk_logprobs.stride(0),
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
            )

        ctx.save_for_backward(logits, lse, indices)
        ctx.n_cols = n_cols
        ctx.k = k
        return topk_logprobs

    @staticmethod
    def backward(ctx, grad_output):
        logits, lse, indices = ctx.saved_tensors
        n_rows, n_cols = logits.shape
        k = ctx.k

        # Precompute dout_sum on GPU (tiny: n_rows scalars)
        dout_sum = grad_output.float().sum(dim=-1)  # (n_rows,) float32

        # Allocate dlogits in original dtype — no float32 vocab tensor
        dlogits = torch.empty_like(logits)

        BLOCK_SIZE = min(triton.next_power_of_2(n_cols), 4 * 1024)
        num_warps = 4 if BLOCK_SIZE < 2048 else (8 if BLOCK_SIZE < 8192 else 16)
        grid = lambda META: (n_rows, triton.cdiv(n_cols, META["BLOCK_SIZE"]))  # noqa

        with torch.cuda.device(logits.device.index):
            topk_logprobs_bwd_kernel[grid](
                dlogits,
                grad_output,
                logits,
                lse,
                indices,
                dout_sum,
                n_cols,
                logits.stride(0),
                dlogits.stride(0),
                indices.stride(0),
                grad_output.stride(0),
                K=k,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
            )

        return dlogits, None


def topk_logprobs_from_logits(logits: torch.Tensor, indices: torch.LongTensor) -> torch.Tensor:
    """Compute log_softmax(logits) at specified indices without materializing full log_softmax.

    Fused Triton kernel: memory O(k) instead of O(vocab) for both forward and backward.
    Internal computation in float32 for numerical stability.

    Args:
        logits: (n_rows, vocab_size) or (batch, seq, vocab_size)
        indices: (n_rows, k) or (batch, seq, k) — indices to gather

    Returns:
        topk_logprobs: same shape as indices, in float32
    """
    orig_shape = indices.shape
    k = orig_shape[-1]

    # Flatten to 2D for the kernel
    logits_2d = logits.reshape(-1, logits.shape[-1])
    indices_2d = indices.reshape(-1, k)

    result = TopkLogprobs.apply(logits_2d, indices_2d)

    return result.reshape(orig_shape)
