/**
 * FlashAttention-2 kernel for HIP/ROCm
 *
 * This kernel implements scaled dot-product attention:
 *   output = softmax(Q @ K^T / sqrt(d)) @ V
 *
 * Optimized for AMD MI300X and other datacenter GPUs.
 * Compile with: hipcc -O3 --genco -o attention_hip.hsaco attention_hip.cpp
 */

#include <hip/hip_runtime.h>

#define BLOCK_SIZE 16
#define MAX_HEAD_DIM 128

// Shared memory tiles
__shared__ float s_Q[BLOCK_SIZE][MAX_HEAD_DIM];
__shared__ float s_K[BLOCK_SIZE][MAX_HEAD_DIM];
__shared__ float s_V[BLOCK_SIZE][MAX_HEAD_DIM];
__shared__ float s_S[BLOCK_SIZE][BLOCK_SIZE];

extern "C" __global__ void attention_forward(
    const float* __restrict__ Q,      // [batch, heads, seq, dim]
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ output,
    uint32_t batch_size,
    uint32_t num_heads,
    uint32_t seq_len,
    uint32_t head_dim,
    float scale
) {
    // Identify which batch/head/row this workgroup handles
    const uint32_t batch_head_idx = blockIdx.x;
    const uint32_t batch_idx = batch_head_idx / num_heads;
    const uint32_t head_idx = batch_head_idx % num_heads;

    const uint32_t row_idx = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    const uint32_t local_row = threadIdx.y;
    const uint32_t local_col = threadIdx.x;

    if (batch_idx >= batch_size || row_idx >= seq_len) return;

    // Base offset for this batch/head
    const uint32_t base_offset = (batch_idx * num_heads + head_idx) * seq_len * head_dim;

    // Load Q row into shared memory
    for (uint32_t d = local_col; d < head_dim; d += BLOCK_SIZE) {
        s_Q[local_row][d] = Q[base_offset + row_idx * head_dim + d];
    }
    __syncthreads();

    // Online softmax variables
    float row_max = -INFINITY;
    float row_sum = 0.0f;
    float output_acc[MAX_HEAD_DIM] = {0};

    // Process K/V in tiles
    for (uint32_t tile_start = 0; tile_start < seq_len; tile_start += BLOCK_SIZE) {
        const uint32_t tile_col = tile_start + local_col;

        // Load K tile into shared memory
        if (tile_col < seq_len) {
            for (uint32_t d = local_row; d < head_dim; d += BLOCK_SIZE) {
                s_K[local_col][d] = K[base_offset + tile_col * head_dim + d];
            }
        }
        __syncthreads();

        // Compute attention scores for this tile: S = Q @ K^T * scale
        float scores[BLOCK_SIZE];
        for (uint32_t j = 0; j < BLOCK_SIZE && (tile_start + j) < seq_len; ++j) {
            float dot = 0.0f;
            for (uint32_t d = 0; d < head_dim; ++d) {
                dot += s_Q[local_row][d] * s_K[j][d];
            }
            scores[j] = dot * scale;
        }

        // Online softmax: update max
        float old_max = row_max;
        for (uint32_t j = 0; j < BLOCK_SIZE && (tile_start + j) < seq_len; ++j) {
            row_max = fmaxf(row_max, scores[j]);
        }

        // Rescale previous sum
        float scale_factor = expf(old_max - row_max);
        row_sum *= scale_factor;
        for (uint32_t d = 0; d < head_dim; ++d) {
            output_acc[d] *= scale_factor;
        }

        // Load V tile into shared memory
        if (tile_col < seq_len) {
            for (uint32_t d = local_row; d < head_dim; d += BLOCK_SIZE) {
                s_V[local_col][d] = V[base_offset + tile_col * head_dim + d];
            }
        }
        __syncthreads();

        // Compute exp(scores - max) and accumulate
        for (uint32_t j = 0; j < BLOCK_SIZE && (tile_start + j) < seq_len; ++j) {
            float exp_score = expf(scores[j] - row_max);
            row_sum += exp_score;

            // Accumulate weighted V
            for (uint32_t d = 0; d < head_dim; ++d) {
                output_acc[d] += exp_score * s_V[j][d];
            }
        }
        __syncthreads();
    }

    // Write output: normalize by sum
    float inv_sum = 1.0f / row_sum;
    for (uint32_t d = local_col; d < head_dim; d += BLOCK_SIZE) {
        output[base_offset + row_idx * head_dim + d] = output_acc[d] * inv_sum;
    }
}

// Simpler kernel for small sequences (no tiling)
extern "C" __global__ void attention_forward_simple(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ output,
    uint32_t batch_size,
    uint32_t num_heads,
    uint32_t seq_len,
    uint32_t head_dim,
    float scale
) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total_rows = batch_size * num_heads * seq_len;

    if (idx >= total_rows) return;

    const uint32_t batch_head_row = idx;
    const uint32_t batch_head = batch_head_row / seq_len;
    const uint32_t row = batch_head_row % seq_len;

    const uint32_t base_offset = batch_head * seq_len * head_dim;
    const float* q_row = Q + base_offset + row * head_dim;

    // Compute attention scores
    float scores[1024]; // Max seq_len
    float max_score = -INFINITY;

    for (uint32_t j = 0; j < seq_len; ++j) {
        const float* k_row = K + base_offset + j * head_dim;
        float dot = 0.0f;
        for (uint32_t d = 0; d < head_dim; ++d) {
            dot += q_row[d] * k_row[d];
        }
        scores[j] = dot * scale;
        max_score = fmaxf(max_score, scores[j]);
    }

    // Softmax
    float sum = 0.0f;
    for (uint32_t j = 0; j < seq_len; ++j) {
        scores[j] = expf(scores[j] - max_score);
        sum += scores[j];
    }
    float inv_sum = 1.0f / sum;
    for (uint32_t j = 0; j < seq_len; ++j) {
        scores[j] *= inv_sum;
    }

    // Output = scores @ V
    float* out_row = output + base_offset + row * head_dim;
    for (uint32_t d = 0; d < head_dim; ++d) {
        float acc = 0.0f;
        for (uint32_t j = 0; j < seq_len; ++j) {
            acc += scores[j] * V[base_offset + j * head_dim + d];
        }
        out_row[d] = acc;
    }
}
