#include <stdlib.h>
#include <stdio.h>

#include "env.h"
#include "mem_info.h"

#include "conv1.h"
#include "conv2.h"
#include "fc1.h"
#include "fc2.h"


static DATA_T mem[MEMORY_SIZE];

static int max(int lhs, int rhs) {
        return (lhs >= rhs)?lhs:rhs;
    }

static int clamp(int v, int lo, int hi) {
    if(v < lo) {
        return lo;
    }
    else if(v > hi) {
        return hi;
    }
    else {
        return v;
    }
}

/*
 * ============================================================
 * 统一 input buffer 辅助函数
 * ============================================================
 *
 * 硬件侧约定：
 *   - input buffer 最大 400 byte = 100 个 32-bit word = 25 个 16-byte block。
 *   - buf4 的 rd 字段不作为真正写回寄存器，而是作为 mode：
 *
 *         active_blocks = rd + 1
 *
 *   - Conv1: rd = x0  -> active_blocks = 1
 *   - Conv2: rd = x24 -> active_blocks = 25
 *   - FC1:   rd = x23 -> active_blocks = 24
 *   - FC2:   rd = x8  -> active_blocks = 9，剩余 6 个 input 走 scalar
 *
 * 注意：
 *   t3 = x28，t4 = x29。硬件里 rs[3]/rs[4] 固定来自 x28/x29，
 *   所以这里继续用 t3/t4 是匹配的。
 */

/* Conv1 专用：把一个 4x4x1 patch buffer 进去。
 * 这里四个指针分别指向 patch 的四行，每行连续 4 个 uint8。
 * buf4 x0 表示 active_blocks = 1，所以后续每个 mac16buf 都会读 block0。
 */
static inline void buffer4_setmode_conv1(void)
{
    asm volatile(

        "buf4 x0, x0, x0, x0, x0 \n\t"
        : 
        : 
        : "cc", "memory"
    );
}

/* Conv2 专用：把一个 pixel 的 16 个 channels buffer 进去。
 * Conv2 一个完整 5x5x16 patch 需要连续执行 25 次这个函数。
 * buf4 x24 表示 active_blocks = 25。
 */
static inline void buffer4_setmode_conv2(void)
{
    asm volatile(

        "buf4 x24, x0, x0, x0, x0 \n\t"
        : 
        : 
        : "cc", "memory"
    );
}

/* FC1 专用：FC1 input = 384 byte = 24 个 16-byte block。
 * buf4 x23 表示 active_blocks = 24。
 */
static inline void buffer4_setmode_fc1(void)
{
    asm volatile(

        "buf4 x23, x0, x0, x0, x0 \n\t"
        : 
        : 
        : "cc", "memory"
    );
}

/* FC2 专用：FC2 input = 150 byte。
 * 前 144 byte = 9 个 16-byte block 用 buffer + mac16buf，
 * 最后 6 byte 用普通 scalar 处理。
 * buf4 x8 表示 active_blocks = 9。
 */
static inline void buffer4_setmode_fc2(void)
{
    asm volatile(

        "buf4 x8, x0, x0, x0, x0 \n\t"
        : 
        : 
        : "cc", "memory"
    );
}

/* 只执行 MAC，不更新 input buffer。
 * 这个函数假设：对应的 input block 已经在硬件 input buffer 中。
 * 硬件每执行一次 mac16buf，会根据 active_blocks 自动移动 read counter。
 */
static inline __attribute__((always_inline))
SUM_T mac16buf_para_conv1_aligned(
    const UDATA_T* __restrict row0,
    const UDATA_T* __restrict row1,
    const UDATA_T* __restrict row2,
    const UDATA_T* __restrict row3,
    const WDATA_T* __restrict weights,
    SUM_T initial_sum)
{
    SUM_T sum = initial_sum;
    uint32_t w0, w1, w2, w3;

    asm volatile(
        "lw %[w0],  0(%[p_wt])\n\t"
        "lw %[w1],  4(%[p_wt])\n\t"
        "lw %[w2],  8(%[p_wt])\n\t"
        "lw %[w3], 12(%[p_wt])\n\t"

        "lw t3, 0(%[row0])\n\t"
        "lw t4, 0(%[row1])\n\t"
        "lw t5, 0(%[row2])\n\t"
        "lw t6, 0(%[row3])\n\t"

        /*
         * active_blocks = 1:
         * 这条指令既是first也是final。
         * 同时计算output0并将input写入buffer。
         */
        "mac16buf_para %[sum], %[w0], %[w1], %[w2], %[w3]\n\t"

        : [sum] "+r"(sum),
          [w0] "=&r"(w0),
          [w1] "=&r"(w1),
          [w2] "=&r"(w2),
          [w3] "=&r"(w3)
        : [row0] "r"(row0),
          [row1] "r"(row1),
          [row2] "r"(row2),
          [row3] "r"(row3),
          [p_wt] "r"(weights)
        : "t3", "t4", "t5", "t6", "memory"
    );

    return sum;
}

static inline __attribute__((always_inline))
SUM_T mac16buf_para_conv1_aligned_wbuf(
    const UDATA_T* __restrict row0,
    const UDATA_T* __restrict row1,
    const UDATA_T* __restrict row2,
    const UDATA_T* __restrict row3,
    SUM_T initial_sum)
{
    SUM_T sum = initial_sum;

    asm volatile(
        /*
         * Input only.
         * Weight is read from the local weight buffer.
         */
        "lw t3, 0(%[row0])\n\t"
        "lw t4, 0(%[row1])\n\t"
        "lw t5, 0(%[row2])\n\t"
        "lw t6, 0(%[row3])\n\t"

        "mac16buf_para %[sum], x0, x0, x0, x0\n\t"

        : [sum] "+r"(sum)
        : [row0] "r"(row0),
          [row1] "r"(row1),
          [row2] "r"(row2),
          [row3] "r"(row3)
        : "t3", "t4", "t5", "t6",
          "cc", "memory"
    );

    return sum;
}

static inline __attribute__((always_inline))
SUM_T mac16buf_para_conv1_unaligned2(
    const UDATA_T* __restrict row0,
    const UDATA_T* __restrict row1,
    const UDATA_T* __restrict row2,
    const UDATA_T* __restrict row3,
    const WDATA_T* __restrict weights,
    SUM_T initial_sum)
{
    SUM_T sum = initial_sum;
    uint32_t w0, w1, w2, w3;

    asm volatile(
        "lw %[w0],  0(%[p_wt])\n\t"
        "lw %[w1],  4(%[p_wt])\n\t"
        "lw %[w2],  8(%[p_wt])\n\t"
        "lw %[w3], 12(%[p_wt])\n\t"

        /* row0: 拼出连续4 bytes */
        "lhu t3, 0(%[row0])\n\t"
        "lhu t0, 2(%[row0])\n\t"
        "slli t0, t0, 16\n\t"
        "or   t3, t3, t0\n\t"

        /* row1 */
        "lhu t4, 0(%[row1])\n\t"
        "lhu t0, 2(%[row1])\n\t"
        "slli t0, t0, 16\n\t"
        "or   t4, t4, t0\n\t"

        /* row2 */
        "lhu t5, 0(%[row2])\n\t"
        "lhu t0, 2(%[row2])\n\t"
        "slli t0, t0, 16\n\t"
        "or   t5, t5, t0\n\t"

        /* row3 */
        "lhu t6, 0(%[row3])\n\t"
        "lhu t0, 2(%[row3])\n\t"
        "slli t0, t0, 16\n\t"
        "or   t6, t6, t0\n\t"

        "mac16buf_para %[sum], %[w0], %[w1], %[w2], %[w3]\n\t"

        : [sum] "+r"(sum),
          [w0] "=&r"(w0),
          [w1] "=&r"(w1),
          [w2] "=&r"(w2),
          [w3] "=&r"(w3)
        : [row0] "r"(row0),
          [row1] "r"(row1),
          [row2] "r"(row2),
          [row3] "r"(row3),
          [p_wt] "r"(weights)
        : "t0", "t3", "t4", "t5", "t6", "memory"
    );

    return sum;
}

static inline __attribute__((always_inline))
SUM_T mac16buf_para_conv1_unaligned2_wbuf(
    const UDATA_T* __restrict row0,
    const UDATA_T* __restrict row1,
    const UDATA_T* __restrict row2,
    const UDATA_T* __restrict row3,
    SUM_T initial_sum)
{
    SUM_T sum = initial_sum;

    asm volatile(
        /* row0 */
        "lhu t3, 0(%[row0])\n\t"
        "lhu t0, 2(%[row0])\n\t"
        "slli t0, t0, 16\n\t"
        "or   t3, t3, t0\n\t"

        /* row1 */
        "lhu t4, 0(%[row1])\n\t"
        "lhu t0, 2(%[row1])\n\t"
        "slli t0, t0, 16\n\t"
        "or   t4, t4, t0\n\t"

        /* row2 */
        "lhu t5, 0(%[row2])\n\t"
        "lhu t0, 2(%[row2])\n\t"
        "slli t0, t0, 16\n\t"
        "or   t5, t5, t0\n\t"

        /* row3 */
        "lhu t6, 0(%[row3])\n\t"
        "lhu t0, 2(%[row3])\n\t"
        "slli t0, t0, 16\n\t"
        "or   t6, t6, t0\n\t"

        /*
         * No weight loads.
         */
        "mac16buf_para %[sum], x0, x0, x0, x0\n\t"

        : [sum] "+r"(sum)
        : [row0] "r"(row0),
          [row1] "r"(row1),
          [row2] "r"(row2),
          [row3] "r"(row3)
        : "t0",
          "t3", "t4", "t5", "t6",
          "cc", "memory"
    );

    return sum;
}


static inline void mac16buf_conv1(const WDATA_T* __restrict weights,
                                 SUM_T* __restrict weightedSum)
{
    int32_t sum = *weightedSum;
    const WDATA_T *p_wt = weights;
    uint32_t w0, w1, w2, w3;
    asm volatile(
        "lw %[w0], 0(%[p_wt]) \n\t"
        "lw %[w1], 4(%[p_wt]) \n\t"
        "lw %[w2], 8(%[p_wt]) \n\t"
        "lw %[w3], 12(%[p_wt]) \n\t"
        "mac16buf %[sum], %[w0], %[w1], %[w2], %[w3] \n\t"
        : [w0] "=&r" (w0),
          [w1] "=&r" (w1),
          [w2] "=&r" (w2),
          [w3] "=&r" (w3),
          [sum] "+r" (sum)
        : [p_wt] "r" (p_wt)
        : "cc", "memory"
    );

    *weightedSum = sum;
}

static inline __attribute__((always_inline))
void mac16buf_conv1_wbuf(
    SUM_T* __restrict weightedSum)
{
    int32_t sum = *weightedSum;

    /*
     * active_blocks = 1
     *
     * This instruction is both first and final:
     *   rd input  = bias / initial accumulator
     *   rd output = final MAC result
     */
    asm volatile(
        "mac16buf %[sum], x0, x0, x0, x0 \n\t"

        : [sum] "+r"(sum)
        :
        : "cc", "memory"
    );

    *weightedSum = sum;
}

static inline void mac16buf_first(const WDATA_T* __restrict weights,
                                 SUM_T* __restrict weightedSum)
{
    int32_t sum = *weightedSum;
    const WDATA_T *p_wt = weights;
    uint32_t w0, w1, w2, w3;
 
    asm volatile(
        "lw %[w0], 0(%[p_wt]) \n\t"
        "lw %[w1], 4(%[p_wt]) \n\t"
        "lw %[w2], 8(%[p_wt]) \n\t"
        "lw %[w3], 12(%[p_wt]) \n\t"
        "mac16buf %[sum], %[w0], %[w1], %[w2], %[w3] \n\t"
        : [w0] "=&r" (w0),
          [w1] "=&r" (w1),
          [w2] "=&r" (w2),
          [w3] "=&r" (w3)
        : [sum] "r" (sum),
          [p_wt] "r" (p_wt)
        : "cc", "memory"
    );
}

static inline void mac16buf_wbuf_first(SUM_T* __restrict weightedSum)
{
    int32_t sum = *weightedSum;
 
    asm volatile(
        "mac16buf %[sum], x0, x0, x0, x0 \n\t"
        : 
        : [sum] "r" (sum)
        : "cc", "memory"
    );
}

static inline void mac16buf_middle(const WDATA_T* __restrict weights)
{
    const WDATA_T *p_wt = weights;
    uint32_t w0, w1, w2, w3;

    asm volatile(
        "lw %[w0], 0(%[p_wt]) \n\t"
        "lw %[w1], 4(%[p_wt]) \n\t"
        "lw %[w2], 8(%[p_wt]) \n\t"
        "lw %[w3], 12(%[p_wt]) \n\t"
        "mac16buf x0, %[w0], %[w1], %[w2], %[w3] \n\t"
        : [w0] "=&r" (w0),
          [w1] "=&r" (w1),
          [w2] "=&r" (w2),
          [w3] "=&r" (w3)
        : [p_wt] "r" (p_wt)
        : "cc", "memory"
    );
}

static inline void mac16buf_wbuf_middle(void)
{
    asm volatile(
        "mac16buf x0, x0, x0, x0, x0 \n\t"
        :
        :
        : "cc", "memory"
    );
}


static inline void mac16buf_middle4(const WDATA_T* __restrict weights)
{
    const WDATA_T *p_wt = weights;
    uint32_t w0, w1, w2, w3;

    asm volatile(
        // block 0: weights[0..15]
        "lw %[w0], 0(%[p_wt]) \n\t"
        "lw %[w1], 4(%[p_wt]) \n\t"
        "lw %[w2], 8(%[p_wt]) \n\t"
        "lw %[w3], 12(%[p_wt]) \n\t"
        "mac16buf x0, %[w0], %[w1], %[w2], %[w3] \n\t"

        // block 1: weights[16..31]
        "lw %[w0], 16(%[p_wt]) \n\t"
        "lw %[w1], 20(%[p_wt]) \n\t"
        "lw %[w2], 24(%[p_wt]) \n\t"
        "lw %[w3], 28(%[p_wt]) \n\t"
        "mac16buf x0, %[w0], %[w1], %[w2], %[w3] \n\t"

        // block 2: weights[32..47]
        "lw %[w0], 32(%[p_wt]) \n\t"
        "lw %[w1], 36(%[p_wt]) \n\t"
        "lw %[w2], 40(%[p_wt]) \n\t"
        "lw %[w3], 44(%[p_wt]) \n\t"
        "mac16buf x0, %[w0], %[w1], %[w2], %[w3] \n\t"

        // block 3: weights[48..63]
        "lw %[w0], 48(%[p_wt]) \n\t"
        "lw %[w1], 52(%[p_wt]) \n\t"
        "lw %[w2], 56(%[p_wt]) \n\t"
        "lw %[w3], 60(%[p_wt]) \n\t"
        "mac16buf x0, %[w0], %[w1], %[w2], %[w3] \n\t"
        : [w0] "=&r" (w0),
          [w1] "=&r" (w1),
          [w2] "=&r" (w2),
          [w3] "=&r" (w3)
        : [p_wt] "r" (p_wt)
        : "cc", "memory"
    );
}

static inline void mac16buf_wbuf_middle4(void)
{
    asm volatile(
        "mac16buf x0, x0, x0, x0, x0 \n\t"
        "mac16buf x0, x0, x0, x0, x0 \n\t"
        "mac16buf x0, x0, x0, x0, x0 \n\t"
        "mac16buf x0, x0, x0, x0, x0 \n\t"
        :
        :
        : "cc", "memory"
    );
}


static inline void mac16buf_final(const WDATA_T* __restrict weights,
                                 SUM_T* __restrict weightedSum)
{
    int32_t sum;
    const WDATA_T *p_wt = weights;
    uint32_t w0, w1, w2, w3;

    asm volatile(
        "lw %[w0], 0(%[p_wt]) \n\t"
        "lw %[w1], 4(%[p_wt]) \n\t"
        "lw %[w2], 8(%[p_wt]) \n\t"
        "lw %[w3], 12(%[p_wt]) \n\t"
        "mac16buf %[sum], %[w0], %[w1], %[w2], %[w3] \n\t"
        : [sum] "=r" (sum),
          [w0] "=&r" (w0),
          [w1] "=&r" (w1),
          [w2] "=&r" (w2),
          [w3] "=&r" (w3)
        : [p_wt] "r" (p_wt)
        : "cc", "memory"
    );

    *weightedSum = sum;
}

static inline void mac16buf_wbuf_final(SUM_T* __restrict weightedSum)
{
    int32_t sum;
    asm volatile(
        "mac16buf %[sum], x0, x0, x0, x0 \n\t"
        : [sum] "=r" (sum)
        :
        : "cc", "memory"
    );

    *weightedSum = sum;
}

static inline __attribute__((always_inline))
void mac16buf_para_middle4(
    const UDATA_T* __restrict inputs,
    const WDATA_T* __restrict weights)
{
    const UDATA_T* p_in = inputs;
    const WDATA_T* p_wt = weights;

    uint32_t w0, w1, w2, w3;

    asm volatile(

        /*
         * ============================================================
         * block 0
         * input  [0..15]
         * weight [0..15]
         * ============================================================
         */

        "lw %[w0],  0(%[p_wt]) \n\t"
        "lw %[w1],  4(%[p_wt]) \n\t"
        "lw %[w2],  8(%[p_wt]) \n\t"
        "lw %[w3], 12(%[p_wt]) \n\t"

        "lw t3,  0(%[p_in]) \n\t"
        "lw t4,  4(%[p_in]) \n\t"
        "lw t5,  8(%[p_in]) \n\t"
        "lw t6, 12(%[p_in]) \n\t"

        "mac16buf_para x0, %[w0], %[w1], %[w2], %[w3] \n\t"


        /*
         * ============================================================
         * block 1
         * input  [16..31]
         * weight [16..31]
         * ============================================================
         */

        "lw %[w0], 16(%[p_wt]) \n\t"
        "lw %[w1], 20(%[p_wt]) \n\t"
        "lw %[w2], 24(%[p_wt]) \n\t"
        "lw %[w3], 28(%[p_wt]) \n\t"

        "lw t3, 16(%[p_in]) \n\t"
        "lw t4, 20(%[p_in]) \n\t"
        "lw t5, 24(%[p_in]) \n\t"
        "lw t6, 28(%[p_in]) \n\t"

        "mac16buf_para x0, %[w0], %[w1], %[w2], %[w3] \n\t"


        /*
         * ============================================================
         * block 2
         * input  [32..47]
         * weight [32..47]
         * ============================================================
         */

        "lw %[w0], 32(%[p_wt]) \n\t"
        "lw %[w1], 36(%[p_wt]) \n\t"
        "lw %[w2], 40(%[p_wt]) \n\t"
        "lw %[w3], 44(%[p_wt]) \n\t"

        "lw t3, 32(%[p_in]) \n\t"
        "lw t4, 36(%[p_in]) \n\t"
        "lw t5, 40(%[p_in]) \n\t"
        "lw t6, 44(%[p_in]) \n\t"

        "mac16buf_para x0, %[w0], %[w1], %[w2], %[w3] \n\t"


        /*
         * ============================================================
         * block 3
         * input  [48..63]
         * weight [48..63]
         * ============================================================
         */

        "lw %[w0], 48(%[p_wt]) \n\t"
        "lw %[w1], 52(%[p_wt]) \n\t"
        "lw %[w2], 56(%[p_wt]) \n\t"
        "lw %[w3], 60(%[p_wt]) \n\t"

        "lw t3, 48(%[p_in]) \n\t"
        "lw t4, 52(%[p_in]) \n\t"
        "lw t5, 56(%[p_in]) \n\t"
        "lw t6, 60(%[p_in]) \n\t"

        "mac16buf_para x0, %[w0], %[w1], %[w2], %[w3] \n\t"

        : [w0] "=&r" (w0),
          [w1] "=&r" (w1),
          [w2] "=&r" (w2),
          [w3] "=&r" (w3)

        : [p_in] "r" (p_in),
          [p_wt] "r" (p_wt)

        : "t3", "t4", "t5", "t6",
          "cc", "memory"
    );
}

static inline __attribute__((always_inline))
void mac16buf_first_offset2(
    const WDATA_T* __restrict weights,
    SUM_T* __restrict weightedSum)
{
    int32_t sum = *weightedSum;
    const WDATA_T* p_wt = weights;

    uint32_t w0, w1, w2, w3;
    uint32_t tmp;

    asm volatile(
        /* w0 = weights[0..3] */
        "lhu %[w0], 0(%[p_wt]) \n\t"
        "lhu %[tmp], 2(%[p_wt]) \n\t"
        "slli %[tmp], %[tmp], 16 \n\t"
        "or %[w0], %[w0], %[tmp] \n\t"

        /* w1 = weights[4..7] */
        "lhu %[w1], 4(%[p_wt]) \n\t"
        "lhu %[tmp], 6(%[p_wt]) \n\t"
        "slli %[tmp], %[tmp], 16 \n\t"
        "or %[w1], %[w1], %[tmp] \n\t"

        /* w2 = weights[8..11] */
        "lhu %[w2], 8(%[p_wt]) \n\t"
        "lhu %[tmp], 10(%[p_wt]) \n\t"
        "slli %[tmp], %[tmp], 16 \n\t"
        "or %[w2], %[w2], %[tmp] \n\t"

        /* w3 = weights[12..15] */
        "lhu %[w3], 12(%[p_wt]) \n\t"
        "lhu %[tmp], 14(%[p_wt]) \n\t"
        "slli %[tmp], %[tmp], 16 \n\t"
        "or %[w3], %[w3], %[tmp] \n\t"

        /*
         * first:
         * CPU rd中的bias进入local accumulator。
         */
        "mac16buf %[sum], %[w0], %[w1], %[w2], %[w3] \n\t"

        : [w0] "=&r"(w0),
          [w1] "=&r"(w1),
          [w2] "=&r"(w2),
          [w3] "=&r"(w3),
          [tmp] "=&r"(tmp)

        : [sum] "r"(sum),
          [p_wt] "r"(p_wt)

        : "cc", "memory"
    );
}


static inline __attribute__((always_inline))
void mac16buf_middle_offset2(
    const WDATA_T* __restrict weights)
{
    const WDATA_T* p_wt = weights;

    uint32_t w0, w1, w2, w3;
    uint32_t tmp;

    asm volatile(
        "lhu %[w0], 0(%[p_wt]) \n\t"
        "lhu %[tmp], 2(%[p_wt]) \n\t"
        "slli %[tmp], %[tmp], 16 \n\t"
        "or %[w0], %[w0], %[tmp] \n\t"

        "lhu %[w1], 4(%[p_wt]) \n\t"
        "lhu %[tmp], 6(%[p_wt]) \n\t"
        "slli %[tmp], %[tmp], 16 \n\t"
        "or %[w1], %[w1], %[tmp] \n\t"

        "lhu %[w2], 8(%[p_wt]) \n\t"
        "lhu %[tmp], 10(%[p_wt]) \n\t"
        "slli %[tmp], %[tmp], 16 \n\t"
        "or %[w2], %[w2], %[tmp] \n\t"

        "lhu %[w3], 12(%[p_wt]) \n\t"
        "lhu %[tmp], 14(%[p_wt]) \n\t"
        "slli %[tmp], %[tmp], 16 \n\t"
        "or %[w3], %[w3], %[tmp] \n\t"

        /*
         * middle:
         * 不读CPU rd，不写CPU rd。
         */
        "mac16buf x0, %[w0], %[w1], %[w2], %[w3] \n\t"

        : [w0] "=&r"(w0),
          [w1] "=&r"(w1),
          [w2] "=&r"(w2),
          [w3] "=&r"(w3),
          [tmp] "=&r"(tmp)

        : [p_wt] "r"(p_wt)

        : "cc", "memory"
    );
}


static inline __attribute__((always_inline))
void mac16buf_final_offset2(
    const WDATA_T* __restrict weights,
    SUM_T* __restrict weightedSum)
{
    int32_t sum;
    const WDATA_T* p_wt = weights;

    uint32_t w0, w1, w2, w3;
    uint32_t tmp;

    asm volatile(
        "lhu %[w0], 0(%[p_wt]) \n\t"
        "lhu %[tmp], 2(%[p_wt]) \n\t"
        "slli %[tmp], %[tmp], 16 \n\t"
        "or %[w0], %[w0], %[tmp] \n\t"

        "lhu %[w1], 4(%[p_wt]) \n\t"
        "lhu %[tmp], 6(%[p_wt]) \n\t"
        "slli %[tmp], %[tmp], 16 \n\t"
        "or %[w1], %[w1], %[tmp] \n\t"

        "lhu %[w2], 8(%[p_wt]) \n\t"
        "lhu %[tmp], 10(%[p_wt]) \n\t"
        "slli %[tmp], %[tmp], 16 \n\t"
        "or %[w2], %[w2], %[tmp] \n\t"

        "lhu %[w3], 12(%[p_wt]) \n\t"
        "lhu %[tmp], 14(%[p_wt]) \n\t"
        "slli %[tmp], %[tmp], 16 \n\t"
        "or %[w3], %[w3], %[tmp] \n\t"

        /*
         * final:
         * local accumulator写回CPU rd。
         */
        "mac16buf %[sum], %[w0], %[w1], %[w2], %[w3] \n\t"

        : [sum] "=r"(sum),
          [w0] "=&r"(w0),
          [w1] "=&r"(w1),
          [w2] "=&r"(w2),
          [w3] "=&r"(w3),
          [tmp] "=&r"(tmp)

        : [p_wt] "r"(p_wt)

        : "cc", "memory"
    );

    *weightedSum = sum;
}


static inline void mac16buf_para_first(const UDATA_T* __restrict inputs,
                                      const WDATA_T* __restrict weights,
                                      SUM_T* __restrict weightedSum)
{
    int32_t sum = *weightedSum;
    const UDATA_T *p_in = inputs;
    const WDATA_T *p_wt = weights;
    uint32_t w0, w1, w2, w3;

    asm volatile(
        // load weight word 0..3
        "lw %[w0], 0(%[p_wt]) \n\t"
        "lw %[w1], 4(%[p_wt]) \n\t"
        "lw %[w2], 8(%[p_wt]) \n\t"
        "lw %[w3], 12(%[p_wt]) \n\t"

        // load input word 0..3 into fixed registers x28..x31
        // t3=x28, t4=x29, t5=x30, t6=x31
        "lw t3, 0(%[p_in]) \n\t"
        "lw t4, 4(%[p_in]) \n\t"
        "lw t5, 8(%[p_in]) \n\t"
        "lw t6, 12(%[p_in]) \n\t"

        // first block: rd carries initial accumulator value
        "mac16buf_para %[sum], %[w0], %[w1], %[w2], %[w3] \n\t"
        : [w0] "=&r" (w0),
          [w1] "=&r" (w1),
          [w2] "=&r" (w2),
          [w3] "=&r" (w3)
        : [sum] "r" (sum),
          [p_in] "r" (p_in),
          [p_wt] "r" (p_wt)
        : "t3", "t4", "t5", "t6", "cc", "memory"
    );
}

static inline void mac16buf_para_middle(const UDATA_T* __restrict inputs,
                                       const WDATA_T* __restrict weights)
{
    const UDATA_T *p_in = inputs;
    const WDATA_T *p_wt = weights;
    uint32_t w0, w1, w2, w3;

    asm volatile(
        "lw %[w0], 0(%[p_wt]) \n\t"
        "lw %[w1], 4(%[p_wt]) \n\t"
        "lw %[w2], 8(%[p_wt]) \n\t"
        "lw %[w3], 12(%[p_wt]) \n\t"

        "lw t3, 0(%[p_in]) \n\t"
        "lw t4, 4(%[p_in]) \n\t"
        "lw t5, 8(%[p_in]) \n\t"
        "lw t6, 12(%[p_in]) \n\t"

        // middle block: x0 as rd, result only stays in local acc
        "mac16buf_para x0, %[w0], %[w1], %[w2], %[w3] \n\t"
        : [w0] "=&r" (w0),
          [w1] "=&r" (w1),
          [w2] "=&r" (w2),
          [w3] "=&r" (w3)
        : [p_in] "r" (p_in),
          [p_wt] "r" (p_wt)
        : "t3", "t4", "t5", "t6", "cc", "memory"
    );
}

static inline void mac16buf_para_final(const UDATA_T* __restrict inputs,
                                      const WDATA_T* __restrict weights,
                                      SUM_T* __restrict weightedSum)
{
    int32_t sum;
    const UDATA_T *p_in = inputs;
    const WDATA_T *p_wt = weights;
    uint32_t w0, w1, w2, w3;

    asm volatile(
        "lw %[w0], 0(%[p_wt]) \n\t"
        "lw %[w1], 4(%[p_wt]) \n\t"
        "lw %[w2], 8(%[p_wt]) \n\t"
        "lw %[w3], 12(%[p_wt]) \n\t"

        "lw t3, 0(%[p_in]) \n\t"
        "lw t4, 4(%[p_in]) \n\t"
        "lw t5, 8(%[p_in]) \n\t"
        "lw t6, 12(%[p_in]) \n\t"

        // final block: hardware writes result back to rd
        "mac16buf_para %[sum], %[w0], %[w1], %[w2], %[w3] \n\t"
        : [sum] "=r" (sum),
          [w0] "=&r" (w0),
          [w1] "=&r" (w1),
          [w2] "=&r" (w2),
          [w3] "=&r" (w3)
        : [p_in] "r" (p_in),
          [p_wt] "r" (p_wt)
        : "t3", "t4", "t5", "t6", "cc", "memory"
    );

    *weightedSum = sum;
}

static inline __attribute__((always_inline))
void mac16buf_para_wbuf_first(
    const UDATA_T* __restrict inputs,
    SUM_T* __restrict weightedSum)
{
    int32_t sum = *weightedSum;
    const UDATA_T* p_in = inputs;

    asm volatile(
        /*
         * Only load input.
         */
        "lw t3,  0(%[p_in]) \n\t"
        "lw t4,  4(%[p_in]) \n\t"
        "lw t5,  8(%[p_in]) \n\t"
        "lw t6, 12(%[p_in]) \n\t"

        /*
         * Weight operands are ignored.
         */
        "mac16buf_para %[sum], x0, x0, x0, x0 \n\t"

        :
        : [sum] "r" (sum),
          [p_in] "r" (p_in)
        : "t3", "t4", "t5", "t6",
          "cc", "memory"
    );
}

static inline __attribute__((always_inline))
void mac16buf_para_wbuf_middle(
    const UDATA_T* __restrict inputs)
{
    const UDATA_T* p_in = inputs;

    asm volatile(
        "lw t3,  0(%[p_in]) \n\t"
        "lw t4,  4(%[p_in]) \n\t"
        "lw t5,  8(%[p_in]) \n\t"
        "lw t6, 12(%[p_in]) \n\t"

        "mac16buf_para x0, x0, x0, x0, x0 \n\t"

        :
        : [p_in] "r" (p_in)
        : "t3", "t4", "t5", "t6",
          "cc", "memory"
    );
}

static inline __attribute__((always_inline))
void mac16buf_para_wbuf_final(
    const UDATA_T* __restrict inputs,
    SUM_T* __restrict weightedSum)
{
    int32_t sum;
    const UDATA_T* p_in = inputs;

    asm volatile(
        "lw t3,  0(%[p_in]) \n\t"
        "lw t4,  4(%[p_in]) \n\t"
        "lw t5,  8(%[p_in]) \n\t"
        "lw t6, 12(%[p_in]) \n\t"

        "mac16buf_para %[sum], x0, x0, x0, x0 \n\t"

        : [sum] "=r" (sum)
        : [p_in] "r" (p_in)
        : "t3", "t4", "t5", "t6",
          "cc", "memory"
    );

    *weightedSum = sum;
}


static inline __attribute__((always_inline))
void mac16buf_conv2_25blocks(
    const WDATA_T* __restrict weights,
    SUM_T* __restrict weightedSum)
{
    /*
     * Conv2:
     *
     * 5 × 5 × 16 = 400 weights
     * 400 / 16 = 25 MAC16 blocks
     *
     * block 0      : first
     * block 1..23  : middle
     * block 24     : final
     */

    /* block 0 */
    mac16buf_first(
        weights + 0,
        weightedSum
    );

    /*
     * blocks 1..20
     *
     * 每个middle4处理4个连续block = 64 weights。
     */
    mac16buf_middle4(weights + 16);   // blocks 1..4
    mac16buf_middle4(weights + 80);   // blocks 5..8
    mac16buf_middle4(weights + 144);  // blocks 9..12
    mac16buf_middle4(weights + 208);  // blocks 13..16
    mac16buf_middle4(weights + 272);  // blocks 17..20

    /* blocks 21..23 */
    mac16buf_middle(weights + 336);   // block 21
    mac16buf_middle(weights + 352);   // block 22
    mac16buf_middle(weights + 368);   // block 23

    /* block 24 */
    mac16buf_final(
        weights + 384,
        weightedSum
    );
}

static inline __attribute__((always_inline))
void mac16buf_wbuf_conv2_25blocks(SUM_T* __restrict weightedSum)
{
    /*
     * Conv2:
     *
     * 5 × 5 × 16 = 400 weights
     * 400 / 16 = 25 MAC16 blocks
     *
     * block 0      : first
     * block 1..23  : middle
     * block 24     : final
     */

    /* block 0 */
    mac16buf_wbuf_first(
        weightedSum
    );

    /*
     * blocks 1..20
     *
     * 每个middle4处理4个连续block = 64 weights。
     */
    mac16buf_wbuf_middle4();   // blocks 1..4
    mac16buf_wbuf_middle4();   // blocks 5..8
    mac16buf_wbuf_middle4();  // blocks 9..12
    mac16buf_wbuf_middle4();  // blocks 13..16
    mac16buf_wbuf_middle4();  // blocks 17..20

    /* blocks 21..23 */
    mac16buf_wbuf_middle();   // block 21
    mac16buf_wbuf_middle();   // block 22
    mac16buf_wbuf_middle();   // block 23

    /* block 24 */
    mac16buf_wbuf_final(
        weightedSum
    );
}

static inline __attribute__((always_inline))
void mac16buf_fc1_24blocks(
    const WDATA_T* __restrict weights,
    SUM_T* __restrict weightedSum)
{
    /* block 0 */
    mac16buf_first(
        weights + 0,
        weightedSum
    );

    /*
     * blocks 1..20
     */
    mac16buf_middle4(weights + 16);   // 1..4
    mac16buf_middle4(weights + 80);   // 5..8
    mac16buf_middle4(weights + 144);  // 9..12
    mac16buf_middle4(weights + 208);  // 13..16
    mac16buf_middle4(weights + 272);  // 17..20

    /*
     * blocks 21..22
     */
    mac16buf_middle(weights + 336);   // 21
    mac16buf_middle(weights + 352);   // 22

    /*
     * block 23
     */
    mac16buf_final(
        weights + 368,
        weightedSum
    );
}


static void macsOnRange(const UDATA_T* __restrict inputs,
                        const WDATA_T* __restrict weights,
                        SUM_T* __restrict weightedSum,
                        int nb_iterations)
{
    for (int iter = 0; iter < nb_iterations; ++iter) {
        *weightedSum += inputs[iter] * weights[iter];
    }
}

static UDATA_T saturate(SUM_T value, uint32_t sat) {
    return clamp(value, (SUM_T)(0), ((SUM_T)(1) << sat) - 1);
}

static UDATA_T sat(SUM_T weightedSum, int output,
                                           ActivationFunction_T func,
                                           /* const Rescaling_T& __restrict rescaling */
                                           int shift)
{
    switch(func) {
        case Linear:
        case Saturation: {
            break;
        }
        case Rectifier: {
            if(weightedSum <= 0) weightedSum = 0;
            break;
        }
        default:
            printf("Unsupported activation function.\n");
            break;
    }

    return saturate(weightedSum>>shift, NB_BITS);
}

static void convcellPropagate1(
    const UDATA_T* __restrict inputs,
    UDATA_T* __restrict outputs,
    const BDATA_T* __restrict biasses,
    const WDATA_T* __restrict weights,
    int rescaling,
    int NB_CHANNELS,
    int CHANNELS_HEIGHT,
    int CHANNELS_WIDTH,
    int NB_OUTPUTS,
    int OUTPUTS_HEIGHT,
    int OUTPUTS_WIDTH,
    int PADDING_Y,
    int PADDING_X,
    int STRIDE_Y,
    int STRIDE_X,
    int KERNEL_HEIGHT,
    int KERNEL_WIDTH,
    ActivationFunction_T ACTIVATION,

    /*
     * Input memory mapping。
     */
    int INPUT_MEM_CONT_OFFSET,
    int INPUT_MEM_CONT_SIZE,
    int INPUT_MEM_WRAP_OFFSET,
    int INPUT_MEM_WRAP_SIZE,
    int INPUT_MEM_STRIDE,

    /*
     * Output memory mapping。
     */
    int OUTPUT_MEM_CONT_OFFSET,
    int OUTPUT_MEM_CONT_SIZE,
    int OUTPUT_MEM_WRAP_OFFSET,
    int OUTPUT_MEM_WRAP_SIZE,
    int OUTPUT_MEM_STRIDE)
{
    /*
     * 当前函数是针对当前固定Conv1配置的加速路径：
     *
     * NB_CHANNELS    = 1
     * KERNEL_WIDTH   = 4
     * KERNEL_HEIGHT  = 4
     * STRIDE_X       = 2
     * STRIDE_Y       = 2
     * PADDING_X      = 0
     * PADDING_Y      = 0
     *
     * 一个Conv1 kernel包含：
     *
     *   4 × 4 × 1 = 16 inputs
     *
     * 正好对应一条MAC16。
     */

    /*
     * 设置Conv1模式：
     *
     * buf4 x0 => active_blocks = 1。
     *
     * 这里只需要设置一次。
     * 后面每个patch的output 0会通过mac16buf_para更新buffer内容。
     */
    buffer4_setmode_conv1();

    /*
     * 每个filter包含16个weights。
     */
    const int filter_size
        = NB_CHANNELS * KERNEL_HEIGHT * KERNEL_WIDTH;

    /*
     * 相邻input行之间的byte距离。
     *
     * 当前Conv1中：
     *
     * CHANNELS_WIDTH = 24
     * INPUT_MEM_STRIDE = 1
     *
     * 因此row_stride = 24 bytes。
     */
    const int row_stride
        = CHANNELS_WIDTH * INPUT_MEM_STRIDE;

    for (int oy = 0; oy < OUTPUTS_HEIGHT; ++oy) {
        /*
         * 当前网络padding为0。
         */
        const int iy
            = oy * STRIDE_Y - PADDING_Y;

        for (int ox = 0; ox < OUTPUTS_WIDTH; ++ox) {
            const int capture_weights = (ox == 0 && oy == 0);
            const int ix
                = ox * STRIDE_X - PADDING_X;

            /*
             * 当前4×4 patch左上角input位置。
             */
            const int input_position
                = ix + CHANNELS_WIDTH * iy;

            int input_offset
                = INPUT_MEM_STRIDE * input_position;

            /*
             * 当前Conv1输入正常情况下不存在wrap。
             * 保留这一判断，避免memory mapping以后发生改变。
             */
            if (INPUT_MEM_WRAP_SIZE > 0
                && input_offset >= INPUT_MEM_CONT_SIZE)
            {
                input_offset +=
                    INPUT_MEM_WRAP_OFFSET
                    - INPUT_MEM_CONT_OFFSET
                    - INPUT_MEM_CONT_SIZE;
            }

            /*
             * 四行input指针。
             */
            const UDATA_T* row0
                = inputs + input_offset;

            const UDATA_T* row1
                = row0 + row_stride;

            const UDATA_T* row2
                = row1 + row_stride;

            const UDATA_T* row3
                = row2 + row_stride;

            /*
             * 当前输出位置。
             */
            const int output_position
                = ox + OUTPUTS_WIDTH * oy;

            int output_offset
                = OUTPUT_MEM_STRIDE * output_position;

            if (OUTPUT_MEM_WRAP_SIZE > 0
                && output_offset >= OUTPUT_MEM_CONT_SIZE)
            {
                output_offset +=
                    OUTPUT_MEM_WRAP_OFFSET
                    - OUTPUT_MEM_CONT_OFFSET
                    - OUTPUT_MEM_CONT_SIZE;
            }

            /*
             * ========================================================
             * Output filter 0
             * ========================================================
             *
             * output 0负责：
             *
             *   1. 根据实际input地址检查alignment；
             *   2. 执行MAC16计算；
             *   3. 同时把4×4 input patch写进硬件buffer。
             *
             * 注意：
             *
             * alignment只在这里检查一次。
             */
            {
                const int output = 0;

                SUM_T weightedSum
                    = biasses[output];

                const uintptr_t input_alignment
                    = ((uintptr_t)row0) & 3u;


                if (capture_weights) {
                
                    const WDATA_T* filter_weights
                        = weights;


                    if (input_alignment == 0u) {
                        /*
                        * 4-byte aligned：
                        * 每行直接使用一个lw。
                        */
                        weightedSum =
                            mac16buf_para_conv1_aligned(
                                row0,
                                row1,
                                row2,
                                row3,
                                filter_weights,
                                weightedSum
                            );
                    }
                    else if (input_alignment == 2u) {
                        /*
                        * 地址为2 mod 4：
                        * 每行使用两个lhu拼接。
                        */
                        weightedSum = 
                            mac16buf_para_conv1_unaligned2(
                                row0,
                                row1,
                                row2,
                                row3,
                                filter_weights,
                                weightedSum
                            );
                    }
                } else {
                    if (input_alignment == 0u) {
                        /*
                        * 4-byte aligned：
                        * 每行直接使用一个lw。
                        */
                        weightedSum = 
                            mac16buf_para_conv1_aligned_wbuf(
                                row0,
                                row1,
                                row2,
                                row3,
                                weightedSum
                            );
                    }
                    else if (input_alignment == 2u) {
                        /*
                        * 地址为2 mod 4：
                        * 每行使用两个lhu拼接。
                        */
                       weightedSum = 
                            mac16buf_para_conv1_unaligned2_wbuf(
                                row0,
                                row1,
                                row2,
                                row3,
                                weightedSum
                            );
                    }
                }
                
                // outputs[output_offset + output]
                //     = sat(
                //         weightedSum,
                //         output,
                //         ACTIVATION,
                //         rescaling
                //     );
                outputs[output_offset + output] = (UDATA_T)weightedSum;
            }

            /*
             * ========================================================
             * Output filters 1～NB_OUTPUTS-1
             * ========================================================
             *
             * output 0已经将当前input patch写入硬件buffer。
             *
             * 因此这里：
             *
             *   - 不再访问原始input；
             *   - 不再检查input alignment；
             *   - 只加载当前filter的16个weights；
             *   - 执行一条mac16buf。
             */
            const WDATA_T* filter_weights
                = weights + filter_size;

            for (int output = 1;
                 output < NB_OUTPUTS;
                 ++output)
            {
                SUM_T weightedSum
                    = biasses[output];
                if (capture_weights) {
                    mac16buf_conv1(
                        filter_weights,
                        &weightedSum
                    );
                    filter_weights += filter_size;
                } else {
                    mac16buf_conv1_wbuf(
                        &weightedSum
                    );
                }

                // outputs[output_offset + output]
                //     = sat(
                //         weightedSum,
                //         output,
                //         ACTIVATION,
                //         rescaling
                //     );
                outputs[output_offset + output] = (UDATA_T)weightedSum;
            }
        }
    }
}


static void convcellPropagate2(
    const UDATA_T* __restrict inputs,
    UDATA_T* __restrict outputs,
    const BDATA_T* __restrict biasses,
    const WDATA_T* __restrict weights,
    int rescaling,
    int NB_CHANNELS, 
    int CHANNELS_HEIGHT, int CHANNELS_WIDTH,
    int NB_OUTPUTS,
    int OUTPUTS_HEIGHT, int OUTPUTS_WIDTH,
    int PADDING_Y, int PADDING_X,
    int STRIDE_Y, int STRIDE_X,
    int KERNEL_HEIGHT, int KERNEL_WIDTH,
    ActivationFunction_T ACTIVATION,
    // Memory mapping: inputs
    int INPUT_MEM_CONT_OFFSET,
    int INPUT_MEM_CONT_SIZE,
    int INPUT_MEM_WRAP_OFFSET,
    int INPUT_MEM_WRAP_SIZE,
    int INPUT_MEM_STRIDE,
    // Memory mapping: outputs
    int OUTPUT_MEM_CONT_OFFSET,
    int OUTPUT_MEM_CONT_SIZE,
    int OUTPUT_MEM_WRAP_OFFSET,
    int OUTPUT_MEM_WRAP_SIZE,
    int OUTPUT_MEM_STRIDE)
{

    int OUTPUTS_HEIGHT_NOPAD
        = (CHANNELS_HEIGHT - KERNEL_HEIGHT + STRIDE_Y) / STRIDE_Y;
    int OUTPUTS_WIDTH_NOPAD
        = (CHANNELS_WIDTH - KERNEL_WIDTH + STRIDE_X) / STRIDE_X;

    buffer4_setmode_conv2();
    /*
     * Conv2 的 buffer 策略：
     *   - Conv2 kernel = 5x5x16 = 400 byte = 25 个 16-byte block。
     *   - 对同一个 output pixel (ox, oy)，完整 400-byte input patch
     *     对 24 个 output filters 都相同。
     *   - 所以先连续执行 25 次 buf4 x24，把完整 input patch 存进硬件 buffer。
     *   - 然后每个 output filter 执行 25 次 mac16buf。
     *   - 硬件 active_blocks=25，所以 25 次 mac16buf 后 read counter 自动回到 0。
     */

    for (int oy = 0; oy < OUTPUTS_HEIGHT; ++oy) {
        const int syMin = (PADDING_Y == 0) ? 0
            : max(PADDING_Y - (oy * STRIDE_Y), 0);
        const int syMax = (PADDING_Y == 0
                && OUTPUTS_HEIGHT == OUTPUTS_HEIGHT_NOPAD) ? KERNEL_HEIGHT
            : clamp(CHANNELS_HEIGHT + PADDING_Y - (oy * STRIDE_Y),
                    0, KERNEL_HEIGHT);
        const int iy = (oy * STRIDE_Y) - PADDING_Y;

        for (int ox = 0; ox < OUTPUTS_WIDTH; ++ox) {
            // In the first output, we capture the weight
            const int capture_weight = (oy == 0 && ox == 0);

            const int sxMin = (PADDING_X == 0) ? 0
                : max(PADDING_X - (ox * STRIDE_X), 0);
            const int sxMax = (PADDING_X == 0
                    && OUTPUTS_WIDTH == OUTPUTS_WIDTH_NOPAD)
                        ? KERNEL_WIDTH
                : clamp(CHANNELS_WIDTH + PADDING_X - (ox * STRIDE_X),
                        0, KERNEL_WIDTH);
            const int ix = (ox * STRIDE_X) - PADDING_X;

            const int oPos = (ox + OUTPUTS_WIDTH * oy);
            int oOffset = OUTPUT_MEM_STRIDE * oPos;

            // if (OUTPUT_MEM_WRAP_SIZE > 0 && oOffset >= OUTPUT_MEM_CONT_SIZE) {
            //     oOffset += OUTPUT_MEM_WRAP_OFFSET - OUTPUT_MEM_CONT_OFFSET
            //                 - OUTPUT_MEM_CONT_SIZE;
            // }
                /*
                 * 填满完整 Conv2 patch。
                 * 连续 25 次 buf4 x24：
                 *   block0, block1, ..., block24
                 * 写完后硬件 write counter 自动回到 0。
                 */
            const int kernel_blocks = 25;   // Conv2 = 25

            /*
            * ============================================================
            * output filter 0:
            * 使用 MAC16BUF_PARA。
            *
            * 每个 block 同时做两件事：
            *   1. 用当前 input block + weight block 做 MAC16
            *   2. 把当前 input block 写进硬件 input_buffer
            *
            * 所以 output 0 算完之后，input_buffer 也已经完整保存了 25 个 blocks。
            * ============================================================
            */
            {
                const int output = 0;
                SUM_T weightedSum = biasses[output];

                const int wBase = NB_CHANNELS * (
                    sxMin + KERNEL_WIDTH * (syMin + KERNEL_HEIGHT * output)
                );

                int block = 0;

                for (int sy = 0; sy < KERNEL_HEIGHT; ++sy) {
                    for (int sx = 0; sx < KERNEL_WIDTH; ++sx) {
                        const int iPos = ((sxMin + sx + ix)
                                        + CHANNELS_WIDTH * (iy + syMin + sy));
                        int iOffset = INPUT_MEM_STRIDE * iPos;

                        if (INPUT_MEM_WRAP_SIZE > 0 && iOffset >= INPUT_MEM_CONT_SIZE) {
                            iOffset += INPUT_MEM_WRAP_OFFSET - INPUT_MEM_CONT_OFFSET
                                      - INPUT_MEM_CONT_SIZE;
                        }

                        const UDATA_T* p_in = inputs + iOffset;

                        if (capture_weight) {
                            const WDATA_T* p_wt = weights + wBase + block * NB_CHANNELS;
                            if (block == 0) {
                                mac16buf_para_first(p_in, p_wt, &weightedSum);
                            }
                            else if (block == kernel_blocks - 1) {
                                mac16buf_para_final(p_in, p_wt, &weightedSum);
                            }
                            else {
                                mac16buf_para_middle(p_in, p_wt);
                            }
                        } else {
                            if (block == 0) {
                                mac16buf_para_wbuf_first(
                                    p_in,
                                    &weightedSum
                                );
                            }
                            else if (block == kernel_blocks - 1) {
                                mac16buf_para_wbuf_final(
                                    p_in,
                                    &weightedSum
                                );
                            }
                            else {
                                mac16buf_para_wbuf_middle(
                                    p_in
                                );
                            }
                        }

                        ++block;
                    }
                }

                // outputs[oOffset + output]
                //     = sat(weightedSum, output, ACTIVATION, rescaling);
                outputs[oOffset + output] = (UDATA_T)weightedSum;
            }


            /*
            * ============================================================
            * output filter 1..NB_OUTPUTS-1:
            * 这里 input_buffer 已经由 output 0 的 MAC16BUF_PARA 填满。
            * 所以后面的 filters 继续用原来的 MAC16BUF 逻辑。
            * ============================================================
            */
            const WDATA_T* filter_weights
                = weights + 400;

            for (int output = 1; output < NB_OUTPUTS; ++output) {

                SUM_T weightedSum = biasses[output];

                if (capture_weight) {
                    mac16buf_conv2_25blocks(
                        filter_weights,
                        &weightedSum
                    );
                    filter_weights += 400;
                } else {
                    mac16buf_wbuf_conv2_25blocks(
                        &weightedSum
                    );
                }

                // outputs[oOffset + output]
                //     = sat(
                //         weightedSum,
                //         output,
                //         ACTIVATION,
                //         rescaling
                //     );
                outputs[oOffset + output] = (UDATA_T)weightedSum;                
            }
        }
    }
}


static void fccellPropagateUDATA_T(
    const UDATA_T* __restrict inputs,
    UDATA_T* __restrict outputs,
    const BDATA_T* __restrict biasses,
    const WDATA_T* __restrict weights,
    const int rescaling,
    int NB_CHANNELS, 
    int CHANNELS_HEIGHT, int CHANNELS_WIDTH,
    int NB_OUTPUTS,
    int OUTPUTS_HEIGHT, int OUTPUTS_WIDTH,
    ActivationFunction_T ACTIVATION,
    // Memory mapping: inputs
    int INPUT_MEM_CONT_OFFSET,
    int INPUT_MEM_CONT_SIZE,
    int INPUT_MEM_WRAP_OFFSET,
    int INPUT_MEM_WRAP_SIZE,
    int INPUT_MEM_STRIDE,
    // Memory mapping: outputs
    int OUTPUT_MEM_CONT_OFFSET,
    int OUTPUT_MEM_CONT_SIZE,
    int OUTPUT_MEM_WRAP_OFFSET,
    int OUTPUT_MEM_WRAP_SIZE,
    int OUTPUT_MEM_STRIDE)
{
    /*
     * FC1 的 buffer 策略：
     *   - FC1 input = Conv2 output = 24x4x4 = 384 byte。
     *   - 384 byte = 24 个 16-byte block。
     *   - 先执行 24 次 buf4 x23，把整个 FC1 input 存进硬件 input buffer。
     *   - 然后每个 output neuron 执行 24 次 mac16buf。
     *   - 硬件 active_blocks=24，所以每个 neuron 算完后 read counter 自动回到 0。
     *
     * 如果输入布局不满足连续/对齐要求，就走原始 scalar fallback。
     */
    const int total_inputs = NB_CHANNELS * CHANNELS_WIDTH * CHANNELS_HEIGHT;
    buffer4_setmode_fc1();

    /*
    * FC1 input = 384 bytes = 24 blocks.
    * output0 用 PARA 填 buffer。
    * output1..NB_OUTPUTS-1 复用 buffer。
    */
    const WDATA_T* neuron_weights = weights;
    for (int och = 0; och < NB_OUTPUTS; och++) {
        SUM_T weightedSum = biasses[och];

        const int wBase = och * total_inputs;

        if (och == 0) {

            /*
            * block 0
            */
            mac16buf_para_first(
                inputs,
                weights + wBase,
                &weightedSum
            );

            /*
            * blocks 1..20
            */
            mac16buf_para_middle4(
                inputs + 16,
                weights + wBase + 16
            );                                  // blocks 1..4

            mac16buf_para_middle4(
                inputs + 80,
                weights + wBase + 80
            );                                  // blocks 5..8

            mac16buf_para_middle4(
                inputs + 144,
                weights + wBase + 144
            );                                  // blocks 9..12

            mac16buf_para_middle4(
                inputs + 208,
                weights + wBase + 208
            );                                  // blocks 13..16

            mac16buf_para_middle4(
                inputs + 272,
                weights + wBase + 272
            );                                  // blocks 17..20

            /*
            * blocks 21、22
            */
            mac16buf_para_middle(
                inputs + 336,
                weights + wBase + 336
            );

            mac16buf_para_middle(
                inputs + 352,
                weights + wBase + 352
            );

            /*
            * block 23
            */
            mac16buf_para_final(
                inputs + 368,
                weights + wBase + 368,
                &weightedSum
            );
        } else {
            /*
            * output1..NB_OUTPUTS-1:
            * input_buffer 已经由 output0 填满。
            * 直接用 mac16buf 计算。
            */
            mac16buf_fc1_24blocks(
                neuron_weights,
                &weightedSum
            );
        }

        // outputs[och] = sat(weightedSum, och, ACTIVATION, rescaling);
        outputs[och] = (UDATA_T)weightedSum;
        neuron_weights += 384;
    }

    return;
}

static void fccellPropagateDATA_T(
    const UDATA_T* __restrict inputs,
    DATA_T* __restrict outputs,
    const BDATA_T* __restrict biasses,
    const WDATA_T* __restrict weights,
    const int rescaling,
    int NB_CHANNELS, 
    int CHANNELS_HEIGHT, int CHANNELS_WIDTH,
    int NB_OUTPUTS,
    int OUTPUTS_HEIGHT, int OUTPUTS_WIDTH,
    ActivationFunction_T ACTIVATION,
    // Memory mapping: inputs
    int INPUT_MEM_CONT_OFFSET,
    int INPUT_MEM_CONT_SIZE,
    int INPUT_MEM_WRAP_OFFSET,
    int INPUT_MEM_WRAP_SIZE,
    int INPUT_MEM_STRIDE,
    // Memory mapping: outputs
    int OUTPUT_MEM_CONT_OFFSET,
    int OUTPUT_MEM_CONT_SIZE,
    int OUTPUT_MEM_WRAP_OFFSET,
    int OUTPUT_MEM_WRAP_SIZE,
    int OUTPUT_MEM_STRIDE)
{
    /*
     * FC2:
     *
     * input = 150 bytes
     *
     * 前144 bytes:
     *     9 × MAC16BUF
     *
     * 最后6 bytes:
     *     scalar
     *
     * 优化：
     *
     * output 0:
     *     使用MAC16BUF_PARA，
     *     一边计算，一边填充9个input buffer blocks。
     *
     * output 1..9:
     *     直接复用input buffer。
     *
     * 对weight:
     *     addr % 4 == 0 -> 普通lw版本
     *     addr % 4 == 2 -> offset2 halfword版本
     */
    const int total_inputs
        = NB_CHANNELS * CHANNELS_WIDTH * CHANNELS_HEIGHT;

    /*
     * 非常重要：
     *
     * 设置active_blocks = 9。
     *
     * 之前你是通过连续9条buf4 x8来完成mode设置+
     * buffer填充。
     *
     * 现在buffer填充交给output0的MAC16BUF_PARA，
     * 所以这里只需要设置一次mode。
     */
    buffer4_setmode_fc2();


    /*
     * ============================================================
     * OUTPUT 0
     * ============================================================
     *
     * output0的weights从整个fc2_weights起始地址开始，
     * 正常情况下这里4-byte aligned。
     *
     * 使用PARA：
     *   calculation + input buffer filling
     */
    {
        const int och = 0;
        const int wBase = 0;

        SUM_T weightedSum = biasses[och];

        /*
         * block 0: first
         */
        mac16buf_para_first(
            inputs + 0,
            weights + wBase + 0,
            &weightedSum
        );

        /*
         * blocks 1..7: middle
         *
         * 这里直接展开，不再写block循环。
         */
        mac16buf_para_middle(
            inputs + 16,
            weights + wBase + 16
        );

        mac16buf_para_middle(
            inputs + 32,
            weights + wBase + 32
        );

        mac16buf_para_middle(
            inputs + 48,
            weights + wBase + 48
        );

        mac16buf_para_middle(
            inputs + 64,
            weights + wBase + 64
        );

        mac16buf_para_middle(
            inputs + 80,
            weights + wBase + 80
        );

        mac16buf_para_middle(
            inputs + 96,
            weights + wBase + 96
        );

        mac16buf_para_middle(
            inputs + 112,
            weights + wBase + 112
        );

        /*
         * block 8: final
         *
         * 结束时：
         *   - local accumulator写回weightedSum；
         *   - input buffer已经完整保存144 bytes。
         */
        mac16buf_para_final(
            inputs + 128,
            weights + wBase + 128,
            &weightedSum
        );

        /*
         * 剩余6个inputs：144..149。
         *
         * 直接展开，避免一个只有6次的循环。
         */
        weightedSum += inputs[144] * weights[wBase + 144];
        weightedSum += inputs[145] * weights[wBase + 145];
        weightedSum += inputs[146] * weights[wBase + 146];
        weightedSum += inputs[147] * weights[wBase + 147];
        weightedSum += inputs[148] * weights[wBase + 148];
        weightedSum += inputs[149] * weights[wBase + 149];

        outputs[och]
            = sat(weightedSum, och, ACTIVATION, rescaling);
        
    }


    /*
     * ============================================================
     * OUTPUT 1 .. NB_OUTPUTS-1
     * ============================================================
     *
     * input_buffer已经由output0填好。
     *
     * 后续outputs只需要读取weights。
     */
    for (int och = 1; och < NB_OUTPUTS; ++och) {

        SUM_T weightedSum = biasses[och];

        const int wBase
            = och * total_inputs;

        const WDATA_T* weight_ptr
            = weights + wBase;

        const uintptr_t weight_alignment
            = ((uintptr_t)weight_ptr) & 3u;


        /*
         * ========================================================
         * CASE 1:
         * Weight address 4-byte aligned
         * ========================================================
         */
        if (weight_alignment == 0u) {

            /*
             * block 0: first
             */
            mac16buf_first(
                weight_ptr + 0,
                &weightedSum
            );

            /*
             * blocks 1..4
             */
            mac16buf_middle4(
                weight_ptr + 16
            );

            /*
             * blocks 5、6、7
             */
            mac16buf_middle(
                weight_ptr + 80
            );

            mac16buf_middle(
                weight_ptr + 96
            );

            mac16buf_middle(
                weight_ptr + 112
            );

            /*
             * block 8: final
             */
            mac16buf_final(
                weight_ptr + 128,
                &weightedSum
            );
        }


        /*
         * ========================================================
         * CASE 2:
         * Weight address = 2 mod 4
         * ========================================================
         *
         * 这是FC2中output 1、3、5、7、9正常会出现的情况。
         *
         * 不再整层scalar fallback。
         */
        else if (weight_alignment == 2u) {

            /*
             * block 0
             */
            mac16buf_first_offset2(
                weight_ptr + 0,
                &weightedSum
            );

            /*
             * blocks 1..7
             */
            mac16buf_middle_offset2(
                weight_ptr + 16
            );

            mac16buf_middle_offset2(
                weight_ptr + 32
            );

            mac16buf_middle_offset2(
                weight_ptr + 48
            );

            mac16buf_middle_offset2(
                weight_ptr + 64
            );

            mac16buf_middle_offset2(
                weight_ptr + 80
            );

            mac16buf_middle_offset2(
                weight_ptr + 96
            );

            mac16buf_middle_offset2(
                weight_ptr + 112
            );

            /*
             * block 8
             */
            mac16buf_final_offset2(
                weight_ptr + 128,
                &weightedSum
            );
        }
        /*
         * ========================================================
         * 剩余6项
         * ========================================================
         *
         * 对齐与offset2两种MAC16路径都在这里统一处理。
         */
        weightedSum += inputs[144] * weight_ptr[144];
        weightedSum += inputs[145] * weight_ptr[145];
        weightedSum += inputs[146] * weight_ptr[146];
        weightedSum += inputs[147] * weight_ptr[147];
        weightedSum += inputs[148] * weight_ptr[148];
        weightedSum += inputs[149] * weight_ptr[149];


        outputs[och]
            = sat(
                weightedSum,
                och,
                ACTIVATION,
                rescaling
            );
    }

    return;
}

static void maxPropagate1(
    const DATA_T* __restrict inputs,
    int32_t* __restrict outputs,
    DATA_T* output_value,
    int NB_CHANNELS,
    int INPUTS_HEIGHT, int INPUTS_WIDTH,
    // Memory mapping: outputs
    int INPUT_MEM_CONT_OFFSET,
    int INPUT_MEM_CONT_SIZE,
    int INPUT_MEM_WRAP_OFFSET,
    int INPUT_MEM_WRAP_SIZE,
    int INPUT_MEM_STRIDE)
{
    int iMaxInput = 0;
    DATA_T maxInput = SCHAR_MIN;

    for (int iy = 0; iy < INPUTS_HEIGHT; ++iy) {
        for (int ix = 0; ix < INPUTS_WIDTH; ++ix) {
            const int oPos = (ix + INPUTS_WIDTH * iy);
            int iOffset = INPUT_MEM_STRIDE * oPos;

            // if (INPUT_MEM_WRAP_SIZE > 0 && iOffset >= INPUT_MEM_CONT_SIZE) {
            //     iOffset += INPUT_MEM_WRAP_OFFSET - INPUT_MEM_CONT_OFFSET
            //                 - INPUT_MEM_CONT_SIZE;
            // }

            if (NB_CHANNELS > 1) {
                for (int ch = 0; ch < NB_CHANNELS; ++ch) {
                    if (inputs[iOffset + ch] > maxInput) {
                        iMaxInput = ch;
                        maxInput = inputs[iOffset + ch];
                    }
                }

                outputs[oPos] = (int32_t)(iMaxInput);
		*output_value = maxInput;
            }
            else {
                outputs[oPos] = (inputs[iOffset] > 0);
		output_value = inputs[iOffset];
            }
        }
    }
}

void propagate(const UDATA_T* inputs, Target_T* outputs, UDATA_T* maxPropagate_val)
{
#ifdef SAVE_OUTPUTS
    FILE* env_stream = fopen("env_output.txt", "w");
    saveOutputs(ENV_NB_OUTPUTS, ENV_SIZE_Y, ENV_SIZE_X, ENV_MEM_CONT_OFFSET, ENV_MEM_CONT_SIZE, ENV_MEM_WRAP_OFFSET, ENV_MEM_WRAP_SIZE, ENV_MEM_STRIDE, inputs, env_stream, Network::Format::CHW);
    fclose(env_stream);
#endif
    // conv1
    UDATA_T* conv1_output = (UDATA_T*) mem + CONV1_MEM_CONT_OFFSET;

#ifdef BENCHMARK
    const Tick_T start_conv1 = tick();
#endif

    convcellPropagate1(inputs , conv1_output, conv1_biases, conv1_weights, 8,
    CONV1_NB_CHANNELS, CONV1_CHANNELS_HEIGHT, CONV1_CHANNELS_WIDTH, CONV1_NB_OUTPUTS, CONV1_OUTPUTS_HEIGHT, 
    CONV1_OUTPUTS_WIDTH, CONV1_PADDING_Y, CONV1_PADDING_X, CONV1_STRIDE_Y, CONV1_STRIDE_X, CONV1_KERNEL_HEIGHT, 
    CONV1_KERNEL_WIDTH, CONV1_ACTIVATION, ENV_MEM_CONT_OFFSET, ENV_MEM_CONT_SIZE, ENV_MEM_WRAP_OFFSET, 
    ENV_MEM_WRAP_SIZE, ENV_MEM_STRIDE, CONV1_MEM_CONT_OFFSET, CONV1_MEM_CONT_SIZE, CONV1_MEM_WRAP_OFFSET, CONV1_MEM_WRAP_SIZE, CONV1_MEM_STRIDE);

    //convcellPropagate1(inputs , conv1_output, conv1_biases, conv1_weights, CONV1_SCALING);

#ifdef BENCHMARK
    const Tick_T end_conv1 = tick();
    static RunningMean_T conv1_timing = {0.0, 0};
    benchmark("conv1", start_conv1, end_conv1, conv1_timing);
#endif

#ifdef SAVE_OUTPUTS
    FILE* conv1_stream = fopen("conv1_output.txt", "w");
    saveOutputs(CONV1_NB_OUTPUTS, CONV1_OUTPUTS_HEIGHT, CONV1_OUTPUTS_WIDTH, CONV1_MEM_CONT_OFFSET, CONV1_MEM_CONT_SIZE, CONV1_MEM_WRAP_OFFSET, CONV1_MEM_WRAP_SIZE, CONV1_MEM_STRIDE, conv1_output , conv1_stream, Network::Format::CHW);
    fclose(conv1_stream);
#endif




    // conv2
    UDATA_T* conv2_output = (UDATA_T*) mem + CONV2_MEM_CONT_OFFSET;

#ifdef BENCHMARK
    const Tick_T start_conv2 = tick();
#endif

    convcellPropagate2(conv1_output , conv2_output, conv2_biases, conv2_weights, 8,
    CONV2_NB_CHANNELS, CONV2_CHANNELS_HEIGHT, CONV2_CHANNELS_WIDTH, 
    CONV2_NB_OUTPUTS, CONV2_OUTPUTS_HEIGHT, CONV2_OUTPUTS_WIDTH, 
    CONV2_PADDING_Y, CONV2_PADDING_X, CONV2_STRIDE_Y, CONV2_STRIDE_X, 
    CONV2_KERNEL_HEIGHT, CONV2_KERNEL_WIDTH, CONV2_ACTIVATION, CONV1_MEM_CONT_OFFSET, 
    CONV1_MEM_CONT_SIZE, CONV1_MEM_WRAP_OFFSET, CONV1_MEM_WRAP_SIZE, 
    CONV1_MEM_STRIDE, CONV2_MEM_CONT_OFFSET, CONV2_MEM_CONT_SIZE, CONV2_MEM_WRAP_OFFSET, 
    CONV2_MEM_WRAP_SIZE, CONV2_MEM_STRIDE);

    //convcellPropagate2(conv1_output , conv2_output, conv2_biases, conv2_weights, CONV2_SCALING);

#ifdef BENCHMARK
    const Tick_T end_conv2 = tick();
    static RunningMean_T conv2_timing = {0.0, 0};
    benchmark("conv2", start_conv2, end_conv2, conv2_timing);
#endif

#ifdef SAVE_OUTPUTS
    FILE* conv2_stream = fopen("conv2_output.txt", "w");
    saveOutputs(CONV2_NB_OUTPUTS, CONV2_OUTPUTS_HEIGHT, CONV2_OUTPUTS_WIDTH, CONV2_MEM_CONT_OFFSET, CONV2_MEM_CONT_SIZE, CONV2_MEM_WRAP_OFFSET, CONV2_MEM_WRAP_SIZE, CONV2_MEM_STRIDE, conv2_output , conv2_stream, Network::Format::CHW);
    fclose(conv2_stream);
#endif




    // fc1
    UDATA_T* fc1_output = (UDATA_T*) mem + FC1_MEM_CONT_OFFSET;

#ifdef BENCHMARK
    const Tick_T start_fc1 = tick();
#endif

    fccellPropagateUDATA_T(conv2_output , fc1_output, fc1_biases, fc1_weights, 8,
    FC1_NB_CHANNELS, FC1_CHANNELS_HEIGHT, 
    FC1_CHANNELS_WIDTH, FC1_NB_OUTPUTS, 
    FC1_OUTPUTS_HEIGHT, FC1_OUTPUTS_WIDTH, FC1_ACTIVATION, 
    CONV2_MEM_CONT_OFFSET, CONV2_MEM_CONT_SIZE, 
    CONV2_MEM_WRAP_OFFSET, CONV2_MEM_WRAP_SIZE, 
    CONV2_MEM_STRIDE, FC1_MEM_CONT_OFFSET, 
    FC1_MEM_CONT_SIZE, FC1_MEM_WRAP_OFFSET, FC1_MEM_WRAP_SIZE, FC1_MEM_STRIDE);

#ifdef BENCHMARK
    const Tick_T end_fc1 = tick();
    static RunningMean_T fc1_timing = {0.0, 0};
    benchmark("fc1", start_fc1, end_fc1, fc1_timing);
#endif

#ifdef SAVE_OUTPUTS
    FILE* fc1_stream = fopen("fc1_output.txt", "w");
    saveOutputs(FC1_NB_OUTPUTS, FC1_OUTPUTS_HEIGHT, FC1_OUTPUTS_WIDTH, FC1_MEM_CONT_OFFSET, FC1_MEM_CONT_SIZE, FC1_MEM_WRAP_OFFSET, FC1_MEM_WRAP_SIZE, FC1_MEM_STRIDE, fc1_output , fc1_stream, Network::Format::CHW);
    fclose(fc1_stream);
#endif




    // fc2
    DATA_T* fc2_output = (DATA_T*) mem + FC2_MEM_CONT_OFFSET;

#ifdef BENCHMARK
    const Tick_T start_fc2 = tick();
#endif

    fccellPropagateDATA_T(fc1_output , fc2_output, fc2_biases, fc2_weights, 11,
    FC2_NB_CHANNELS, FC2_CHANNELS_HEIGHT, 
    FC2_CHANNELS_WIDTH, FC2_NB_OUTPUTS, 
    FC2_OUTPUTS_HEIGHT, FC2_OUTPUTS_WIDTH, 
    FC2_ACTIVATION, FC1_MEM_CONT_OFFSET, 
    FC1_MEM_CONT_SIZE, FC1_MEM_WRAP_OFFSET, 
    FC1_MEM_WRAP_SIZE, FC1_MEM_STRIDE, 
    FC2_MEM_CONT_OFFSET, FC2_MEM_CONT_SIZE, 
    FC2_MEM_WRAP_OFFSET, FC2_MEM_WRAP_SIZE, FC2_MEM_STRIDE);

#ifdef BENCHMARK
    const Tick_T end_fc2 = tick();
    static RunningMean_T fc2_timing = {0.0, 0};
    benchmark("fc2", start_fc2, end_fc2, fc2_timing);
#endif

#ifdef SAVE_OUTPUTS
    FILE* fc2_stream = fopen("fc2_output.txt", "w");
    saveOutputs(FC2_NB_OUTPUTS, FC2_OUTPUTS_HEIGHT, FC2_OUTPUTS_WIDTH, FC2_MEM_CONT_OFFSET, FC2_MEM_CONT_SIZE, FC2_MEM_WRAP_OFFSET, FC2_MEM_WRAP_SIZE, FC2_MEM_STRIDE, fc2_output , fc2_stream, Network::Format::CHW);
    fclose(fc2_stream);
#endif
//modifcation debug
    // printf("fc2_output = ");
    // for (int i = 0; i < 10; i++) {
    //     printf("%d ", fc2_output[i]);
    // }
    // printf("\n");
    maxPropagate1(fc2_output, outputs, maxPropagate_val, FC2_NB_OUTPUTS, FC2_OUTPUTS_HEIGHT, FC2_OUTPUTS_WIDTH, FC2_MEM_CONT_OFFSET, FC2_MEM_CONT_SIZE, FC2_MEM_WRAP_OFFSET, FC2_MEM_WRAP_SIZE, FC2_MEM_STRIDE);

#ifdef SAVE_OUTPUTS
    FILE* max_stream = fopen("max_output.txt", "w");
    saveOutputs(FC2_NB_OUTPUTS, FC2_OUTPUTS_HEIGHT, FC2_OUTPUTS_WIDTH, FC2_MEM_CONT_OFFSET, FC2_MEM_CONT_SIZE, FC2_MEM_WRAP_OFFSET, FC2_MEM_WRAP_SIZE, FC2_MEM_STRIDE, outputs, max_stream, Network::Format::CHW);
    fclose(max_stream);
#endif

}

/*template<>
float Network::backpropagate(const DATA_T* input, const std::int32_t* labels){
   const float loss = 0.0f;
   return loss;
 }

int Network::gradientCheck(){
   return(0);
}*/