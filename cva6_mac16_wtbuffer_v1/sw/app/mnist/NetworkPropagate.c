#include <stdlib.h>
#include <stdio.h>

#include "env.h"
#include "mem_info.h"

#include "conv1.h"
#include "conv2.h"
#include "fc1.h"
#include "fc2.h"


/*
 * ============================================================================
 * Final CVA6 MNIST accelerator software path
 * ============================================================================
 *
 * This file is the software side of the final accelerator implementation.  The
 * optimized path uses three custom instructions and three hardware mechanisms:
 *
 *   BUF4
 *     Configures the number of active 16-byte blocks used by the current layer.
 *     The rd field encodes active_blocks - 1.
 *
 *   MAC16BUF_PARA
 *     Performs one 16-element INT8 MAC while the four 32-bit input words are
 *     supplied by the CPU.  The same input words are written into the hardware
 *     input buffer, so computation and buffer filling happen at the same time.
 *
 *   MAC16BUF
 *     Performs the same 16-element INT8 MAC but reads the input words from the
 *     hardware input buffer.  It is therefore used by the later output
 *     filters/neurons that reuse the same input vector or convolution patch.
 *
 *   Input buffer
 *     Holds up to 25 blocks = 400 bytes.  It is refreshed for each new input
 *     patch/vector by the first output filter/neuron and then reused by the
 *     remaining outputs.
 *
 *   Weight buffer
 *     Used only by Conv1 and Conv2.  During the first spatial output position,
 *     weight blocks are consumed by the MAC datapath and captured in hardware
 *     at the same time.  Later spatial positions reuse those buffered weights,
 *     removing repeated weight loads from software.  Conv1 stores 16 blocks
 *     (256 bytes); Conv2 stores 600 blocks (9.6 kB).  FC1 and FC2 weights are
 *     still supplied directly by software.
 *
 *   Local accumulator
 *     A multi-block dot product is split into first/middle/final MAC16 blocks.
 *     The first block starts from the bias carried in rd; middle blocks keep the
 *     partial sum in the coprocessor; only the final block returns a result to
 *     the architectural register file.
 *
 * Post-processing is also split between hardware and software.  For Conv1,
 * Conv2 and FC1, the final MAC block already applies ReLU, right shift by 8 and
 * unsigned 8-bit saturation in hardware, so the returned value is stored
 * directly.  FC2 is different: only 144 of its 150 inputs are covered by the
 * nine MAC16 blocks, therefore hardware returns the full 32-bit accumulator;
 * software adds the remaining six scalar MAC terms and then calls sat().
 * ============================================================================
 */

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
 * ============================================================================
 * Accelerator layer configuration
 * ============================================================================
 *
 * BUF4 programs the active input-buffer length through its rd field:
 *
 *     active_blocks = rd + 1
 *
 *   Conv1: rd = x0  ->  1 block  =  16 bytes
 *   Conv2: rd = x24 -> 25 blocks = 400 bytes
 *   FC1:   rd = x23 -> 24 blocks = 384 bytes
 *   FC2:   rd = x8  ->  9 blocks = 144 bytes (+ 6 scalar bytes)
 *
 * These values also identify the fixed layer modes used by the coprocessor:
 * active_blocks = 1 and 25 enable the Conv1/Conv2 weight-buffer paths, while
 * 1, 25 and 24 enable hardware post-processing.  FC2 uses 9 blocks and is
 * intentionally excluded from hardware post-processing because six scalar
 * products still have to be accumulated in software.
 *
 * MAC16BUF_PARA supplies its four input words through fixed registers x28-x31;
 * the software register convention must therefore match issue_read_operands.sv.
 */

/*
 * Configure Conv1 for one active 16-byte block.
 * Each 4x4x1 input patch contains exactly 16 bytes and therefore maps to one
 * MAC16 block. The patch itself is filled by MAC16BUF_PARA.
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

/*
 * Configure Conv2 for 25 active 16-byte blocks.
 * A 5x5x16 input patch contains 400 bytes = 25 MAC16 blocks.
 * The 25 blocks are filled while output filter 0 is computed.
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

/*
 * Configure FC1 for 24 active 16-byte blocks.
 * FC1 consumes 384 input bytes = 24 MAC16 blocks.
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

/*
 * Configure FC2 for 9 active 16-byte blocks.
 * The first 144 input bytes are handled by 9 MAC16 blocks; the remaining
 * 6 bytes are processed with scalar MAC operations.
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

/*

/*
 * ============================================================================
 * Low-level MAC16BUF primitives
 * ============================================================================
 *
 * The helpers below are ordered from the most generic execution primitives to
 * layer-specific wrappers.  Keeping this order makes the software data path
 * easier to follow:
 *
 *   1. MAC16BUF first/middle/final primitives
 *   2. MAC16BUF_PARA primitives
 *   3. Unrolled and offset-2 variants
 *   4. Conv1-specific helpers
 *   5. Conv2/FC1 multi-block wrappers
 */

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


/*
 * ----------------------------------------------------------------------------
 * MAC16BUF_PARA primitives
 * ----------------------------------------------------------------------------
 * These variants provide the input block through x28-x31.  The coprocessor
 * computes the MAC and stores the same input words into the input buffer.
 */

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



/*
 * ----------------------------------------------------------------------------
 * Unrolled and alignment-specialized helpers
 * ----------------------------------------------------------------------------
 */

/*
 * ============================================================================
 * MAC16BUF_PARA helpers: input-buffer fill path
 * ============================================================================
 */

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

/*
 * ============================================================================
 * FC2 unaligned-weight helpers
 * ============================================================================
 */

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
         * First block: initialize the local accumulator from rd (bias).
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
         * Middle block: keep the partial sum in the local accumulator.
         * No architectural register write-back is requested.
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
         * Final block: write the completed local accumulator back to rd.
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



/*
 * ----------------------------------------------------------------------------
 * Conv1-specific helpers
 * ----------------------------------------------------------------------------
 * Conv1 uses one 16-byte block per output.  The aligned and +2-byte variants
 * differ only in how the 4x4 patch is loaded; the *_wbuf variants reuse the
 * Conv1 weights already captured by the coprocessor.
 */

/*
 * Conv1 MAC16BUF_PARA helpers.
 *
 * The first spatial patch uses the variants that also supply weight words, so
 * the coprocessor can capture Conv1 weights while computing.  Once the 16
 * Conv1 weight blocks have been captured, the *_wbuf variants supply only the
 * input words; the weight words come from the local hardware weight buffer.
 * In both cases the current 4x4 input patch is written into input-buffer block 0.
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
         * Conv1 uses active_blocks = 1, so this instruction is both the first
         * and final block. It computes the MAC16 result and fills input block 0.
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

        /* Reconstruct four consecutive bytes from two halfword loads. */
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


/*
 * ----------------------------------------------------------------------------
 * Layer-sized multi-block wrappers
 * ----------------------------------------------------------------------------
 * Conv2 consumes 25 MAC16 blocks per output and FC1 consumes 24.  These
 * wrappers sequence first/middle/final operations while preserving the local
 * accumulator protocol implemented in hardware.
 */

/*
 * ============================================================================
 * Layer-level MAC16 block sequences
 * ============================================================================
 */

static inline __attribute__((always_inline))
void mac16buf_conv2_25blocks(
    const WDATA_T* __restrict weights,
    SUM_T* __restrict weightedSum)
{
    /*
     * Conv2 uses 5 x 5 x 16 = 400 weights = 25 MAC16 blocks:
     *   block 0      : first
     *   blocks 1..23 : middle
     *   block 24     : final
     */

    /* block 0 */
    mac16buf_first(
        weights + 0,
        weightedSum
    );

    /*
     * Blocks 1..20 are grouped in five calls to mac16buf_middle4().
     * Each call processes four consecutive 16-byte weight blocks.
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
     * Conv2 weight-buffer path: 25 MAC16 blocks are read directly from the
     * local weight buffer using the same first/middle/final sequence.
     */

    /* block 0 */
    mac16buf_wbuf_first(
        weightedSum
    );

    /*
     * Blocks 1..20 are grouped in five calls to mac16buf_wbuf_middle4().
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



/*
 * ============================================================================
 * Scalar reference and software post-processing helpers
 * ============================================================================
 */

/* Scalar reference helper retained from the baseline implementation. */
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


/*
 * ============================================================================
 * Layer propagation functions
 * ============================================================================
 */

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

    /* Input memory-mapping parameters. */
    int INPUT_MEM_CONT_OFFSET,
    int INPUT_MEM_CONT_SIZE,
    int INPUT_MEM_WRAP_OFFSET,
    int INPUT_MEM_WRAP_SIZE,
    int INPUT_MEM_STRIDE,

    /* Output memory-mapping parameters. */
    int OUTPUT_MEM_CONT_OFFSET,
    int OUTPUT_MEM_CONT_SIZE,
    int OUTPUT_MEM_WRAP_OFFSET,
    int OUTPUT_MEM_WRAP_SIZE,
    int OUTPUT_MEM_STRIDE)
{
    /*
     * Specialized Conv1 acceleration path for the current network:
     *   NB_CHANNELS   = 1
     *   KERNEL        = 4 x 4
     *   STRIDE        = 2 x 2
     *   PADDING       = 0
     *
     * One kernel contains 4 x 4 x 1 = 16 input values, exactly one MAC16 block.
     */

    /*
     * Conv1 uses one 16-byte input block per 4x4 patch.  For each spatial
     * position, output filter 0 executes MAC16BUF_PARA: it computes the first
     * output and refreshes input-buffer block 0 at the same time.  Filters
     * 1..15 then reuse that input block with MAC16BUF.
     *
     * At the first spatial position only, all 16 filter weight blocks are also
     * captured into the hardware weight buffer while they are being used.  All
     * later spatial positions therefore reuse both buffered inputs and buffered
     * weights instead of loading the Conv1 weights again from memory.
     */
    buffer4_setmode_conv1();

    /* Each Conv1 output filter contains 16 weights. */
    const int filter_size
        = NB_CHANNELS * KERNEL_HEIGHT * KERNEL_WIDTH;

    /*
     * Byte distance between two adjacent input rows.
     * For the current Conv1 configuration:
     *   CHANNELS_WIDTH   = 24
     *   INPUT_MEM_STRIDE = 1
     * therefore row_stride = 24 bytes.
     */
    const int row_stride
        = CHANNELS_WIDTH * INPUT_MEM_STRIDE;

    for (int oy = 0; oy < OUTPUTS_HEIGHT; ++oy) {
        /* The current network uses zero padding for Conv1. */
        const int iy
            = oy * STRIDE_Y - PADDING_Y;

        for (int ox = 0; ox < OUTPUTS_WIDTH; ++ox) {
            /* Capture all Conv1 filter weights only at the first spatial position. */
            const int capture_weights = (ox == 0 && oy == 0);
            const int ix
                = ox * STRIDE_X - PADDING_X;

            /* Top-left input position of the current 4x4 patch. */
            const int input_position
                = ix + CHANNELS_WIDTH * iy;

            int input_offset
                = INPUT_MEM_STRIDE * input_position;

            /*
             * Wrapping is not expected for the current Conv1 memory layout.
             * Keep the check to preserve compatibility with the mapping parameters.
             */
            if (INPUT_MEM_WRAP_SIZE > 0
                && input_offset >= INPUT_MEM_CONT_SIZE)
            {
                input_offset +=
                    INPUT_MEM_WRAP_OFFSET
                    - INPUT_MEM_CONT_OFFSET
                    - INPUT_MEM_CONT_SIZE;
            }

            /* Pointers to the four rows of the current 4x4 patch. */
            const UDATA_T* row0
                = inputs + input_offset;

            const UDATA_T* row1
                = row0 + row_stride;

            const UDATA_T* row2
                = row1 + row_stride;

            const UDATA_T* row3
                = row2 + row_stride;

            /* Output-memory position for the current spatial location. */
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
             * Output filter 0:
             *   1. Check the actual input alignment.
             *   2. Execute one MAC16 operation.
             *   3. Fill the hardware input buffer with the current 4x4 patch.
             *
             * Alignment is checked only on this path because later filters reuse
             * the buffered input data.
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
                         * 4-byte-aligned input: load one 32-bit word per row.
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
                         * Address is 2 mod 4: reconstruct each row with two
                         * halfword loads.
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
                         * 4-byte-aligned input: load one 32-bit word per row.
                         * Weights are read from the local weight buffer.
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
                         * Address is 2 mod 4: reconstruct each row with two
                         * halfword loads. Weights come from the local buffer.
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
                /* Conv1 final MAC already returns ReLU >> 8, saturated to u8. */
                outputs[output_offset + output] = (UDATA_T)weightedSum;
            }

            /*
             * Output filters 1..NB_OUTPUTS-1:
             *   - the current input patch is already buffered by output filter 0;
             *   - no further input-memory access or alignment check is required;
             *   - only the corresponding weights are supplied (or read from the
             *     weight buffer);
             *   - one MAC16BUF instruction produces each Conv1 output.
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
                /* Conv1 final MAC already returns ReLU >> 8, saturated to u8. */
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
     * Conv2 buffering strategy:
     *   - One 5x5x16 patch = 400 bytes = 25 x 16-byte blocks.
     *   - BUF4 configures active_blocks = 25 once for the layer.
     *   - For every spatial position, output filter 0 executes 25
     *     MAC16BUF_PARA operations.  Those operations compute filter 0 while
     *     filling the complete input buffer; filters 1..23 then reuse the same
     *     25 input blocks through MAC16BUF.
     *   - At the first spatial position only, all 24 x 25 = 600 Conv2 weight
     *     blocks are captured while the outputs are computed.  Later spatial
     *     positions read those weights from the local weight buffer.
     *   - The local accumulator keeps the 25 partial MAC16 blocks inside the
     *     coprocessor and only the final block writes the completed output.
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
            /* Capture the complete Conv2 weight set only at the first spatial position. */
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
            /* The generated Conv2 output layout is contiguous in this final network. */
                /*
                 * The current Conv2 patch is filled block by block by the
                 * MAC16BUF_PARA operations executed for output filter 0.
                 */
            const int kernel_blocks = 25;   // Conv2 = 25

            /*
             * Output filter 0:
             * each MAC16BUF_PARA block performs two operations simultaneously:
             *   1. MAC16(input block, weight block)
             *   2. store the current input block in the hardware input buffer
             *
             * After 25 blocks, both the first output and the full input-buffer
             * contents for this spatial position are available.
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
                /* Conv2 final MAC already returns ReLU >> 8, saturated to u8. */
                outputs[oOffset + output] = (UDATA_T)weightedSum;
            }


            /*
             * Output filters 1..NB_OUTPUTS-1:
             * the input buffer has already been filled by output filter 0, so
             * these filters use MAC16BUF and reuse the buffered input blocks.
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
                /* Conv2 final MAC already returns ReLU >> 8, saturated to u8. */
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
     * FC1 buffering strategy:
     *   - FC1 input = 24 x 4 x 4 = 384 bytes = 24 x 16-byte blocks.
     *   - BUF4 configures active_blocks = 24 once.
     *   - Neuron 0 computes with MAC16BUF_PARA and fills all 24 input-buffer
     *     blocks at the same time.
     *   - Neurons 1..63 reuse those input blocks with MAC16BUF.
     *   - FC1 weights are not buffered: each neuron supplies its own 384-byte
     *     weight vector from software.
     *   - Hardware applies ReLU, >>8 and u8 saturation on the final block.
     */
    const int total_inputs = NB_CHANNELS * CHANNELS_WIDTH * CHANNELS_HEIGHT;
    buffer4_setmode_fc1();

    /*
     * FC1 input = 384 bytes = 24 blocks.
     * Neuron 0 fills the input buffer through the PARA path; later neurons
     * reuse the buffered inputs.
     */
    const WDATA_T* neuron_weights = weights;
    for (int och = 0; och < NB_OUTPUTS; och++) {
        SUM_T weightedSum = biasses[och];

        const int wBase = och * total_inputs;

        if (och == 0) {

            /* Block 0: first. */
            mac16buf_para_first(
                inputs,
                weights + wBase,
                &weightedSum
            );

            /* Blocks 1..20, unrolled four at a time. */
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

            /* Blocks 21 and 22. */
            mac16buf_para_middle(
                inputs + 336,
                weights + wBase + 336
            );

            mac16buf_para_middle(
                inputs + 352,
                weights + wBase + 352
            );

            /* Block 23: final. */
            mac16buf_para_final(
                inputs + 368,
                weights + wBase + 368,
                &weightedSum
            );
        } else {
            /*
             * Neurons 1..NB_OUTPUTS-1 reuse the input buffer filled by neuron 0.
             */
            mac16buf_fc1_24blocks(
                neuron_weights,
                &weightedSum
            );
        }
        /* FC1 final MAC already returns ReLU >> 8, saturated to u8. */
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
     * FC2 acceleration:
     *   - Total input size: 150 bytes.
     *   - First 144 bytes: 9 MAC16 blocks.
     *   - Remaining 6 bytes: scalar MAC operations.
     *
     * Output 0 uses MAC16BUF_PARA to compute while filling the 9 input-buffer
     * blocks. Outputs 1..9 reuse these buffered inputs with MAC16BUF.  FC2
     * weights are not stored in the hardware weight buffer.
     *
     * Unlike Conv1/Conv2/FC1, active_blocks = 9 disables hardware
     * post-processing.  The final MAC16 block returns the full 32-bit partial
     * sum, software adds inputs 144..149, and sat() is applied only after those
     * six scalar terms have been included.
     *
     * Weight alignment:
     *   - address % 4 == 0 : regular 32-bit load path
     *   - address % 4 == 2 : halfword reconstruction path
     */
    const int total_inputs
        = NB_CHANNELS * CHANNELS_WIDTH * CHANNELS_HEIGHT;

    /*
     * Configure active_blocks = 9 once for FC2.
     * Buffer filling is performed by the MAC16BUF_PARA sequence of output 0.
     */
    buffer4_setmode_fc2();


    /*
     * Output 0:
     * its weight array starts at the base of fc2_weights and is normally
     * 4-byte aligned. The PARA path computes the output and fills the input
     * buffer simultaneously.
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
         * Blocks 1..7: middle blocks, explicitly unrolled.
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
         * Block 8: final block.
         * The local accumulator is written back to weightedSum and all
         * 144 buffered input bytes are now available for the next outputs.
         */
        mac16buf_para_final(
            inputs + 128,
            weights + wBase + 128,
            &weightedSum
        );

        /*
         * Remaining inputs 144..149 are handled with scalar MAC operations.
         */
        weightedSum += inputs[144] * weights[wBase + 144];
        weightedSum += inputs[145] * weights[wBase + 145];
        weightedSum += inputs[146] * weights[wBase + 146];
        weightedSum += inputs[147] * weights[wBase + 147];
        weightedSum += inputs[148] * weights[wBase + 148];
        weightedSum += inputs[149] * weights[wBase + 149];

        /* FC2 completes six scalar MACs in software, so saturation stays here. */
        outputs[och]
            = sat(weightedSum, och, ACTIVATION, rescaling);
        
    }


    /*
     * Outputs 1..NB_OUTPUTS-1:
     * reuse the input buffer filled while output 0 was computed. Only weights
     * need to be supplied from software.
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

            /* Blocks 5, 6 and 7. */
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
         * Case 2: weight address is 2 mod 4.
         *
         * With a 150-byte weight vector, this alignment occurs naturally for
         * alternating FC2 output neurons. Halfword loads reconstruct each
         * 32-bit weight word instead of falling back to a scalar implementation.
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
         * Remaining six scalar terms, shared by both weight-alignment paths.
         */
        weightedSum += inputs[144] * weight_ptr[144];
        weightedSum += inputs[145] * weight_ptr[145];
        weightedSum += inputs[146] * weight_ptr[146];
        weightedSum += inputs[147] * weight_ptr[147];
        weightedSum += inputs[148] * weight_ptr[148];
        weightedSum += inputs[149] * weight_ptr[149];


        /* FC2 completes six scalar MACs in software, so saturation stays here. */
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
            /* The final FC2 output layout is contiguous; no wrap adjustment is used here. */

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
    maxPropagate1(fc2_output, outputs, maxPropagate_val, FC2_NB_OUTPUTS, FC2_OUTPUTS_HEIGHT, FC2_OUTPUTS_WIDTH, FC2_MEM_CONT_OFFSET, FC2_MEM_CONT_SIZE, FC2_MEM_WRAP_OFFSET, FC2_MEM_WRAP_SIZE, FC2_MEM_STRIDE);

#ifdef SAVE_OUTPUTS
    FILE* max_stream = fopen("max_output.txt", "w");
    saveOutputs(FC2_NB_OUTPUTS, FC2_OUTPUTS_HEIGHT, FC2_OUTPUTS_WIDTH, FC2_MEM_CONT_OFFSET, FC2_MEM_CONT_SIZE, FC2_MEM_WRAP_OFFSET, FC2_MEM_WRAP_SIZE, FC2_MEM_STRIDE, outputs, max_stream, Network::Format::CHW);
    fclose(max_stream);
#endif

}

