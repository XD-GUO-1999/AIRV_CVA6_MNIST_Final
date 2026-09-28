// Copyright 2021 Thales DIS design services SAS
//
// Licensed under the Solderpad Hardware Licence, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.0
// You may obtain a copy of the License at https://solderpad.org/licenses/
//
// Original Author: Guillaume Chauvon (guillaume.chauvon@thalesgroup.com)
// CV-X-IF coprocessor adapted for the final MNIST accelerator.
//
//   BUF4          : configures the active number of 16-byte input blocks.
//   MAC16BUF_PARA : performs MAC16 from CPU-provided inputs while filling the input buffer.
//   MAC16BUF      : performs MAC16 from the local input buffer.
//
// Conv1/Conv2 weights are captured into a four-bank BRAM weight buffer and reused.
// Multi-block dot products use a local first/middle/final accumulator. Conv1, Conv2
// and FC1 are post-processed in hardware and four 8-bit outputs are packed into one
// 32-bit write-back. A pipeline register after partial_sum splits the MAC reduction
// from accumulation/post-processing/packing while preserving an initiation interval of 1.

module cvxif_example_coprocessor
  import cvxif_pkg::*;
  import cvxif_instr_pkg::*;
(
    input  logic        clk_i,        // Clock
    input  logic        rst_ni,       // Asynchronous reset active low
    input  cvxif_req_t  cvxif_req_i,
    output cvxif_resp_t cvxif_resp_o
);

  // Compressed-instruction interface
  logic               x_compressed_valid_i;
  logic               x_compressed_ready_o;
  x_compressed_req_t  x_compressed_req_i;
  x_compressed_resp_t x_compressed_resp_o;
  // Issue interface
  logic               x_issue_valid_i;
  logic               x_issue_ready_o;
  x_issue_req_t       x_issue_req_i;
  x_issue_resp_t      x_issue_resp_o;
  x_issue_resp_t      x_issue_resp_dec; // Raw response from the instruction-table decoder
  // Commit interface
  logic               x_commit_valid_i;
  x_commit_t          x_commit_i;
  // Memory interface
  logic               x_mem_valid_o;
  logic               x_mem_ready_i;
  x_mem_req_t         x_mem_req_o;
  x_mem_resp_t        x_mem_resp_i;
  // Memory-result interface
  logic               x_mem_result_valid_i;
  x_mem_result_t      x_mem_result_i;
  // Result interface
  logic               x_result_valid_o;
  logic               x_result_ready_i;
  x_result_t          x_result_o;

  assign x_compressed_valid_i            = cvxif_req_i.x_compressed_valid;
  assign x_compressed_req_i              = cvxif_req_i.x_compressed_req;
  assign x_issue_valid_i                 = cvxif_req_i.x_issue_valid;
  assign x_issue_req_i                   = cvxif_req_i.x_issue_req;
  assign x_commit_valid_i                = cvxif_req_i.x_commit_valid;
  assign x_commit_i                      = cvxif_req_i.x_commit;
  assign x_mem_ready_i                   = cvxif_req_i.x_mem_ready;
  assign x_mem_resp_i                    = cvxif_req_i.x_mem_resp;
  assign x_mem_result_valid_i            = cvxif_req_i.x_mem_result_valid;
  assign x_mem_result_i                  = cvxif_req_i.x_mem_result;
  assign x_result_ready_i                = cvxif_req_i.x_result_ready;

  assign cvxif_resp_o.x_compressed_ready = x_compressed_ready_o;
  assign cvxif_resp_o.x_compressed_resp  = x_compressed_resp_o;
  assign cvxif_resp_o.x_issue_ready      = x_issue_ready_o;
  assign cvxif_resp_o.x_issue_resp       = x_issue_resp_o;
  assign cvxif_resp_o.x_mem_valid        = x_mem_valid_o;
  assign cvxif_resp_o.x_mem_req          = x_mem_req_o;
  assign cvxif_resp_o.x_result_valid     = x_result_valid_o;
  assign cvxif_resp_o.x_result           = x_result_o;

  // Compressed-instruction interface
  assign x_compressed_ready_o            = '0;
  assign x_compressed_resp_o.instr       = '0;
  assign x_compressed_resp_o.accept      = '0;

  // --------------------------------------------------------------------------
  // Instruction decode and issue metadata
  // --------------------------------------------------------------------------
  instr_decoder #(
      .NbInstr   (cvxif_instr_pkg::NbInstr),
      .CoproInstr(cvxif_instr_pkg::CoproInstr)
  ) instr_decoder_i (
      .clk_i         (clk_i),
      .x_issue_req_i (x_issue_req_i),
      .x_issue_resp_o(x_issue_resp_dec)
  );

  // Extend each FIFO entry with accelerator metadata required after issue.
  typedef struct packed {
    x_issue_req_t  req;
    x_issue_resp_t resp;
    logic          is_buf4;
    logic          is_mac16buf;
    logic          is_mac16buf_para;
    logic          is_first_block;
    logic          is_final_block;

    // Apply layer-specific post-processing before architectural write-back.
    logic          postprocessed_en;

    // Pack four post-processed 8-bit outputs into one 32-bit write-back.
    // Conv1/Conv2 use complete groups of four. FC1 also uses pack4,
    // with a final 2-output tail flushed after output 149.
    logic          pack4_en;
    logic [1:0]    pack_idx;
  } x_issue_t;


  logic fifo_full, fifo_empty;
  logic x_issue_ready_q;
  logic instr_push, instr_pop;
  x_issue_t req_i;
  x_issue_t req_o;

  /*
   * ============================================================
   * Two-stage MAC pipeline (cut after partial_sum)
   * ============================================================
   *
   * Stage 1 performs input/weight selection, 16 multiplications and the
   * reduction tree that produces partial_sum.  partial_sum plus the matching
   * instruction metadata are captured in the pipeline register.
   *
   * Stage 2 performs the local accumulation (bias/acc_q + partial_sum), then
   * post-processing / output packing and drives CV-X-IF x_result.  This is the
   * intended timing cut:
   *
   *   multipliers + reduction -> [REGISTER] -> accumulator + sat + pack/result
   *
   * The register is elastic: when Stage 2 is accepted in the same cycle,
   * Stage 1 may immediately load the next instruction.  Thus the intended
   * initiation interval is one cycle (II=1).
   */
  logic       result_pipe_valid_q;
  x_issue_t   result_pipe_req_q;
  logic signed [31:0] partial_sum_pipe_q; // Stage-1 registered reduction result

  logic result_pipe_ready;
  logic stage1_fire;
  logic stage2_fire;

  // Issue-side counters tag each MAC as first/middle/final before it enters the FIFO.
  // It is used only to decide whether a MAC16BUF instruction should request
  // an architectural writeback. The actual accumulation is still performed
  // later when the instruction reaches the FIFO output/result stage.
  logic [4:0] issue_active_blocks_q;
  logic [4:0] issue_block_cnt_q;
  logic [4:0] issue_buf_active_blocks;
  logic       issue_is_buf4;
  logic       issue_is_mac16buf;
  logic       issue_is_mac16buf_para; // MAC16 plus input-buffer fill
  logic       issue_mac_op; // MAC16BUF or MAC16BUF_PARA
  logic       issue_is_first_block;
  logic       issue_is_final_block;

  // Layer-specific post-processing and packed-output control.
  logic       issue_postprocessed_en;

  // Conv1: 16 outputs/channel group  -> exact groups of four.
  // Conv2: 24 outputs/channel group  -> exact groups of four.
  // FC1  : 150 output neurons        -> 37 groups of four + 2-output tail.
  logic       issue_pack4_en;
  logic [1:0] issue_pack_cnt_q;
  logic       issue_is_fc1;
  logic [7:0] issue_fc1_output_cnt_q;
  logic       issue_pack_writeback;

  assign issue_postprocessed_en = issue_mac_op &&
      ((issue_active_blocks_q == 5'd1) ||
       (issue_active_blocks_q == 5'd25) ||
       (issue_active_blocks_q == 5'd24));

  assign issue_pack4_en = issue_mac_op &&
      ((issue_active_blocks_q == 5'd1) ||
       (issue_active_blocks_q == 5'd25) ||
       (issue_active_blocks_q == 5'd24));

  // FC1 is identified by its 24 MAC16 blocks per output neuron.
  assign issue_is_fc1 = issue_mac_op &&
      (issue_active_blocks_q == 5'd24);

  // Detect accelerator instructions from the opcode field.
  assign issue_is_buf4          = (x_issue_req_i.instr[6:0] == 7'b0101011);
  assign issue_is_mac16buf      = (x_issue_req_i.instr[6:0] == 7'b0001011);
  assign issue_is_mac16buf_para = (x_issue_req_i.instr[6:0] == 7'b1011011);

  assign issue_mac_op = issue_is_mac16buf || issue_is_mac16buf_para;
  assign issue_buf_active_blocks = x_issue_req_i.instr[11:7] + 5'd1;
  assign issue_is_first_block   = issue_mac_op && (issue_block_cnt_q == 5'd0);
  assign issue_is_final_block   = issue_mac_op && (issue_block_cnt_q == (issue_active_blocks_q - 5'd1));


  // A packed group normally writes back on byte 3 (four outputs collected).
  // FC1 has 150 outputs, so output 149 must also force a writeback even though
  // it is only byte 1 of the final partial group (outputs 148 and 149).
  assign issue_pack_writeback =
      issue_is_final_block &&
      issue_pack4_en &&
      ((issue_pack_cnt_q == 2'd3) ||
       (issue_is_fc1 && (issue_fc1_output_cnt_q == 8'd149)));

  // Start from the table decoder response, then override MAC16BUF writeback.
  // Normal mode: only the final block of each output writes back.
  // Pack4 mode: normally every fourth final output writes back.
  // FC1 additionally flushes its final two outputs (148/149) on output 149.
  // Outputs that do not write back are kept locally in output_pack_q.
  always_comb begin
    x_issue_resp_o = x_issue_resp_dec;
    if (issue_mac_op && x_issue_resp_dec.accept) begin
      if (!issue_is_final_block) begin
        x_issue_resp_o.writeback = 1'b0;
      end else if (issue_pack4_en) begin
        x_issue_resp_o.writeback = issue_pack_writeback;
      end else begin
        x_issue_resp_o.writeback = 1'b1;
      end
    end
  end

  assign instr_push = x_issue_valid_i && x_issue_ready_o && x_issue_resp_o.accept;

  // Stage 2 can be replaced in the same cycle in which its result is accepted.
  assign result_pipe_ready = ~result_pipe_valid_q || x_result_ready_i;

  // Accept the FIFO head into Stage 1 only when the result pipeline has room.
  assign stage1_fire = ~fifo_empty
                    && result_pipe_ready
                    && ~(x_commit_i.x_commit_kill && x_commit_valid_i);

  assign stage2_fire = result_pipe_valid_q
                    && x_result_ready_i
                    && ~(x_commit_i.x_commit_kill && x_commit_valid_i);

  // IMPORTANT: the FIFO now pops when Stage 1 accepts the instruction, not
  // when Stage 2 returns the result.  This is what allows one new MAC per cycle.
  assign instr_pop  = (x_commit_i.x_commit_kill && x_commit_valid_i) ||
                      stage1_fire;
  assign x_issue_ready_q = ~fifo_full;

  // Keep issue-time accelerator metadata associated with the instruction in the FIFO.
  assign req_i.req            = x_issue_req_i;
  assign req_i.resp           = x_issue_resp_o;
  assign req_i.is_buf4        = issue_is_buf4;
  assign req_i.is_mac16buf    = issue_is_mac16buf;
  assign req_i.is_mac16buf_para = issue_is_mac16buf_para;
  assign req_i.is_first_block   = issue_is_first_block;
  assign req_i.is_final_block   = issue_is_final_block;
  assign req_i.postprocessed_en = issue_postprocessed_en;
  assign req_i.pack4_en         = issue_pack4_en;
  assign req_i.pack_idx         = issue_pack_cnt_q;

  // Advance/reset issue-side block and packing counters as instructions are accepted.
  always_ff @(posedge clk_i or negedge rst_ni) begin : issue_block_counter
    if (!rst_ni) begin
      issue_active_blocks_q     <= 5'd1;
      issue_block_cnt_q         <= 5'd0;
      issue_pack_cnt_q          <= 2'd0;
      issue_fc1_output_cnt_q    <= 8'd0;
    end else if (instr_push) begin
      if (issue_is_buf4) begin
        issue_active_blocks_q  <= issue_buf_active_blocks;
        issue_block_cnt_q      <= 5'd0;
        issue_pack_cnt_q       <= 2'd0;
        issue_fc1_output_cnt_q <= 8'd0;
      end else if (issue_is_mac16buf || issue_is_mac16buf_para) begin
        if (issue_is_final_block) begin
          issue_block_cnt_q <= 5'd0;

          // Advance one packed-output position only after a complete output
          // neuron/channel has finished. A normal full group or the FC1 tail
          // writeback resets the byte index to zero for the next group.
          if (issue_pack4_en) begin
            if (issue_pack_writeback) begin
              issue_pack_cnt_q <= 2'd0;
            end else begin
              issue_pack_cnt_q <= issue_pack_cnt_q + 2'd1;
            end
          end

          // FC1 has exactly 150 output neurons (0..149). This counter is used
          // only to detect the final 2-output tail and is reset after output149.
          if (issue_is_fc1) begin
            if (issue_fc1_output_cnt_q == 8'd149) begin
              issue_fc1_output_cnt_q <= 8'd0;
            end else begin
              issue_fc1_output_cnt_q <= issue_fc1_output_cnt_q + 8'd1;
            end
          end
        end else begin
          issue_block_cnt_q <= issue_block_cnt_q + 5'd1;
        end
      end
    end
  end

  always_ff @(posedge clk_i or negedge rst_ni) begin : regs
    if (!rst_ni) begin
      x_issue_ready_o <= 1;
    end else begin
      x_issue_ready_o <= x_issue_ready_q;
    end
  end

  // --------------------------------------------------------------------------
  // CV-X-IF instruction FIFO
  // --------------------------------------------------------------------------
  fifo_v3 #(
      .FALL_THROUGH(1),         // data_o ready and pop in the same cycle
      .DATA_WIDTH  ($bits(x_issue_t)),
      .DEPTH       (8),
      .dtype       (x_issue_t)
  ) fifo_commit_i (
      .clk_i     (clk_i),
      .rst_ni    (rst_ni),
      .flush_i   (1'b0),
      .testmode_i(1'b0),
      .full_o    (fifo_full),
      .empty_o   (fifo_empty),
      .usage_o   (),
      .data_i    (req_i),
      .push_i    (instr_push),
      .data_o    (req_o),
      .pop_i     (instr_pop)
  );

  logic [3:0] c;
  counter #(
      .WIDTH(4)
  ) counter_i (
      .clk_i     (clk_i),
      .rst_ni    (rst_ni),
      .clear_i   (~x_commit_i.x_commit_kill && x_commit_valid_i),
      .en_i      (1'b1),
      .load_i    (),
      .down_i    (),
      .d_i       (),
      .q_o       (c),
      .overflow_o()
  );

  // --------------------------------------------------------------------------
  // Accelerator execution state
  // --------------------------------------------------------------------------
  localparam int unsigned INPUT_BUF_WORDS = 25;

  localparam logic [4:0] CONV1_ACTIVE_BLOCKS = 5'd1;
  localparam logic [9:0] CONV1_WEIGHT_LAST   = 10'd15;
  localparam logic [4:0] CONV2_ACTIVE_BLOCKS = 5'd25;
  localparam logic [9:0] CONV2_WEIGHT_LAST   = 10'd599;

  // --------------------------------------------------------------------------
  // Weight buffer: 4 x 32-bit banks x 600 entries
  // --------------------------------------------------------------------------
  /*
   * ============================================================
   * Weight buffer
   * ============================================================
   *
   * Conv1 uses entries 0..15.
   * Conv2 uses entries 0..599.
   *
   * Four independent 32-bit banks provide the 128-bit weight
   * block required by one MAC16 operation.
   *
   * IMPORTANT: reads are synchronous so Vivado can infer BRAM.
   */
  localparam int unsigned WEIGHT_BUF_DEPTH = 600;

  (* ram_style = "block" *)
  logic [31:0] weight_buffer0 [0:WEIGHT_BUF_DEPTH-1];

  (* ram_style = "block" *)
  logic [31:0] weight_buffer1 [0:WEIGHT_BUF_DEPTH-1];

  (* ram_style = "block" *)
  logic [31:0] weight_buffer2 [0:WEIGHT_BUF_DEPTH-1];

  (* ram_style = "block" *)
  logic [31:0] weight_buffer3 [0:WEIGHT_BUF_DEPTH-1];

  /* Registered outputs from synchronous BRAM reads. */
  logic [31:0] weight_rd0_q;
  logic [31:0] weight_rd1_q;
  logic [31:0] weight_rd2_q;
  logic [31:0] weight_rd3_q;

  logic [9:0] weight_block_cnt_q;
  logic       weight_buffer_valid_q;

  logic conv2_weight_mode;
  logic conv1_weight_mode;
  logic weight_buffer_mode;
  logic use_weight_buffer;
  logic [9:0] weight_last_block;

  /* BRAM capture/prefetch control. */
  logic       mac_done;
  logic       weight_capture_en;
  logic       weight_prefetch_en;
  logic [9:0] weight_prefetch_addr;


  logic signed [31:0] acc_q;
  logic signed [31:0] partial_sum;
  logic signed [31:0] mac_base_acc;
  logic signed [31:0] mac_next_acc;
  logic        [7:0] sat_result_u8;
  logic signed [31:0] mac_writeback_data;

  // Packed-output state. Each completed output contributes one byte.
  logic [31:0] output_pack_q;
  logic [31:0] output_pack_next;

  function automatic logic [7:0] sat_shift8_u8(
      input logic signed [31:0] value
  );
  begin
      /*
      * Equivalent to:
      *
      *   ReLU(value)
      *   value >> 8
      *   clamp(value, 0, 255)
      *
      * Conv1 / Conv2 / FC1 all use shift = 8
      * and unsigned 8-bit outputs.
      */

      if (value[31]) begin
          // Negative -> ReLU / lower saturation
          sat_shift8_u8 = 8'd0;
      end
      else if (|value[30:16]) begin
          // (value >> 8) > 255
          sat_shift8_u8 = 8'd255;
      end
      else begin
          // Equivalent to value >> 8
          sat_shift8_u8 = value[15:8];
      end
  end
  endfunction

  // --------------------------------------------------------------------------
  // Stage 2: local accumulation, post-processing and output packing
  // --------------------------------------------------------------------------
  /*
   * Stage 2: local accumulation + post-processing + packing.
   *
   * IMPORTANT: partial_sum_pipe_q and result_pipe_req_q belong to the SAME
   * MAC instruction.  For a first block we use its saved architectural rd/bias;
   * otherwise we use the local accumulator produced by the previous Stage-2 MAC.
   */
  always_comb begin
      mac_base_acc = result_pipe_req_q.is_first_block
                   ? $signed(result_pipe_req_q.req.rs[2])
                   : acc_q;

      mac_next_acc = mac_base_acc + partial_sum_pipe_q;

      sat_result_u8 = sat_shift8_u8(mac_next_acc);

      // Build the packed word including the CURRENT output byte.
      output_pack_next = output_pack_q;
      case (result_pipe_req_q.pack_idx)
        2'd0: output_pack_next[7:0]   = sat_result_u8;
        2'd1: output_pack_next[15:8]  = sat_result_u8;
        2'd2: output_pack_next[23:16] = sat_result_u8;
        2'd3: output_pack_next[31:24] = sat_result_u8;
        default: ;
      endcase

      /*
       * Default: return the full 32-bit accumulated result.
       * FC2 still needs this because six scalar MACs follow MAC16.
       */
      mac_writeback_data = mac_next_acc;

      if (result_pipe_req_q.is_final_block && result_pipe_req_q.postprocessed_en) begin
        if (result_pipe_req_q.pack4_en) begin
          mac_writeback_data = output_pack_next;
        end else begin
          mac_writeback_data = {24'd0, sat_result_u8};
        end
      end
  end

  // --------------------------------------------------------------------------
  // Input buffer: 4 x 32-bit banks x 25 entries = 400 B
  // --------------------------------------------------------------------------
  logic [31:0] input_buffer0 [0:INPUT_BUF_WORDS-1];
  logic [31:0] input_buffer1 [0:INPUT_BUF_WORDS-1];
  logic [31:0] input_buffer2 [0:INPUT_BUF_WORDS-1];
  logic [31:0] input_buffer3 [0:INPUT_BUF_WORDS-1];

  logic [4:0] wr_block_cnt_q;
  logic [4:0] rd_block_cnt_q;
  logic [4:0] active_blocks_q;

  // Conv1/Conv2 reuse locally buffered weights; FC layers use CPU-provided weights.
  assign conv1_weight_mode = (active_blocks_q == CONV1_ACTIVE_BLOCKS);
  assign conv2_weight_mode = (active_blocks_q == CONV2_ACTIVE_BLOCKS);
  assign weight_buffer_mode = conv1_weight_mode || conv2_weight_mode;
  assign use_weight_buffer = weight_buffer_mode && weight_buffer_valid_q;
  assign weight_last_block = conv1_weight_mode ? CONV1_WEIGHT_LAST : CONV2_WEIGHT_LAST;


  logic [4:0] buf_active_blocks;
  logic [4:0] wr_block_sel;
  assign buf_active_blocks = req_o.req.instr[11:7] + 5'd1;

  logic is_buf4_ex;
  logic is_mac16buf_ex;
  logic is_mac16buf_para_ex;

  assign is_buf4_ex          = req_o.is_buf4;
  assign is_mac16buf_ex      = req_o.is_mac16buf;
  assign is_mac16buf_para_ex = req_o.is_mac16buf_para;

  /*
   * Stage 2 is now driven from the result pipeline register.  The FIFO head
   * has already been consumed by Stage 1 one cycle earlier.
   */
  assign x_result_valid_o = result_pipe_valid_q
                         && ~(x_commit_i.x_commit_kill && x_commit_valid_i);

  /*
   * Internal MAC-side state advances when Stage 1 accepts a MAC.  This keeps
   * the input/weight-buffer address stream aligned with an II=1 pipeline.
   */
  assign mac_done = stage1_fire
                 && (is_mac16buf_ex || is_mac16buf_para_ex);

  /*
   * Capture phase: when Stage 1 accepts a MAC, use CPU weight operands
   * directly for the MAC and simultaneously write them into local weight BRAM.
   */
  assign weight_capture_en =
      mac_done
      && weight_buffer_mode
      && !weight_buffer_valid_q;

  /*
   * Prefetch phase:
   *   - normal reuse: completed block n prefetches block n+1;
   *   - final capture block: prefetch block 0 so the first reuse MAC
   *     can execute immediately on the next cycle.
   */
  assign weight_prefetch_en =
      mac_done
      && weight_buffer_mode
      && (
           weight_buffer_valid_q
           ||
           (!weight_buffer_valid_q
            && (weight_block_cnt_q == weight_last_block))
         );

  /* Circular address sequence for Conv1 (0..15) or Conv2 (0..599). */
  always_comb begin
    if (weight_block_cnt_q == weight_last_block) begin
      weight_prefetch_addr = 10'd0;
    end else begin
      weight_prefetch_addr = weight_block_cnt_q + 10'd1;
    end
  end

  assign wr_block_sel = (is_buf4_ex && (buf_active_blocks != active_blocks_q)) ||
                        (is_mac16buf_para_ex && req_o.is_first_block) ? 5'd0 : wr_block_cnt_q;

  /*
   * ============================================================
   * Synchronous weight BRAM
   * ============================================================
   *
   * Do NOT reset the RAM arrays. weight_buffer_valid_q guarantees
   * that uninitialized contents are never consumed.
   *
   * Write and read are independent on purpose. On the final capture
   * MAC, the last block is written while block 0 is prefetched.
   */
  always_ff @(posedge clk_i) begin : weight_bram
    if (weight_capture_en) begin
      weight_buffer0[weight_block_cnt_q] <= req_o.req.rs[0];
      weight_buffer1[weight_block_cnt_q] <= req_o.req.rs[1];
      weight_buffer2[weight_block_cnt_q] <= req_o.req.rs[3];
      weight_buffer3[weight_block_cnt_q] <= req_o.req.rs[4];
    end

    if (weight_prefetch_en) begin
      weight_rd0_q <= weight_buffer0[weight_prefetch_addr];
      weight_rd1_q <= weight_buffer1[weight_prefetch_addr];
      weight_rd2_q <= weight_buffer2[weight_prefetch_addr];
      weight_rd3_q <= weight_buffer3[weight_prefetch_addr];
    end
  end

  // Execution-side buffer, weight-buffer, accumulator and packing state.
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      active_blocks_q <= 5'd1;
      wr_block_cnt_q <= 5'd0;
      rd_block_cnt_q <= 5'd0;
      acc_q <= '0;
      output_pack_q <= '0;

      weight_buffer_valid_q <= 1'b0;
      weight_block_cnt_q <= 10'd0;

      for (int i = 0; i < INPUT_BUF_WORDS; i++) begin
        input_buffer0[i] <= '0;
        input_buffer1[i] <= '0;
        input_buffer2[i] <= '0;
        input_buffer3[i] <= '0;
      end
    end else begin
      /*
       * Stage 1 input/buffer state updates.
       *
       * These input/buffer updates used to occur on x_result handshake. They must now
       * occur when Stage 1 consumes req_o, otherwise the following MAC would
       * see stale input/weight-buffer addresses and the pipeline could not
       * accept one instruction per cycle.
       */
      if (stage1_fire) begin
        if (is_buf4_ex) begin
          active_blocks_q <= buf_active_blocks;
          output_pack_q <= '0;

          if ((buf_active_blocks == CONV2_ACTIVE_BLOCKS) ||
              (buf_active_blocks == CONV1_ACTIVE_BLOCKS)) begin
            weight_block_cnt_q    <= 10'd0;
            weight_buffer_valid_q <= 1'b0;
          end

          if (wr_block_sel == (buf_active_blocks - 5'd1)) begin
            wr_block_cnt_q <= 5'd0;
          end else begin
            wr_block_cnt_q <= wr_block_sel + 5'd1;
          end

          if (buf_active_blocks != active_blocks_q) begin
            rd_block_cnt_q <= 5'd0;
          end
        end else if (is_mac16buf_ex || is_mac16buf_para_ex) begin
          if (is_mac16buf_para_ex) begin
            input_buffer0[wr_block_sel] <= req_o.req.rs[5];
            input_buffer1[wr_block_sel] <= req_o.req.rs[6];
            input_buffer2[wr_block_sel] <= req_o.req.rs[7];
            input_buffer3[wr_block_sel] <= req_o.req.rs[8];

            if (req_o.is_final_block) begin
              wr_block_cnt_q <= 5'd0;
            end else begin
              wr_block_cnt_q <= wr_block_sel + 5'd1;
            end
          end

          // Weight-buffer address/state tracking follows Stage 1 acceptance.
          if (weight_buffer_mode) begin
            if (weight_block_cnt_q == weight_last_block) begin
              weight_block_cnt_q <= 10'd0;
              if (!weight_buffer_valid_q) begin
                weight_buffer_valid_q <= 1'b1;
              end
            end else begin
              weight_block_cnt_q <= weight_block_cnt_q + 10'd1;
            end
          end

          if (req_o.is_final_block) begin
            rd_block_cnt_q <= 5'd0;
          end else begin
            rd_block_cnt_q <= rd_block_cnt_q + 5'd1;
          end
        end
      end

      /*
       * Stage 2 state update.
       *
       * The accumulation recurrence lives here.  With one Stage-2 MAC accepted
       * every cycle, acc_q is updated on the edge and is therefore available to
       * the following middle/final MAC on the next cycle (II=1).
       */
      if (stage2_fire &&
          (result_pipe_req_q.is_mac16buf || result_pipe_req_q.is_mac16buf_para)) begin
        acc_q <= mac_next_acc;

        // Packed-output state belongs to the Stage-2 instruction.
        if (result_pipe_req_q.is_final_block &&
            result_pipe_req_q.postprocessed_en &&
            result_pipe_req_q.pack4_en) begin
          if (result_pipe_req_q.resp.writeback) begin
            output_pack_q <= '0;
          end else begin
            output_pack_q <= output_pack_next;
          end
        end
      end
    end
  end

  // --------------------------------------------------------------------------
  // Stage 1: MAC16 datapath and reduction tree
  // --------------------------------------------------------------------------
  logic signed [15:0] p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15;
  logic signed [31:0] input1, input2, input3, input4, weight1, weight2, weight3, weight4;

  always_comb begin
    weight1 = '0;
    weight2 = '0;
    weight3 = '0;
    weight4 = '0;
    input1  = '0;
    input2  = '0;
    input3  = '0;
    input4  = '0;
    p0      = '0;
    p1      = '0;
    p2      = '0;
    p3      = '0;
    p4      = '0;
    p5      = '0;
    p6      = '0;
    p7      = '0;
    p8      = '0;
    p9      = '0;
    p10     = '0;
    p11     = '0;
    p12     = '0;
    p13     = '0;
    p14     = '0;
    p15     = '0;
    partial_sum  = '0;

    if (is_mac16buf_ex || is_mac16buf_para_ex) begin
      if (use_weight_buffer) begin
        /*
         * Reuse phase: consume the weight block prefetched by the
         * previous completed MAC. No combinational RAM read remains
         * on the MAC critical path.
         */
        weight1 = $signed(weight_rd0_q);
        weight2 = $signed(weight_rd1_q);
        weight3 = $signed(weight_rd2_q);
        weight4 = $signed(weight_rd3_q);
      end else begin
        weight1 = $signed(req_o.req.rs[0]);
        weight2 = $signed(req_o.req.rs[1]);
        weight3 = $signed(req_o.req.rs[3]);
        weight4 = $signed(req_o.req.rs[4]);
      end

      if (is_mac16buf_para_ex) begin
        input1 = $signed(req_o.req.rs[5]);
        input2 = $signed(req_o.req.rs[6]);
        input3 = $signed(req_o.req.rs[7]);
        input4 = $signed(req_o.req.rs[8]);
      end else begin
        input1 = $signed(input_buffer0[rd_block_cnt_q]);
        input2 = $signed(input_buffer1[rd_block_cnt_q]);
        input3 = $signed(input_buffer2[rd_block_cnt_q]);
        input4 = $signed(input_buffer3[rd_block_cnt_q]);
      end

      p0 = $signed({1'b0, input1[7:0]}) * $signed(weight1[7:0]);
      p1 = $signed({1'b0, input1[15:8]}) * $signed(weight1[15:8]);
      p2 = $signed({1'b0, input1[23:16]}) * $signed(weight1[23:16]);
      p3 = $signed({1'b0, input1[31:24]}) * $signed(weight1[31:24]);
      p4 = $signed({1'b0, input2[7:0]}) * $signed(weight2[7:0]);
      p5 = $signed({1'b0, input2[15:8]}) * $signed(weight2[15:8]);
      p6 = $signed({1'b0, input2[23:16]}) * $signed(weight2[23:16]);
      p7 = $signed({1'b0, input2[31:24]}) * $signed(weight2[31:24]);
      p8 = $signed({1'b0, input3[7:0]}) * $signed(weight3[7:0]);
      p9 = $signed({1'b0, input3[15:8]}) * $signed(weight3[15:8]);
      p10 = $signed({1'b0, input3[23:16]}) * $signed(weight3[23:16]);
      p11 = $signed({1'b0, input3[31:24]}) * $signed(weight3[31:24]);
      p12 = $signed({1'b0, input4[7:0]}) * $signed(weight4[7:0]);
      p13 = $signed({1'b0, input4[15:8]}) * $signed(weight4[15:8]);
      p14 = $signed({1'b0, input4[23:16]}) * $signed(weight4[23:16]);
      p15 = $signed({1'b0, input4[31:24]}) * $signed(weight4[31:24]);

      partial_sum = 32'(p0)+ 32'(p1)+ 32'(p2)+ 32'(p3) + 32'(p4) + 32'(p5) + 32'(p6) + 32'(p7) + 32'(p8) 
                    + 32'(p9) + 32'(p10) + 32'(p11) + 32'(p12) + 32'(p13) + 32'(p14) + 32'(p15);

    end
  end

  // --------------------------------------------------------------------------
  // Stage 1 -> Stage 2 pipeline register
  // --------------------------------------------------------------------------
  // This is the timing cut: Stage 1 ends at
  // partial_sum, while Stage 2 starts from partial_sum_pipe_q.
  always_ff @(posedge clk_i or negedge rst_ni) begin : result_pipeline_reg
    if (!rst_ni) begin
      result_pipe_valid_q <= 1'b0;
      result_pipe_req_q   <= '0;
      partial_sum_pipe_q  <= '0;
    end else if (x_commit_i.x_commit_kill && x_commit_valid_i) begin
      result_pipe_valid_q <= 1'b0;
    end else if (result_pipe_ready) begin
      result_pipe_valid_q <= stage1_fire;

      if (stage1_fire) begin
        result_pipe_req_q  <= req_o;
        partial_sum_pipe_q <= (is_mac16buf_ex || is_mac16buf_para_ex)
                            ? partial_sum
                            : 32'sd0;
      end
    end
  end


  // --------------------------------------------------------------------------
  // CV-X-IF result interface
  // --------------------------------------------------------------------------
  always_comb begin
    x_result_o.data = (result_pipe_req_q.is_mac16buf ||
                       result_pipe_req_q.is_mac16buf_para)
                    ? mac_writeback_data
                    : '0;

    x_result_o.id = result_pipe_req_q.req.id;
    x_result_o.rd = result_pipe_req_q.req.instr[11:7];

    // Stage 2 architectural writeback.  Metadata was captured together with
    // the MAC result, so it always belongs to the same instruction.
    x_result_o.we = result_pipe_req_q.resp.writeback
                  & x_result_valid_o
                  & (result_pipe_req_q.is_mac16buf ||
                     result_pipe_req_q.is_mac16buf_para)
                  & result_pipe_req_q.is_final_block;

    x_result_o.exc     = 1'b0;
    x_result_o.exccode = '0;
  end

endmodule