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
//   MAC16BUF_PARA : MAC16 using CPU-provided input words while simultaneously
//                   filling the input buffer.
//   MAC16BUF      : MAC16 using the buffered input words.
//
// Conv1/Conv2 weights are captured during their first spatial output position
// and then reused from a four-bank BRAM weight buffer.  Multi-block dot products
// use a local first/middle/final accumulator so only the final block writes back.
// Conv1/Conv2/FC1 also receive ReLU + >>8 + u8 saturation in hardware; FC2
// returns a full 32-bit accumulator because software still adds six scalar MACs.

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
  // Accelerator layer configuration
  // --------------------------------------------------------------------------
  // Active-block counts are programmed by BUF4 and are also used to select
  // layer-specific hardware behavior.
  localparam logic [4:0] CONV1_ACTIVE_BLOCKS = 5'd1;
  localparam logic [4:0] CONV2_ACTIVE_BLOCKS = 5'd25;
  localparam logic [4:0] FC1_ACTIVE_BLOCKS   = 5'd24;

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

  // Extend each FIFO entry with accelerator metadata required at execution/result time.
  typedef struct packed {
    x_issue_req_t  req;
    x_issue_resp_t resp;
    logic          is_buf4;
    logic          is_mac16buf;
    logic          is_mac16buf_para;
    logic          is_first_block;
    logic          is_final_block;

    // Apply layer-specific post-processing before architectural write-back.
    logic postprocessed_en;
  } x_issue_t;


  logic fifo_full, fifo_empty;
  logic x_issue_ready_q;
  logic instr_push, instr_pop;
  x_issue_t req_i;
  x_issue_t req_o;

  // --------------------------------------------------------------------------
  // Issue-stage block tracking
  // --------------------------------------------------------------------------
  // The issue-side counters tag each MAC instruction as first/middle/final.
  // These tags determine architectural write-back behavior; the actual MAC
  // accumulation is performed later at the FIFO output/result stage.
  logic [4:0] issue_active_blocks_q;
  logic [4:0] issue_block_cnt_q;
  logic [4:0] issue_buf_active_blocks;
  logic       issue_is_buf4;
  logic       issue_is_mac16buf;
  logic       issue_is_mac16buf_para; // MAC16 plus input-buffer fill
  logic       issue_mac_op; // MAC16BUF or MAC16BUF_PARA
  logic       issue_is_first_block;
  logic       issue_is_final_block;

  // Enable hardware ReLU + >>8 + u8 saturation for Conv1 (1 block),
  // Conv2 (25 blocks) and FC1 (24 blocks).  FC2 uses 9 blocks and stays full-precision.
  logic       issue_postprocessed_en;
  assign issue_postprocessed_en = issue_mac_op
      && ((issue_active_blocks_q == CONV1_ACTIVE_BLOCKS)
          || (issue_active_blocks_q == CONV2_ACTIVE_BLOCKS)
          || (issue_active_blocks_q == FC1_ACTIVE_BLOCKS));

  // Detect accelerator instructions from the opcode field.
  assign issue_is_buf4          = (x_issue_req_i.instr[6:0] == 7'b0101011);
  assign issue_is_mac16buf      = (x_issue_req_i.instr[6:0] == 7'b0001011);
  assign issue_is_mac16buf_para = (x_issue_req_i.instr[6:0] == 7'b1011011);

  assign issue_mac_op = issue_is_mac16buf || issue_is_mac16buf_para;
  assign issue_buf_active_blocks = x_issue_req_i.instr[11:7] + 5'd1;
  assign issue_is_first_block   = issue_mac_op && (issue_block_cnt_q == 5'd0);
  assign issue_is_final_block   = issue_mac_op && (issue_block_cnt_q == (issue_active_blocks_q - 5'd1));

  // Start from the table-decoder response and override write-back only for
  // accelerator MAC instructions. First/middle blocks complete through the
  // CV-X-IF result handshake but do not write the architectural register file;
  // only the final block writes back.
  always_comb begin
    x_issue_resp_o = x_issue_resp_dec;
    if (issue_mac_op && x_issue_resp_dec.accept) begin
      x_issue_resp_o.writeback = issue_is_final_block;
    end
  end

  assign instr_push = x_issue_valid_i && x_issue_ready_o && x_issue_resp_o.accept;
  assign instr_pop  = (x_commit_i.x_commit_kill && x_commit_valid_i) ||
                      (x_result_valid_o && x_result_ready_i);
  assign x_issue_ready_q = ~fifo_full;

  // Store accelerator metadata in the FIFO so issue-stage decisions remain
  // associated with the instruction until execution/result time.
  assign req_i.req            = x_issue_req_i;
  assign req_i.resp           = x_issue_resp_o;
  assign req_i.is_buf4        = issue_is_buf4;
  assign req_i.is_mac16buf    = issue_is_mac16buf;
  assign req_i.is_mac16buf_para = issue_is_mac16buf_para;
  assign req_i.is_first_block = issue_is_first_block;
  assign req_i.is_final_block = issue_is_final_block;
  assign req_i.postprocessed_en  = issue_postprocessed_en;

  // Advance/reset the issue-side block counter as instructions are accepted.
  always_ff @(posedge clk_i or negedge rst_ni) begin : issue_block_counter
    if (!rst_ni) begin
      issue_active_blocks_q <= 5'd1;
      issue_block_cnt_q     <= 5'd0;
    end else if (instr_push) begin
      if (issue_is_buf4) begin
        issue_active_blocks_q <= issue_buf_active_blocks;
        issue_block_cnt_q     <= 5'd0;
      end else if (issue_is_mac16buf || issue_is_mac16buf_para) begin
        if (issue_is_final_block) begin
          issue_block_cnt_q <= 5'd0;
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
      .FALL_THROUGH(1),         // data_o is available in the same cycle as pop
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

  // --------------------------------------------------------------------------
  // Accelerator execution state
  // --------------------------------------------------------------------------
  // The accelerator state is grouped by function below so that each storage
  // structure sits next to its size, pointers, validity flags and control.

  // Execution-stage aliases from the FIFO entry.
  logic is_buf4_ex;
  logic is_mac16buf_ex;
  logic is_mac16buf_para_ex;

  assign is_buf4_ex          = req_o.is_buf4;
  assign is_mac16buf_ex      = req_o.is_mac16buf;
  assign is_mac16buf_para_ex = req_o.is_mac16buf_para;

  // --------------------------------------------------------------------------
  // Input buffer: 4 x 32-bit banks x 25 entries = 25 x 16-byte blocks = 400 B
  // --------------------------------------------------------------------------
  localparam int unsigned INPUT_BUF_DEPTH = 25;

  logic [31:0] input_buffer0 [0:INPUT_BUF_DEPTH-1];
  logic [31:0] input_buffer1 [0:INPUT_BUF_DEPTH-1];
  logic [31:0] input_buffer2 [0:INPUT_BUF_DEPTH-1];
  logic [31:0] input_buffer3 [0:INPUT_BUF_DEPTH-1];

  logic [4:0] active_blocks_q;
  logic [4:0] wr_block_cnt_q;
  logic [4:0] rd_block_cnt_q;
  logic [4:0] buf_active_blocks;
  logic [4:0] wr_block_sel;

  // BUF4 encodes active_blocks - 1 in rd.
  assign buf_active_blocks = req_o.req.instr[11:7] + 5'd1;

  // A layer change restarts input-buffer filling at block 0.  PARA first-block
  // operations also restart the write pointer before capturing a new patch.
  assign wr_block_sel = (is_buf4_ex && (buf_active_blocks != active_blocks_q)) ||
                        (is_mac16buf_para_ex && req_o.is_first_block) ? 5'd0 : wr_block_cnt_q;

  // --------------------------------------------------------------------------
  // Weight buffer: 4 x 32-bit banks x 600 entries = 600 x 16-byte blocks
  // --------------------------------------------------------------------------
  // Conv1 stores 16 blocks = 256 B. Conv2 stores 600 blocks = 9.6 kB.
  // FC1 and FC2 do not use the weight buffer.
  localparam logic [9:0] CONV1_WEIGHT_LAST = 10'd15;
  localparam logic [9:0] CONV2_WEIGHT_LAST = 10'd599;
  localparam int unsigned WEIGHT_BUF_DEPTH   = 600;

  (* ram_style = "block" *)
  logic [31:0] weight_buffer0 [0:WEIGHT_BUF_DEPTH-1];
  (* ram_style = "block" *)
  logic [31:0] weight_buffer1 [0:WEIGHT_BUF_DEPTH-1];
  (* ram_style = "block" *)
  logic [31:0] weight_buffer2 [0:WEIGHT_BUF_DEPTH-1];
  (* ram_style = "block" *)
  logic [31:0] weight_buffer3 [0:WEIGHT_BUF_DEPTH-1];

  // Registered outputs from synchronous BRAM reads.
  logic [31:0] weight_rd0_q;
  logic [31:0] weight_rd1_q;
  logic [31:0] weight_rd2_q;
  logic [31:0] weight_rd3_q;

  logic [9:0] weight_block_cnt_q;
  logic       weight_buffer_valid_q;
  logic       conv1_weight_mode;
  logic       conv2_weight_mode;
  logic       weight_buffer_mode;
  logic       use_weight_buffer;
  logic [9:0] weight_last_block;

  // BRAM capture/prefetch control.
  logic       mac_done;
  logic       weight_capture_en;
  logic       weight_prefetch_en;
  logic [9:0] weight_prefetch_addr;

  // Only Conv1 and Conv2 reuse weights from the local BRAM.
  assign conv1_weight_mode = (active_blocks_q == CONV1_ACTIVE_BLOCKS);
  assign conv2_weight_mode = (active_blocks_q == CONV2_ACTIVE_BLOCKS);
  assign weight_buffer_mode = conv1_weight_mode || conv2_weight_mode;
  assign use_weight_buffer = weight_buffer_mode && weight_buffer_valid_q;
  assign weight_last_block = conv1_weight_mode ? CONV1_WEIGHT_LAST : CONV2_WEIGHT_LAST;

  // --------------------------------------------------------------------------
  // Local accumulator and final-layer post-processing
  // --------------------------------------------------------------------------
  logic signed [31:0] acc_q;
  logic signed [31:0] partial_sum;
  logic signed [31:0] mac_base_acc;
  logic signed [31:0] mac_next_acc;
  logic        [7:0]  sat_result_u8;
  logic signed [31:0] mac_writeback_data;

  function automatic logic [7:0] sat_shift8_u8(
      input logic signed [31:0] value
  );
  begin
    // Equivalent to ReLU(value), value >> 8, clamp(value, 0, 255).
    if (value[31]) begin
      sat_shift8_u8 = 8'd0;
    end else if (|value[30:16]) begin
      sat_shift8_u8 = 8'd255;
    end else begin
      sat_shift8_u8 = value[15:8];
    end
  end
  endfunction

  // Conv1/Conv2/FC1 return the post-processed 8-bit result on the final block.
  // FC2 returns the full 32-bit accumulator because software still adds six
  // scalar MAC terms before applying sat().
  always_comb begin
    sat_result_u8    = sat_shift8_u8(mac_next_acc);
    mac_writeback_data = mac_next_acc;

    if (req_o.is_final_block && req_o.postprocessed_en) begin
      mac_writeback_data = {24'd0, sat_result_u8};
    end
  end

  // --------------------------------------------------------------------------
  // Result handshake and weight-buffer capture/prefetch control
  // --------------------------------------------------------------------------
  // Keep the original CV-X-IF result timing. The synchronous BRAM read latency
  // is hidden by prefetching the next weight block.
  assign x_result_valid_o = ~fifo_empty && ~x_commit_i.x_commit_kill;

  assign mac_done = x_result_valid_o
                 && x_result_ready_i
                 && (is_mac16buf_ex || is_mac16buf_para_ex);

  // During the first Conv1/Conv2 position, CPU weight operands feed the MAC and
  // are captured into BRAM at the same time.
  assign weight_capture_en = mac_done
                          && weight_buffer_mode
                          && !weight_buffer_valid_q;

  // During reuse, completed block n prefetches block n+1.  The final capture
  // block prefetches block 0 so the first reuse MAC can start immediately.
  assign weight_prefetch_en = mac_done
                           && weight_buffer_mode
                           && (weight_buffer_valid_q
                               || (!weight_buffer_valid_q
                                   && (weight_block_cnt_q == weight_last_block)));

  always_comb begin
    if (weight_block_cnt_q == weight_last_block) begin
      weight_prefetch_addr = 10'd0;
    end else begin
      weight_prefetch_addr = weight_block_cnt_q + 10'd1;
    end
  end

  // Synchronous BRAM: arrays are intentionally not reset.  The validity flag
  // guarantees that uninitialized contents are never consumed.
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

  // --------------------------------------------------------------------------
  // Accelerator state update
  // --------------------------------------------------------------------------
  // Update input-buffer pointers, weight-buffer state, and the local accumulator
  // only when the current CV-X-IF result is accepted.
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      active_blocks_q <= 5'd1;
      wr_block_cnt_q <= 5'd0;
      rd_block_cnt_q <= 5'd0;
      acc_q <= '0;

      // Weight-buffer contents are not reset; validity controls their use.
      weight_buffer_valid_q <= 1'b0;
      weight_block_cnt_q <= 10'd0;

      for (int i = 0; i < INPUT_BUF_DEPTH; i++) begin
        input_buffer0[i] <= '0;
        input_buffer1[i] <= '0;
        input_buffer2[i] <= '0;
        input_buffer3[i] <= '0;
      end
    end else if (x_result_valid_o && x_result_ready_i) begin
      if(is_buf4_ex) begin
        active_blocks_q <= buf_active_blocks;
        // Reconfigure weight capture when entering a buffered layer.
        if ((buf_active_blocks == CONV2_ACTIVE_BLOCKS) || (buf_active_blocks == CONV1_ACTIVE_BLOCKS))
        begin
          weight_block_cnt_q    <= 10'd0;
          weight_buffer_valid_q <= 1'b0;
        end

        if (wr_block_sel == (buf_active_blocks - 5'd1)) begin
          wr_block_cnt_q <= 5'd0;
        end else begin
          wr_block_cnt_q <= wr_block_sel + 5'd1;
        end

        if (buf_active_blocks != active_blocks_q) begin
          rd_block_cnt_q <= 5'd0; // Restart input-buffer reads after a mode change
        end
      end else if (is_mac16buf_ex || is_mac16buf_para_ex) begin
        if (is_mac16buf_para_ex) begin
          input_buffer0[wr_block_sel] <= req_o.req.rs[5];
          input_buffer1[wr_block_sel] <= req_o.req.rs[6];
          input_buffer2[wr_block_sel] <= req_o.req.rs[7];
          input_buffer3[wr_block_sel] <= req_o.req.rs[8];

          if(req_o.is_final_block) begin
            wr_block_cnt_q <= 5'd0;
          end else begin
            wr_block_cnt_q <= wr_block_sel + 5'd1;
          end
        end
        /*
         * Weight-buffer address/state tracking.
         * Actual RAM accesses are handled only by weight_bram above.
         */
        if (weight_buffer_mode) begin
          if (weight_block_cnt_q == weight_last_block) begin
            weight_block_cnt_q <= 10'd0;

            if (!weight_buffer_valid_q) begin
              weight_buffer_valid_q <= 1'b1;
            end
          end
          else begin
            weight_block_cnt_q <= weight_block_cnt_q + 10'd1;
          end
        end


        // Local accumulator:
        //   first  : initialize from the CPU rd/bias and add partial_sum;
        //   middle : continue from acc_q without architectural write-back;
        //   final  : produce the completed result for architectural write-back.
        acc_q <= mac_next_acc;

        if (req_o.is_final_block) begin
          rd_block_cnt_q <= 5'd0;
        end else begin
          rd_block_cnt_q <= rd_block_cnt_q + 5'd1;
        end
      end
    end
  end

  // --------------------------------------------------------------------------
  // MAC16 datapath
  // --------------------------------------------------------------------------
  // Sixteen unsigned-input x signed-weight 8-bit products are reduced into a
  // 32-bit partial sum and accumulated with either rd (first block) or acc_q.
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
    mac_base_acc = '0;
    mac_next_acc = '0;

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

      mac_base_acc = req_o.is_first_block ? $signed(req_o.req.rs[2]) : acc_q;
      mac_next_acc = mac_base_acc + partial_sum;
    end
  end


  // --------------------------------------------------------------------------
  // CV-X-IF result generation
  // --------------------------------------------------------------------------
  always_comb begin
    x_result_o.data    = (is_mac16buf_ex || is_mac16buf_para_ex) ? mac_writeback_data : '0;
    x_result_o.id      = req_o.req.id;
    x_result_o.rd      = req_o.req.instr[11:7];

    // Only the final accelerator MAC block writes the architectural register file.
    // First/middle blocks update only the local accumulator.
    x_result_o.we = req_o.resp.writeback & x_result_valid_o & (is_mac16buf_ex || is_mac16buf_para_ex) & req_o.is_final_block;
    x_result_o.exc     = 1'b0;
    x_result_o.exccode = '0;
  end

endmodule