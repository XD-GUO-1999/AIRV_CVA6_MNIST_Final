// Copyright 2021 Thales DIS design services SAS
//
// Licensed under the Solderpad Hardware Licence, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.0
// You may obtain a copy of the License at https://solderpad.org/licenses/
//
// Original Author: Guillaume Chauvon (guillaume.chauvon@thalesgroup.com)
// Example coprocessor adds rs1,rs2(,rs3) together and gives back the result to the CPU via the CoreV-X-Interface.
// Coprocessor delays the sending of the result depending on result least significant bits.

module cvxif_example_coprocessor
  import cvxif_pkg::*;
  import cvxif_instr_pkg::*;
(
    input  logic        clk_i,        // Clock
    input  logic        rst_ni,       // Asynchronous reset active low
    input  cvxif_req_t  cvxif_req_i,
    output cvxif_resp_t cvxif_resp_o
);

  //Compressed interface
  logic               x_compressed_valid_i;
  logic               x_compressed_ready_o;
  x_compressed_req_t  x_compressed_req_i;
  x_compressed_resp_t x_compressed_resp_o;
  //Issue interface
  logic               x_issue_valid_i;
  logic               x_issue_ready_o;
  x_issue_req_t       x_issue_req_i;
  x_issue_resp_t      x_issue_resp_o;
  x_issue_resp_t      x_issue_resp_dec; // modification: decoded issue response from the table
  // Commit interface
  logic               x_commit_valid_i;
  x_commit_t          x_commit_i;
  //Memory interface
  logic               x_mem_valid_o;
  logic               x_mem_ready_i;
  x_mem_req_t         x_mem_req_o;
  x_mem_resp_t        x_mem_resp_i;
  //Memory result interface
  logic               x_mem_result_valid_i;
  x_mem_result_t      x_mem_result_i;
  //Result interface
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

  //Compressed interface
  assign x_compressed_ready_o            = '0;
  assign x_compressed_resp_o.instr       = '0;
  assign x_compressed_resp_o.accept      = '0;

  instr_decoder #(
      .NbInstr   (cvxif_instr_pkg::NbInstr),
      .CoproInstr(cvxif_instr_pkg::CoproInstr)
  ) instr_decoder_i (
      .clk_i         (clk_i),
      .x_issue_req_i (x_issue_req_i),
      .x_issue_resp_o(x_issue_resp_dec)
  );

  // modification: extend the issue entry with MAC16BUF/BUF4/MAC16BUF_PARA block tracking flags
  typedef struct packed {
    x_issue_req_t  req;
    x_issue_resp_t resp;
    logic          is_buf4;
    logic          is_mac16buf;
    logic          is_mac16buf_para;
    logic          is_first_block;
    logic          is_final_block;

    //modification: post processed before write back to the CPU
    logic postprocessed_en;
  } x_issue_t;


  logic fifo_full, fifo_empty;
  logic x_issue_ready_q;
  logic instr_push, instr_pop;
  x_issue_t req_i;
  x_issue_t req_o;

  // modification: issue-side block counter and flags for MAC16BUF/BUF4 processing
  // It is used only to decide whether a MAC16BUF instruction should request
  // an architectural writeback. The actual accumulation is still performed
  // later when the instruction reaches the FIFO output/result stage.
  logic [4:0] issue_active_blocks_q;
  logic [4:0] issue_block_cnt_q;
  logic [4:0] issue_buf_active_blocks;
  logic       issue_is_buf4;
  logic       issue_is_mac16buf;
  logic       issue_is_mac16buf_para; //add new instruction which can do both buffer and mac
  logic       issue_mac_op; //mac16buf and mac16buf_para
  logic       issue_is_first_block;
  logic       issue_is_final_block;

  //modification: post processed before write back to the CPU
  logic       issue_postprocessed_en;
  assign issue_postprocessed_en = issue_mac_op && ((issue_active_blocks_q == 5'd1) || (issue_active_blocks_q == 5'd25) || (issue_active_blocks_q == 5'd24));

  // modification: detect whether the incoming issue request is BUF4 or MAC16BUF or MAC16BUF_PARA
  assign issue_is_buf4          = (x_issue_req_i.instr[6:0] == 7'b0101011);
  assign issue_is_mac16buf      = (x_issue_req_i.instr[6:0] == 7'b0001011);
  assign issue_is_mac16buf_para = (x_issue_req_i.instr[6:0] == 7'b1011011);

  assign issue_mac_op = issue_is_mac16buf || issue_is_mac16buf_para;
  assign issue_buf_active_blocks = x_issue_req_i.instr[11:7] + 5'd1;
  assign issue_is_first_block   = issue_mac_op && (issue_block_cnt_q == 5'd0);
  assign issue_is_final_block   = issue_mac_op && (issue_block_cnt_q == (issue_active_blocks_q - 5'd1));

  // Start from the table decoder response, then override only MAC16BUF.writeback.
  // A MAC16BUF is CPU-visible only for the final block of one output element.
  // Non-final MAC16BUF instructions still complete through x_result_valid, but
  // they do not write the register file.
  // modification: override the decoded issue response for MAC16BUF writeback semantics
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

  // modification: stash MAC16BUF/BUF4 metadata in the FIFO entry for later processing
  assign req_i.req            = x_issue_req_i;
  assign req_i.resp           = x_issue_resp_o;
  assign req_i.is_buf4        = issue_is_buf4;
  assign req_i.is_mac16buf    = issue_is_mac16buf;
  assign req_i.is_mac16buf_para = issue_is_mac16buf_para;
  assign req_i.is_first_block = issue_is_first_block;
  assign req_i.is_final_block = issue_is_final_block;
  assign req_i.postprocessed_en  = issue_postprocessed_en;

  // modification: track issue-side buffer block counters for MAC16BUF/BUF4 execution
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

  fifo_v3 #(
      .FALL_THROUGH(1),         //data_o ready and pop in the same cycle
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

  localparam int unsigned  INPUT_BUF_WORDS = 25; 
  // Add weight buffer for Conv2/Conv1
  localparam logic [4:0] CONV1_ACTIVE_BLOCKS = 5'd1;
  localparam logic [9:0] CONV1_WEIGHT_LAST = 10'd15;

  localparam logic [4:0] CONV2_ACTIVE_BLOCKS = 5'd25;  
  localparam logic [9:0] CONV2_WEIGHT_LAST = 10'd599;

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
 /////

  logic signed [31:0] acc_q;
  logic signed [31:0] partial_sum;
  logic signed [31:0] mac_base_acc;
  logic signed [31:0] mac_next_acc;
  logic        [7:0] sat_result_u8;
  logic signed [31:0] mac_writeback_data;

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

  always_comb begin
      sat_result_u8 = sat_shift8_u8(mac_next_acc);

      /*
      * Default:
      * return full 32-bit accumulator.
      *
      * This is important for FC2 because its final MAC16BUF
      * is followed by 6 scalar MAC operations in software.
      */
      mac_writeback_data = mac_next_acc;

      /*
      * Conv1 / Conv2 / FC1:
      * only the final block returns the post-processed 8-bit result.
      */
      if (req_o.is_final_block && req_o.postprocessed_en) begin
          mac_writeback_data = {24'd0, sat_result_u8};
      end
  end

  logic [31:0] input_buffer0 [0:INPUT_BUF_WORDS-1];
  logic [31:0] input_buffer1 [0:INPUT_BUF_WORDS-1];
  logic [31:0] input_buffer2 [0:INPUT_BUF_WORDS-1];
  logic [31:0] input_buffer3 [0:INPUT_BUF_WORDS-1];

  logic [4:0] wr_block_cnt_q;
  logic [4:0] rd_block_cnt_q;
  logic [4:0] active_blocks_q;

  //set the mode of weight buffer of conv2
  assign conv1_weight_mode = (active_blocks_q == CONV1_ACTIVE_BLOCKS);
  assign conv2_weight_mode = (active_blocks_q == CONV2_ACTIVE_BLOCKS);
  assign weight_buffer_mode = conv1_weight_mode || conv2_weight_mode;
  assign use_weight_buffer = weight_buffer_mode && weight_buffer_valid_q;
  assign weight_last_block = conv1_weight_mode ? CONV1_WEIGHT_LAST : CONV2_WEIGHT_LAST;
    ///

  logic [4:0] buf_active_blocks;
  logic [4:0] wr_block_sel;
  // logic [6:0] wr_base;
  // logic [6:0] rd_base;

  assign buf_active_blocks = req_o.req.instr[11:7] + 5'd1;
  
  // assign wr_base = {wr_block_sel, 2'b00};
  // assign rd_base = {rd_block_cnt_q, 2'b00};


  logic is_buf4_ex;
  logic is_mac16buf_ex;
  logic is_mac16buf_para_ex;

  assign is_buf4_ex     = req_o.is_buf4;
  assign is_mac16buf_ex = req_o.is_mac16buf;
  assign is_mac16buf_para_ex = req_o.is_mac16buf_para;

  /*
   * Keep the original CV-X-IF result timing.
   *
   * The one-cycle synchronous BRAM latency is hidden by prefetching
   * the NEXT weight block, so no extra x_result_valid wait state is
   * introduced.
   */
  assign x_result_valid_o = ~fifo_empty && ~x_commit_i.x_commit_kill;

  /* A MAC really completes only when the result handshake occurs. */
  assign mac_done =
      x_result_valid_o
      && x_result_ready_i
      && (is_mac16buf_ex || is_mac16buf_para_ex);

  /*
   * Capture phase: use CPU weight operands directly for the MAC and
   * simultaneously write them into the local weight BRAM.
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

  // modification: buffer write state machine for BUF4 instructions
  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      active_blocks_q <= 5'd1;
      wr_block_cnt_q <= 5'd0;
      rd_block_cnt_q <= 5'd0;
      acc_q <= '0;

      //keep the weight buffer at the moment
      weight_buffer_valid_q <= 1'b0;
      weight_block_cnt_q <= 10'd0;

      for (int i = 0; i < INPUT_BUF_WORDS; i++) begin
        input_buffer0[i] <= '0;
        input_buffer1[i] <= '0;
        input_buffer2[i] <= '0;
        input_buffer3[i] <= '0;
      end
    end else if (x_result_valid_o && x_result_ready_i) begin
      if(is_buf4_ex) begin
        active_blocks_q <= buf_active_blocks;
        //enter the weight buffer
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
          rd_block_cnt_q <= 5'd0; // Reset read block counter if active blocks change
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
        ////

        // modification: auto local accumulator for MAC16BUF blocks
        //   first block : acc = old rd + partial_sum
        //   middle      : acc = acc_q + partial_sum
        //   final       : acc = acc_q + partial_sum, then write back
        acc_q <= mac_next_acc;

        if (req_o.is_final_block) begin
          rd_block_cnt_q <= 5'd0;
        end else begin
          rd_block_cnt_q <= rd_block_cnt_q + 5'd1;
        end
      end
    end
  end

  // modification: MAC16 parallel multiply-accumulate logic (for MAC16BUF)
  //logic signed [31:0] mac_result;
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
                    + 32'(p9) + 32'(p10) + 32'(p11) + 32'(p12) + 32'(p13) + 32'(p14) + 32'(p15); // we add the result together

      mac_base_acc = req_o.is_first_block ? $signed(req_o.req.rs[2]) : acc_q;
      mac_next_acc = mac_base_acc + partial_sum;
    end
  end


  always_comb begin
    x_result_o.data    = (is_mac16buf_ex || is_mac16buf_para_ex) ? mac_writeback_data : '0;
    x_result_o.id      = req_o.req.id;
    x_result_o.rd      = req_o.req.instr[11:7];

    // modification: for auto-accumulator mode, only the final MAC16BUF block writes back
    // Non-final MAC16BUF instructions only update acc_q locally.
    x_result_o.we = req_o.resp.writeback & x_result_valid_o & (is_mac16buf_ex || is_mac16buf_para_ex) & req_o.is_final_block;
    x_result_o.exc     = 1'b0;
    x_result_o.exccode = '0;
  end

endmodule