// Copyright 2021 Thales DIS design services SAS
//
// Licensed under the Solderpad Hardware Licence, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// SPDX-License-Identifier: Apache-2.0 WITH SHL-2.0
// You may obtain a copy of the License at https://solderpad.org/licenses/
//
// Original Author: Guillaume Chauvon (guillaume.chauvon@thalesgroup.com)

// Final accelerator instruction table.  Only the 7-bit custom opcode is
// masked for these three entries; the remaining instruction fields are free to
// carry rd, rs1, rs2 and the two extra weight-register addresses.
package cvxif_instr_pkg;

  typedef struct packed {
    logic [31:0]              instr;
    logic [31:0]              mask;
    cvxif_pkg::x_issue_resp_t resp;
  } copro_issue_resp_t;

  // Custom instructions accepted by the MNIST accelerator coprocessor.
  // Only opcode[6:0] is matched; the remaining fields carry register indices.
  parameter int unsigned NbInstr = 3;
  parameter copro_issue_resp_t CoproInstr[NbInstr] = '{
      '{
          instr: 32'b00000_00_00000_00000_0_00_00000_0101011,  // custom-1: BUF4
          mask: 32'b00000_00_00000_00000_0_00_00000_1111111,
          resp : '{
              accept : 1'b1,
              writeback : 1'b0,
              dualwrite : 1'b0,
              dualread : 1'b0,
              loadstore : 1'b0,
              exc : 1'b0
          }
      },
      '{
          instr: 32'b00000_00_00000_00000_0_00_00000_0001011,  // custom-0: MAC16BUF
          mask: 32'b00000_00_00000_00000_0_00_00000_1111111,
          resp : '{
              accept : 1'b1,
              writeback : 1'b1,
              dualwrite : 1'b0,
              dualread : 1'b0,
              loadstore : 1'b0,
              exc : 1'b0
          }
        },
        '{
          instr: 32'b00000_00_00000_00000_0_00_00000_1011011,  // custom-2: MAC16BUF_PARA
          mask: 32'b00000_00_00000_00000_0_00_00000_1111111,
          resp : '{
              accept : 1'b1,
              writeback : 1'b1,
              dualwrite : 1'b0,
              dualread : 1'b0,
              loadstore : 1'b0,
              exc : 1'b0
          }
      }
  };

endpackage
