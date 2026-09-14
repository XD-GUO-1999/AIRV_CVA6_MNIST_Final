# AIRV CVA6 MNIST Accelerator — User Guide

This guide explains how to configure the environment, compile the MNIST application, run RTL simulation with Questa, and execute the accelerator on the Zybo Z7-20 FPGA platform.

The repository contains the complete AIRV project, including the modified RISC-V toolchain used to assemble the custom instructions (`mac16buf`, `buf4`, and `mac16buf_para`).

---

## 1. Project Structure

The main project directory is:

```text
cva6_mac16_wtbuffer_v1/
```

Important subdirectories include:

```text
cva6_mac16_wtbuffer_v1/
├── core/                         # CVA6 RTL and CV-X-IF accelerator
├── sw/
│   └── app/                      # Software applications, including MNIST
├── util/
│   ├── gcc-toolchain-builder/    # Modified RISC-V GNU toolchain
│   ├── riscv-opcodes/            # RISC-V instruction encoding definitions
│   └── openocd/                  # OpenOCD
└── setup.sh                      # Environment configuration script
```

The complete repository version already contains the modified RISC-V toolchain sources and the built `riscv_toolchain/` directory. A normal user therefore does **not** need to rebuild the toolchain before compiling MNIST.

---

## 2. Main Requirements

The validated development environment uses:

- Linux
- Xilinx Vivado / Vitis 2024.1
- QuestaSim
- RISC-V GNU toolchain included in the repository
- OpenOCD
- Zybo Z7-20 FPGA board for hardware execution
- Digilent JTAG-HS2 cable
- Pmod USB-UART module

For RTL-only simulation, the FPGA board is not required.

---

## 3. Environment Setup

The project uses a `setup.sh` script to configure the main environment variables and tool paths.

### 3.1 Current `setup.sh`

```bash
#!/usr/bin/env bash

# =========================
# AIRV global environment
# =========================

# Project root
export PROJECTROOT=/home/imtuser/AIRV_TEST/AIRV_CVA6_MNIST_Final/cva6_mac16_wtbuffer_v1

# =========================
# Vivado / Vitis 2024.1
# =========================

if [ -f /opt/Xilinx/Vivado/2024.1/settings64.sh ]; then
    source /opt/Xilinx/Vivado/2024.1/settings64.sh
fi

# Add xsct if needed
export PATH="$PATH:/opt/Xilinx/Vivado/2024.1/xsct-trim/bin"

# =========================
# Questa
# =========================

# License server
export MGLS_LICENSE_FILE="29000@eda-lic.imta.fr"
export LM_LICENSE_FILE="29000@eda-lic.imta.fr"

# Questa installation path
export QUESTA_PATH="/opt/Questa/questasim"

# Questa binaries
export PATH="$PATH:$QUESTA_PATH/linux_x86_64"

# Questa GCC support, if installed
export PATH="$PATH:$QUESTA_PATH/gcc-64-linux_x86_64/bin"

# =========================
# OpenOCD
# =========================

export PATH="$PATH:$PROJECTROOT/util/openocd/build/bin"

# =========================
# RISC-V toolchain
# =========================

export RISCV=riscv_toolchain
export PATH="$PATH:$PROJECTROOT/util/gcc-toolchain-builder/riscv_toolchain/bin"

# =========================
# Quick information
# =========================

echo "AIRV environment loaded."
echo "PROJECTROOT = $PROJECTROOT"
echo "Vivado:  $(which vivado 2>/dev/null)"
echo "XSCT:    $(which xsct 2>/dev/null)"
echo "Questa:  $(which vsim 2>/dev/null)"
echo "OpenOCD: $(which openocd 2>/dev/null)"
echo "RISC-V:  $(which riscv-none-elf-gcc 2>/dev/null)"
```

### 3.2 Important: update `PROJECTROOT`

The `PROJECTROOT` path above corresponds to the validated development machine.

After cloning the repository on another machine, edit this line:

```bash
export PROJECTROOT=/home/imtuser/AIRV_TEST/AIRV_CVA6_MNIST_Final/cva6_mac16_wtbuffer_v1
```

and replace it with the absolute path of your local project.

For example:

```bash
export PROJECTROOT=/home/user/AIRV_CVA6_MNIST_Final/cva6_mac16_wtbuffer_v1
```

### 3.3 Load the environment

From a terminal:

```bash
source /path/to/setup.sh
```

The script should print something similar to:

```text
AIRV environment loaded.
PROJECTROOT = ...
Vivado:  /opt/Xilinx/...
XSCT:    /opt/Xilinx/...
Questa:  /opt/Questa/...
OpenOCD: .../util/openocd/build/bin/openocd
RISC-V:  .../util/gcc-toolchain-builder/riscv_toolchain/bin/riscv-none-elf-gcc
```

Check the environment manually if needed:

```bash
echo $PROJECTROOT
which vivado
which vsim
which openocd
which riscv-none-elf-gcc
```

### 3.4 Recommended toolchain PATH ordering

The tested script appends the project RISC-V toolchain to `PATH`.

If another `riscv-none-elf-gcc` is already installed on the machine, it is safer to force the project version to have priority by changing:

```bash
export PATH="$PATH:$PROJECTROOT/util/gcc-toolchain-builder/riscv_toolchain/bin"
```

to:

```bash
export PATH="$PROJECTROOT/util/gcc-toolchain-builder/riscv_toolchain/bin:$PATH"
```

Then verify:

```bash
which riscv-none-elf-gcc
```

The selected compiler should be the one inside this repository.

---

## 4. Compile the MNIST Application

Load the environment first:

```bash
source /path/to/setup.sh
```

Then:

```bash
cd $PROJECTROOT/sw/app
make mnist
```

The compilation should generate:

```text
mnist.riscv
```

If a clean rebuild is required:

```bash
make clean
make mnist
```

The MNIST software uses the custom accelerator instructions supported by the modified GNU assembler contained in the repository.

---

## 5. Run the MNIST RTL Simulation with Questa

After loading the environment:

```bash
cd $PROJECTROOT
make sim APP=mnist
```

This command performs three main operations:

1. builds the CVA6 RTL and testbench;
2. compiles the MNIST software with the RISC-V toolchain;
3. launches the RTL simulation in Questa.

The simulation can generate several useful files, including:

```text
vsim.wlf
trace_hart_0.log
uart
```

`vsim.wlf` contains the saved Questa waveform and can become very large.

The custom accelerator can be inspected in Questa through signals such as:

```text
issue_is_mac16buf
issue_block_cnt_q
first
final
acc_q
mac_base_acc
partial_sum
mac_next_acc
writeback
```

These signals are useful for checking:

- custom instruction issue;
- input/weight block progression;
- first / middle / final accumulation phases;
- local accumulator behavior;
- partial MAC16 result;
- final write-back.

---

## 6. FPGA Execution on Zybo Z7-20

### 6.1 Hardware connection

The validated setup uses:

1. Zybo Z7-20 power supply;
2. Pmod USB-UART module;
3. Digilent JTAG-HS2 cable.

After connecting the hardware, power on the board.

### 6.2 Generate the FPGA bitstream

Load the environment and run:

```bash
cd $PROJECTROOT
make cva6_fpga
```

### 6.3 Program the FPGA

Once the bitstream has been generated:

```bash
cd $PROJECTROOT
make program_cva6_fpga
```

After successful programming, the FPGA DONE LED should indicate that the bitstream has been loaded.

### 6.4 Open the UART terminal

First identify the serial device:

```bash
ls -l /dev/serial/by-id/ | grep -i uart
```

Then open it, for example:

```bash
tio /dev/ttyUSB0
```

If permission is denied:

```bash
sudo tio /dev/ttyUSB0
```

Keep this terminal open while running the application.

### 6.5 Start OpenOCD

Open another terminal and load the environment:

```bash
source /path/to/setup.sh
```

Then:

```bash
cd $PROJECTROOT/sw/app
openocd -f openocd_digilent_hs2.cfg
```

A successful connection should report that the RISC-V hart has been detected and that OpenOCD is listening for GDB connections on port `3333`.

### 6.6 Start GDB and load MNIST

Open a third terminal:

```bash
source /path/to/setup.sh
cd $PROJECTROOT/sw/app
riscv-none-elf-gdb mnist.riscv
```

Inside GDB:

```gdb
target extended-remote :3333
load
c
```

The MNIST output should then appear on the UART terminal.

To stop execution:

```text
Ctrl+C
```

Then inside GDB:

```gdb
kill
q
```

---

## 7. Modified RISC-V Toolchain

The repository includes the custom GNU assembler support required by the accelerator.

The three custom instructions are:

```text
mac16buf
buf4
mac16buf_para
```

Their 7-bit opcodes are:

| Instruction | Opcode |
|---|---:|
| `mac16buf` | `0x0b` |
| `buf4` | `0x2b` |
| `mac16buf_para` | `0x5b` |

The assembler operand format is:

```text
d,W1,W2,W3,W4
```

with the instruction fields:

```text
d   -> bits [11:7]
W1  -> bits [16:12]
W2  -> bits [21:17]
W3  -> bits [26:22]
W4  -> bits [31:27]
```

The project-specific toolchain modifications are mainly located in:

```text
util/riscv-opcodes/extensions/rv_i

util/gcc-toolchain-builder/src/binutils-gdb/
├── include/opcode/riscv-opc.h
├── include/opcode/riscv.h
├── opcodes/riscv-opc.c
└── gas/config/tc-riscv.c
```

`tc-riscv.h` does not contain an AIRV-specific modification.

Because the complete toolchain is already included in this repository, users normally do not need to run `get-toolchain.sh`.

### Rebuilding the included toolchain

If the custom GNU toolchain sources are intentionally modified, rebuild it from:

```bash
cd $PROJECTROOT/util/gcc-toolchain-builder

export RISCV=riscv_toolchain
bash ./build-toolchain.sh "$RISCV"
```

After rebuilding:

```bash
export PATH="$PROJECTROOT/util/gcc-toolchain-builder/riscv_toolchain/bin:$PATH"
which riscv-none-elf-gcc
```

> **Warning:** do not run `get-toolchain.sh` unless you intentionally want to download a new upstream toolchain source tree. The repository already contains the AIRV-modified source tree.

---

## 8. Quick Validation Procedure

A minimal validation after cloning the complete repository is:

```bash
# 1. Configure the local PROJECTROOT in setup.sh
source /path/to/setup.sh

# 2. Verify tools
which riscv-none-elf-gcc
which vsim
which vivado
which openocd

# 3. Compile MNIST
cd $PROJECTROOT/sw/app
make mnist

# 4. Run simulation
cd $PROJECTROOT
make sim APP=mnist
```

For FPGA validation:

```bash
cd $PROJECTROOT
make cva6_fpga
make program_cva6_fpga
```

Then use UART + OpenOCD + GDB as described above.

---

## 9. Troubleshooting

### `PROJECTROOT` points to the wrong directory

Check:

```bash
echo $PROJECTROOT
```

If incorrect, edit `setup.sh` and reload it:

```bash
source /path/to/setup.sh
```

### Wrong RISC-V compiler is selected

Check:

```bash
which riscv-none-elf-gcc
```

If it does not point to the project toolchain, prepend it to `PATH`:

```bash
export PATH="$PROJECTROOT/util/gcc-toolchain-builder/riscv_toolchain/bin:$PATH"
```

### Questa does not start

Check:

```bash
which vsim
echo $MGLS_LICENSE_FILE
echo $LM_LICENSE_FILE
```

The provided setup currently uses the IMT Atlantique license server:

```text
29000@eda-lic.imta.fr
```

A user outside this environment must configure an appropriate Questa installation and license.

### UART shows no output

Check the serial device:

```bash
ls -l /dev/serial/by-id/
```

Try the detected `/dev/ttyUSBx` device and verify permissions. If necessary:

```bash
sudo tio /dev/ttyUSBx
```

A board power cycle can also be useful before repeating the FPGA/OpenOCD/GDB sequence.

### OpenOCD cannot detect the target

Check:

- board power;
- JTAG-HS2 connection;
- USB permissions / udev rules;
- whether another OpenOCD process is already running.

To find old OpenOCD processes:

```bash
ps aux | grep openocd
```

### `vsim.wlf` becomes very large

RTL simulation can generate a multi-gigabyte waveform file. Delete old waveform files when they are no longer needed:

```bash
rm -f $PROJECTROOT/vsim.wlf
```

---

## 10. Typical Workflow

For software or RTL development:

```text
source setup.sh
      ↓
modify software / RTL
      ↓
make mnist
      ↓
make sim APP=mnist
      ↓
inspect result + Questa waveform
```

For final FPGA validation:

```text
source setup.sh
      ↓
make mnist
      ↓
make cva6_fpga
      ↓
make program_cva6_fpga
      ↓
UART terminal
      ↓
OpenOCD
      ↓
GDB: load + continue
      ↓
check MNIST result
```

---

## 11. Notes for Reproducibility

The repository is currently distributed as a **complete development version**. It contains both the modified toolchain source and the compiled RISC-V toolchain in order to simplify reuse on the validated environment.

For a future lightweight release, the large generated/toolchain directories can be replaced by a smaller patch-based or overlay-based toolchain setup. That is not required for the present complete release.

---

# 12. Implementation Guide — Modified Files and Code Locations

This section is intended for the next developer who needs to understand or extend the accelerator without reconstructing the whole development history.

> **Important:** the line numbers below refer to the validated final source snapshot used to prepare this guide.  
> If a file is reformatted later, the line number may move. Always use the **symbol / anchor name** shown together with the line number.  
> Before making a final release, it is recommended to freeze the repository with a Git tag so these line references remain stable.

A useful command for locating the files is:

```bash
cd $PROJECTROOT

find . \( \
  -name "NetworkPropagate.c" -o \
  -name "decoder.sv" -o \
  -name "ariane_pkg.sv" -o \
  -name "cva6.sv" -o \
  -name "issue_read_operands.sv" -o \
  -name "scoreboard.sv" -o \
  -name "issue_stage.sv" -o \
  -name "cvxif_fu.sv" -o \
  -name "cvxif_pkg.sv" -o \
  -name "cvxif_instr_pkg.sv" -o \
  -name "cvxif_example_coprocessor.sv" -o \
  -name "cv32a6_ima_sv32_fpga_config_pkg.sv" \
\) -print
```

The overall data path is:

```text
NetworkPropagate.c
       │
       │ custom mnemonic
       ▼
GNU assembler / binutils
       │
       │ 32-bit custom instruction
       ▼
decoder.sv
       │
       │ rs1 / rs2 / rd + result[9:0]
       ▼
issue_read_operands.sv
       │
       ├── scoreboard.sv  (dependency / forwarding)
       │
       └── issue_stage.sv (wiring)
       │
       ▼
cvxif_fu.sv
       │
       │ x_issue_req.rs[0:8]
       ▼
CV-X-IF
       │
       ▼
cvxif_example_coprocessor.sv
       │
       ├── input buffer
       ├── weight buffer
       ├── local accumulator
       └── MAC16 datapath / post-processing
       │
       ▼
final result write-back
```

---

## 12.1 Software: `NetworkPropagate.c`

This is the software side of the accelerator. It replaces the original scalar CNN inner loops with custom instructions and coordinates input buffering, weight buffering, and local accumulation.

### Main modified areas

| Current line(s) | Anchor / function | Purpose |
|---:|---|---|
| 12–61 | file header comments | Documents the final accelerator model and the role of `BUF4`, `MAC16BUF`, and `MAC16BUF_PARA`. |
| 81–166 | `buffer4_setmode_conv1/conv2/fc1/fc2()` | Configures `active_blocks` for each layer through the `rd` field of `BUF4`. |
| 186–347 | `mac16buf_first`, `mac16buf_middle`, `mac16buf_final`, `middle4`, weight-buffer variants | Basic MAC16 operations using the input buffer. |
| 355–545 | `mac16buf_para_*` | MAC16 while input data are supplied from the CPU through x28–x31 and simultaneously written into the input buffer. |
| 547–659 | `mac16buf_para_middle4()` | Four-block unrolled PARA path used to reduce loop/control overhead. |
| 661–835 | `*_offset2` helpers | Handles FC2 / special data alignment cases with a two-byte offset. |
| 837–1027 | `mac16buf_para_conv1_*` | Conv1-specific aligned and unaligned input-loading paths. |
| 1028–1091 | `mac16buf_conv1*` | Conv1 reuse path after the input block has already been buffered. |
| 1093–1164 | `mac16buf_conv2_25blocks*` | Conv2 25-block reuse path. |
| 1166–1208 | `mac16buf_fc1_24blocks()` | FC1 24-block reuse path. |
| 1252–1505 | `convcellPropagate1()` | Final accelerated Conv1 implementation. |
| 1507–1687 | `convcellPropagate2()` | Final accelerated Conv2 implementation. |
| 1689–1805 | `fccellPropagateUDATA_T()` | Final accelerated FC1 implementation. |
| 1807–2115 | `fccellPropagateDATA_T()` | Final accelerated FC2 implementation, including the 6-element scalar tail. |

### `BUF4` layer configuration

The four helper functions begin around lines 110–166.

The configuration is:

```text
Conv1 : rd = x0  -> active_blocks = 1
Conv2 : rd = x24 -> active_blocks = 25
FC1   : rd = x23 -> active_blocks = 24
FC2   : rd = x8  -> active_blocks = 9
```

The hardware interprets:

```text
active_blocks = rd_index + 1
```

This means the register number is used as a small configuration field rather than as a normal destination register for `BUF4`.

### Conv1: lines 1252–1505

Anchor:

```c
static void convcellPropagate1(...)
```

Important behavior:

```text
one 4×4×1 patch = 16 bytes = 1 MAC16 block
```

For the first output filter, `MAC16BUF_PARA` computes the output while filling the input buffer. Later output filters reuse the buffered input with `MAC16BUF`.

The code also keeps the special aligned / unaligned Conv1 load paths.

Important anchors:

```c
mac16buf_para_conv1_aligned(...)
mac16buf_para_conv1_unaligned2(...)
mac16buf_para_conv1_aligned_wbuf(...)
mac16buf_para_conv1_unaligned2_wbuf(...)
mac16buf_conv1(...)
mac16buf_conv1_wbuf(...)
```

Hardware performs the final:

```text
ReLU -> >> 8 -> clamp to [0,255]
```

so software does not call `saturate()` after the accelerated Conv1 result.

### Conv2: lines 1507–1687

Anchor:

```c
static void convcellPropagate2(...)
```

The important configuration is:

```text
5 × 5 × 16 inputs
= 400 bytes
= 25 × 16-byte blocks
```

The first output filter performs 25 `MAC16BUF_PARA` operations. These operations do two things at the same time:

```text
compute filter 0
+
fill the 25-block input buffer
```

Later filters reuse the same 25 buffered input blocks through `MAC16BUF`.

At the first spatial position, the Conv2 weights are also captured into the weight buffer:

```text
24 filters × 25 blocks = 600 blocks
600 × 16 bytes = 9.6 kB
```

Important anchors:

```c
mac16buf_para_first(...)
mac16buf_para_middle(...)
mac16buf_para_final(...)
mac16buf_para_wbuf_first(...)
mac16buf_para_wbuf_middle(...)
mac16buf_para_wbuf_final(...)
mac16buf_conv2_25blocks(...)
mac16buf_wbuf_conv2_25blocks(...)
```

### FC1: lines 1689–1805

Anchor:

```c
static void fccellPropagateUDATA_T(...)
```

FC1 input size:

```text
24 × 4 × 4 = 384 bytes
384 / 16 = 24 blocks
```

Neuron 0 uses `MAC16BUF_PARA` to compute and fill the input buffer.

Neurons 1–63 reuse the same 24 input blocks through `MAC16BUF`.

FC1 weights are not stored in the weight buffer.

The unrolled helper:

```c
mac16buf_para_middle4(...)
```

is used to issue four middle blocks together and reduce software loop overhead.

### FC2: lines 1807–2115

Anchor:

```c
static void fccellPropagateDATA_T(...)
```

FC2 has 150 input values:

```text
9 × 16 = 144 values handled by MAC16
6 values handled by scalar software
```

Therefore:

```text
active_blocks = 9
```

The hardware intentionally **does not perform final ReLU / shift / saturation for FC2**, because software still has six scalar MAC operations to add afterward.

The final FC2 flow is:

```text
9 MAC16 blocks
      ↓
full 32-bit hardware accumulator
      ↓
6 scalar MACs in software
      ↓
software saturation
```

---

## 12.2 CPU Operation Definitions: `core/include/ariane_pkg.sv`

This file defines the architectural operation types and the operand bundle transported through the pipeline.

### Line 75 — register-file read-port count

Anchor:

```systemverilog
localparam NR_RGPR_PORTS = 9;
```

Final accelerator requirement:

```text
rs1 + rs2
+ rd/bias
+ two extra weight registers
+ x28 + x29 + x30 + x31
= 9 GPR reads
```

### Lines 446–450 — custom operation enumeration

Anchor:

```systemverilog
MAC16BUF,
MAC16BUF_PARA,
BUF4,
```

These values allow the decoder and pipeline to distinguish the three AIRV custom instructions.

### Lines 581–589 — additional operand fields

Anchor:

```systemverilog
operand_d
operand_e
operand_f
operand_g
operand_h
operand_i
```

These fields extend `fu_data_t` so that the extra register operands can travel from issue/read-operands to the CV-X-IF functional unit.

Mapping:

```text
operand_a -> weight word 0
operand_b -> weight word 1
imm       -> rd / initial accumulator
operand_d -> weight word 2
operand_e -> weight word 3
operand_f -> input word 0 / x28
operand_g -> input word 1 / x29
operand_h -> input word 2 / x30
operand_i -> input word 3 / x31
```

---

## 12.3 Top-Level Register-File Configuration: `core/cva6.sv`

### Lines 164–166

Anchor:

```systemverilog
localparam NrRgprPorts = 9;
```

This is the top-level configuration that instantiates the integer register file with nine read ports.

If this value is changed without updating `issue_read_operands.sv`, `scoreboard.sv`, `issue_stage.sv`, `cvxif_pkg.sv`, and `cvxif_fu.sv`, the accelerator operand path will no longer be consistent.

---

## 12.4 Custom Instruction Decode: `core/decoder.sv`

### Lines 1191–1233 — custom opcode decode

The instruction layout is:

```text
31          27 26          22 21          17 16          12 11           7 6          0
+-------------+--------------+--------------+--------------+--------------+------------+
| extra reg 2 | extra reg 1  |     rs2      |     rs1      |      rd      |   opcode   |
+-------------+--------------+--------------+--------------+--------------+------------+
```

The three opcodes are decoded around lines 1208–1233:

```systemverilog
7'b0001011 -> MAC16BUF
7'b0101011 -> BUF4
7'b1011011 -> MAC16BUF_PARA
```

All three are sent to:

```systemverilog
instruction_o.fu = CVXIF;
```

### Lines 1313–1325 — carry the two extra register addresses

Anchor:

```systemverilog
RS3: begin
```

For the custom accelerator instructions:

```systemverilog
instruction_o.result =
    {{riscv::XLEN - 10{1'b0}},
     instruction_i[31:27],
     instruction_i[26:22]};
```

Therefore:

```text
result[4:0] = instruction[26:22]
result[9:5] = instruction[31:27]
```

These two fields later become the third and fourth weight-register addresses.

---

## 12.5 Register Reading and Operand Mapping: `core/issue_read_operands.sv`

This is one of the most important modified CPU files.

### Lines 45–65 — additional GPR interfaces

Adds:

```text
rs4
rs5
rs6
rs7
rs8
rs9
```

to the issue/read-operands interface.

### Lines 124–130 — pipeline storage for extra operands

Adds the internal operand registers corresponding to the additional register-file ports.

### Lines 218–258 — accelerator register-address selection

Important behavior:

```text
MAC16BUF / BUF4:
    rs4 = result[4:0]
    rs5 = result[9:5]

MAC16BUF_PARA:
    rs4 = result[4:0]
    rs5 = result[9:5]
    rs6 = x28
    rs7 = x29
    rs8 = x30
    rs9 = x31
```

For `MAC16BUF` and `MAC16BUF_PARA`, `rd` is also read as the accumulator/bias source.

### Lines 300–343 — dependency and forwarding checks

The additional operands participate in scoreboard dependency detection.

This is necessary because a custom instruction must not read an old value if one of its source registers is still waiting for a previous instruction to write back.

### Lines 392–413 — forwarded operand selection

If the scoreboard reports a forwarded value for rs4–rs9, that forwarded value replaces the value read from the register file.

### Lines 595–619 — nine-port register-file address packing

This is the most useful block for understanding the complete register mapping.

Anchor:

```systemverilog
if (CVA6Cfg.NrRgprPorts == 9) begin
```

Final mapping:

```text
rdata[0] = rs1                     -> weight0
rdata[1] = rs2                     -> weight1
rdata[2] = rd                      -> bias / initial accumulator
rdata[3] = result[4:0]             -> weight2
rdata[4] = result[9:5]             -> weight3
rdata[5] = x28                     -> input0
rdata[6] = x29                     -> input1
rdata[7] = x30                     -> input2
rdata[8] = x31                     -> input3
```

The actual packing around lines 602–612 is:

```systemverilog
{
    x31,
    x30,
    x29,
    x28,
    result[9:5],
    result[4:0],
    rd,
    rs2,
    rs1
}
```

### Lines 711–734 — map register-file outputs to CV-X-IF operands

Anchor:

```systemverilog
operand_d_regfile
operand_e_regfile
operand_f_regfile
operand_g_regfile
operand_h_regfile
operand_i_regfile
```

This converts the nine read ports into the `fu_data_t` operand fields.

### Lines 740–768 — ID/EX pipeline registers

The additional operands are registered together with the normal operands before execution.

---

## 12.6 Scoreboard Forwarding: `core/scoreboard.sv`

The scoreboard was extended so that rs4–rs9 behave like normal integer source registers with dependency and forwarding support.

### Lines 44–68 — additional ports

Adds input address, output data, and valid signals for:

```text
rs4 ... rs9
```

### Lines 336–402 — dependency / forwarding requests

Adds:

```text
rs4_fwd_req
rs5_fwd_req
rs6_fwd_req
rs7_fwd_req
rs8_fwd_req
rs9_fwd_req
```

There are two sources of forwarded values:

```text
write-back ports
scoreboard entries still in the pipeline
```

The valid signals at approximately lines 397–402 also prevent x0 from being treated as a real dependency.

### Lines 465–579 — forwarding arbiters

Six additional `rr_arb_tree` instances select the newest valid forwarded value for rs4–rs9.

Anchors:

```text
i_sel_rs4
i_sel_rs5
i_sel_rs6
i_sel_rs7
i_sel_rs8
i_sel_rs9
```

This file is especially important when debugging stalls or incorrect register values.

---

## 12.7 Issue-Stage Wiring: `core/issue_stage.sv`

This file mainly connects the extended scoreboard interface to `issue_read_operands.sv`.

### Line 100

The `rs3_len_t` width is selected so the third integer operand remains XLEN-wide when the final nine-port configuration is active.

### Lines 116–140

Declares the rs4–rs9 signals between:

```text
Scoreboard
    ↕
Issue / Read Operands
```

### Lines 178–202

Connects rs4–rs9 to the scoreboard instance.

### Lines 246–264

Connects rs4–rs9 to the `issue_read_operands` instance.

If a new register operand is added in the future, all three parts must remain consistent:

```text
signal declaration
scoreboard connection
issue_read_operands connection
```

---

## 12.8 CV-X-IF Operand Count: `cvxif_pkg.sv`

### Line 15

Anchor:

```systemverilog
localparam X_NUM_RS = ariane_pkg::NR_RGPR_PORTS;
```

This makes the CV-X-IF request width follow the CPU register-file read-port configuration.

With the final accelerator:

```text
X_NUM_RS = 9
```

### Lines 36–37

The request therefore contains:

```systemverilog
logic [X_NUM_RS-1:0][X_RFR_WIDTH-1:0] rs;
logic [X_NUM_RS-1:0]                  rs_valid;
```

---

## 12.9 Packing Operands into CV-X-IF: `cvxif_fu.sv`

### Around lines 36–44

Defines the number of source registers and the `rs_valid` vector for the nine-operand configuration.

### Around lines 56–69

Anchor:

```systemverilog
if (cvxif_pkg::X_NUM_RS == 9)
```

The request mapping is:

```text
x_issue_req.rs[0] = operand_a
x_issue_req.rs[1] = operand_b
x_issue_req.rs[2] = imm
x_issue_req.rs[3] = operand_d
x_issue_req.rs[4] = operand_e
x_issue_req.rs[5] = operand_f
x_issue_req.rs[6] = operand_g
x_issue_req.rs[7] = operand_h
x_issue_req.rs[8] = operand_i
```

For the accelerator this becomes:

```text
rs[0] = weight0
rs[1] = weight1
rs[2] = rd / bias / initial accumulator
rs[3] = weight2
rs[4] = weight3
rs[5] = input0 / x28
rs[6] = input1 / x29
rs[7] = input2 / x30
rs[8] = input3 / x31
```

---

## 12.10 CV-X-IF Instruction Table: `core/cvxif_example/include/cvxif_instr_pkg.sv`

### Lines 21–60

The final instruction table contains three entries:

```text
BUF4          : 0101011
MAC16BUF      : 0001011
MAC16BUF_PARA : 1011011
```

Only the low seven opcode bits are masked.

This is intentional because the remaining fields contain register indices and, for `BUF4`, the active-block configuration in `rd`.

Important write-back behavior in this table:

```text
BUF4          -> writeback = 0
MAC16BUF      -> writeback = 1
MAC16BUF_PARA -> writeback = 1
```

The coprocessor later dynamically suppresses architectural write-back for first/middle MAC blocks so only the final block writes the completed accumulation result.

---

## 12.11 CV-X-IF Coprocessor: `core/cvxif_example/cvxif_example_coprocessor.sv`

This is the core hardware accelerator implementation.

### Lines 86–205 — layer configuration and block tracking

Contains:

```text
active_blocks
instruction identification
first / middle / final block detection
post-processing enable
issue-side block counter
```

The opcode tests are around lines 152–154:

```systemverilog
BUF4          -> 7'b0101011
MAC16BUF      -> 7'b0001011
MAC16BUF_PARA -> 7'b1011011
```

`BUF4` sets:

```text
active_blocks = instr[11:7] + 1
```

The issue counter determines:

```text
first  : block == 0
middle : 0 < block < active_blocks - 1
final  : block == active_blocks - 1
```

Only the final block performs architectural result write-back.

### Lines 253–275 — input buffer

The final input buffer is split into four banks:

```systemverilog
input_buffer0
input_buffer1
input_buffer2
input_buffer3
```

Each bank contains 25 × 32-bit words.

Together:

```text
4 banks × 25 words × 4 bytes = 400 bytes
```

which is exactly one Conv2 5×5×16 patch.

`MAC16BUF_PARA` writes these banks while performing the MAC.

`MAC16BUF` reads the buffered values for later filters / neurons.

### Lines 277–319 — weight buffer

The weight buffer is also split into four 32-bit banks.

Depth:

```text
600 entries per bank grouping
= 600 × 16-byte blocks
= 9.6 kB total Conv2 weights
```

Supported reuse:

```text
Conv1 -> 16 blocks
Conv2 -> 600 blocks
FC1   -> no weight buffering
FC2   -> no weight buffering
```

Important controls:

```text
weight_buffer_valid_q
weight_buffer_mode
use_weight_buffer
weight_block_cnt_q
weight_prefetch_addr
```

### Lines 321–356 — local accumulator and post-processing

Anchor:

```systemverilog
acc_q
partial_sum
```

The local accumulator keeps intermediate first/middle block results inside the coprocessor instead of writing them back to the CPU register file.

For Conv1, Conv2, and FC1, the final block applies:

```text
ReLU
>>
8
clamp to unsigned 8-bit
```

FC2 is excluded because six scalar terms still have to be added in software.

### Lines 358–408 — weight capture and BRAM prefetch

This block controls:

```text
weight capture
synchronous BRAM reads
prefetch of the next weight block
```

During the first Conv1/Conv2 use, CPU-provided weights are both:

```text
used immediately for MAC
+
captured into the weight buffer
```

During later reuse, the next weight block is prefetched so BRAM latency can be hidden.

### Lines 410–495 — accelerator state update

Contains the sequential updates for:

```text
input-buffer write pointer
weight-buffer state
weight-buffer valid flag
weight counter
local accumulator
```

The state is updated only when the CV-X-IF result handshake accepts the current operation.

### Lines 497–588 — MAC16 datapath

This is the arithmetic datapath.

It forms sixteen INT8 products:

```text
input[0]  × weight[0]
...
input[15] × weight[15]
```

and reduces them to:

```text
partial_sum
```

The result is then combined with either:

```text
initial rd/bias     for first block
acc_q               for middle/final blocks
```

### Lines 590–603 — result generation

Final result behavior:

```text
first block  -> local accumulator only
middle block -> local accumulator only
final block  -> architectural write-back
```

This is what removes repeated CPU-register accumulation traffic.

---

## 12.12 FPGA Configuration: `cv32a6_ima_sv32_fpga_config_pkg.sv`

### Line 21

Anchor:

```systemverilog
localparam CVA6ConfigCvxifEn = 1;
```

This enables CV-X-IF for the FPGA configuration.

### Line 91

The value is propagated into:

```systemverilog
CvxifEn: bit'(CVA6ConfigCvxifEn)
```

If CV-X-IF is disabled here, the custom instructions will not reach the coprocessor even if the rest of the RTL is present.

---

# 13. GNU Toolchain Modifications

The hardware alone is not sufficient. The GNU assembler must understand:

```text
mac16buf
buf4
mac16buf_para
```

and must know how to encode the four custom register operands.

The final instruction syntax is:

```text
mnemonic d, W1, W2, W3, W4
```

where:

```text
d  -> instruction[11:7]
W1 -> instruction[16:12]
W2 -> instruction[21:17]
W3 -> instruction[26:22]
W4 -> instruction[31:27]
```

The following line numbers correspond to the cleaned toolchain source prepared for this project. If the full upstream source has shifted, search for the listed symbols.

---

## 13.1 `util/riscv-opcodes/extensions/rv_i`

### Lines 27–32

Defines the three custom instruction opcodes:

```text
mac16buf       6..2=0x02 1..0=3
buf4           6..2=0x0A 1..0=3
mac16buf_para  6..2=0x16 1..0=3
```

These produce:

```text
MAC16BUF      = 0x0b
BUF4          = 0x2b
MAC16BUF_PARA = 0x5b
```

---

## 13.2 `util/gcc-toolchain-builder/src/binutils-gdb/include/opcode/riscv-opc.h`

### Lines 25–32

Defines:

```c
MATCH_MAC16BUF
MASK_MAC16BUF
MATCH_BUF4
MASK_BUF4
MATCH_MAC16BUF_PARA
MASK_MAC16BUF_PARA
```

The mask is:

```text
0x7f
```

because only opcode bits `[6:0]` are fixed.

### Lines 2796–2799

Registers the three instruction declarations:

```c
DECLARE_INSN(mac16buf, ...)
DECLARE_INSN(buf4, ...)
DECLARE_INSN(mac16buf_para, ...)
```

---

## 13.3 `util/gcc-toolchain-builder/src/binutils-gdb/opcodes/riscv-opc.c`

### Lines 323–327

Adds the three mnemonics to the RISC-V opcode table:

```c
{"mac16buf",      ..., "d,W1,W2,W3,W4", ...}
{"mac16buf_para", ..., "d,W1,W2,W3,W4", ...}
{"buf4",          ..., "d,W1,W2,W3,W4", ...}
```

The key part is:

```text
"d,W1,W2,W3,W4"
```

which tells GAS to use the custom `W` operand parser.

---

## 13.4 `util/gcc-toolchain-builder/src/binutils-gdb/include/opcode/riscv.h`

### Lines 110–137

Defines the custom field extraction and encoding helpers.

Cleaned names:

```c
EXTRACT_AIRV_W1
EXTRACT_AIRV_W2
EXTRACT_AIRV_W3
EXTRACT_AIRV_W4

ENCODE_AIRV_W1
ENCODE_AIRV_W2
ENCODE_AIRV_W3
ENCODE_AIRV_W4
```

Bit positions:

```text
W1 -> shift 12
W2 -> shift 17
W3 -> shift 22
W4 -> shift 27
```

If an older source snapshot still contains names such as:

```text
ENCODE_MAC8_RS1
...
ENCODE_MAC8_RS4
```

those are historical names for the same final bit fields.

---

## 13.5 `util/gcc-toolchain-builder/src/binutils-gdb/gas/config/tc-riscv.c`

Two blocks are important.

### Lines 1392–1405 — operand validation

Anchor:

```c
case 'W':
```

This marks the custom W1–W4 instruction fields as occupied/valid when GAS checks the instruction format.

### Lines 3308–3330 — operand parser

The second:

```c
case 'W':
```

reads a normal GPR name such as:

```text
x5
t0
a1
```

and places its register number into W1, W2, W3, or W4.

For example:

```text
W1 -> ENCODE_AIRV_W1(regno)
W2 -> ENCODE_AIRV_W2(regno)
W3 -> ENCODE_AIRV_W3(regno)
W4 -> ENCODE_AIRV_W4(regno)
```

`tc-riscv.h` does not contain an AIRV-specific modification.

---

# 14. Final Register and Instruction Mapping

The final custom instruction encoding is:

```text
bits [31:27] -> W4 / extra weight register
bits [26:22] -> W3 / extra weight register
bits [21:17] -> W2
bits [16:12] -> W1
bits [11:7]  -> d / rd field
bits [6:0]   -> custom opcode
```

For the MAC instructions, the CPU-side register mapping is:

```text
instruction W1 -> CV-X-IF rs[0] -> weight word 0
instruction W2 -> CV-X-IF rs[1] -> weight word 1
instruction d  -> CV-X-IF rs[2] -> initial accumulator / bias
instruction W3 -> CV-X-IF rs[3] -> weight word 2
instruction W4 -> CV-X-IF rs[4] -> weight word 3

x28            -> CV-X-IF rs[5] -> input word 0
x29            -> CV-X-IF rs[6] -> input word 1
x30            -> CV-X-IF rs[7] -> input word 2
x31            -> CV-X-IF rs[8] -> input word 3
```

This mapping must remain identical in:

```text
riscv.h
tc-riscv.c
riscv-opc.c
decoder.sv
issue_read_operands.sv
ariane_pkg.sv
cvxif_fu.sv
cvxif_example_coprocessor.sv
NetworkPropagate.c
```

A mismatch between any two of these files can produce a program that assembles correctly but uses the wrong registers in hardware.

---

# 15. What to Change When Extending the Accelerator

If a future developer changes the CNN dimensions, the first values to verify are:

```text
Conv1 active blocks
Conv2 active blocks
FC1 active blocks
FC2 active blocks
input-buffer depth
weight-buffer depth
hardware post-processing condition
software scalar tail
```

For a layer containing `N` input bytes handled entirely by MAC16:

```text
active_blocks = ceil(N / 16)
```

However, if the final block is incomplete, the current accelerator may require a software tail or a new masking mechanism. Do not simply increase `active_blocks` without checking memory safety and the number of valid input elements.

The present design intentionally keeps the FC2 final six elements in software.

---

# 16. Recommended Debugging Order

When a custom instruction produces a wrong result, debug in this order:

```text
1. Check the generated disassembly
   ↓
2. Check decoder.sv opcode and register fields
   ↓
3. Check issue_read_operands.sv raddr_pack
   ↓
4. Check scoreboard forwarding / stalls
   ↓
5. Check cvxif_fu.sv x_issue_req.rs[0:8]
   ↓
6. Check coprocessor issue flags
   ↓
7. Check first / final / block counter
   ↓
8. Check partial_sum
   ↓
9. Check acc_q
   ↓
10. Check final writeback
```

Useful Questa signals:

```text
issue_is_mac16buf
issue_is_mac16buf_para
issue_block_cnt_q
issue_is_first_block
issue_is_final_block
acc_q
partial_sum
mac_base_acc
mac_next_acc
writeback
weight_buffer_valid_q
weight_block_cnt_q
```

---

# 17. Validation Before Modifying the Final Release

Before editing the final validated version, create a Git branch:

```bash
git switch -c experiment_name
```

After a modification, validate at least:

```bash
source /path/to/setup.sh

cd $PROJECTROOT/sw/app
make clean
make mnist

cd $PROJECTROOT
make sim APP=mnist
```

Then compare:

```text
Predicted class
output values
instruction count
cycle count
Questa accelerator waveform
```

For hardware changes, also run FPGA synthesis / implementation and inspect:

```text
LUT
FF
BRAM
WNS
TNS
```

Finally, run the application on the Zybo Z7-20 if the change is intended for the FPGA release.

---

# 18. Recommended Repository Freeze

Once this guide and the source code are confirmed to match, freeze the validated state:

```bash
git status
git add USER_GUIDE.md
git commit -m "Add final AIRV user and implementation guide"
git push

git tag final-v1.0
git push origin final-v1.0
```

The tag is important because all line numbers in this implementation guide can then be interpreted relative to one immutable source version.

