# AIRV CVA6 MNIST Accelerator

Hardware/software co-design for accelerating a quantized MNIST CNN on the CVA6 RISC-V processor.

The final implementation includes:

- Custom RISC-V instructions: `MAC16BUF`, `MAC16BUF_PARA`, and `BUF4`
- 16-way INT8 MAC acceleration
- Input buffer
- Weight buffer
- Local accumulator
- CV-X-IF coprocessor integration
- Modified GNU assembler/toolchain
- Questa RTL simulation
- Zybo Z7-20 FPGA execution

## Documentation

For environment setup, simulation, FPGA execution, toolchain modifications, and a detailed description of every modified source file:

👉 [User and Implementation Guide](USER_GUIDE_AIRV.md)

## Repository

The main project is located in:

```text
cva6_mac16_wtbuffer_v1/
