#!/usr/bin/env bash

# =========================
# AIRV global environment
# =========================

# Project root
# export PROJECTROOT=/home/imtuser/AIRV/cva6-softcore-contest
export PROJECTROOT=/home/imtuser/AIRV/cva6_mac16_wtbuffer_v1

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