#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src/binutils-gdb"
OVERLAY_DIR="${SCRIPT_DIR}/airv-overlay/binutils-gdb"

if [[ ! -d "${SRC_DIR}" ]]; then
  echo "ERROR: ${SRC_DIR} does not exist."
  echo "Run get-toolchain.sh first."
  exit 1
fi

files=(
  "include/opcode/riscv-opc.h"
  "include/opcode/riscv.h"
  "opcodes/riscv-opc.c"
  "gas/config/tc-riscv.c"
)

for rel in "${files[@]}"; do
  install -D -m 0644 "${OVERLAY_DIR}/${rel}" "${SRC_DIR}/${rel}"
  echo "Applied AIRV overlay: ${rel}"
done
