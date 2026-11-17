#!/bin/bash
# ---------------------------------------------------------------------------
# Load the LoCod bitstream on a PYNQ-Z2 (Zynq-7000) via the kernel
# fpga_manager interface.
#
# The mainline zynq-fpga driver requires a BYTE-SWAPPED .bin file
# (sync word 66 55 99 AA). Xilinx fpgautil does this swap internally;
# this script does it explicitly.
#
# Usage:  sudo ./load_fpga_pynqz2.sh [bitstream.bin]
#         (defaults to fpga.bin next to this script)
# ---------------------------------------------------------------------------
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
    echo "ERROR: run this script with sudo" >&2
    exit 1
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BIT="${1:-${SCRIPT_DIR}/fpga.bin}"
SWAPPED=/lib/firmware/fpga_swapped.bin
MGR=/sys/class/fpga_manager/fpga0

if [ ! -f "$BIT" ]; then
    echo "ERROR: bitstream not found: $BIT" >&2
    exit 1
fi

echo "Swapping bitstream '${BIT}' -> '${SWAPPED}' ..."
python3 - "$BIT" "$SWAPPED" <<'PY'
import sys
src, dst = sys.argv[1], sys.argv[2]
data = open(src, 'rb').read()
if len(data) % 4:
    sys.exit("ERROR: bitstream size is not a multiple of 4")
out = bytearray(len(data))
for i in range(0, len(data), 4):
    out[i:i+4] = data[i:i+4][::-1]
open(dst, 'wb').write(out)
PY

echo "Programming the PL ..."
echo 0 > "${MGR}/flags"
echo "$(basename "$SWAPPED")" > "${MGR}/firmware"

STATE=$(cat "${MGR}/state")
echo "FPGA manager state: ${STATE}"

if [ "$STATE" = "operating" ]; then
    echo "Successively done uploading FPGA.bin to PL!!"
else
    echo "ERROR: FPGA load failed (state: ${STATE})" >&2
    dmesg | tail -15 >&2
    exit 1
fi
