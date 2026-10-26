#!/usr/bin/env bash
# C-20 proposal A — the "reference probe" (PREPARED, NEVER RUN BY US).
# Runs ONLY in the workstation window, with Adrian present, from a Linux live
# stick (not Cyber Universe), as root, over SSH from the laptop (the screen
# goes black when tinygrad takes the RTX 3070 Ti: it is the only GPU).
#
# What it does: checks the board and the firmware files, installs tinygrad at
# a fixed commit, runs ONE small addition on the GPU through tinygrad's own
# PCI driver (DEV=PCI+NV:NAK) with every BAR0 register write logged
# (NV_DEBUG=4) and every RPC printed (DEBUG=3), and saves everything on the
# stick. THIS WRITES TO THE REAL BOARD (through tinygrad). Nothing is flashed;
# a reboot returns the board to its power-on state.
#
#   sudo bash proba.sh <stick-dir>        e.g. sudo bash proba.sh /media/ubuntu/CYBER
set -euo pipefail
STICK="${1:?usage: proba.sh <stick-dir>}"
TINYGRAD_COMMIT=9cf7829a6            # read in session 6 (26 oct 2026), MIT
TINYMESA_VERSION=25.2.7.2            # tinygrad's NAK build (Mesa, MIT)
FW_SRC="$STICK/firmware/nvidia/ga102/gsp"
OUT="$STICK/proba-tinygrad/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUT"
exec > >(tee "$OUT/consola.txt") 2>&1

echo "== 0. Confirmare"
[ "${CONFIRM_ADRIAN:-}" = "da" ] || { echo "Seteaza CONFIRM_ADRIAN=da (Adrian e de fata). Opresc."; exit 2; }
[ "$(id -u)" = 0 ] || { echo "Trebuie rulat ca root."; exit 2; }

echo "== 1. Placa (doar citire)"
lspci -nn -d 10de:2482 | tee "$OUT/lspci.txt"
[ -s "$OUT/lspci.txt" ] || { echo "RTX 3070 Ti (10de:2482) negasita. Opresc."; exit 3; }
BDF="0000:$(cut -d' ' -f1 "$OUT/lspci.txt" | head -1)"
lspci -vvv -s "$BDF" > "$OUT/lspci-vvv.txt" || true
if lsmod | grep -Eq '^(nouveau|nvidia) '; then
  echo "nouveau/nvidia e incarcat: porneste stick-ul cu modprobe.blacklist=nouveau,nvidia. Opresc."; exit 3
fi

echo "== 2. Firmware 570.144 de pe stick (sumele din docs/COMPATIBILITATE_FIRMWARE.md)"
cat > "$OUT/sume.txt" <<SUMS
4497e3eff7e95c774b8a569d17b27c08c9650158d10b229d2be81cdcad9a085b  booter_load-570.144.bin
82428f532240727e95bb3083fbaaba9b2cc7b937314323f2d546ce7245f27fad  bootloader-570.144.bin
a8c3ebeed280323aedb51c061f321e73379cce7a9ae643a33dd03915df027f7f  gsp-570.144.bin
SUMS
(cd "$FW_SRC" && sha256sum -c "$OUT/sume.txt") || { echo "Firmware lipsa sau gresit. Opresc."; exit 4; }
mkdir -p /lib/firmware/nvidia/ga102/gsp
for f in booter_load bootloader gsp; do
  zstd -q -f "$FW_SRC/$f-570.144.bin" -o "/lib/firmware/nvidia/ga102/gsp/$f-570.144.bin.zst"
done

echo "== 3. tinygrad $TINYGRAD_COMMIT + tinymesa $TINYMESA_VERSION (necesita retea)"
python3 -m venv /tmp/tg && . /tmp/tg/bin/activate
pip install -q "git+https://github.com/tinygrad/tinygrad@$TINYGRAD_COMMIT" "tinymesa==$TINYMESA_VERSION" zstandard
python3 -c "import tinygrad, sys; print('tinygrad', tinygrad.__file__, sys.version)"

echo "== 4. Proba: o adunare de 64 de numere pe GPU (SCRIERI IN PLACA, prin tinygrad)"
sync
set +e
DEV=PCI+NV:NAK NV_DEBUG=4 DEBUG=3 timeout 600 python3 - > "$OUT/jurnal.txt" 2>&1 <<'PY'
from tinygrad import Tensor
a = [float(i) for i in range(64)]
b = [0.5 * i for i in range(64)]
got = (Tensor(a) + Tensor(b)).tolist()
want = [x + y for x, y in zip(a, b)]
print("PROBA:", "OK" if got == want else "GRESIT", got[:4], "...")
PY
rc=$?
set -e
sync
echo "cod de iesire: $rc"
grep -c '^wreg: ' "$OUT/jurnal.txt" | sed 's/^/scrieri in registre jurnalizate: /'
grep '^PROBA:' "$OUT/jurnal.txt" || echo "PROBA: fara rezultat (vezi jurnal.txt)"
sha256sum "$OUT"/*.txt > "$OUT/SHA256SUMS"
echo "== gata. Fisierele: $OUT  — reporneste in Windows (nu opri de la buton inainte de sync)."
