#!/usr/bin/env sh
# Host tool (C-18: compiler = Mesa NAK, on the build machine, never in the OS).
# Builds Mesa's NVK/NAK + the nouveau drm-shim once, then compiles a GLSL
# compute shader for an NVIDIA chip WITHOUT a GPU and extracts the SM86 code.
#
#   sh tools/nak/build.sh mesa     # once: download + build Mesa 25.0.7 into $NAK_HOME (~15 min, 2 cores)
#   sh tools/nak/build.sh add64    # compile add64.comp for GA104 -> tools/nak/out/add64.sm86.{sass,txt}
#   sh tools/nak/build.sh matmul_naive   # the same for matmul_naive.comp (C-20 E, step 1)
#   sh tools/nak/build.sh kernel <name> <push-constant bytes>
#
# Needs (Ubuntu 24.04): apt install glslang-tools spirv-tools libvulkan-dev libdrm-dev
#   libclc-18-dev llvm-18-dev libclang-18-dev libllvmspirvlib-18-dev libwayland-dev
#   wayland-protocols byacc flex libexpat1-dev zlib1g-dev libelf-dev;
#   pip install meson==1.6.1 mako pyyaml packaging; cargo install bindgen-cli cbindgen
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
NAK_HOME=${NAK_HOME:-$HOME/.cache/cyber-gpu-nak}
MESA_VER=25.0.7
MESA_SHA256=592272df3cf01e85e7db300c449df5061092574d099da275d19e97ef0510f8a6
SRC=$NAK_HOME/mesa-$MESA_VER
INST=$NAK_HOME/inst
CHIP=${NOUVEAU_CHIPSET:-174}   # 0x174 = GA104 (RTX 3070 Ti) -> SM 8.6

case "${1:-}" in
mesa)
    mkdir -p "$NAK_HOME"; cd "$NAK_HOME"
    [ -f mesa.tar.xz ] || curl -fsSL -o mesa.tar.xz "https://archive.mesa3d.org/mesa-$MESA_VER.tar.xz"
    echo "$MESA_SHA256  mesa.tar.xz" | sha256sum -c -
    [ -d "$SRC" ] || tar xf mesa.tar.xz
    cd "$SRC"
    PATH=/usr/lib/llvm-18/bin:$PATH meson setup build -Dprefix="$INST" \
        -Dvulkan-drivers=nouveau -Dgallium-drivers= -Dtools=drm-shim -Dplatforms= \
        -Dglx=disabled -Degl=disabled -Dgles1=disabled -Dgles2=disabled -Dopengl=false \
        -Dllvm=enabled -Dbuildtype=release -Dvideo-codecs= -Dshared-glapi=disabled
    PATH=/usr/lib/llvm-18/bin:$PATH ninja -C build install
    ;;
add64)
    exec sh "$0" kernel add64 24
    ;;
matmul_naive)
    exec sh "$0" kernel matmul_naive 40
    ;;
kernel)
    NAME=${2:?nume}; PUSH=${3:?octeti push constants}
    OUT=$HERE/out; TMP=$(mktemp -d); mkdir -p "$OUT"
    LIB=$(dirname "$(find -L "$INST" -name libnouveau_noop_drm_shim.so | head -1)")
    glslangValidator -V --target-env vulkan1.3 "$HERE/$NAME.comp" -o "$TMP/$NAME.spv"
    spirv-val --target-env vulkan1.3 "$TMP/$NAME.spv"
    cc -O1 -Wall -o "$TMP/nak_dump" "$HERE/nak_dump.c" -lvulkan
    VK_ICD_FILENAMES="$INST/share/vulkan/icd.d/nouveau_icd.x86_64.json" \
    LD_PRELOAD="$LIB/libnouveau_noop_drm_shim.so" NOUVEAU_CHIPSET=$CHIP NAK_DEBUG=print MESA_SHADER_CACHE_DISABLE=true \
        "$TMP/nak_dump" "$TMP/$NAME.spv" "$PUSH" > "$TMP/nak.log" 2>&1
    grep -q "pipeline compiled" "$TMP/nak.log"
    grep -q "^Encoded shader:" "$TMP/nak.log" || { echo "NAK nu a tiparit codul (cache?)" >&2; exit 1; }
    # Text: the final NAK IR (assembly) + statistics + the encoded words.
    awk '/NAK IR after/{n=NR} {l[NR]=$0} END{for(i=n;i<=NR;i++) if (l[i] !~ /^pipeline compiled/) print l[i]}' \
        "$TMP/nak.log" > "$OUT/$NAME.sm86.txt"
    # Binary: the encoded 32-bit words, little-endian, 16 bytes per instruction.
    awk '/^Encoded shader:/{f=1;next} f&&/^ /{for(i=1;i<=NF;i++) print $i} f&&!/^ /{f=0}' "$TMP/nak.log" \
        | python3 -c "import sys,struct; sys.stdout.buffer.write(b''.join(struct.pack('<I',int(w,16)) for w in sys.stdin.read().split()))" \
        > "$OUT/$NAME.sm86.sass"
    [ -s "$OUT/$NAME.sm86.sass" ] || { echo "binar gol" >&2; exit 1; }
    sha256sum "$OUT/$NAME.sm86.sass"
    rm -rf "$TMP"
    ;;
*)
    echo "usage: $0 mesa|add64|matmul_naive|kernel <nume> <octeti>" >&2; exit 2
    ;;
esac
