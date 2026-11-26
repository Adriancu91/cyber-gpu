#!/usr/bin/env sh
# SECOND host NAK tool (C-147 point 5: tensor cores, preparation only).
# Same idea as build.sh (NVK + NAK + nouveau drm-shim, no GPU), but from a
# current Mesa that has VK_KHR_cooperative_matrix in NVK. It lives in its own
# directory ($NAK2_HOME) and writes to tools/nak/out2/, so the first tool
# (Mesa 25.0.7, build.sh, tools/nak/out/) and the checksums of the programs on
# the board today are never touched.
#
#   sh tools/nak/build2.sh mesa         # once: download + build Mesa 26.3.1 into $NAK2_HOME (~20 min, 2 cores)
#   sh tools/nak/build2.sh mma16_f16    # 16x16x16 f16 -> f32 coopMatMulAdd -> tools/nak/out2/mma16_f16.sm86.{sass,txt}
#   sh tools/nak/build2.sh mma16_s8     # 16x16x32 s8 -> s32 (IMMA probe; compiled, not modelled)
#   sh tools/nak/build2.sh mm_tc        # C-160 point 7: training matmul on the tensor cores, f32 in/out, f16 products (session 157)
#   sh tools/nak/build2.sh mm_tc_gelu   # C-157 p.2: mm_tc with gelu_tr mode 1 in the store, mixed step (session 199)
#   sh tools/nak/build2.sh mm_tc128     # C-165 after os-416: mm_tc on 128x128 tiles, 256 threads, same bits (session 220)
#   sh tools/nak/build2.sh mm_tc_pf     # C-157 p.2 after os-423: mm_tc with the global loads issued ahead, same bits (session 229)
#   sh tools/nak/build2.sh mm_tc_v4     # C-157 p.2: mm_tc_pf with 128-bit loads, aligned products only, same bits (session 230)
#   sh tools/nak/build2.sh mm_tc128_v4  # C-157 p.2: mm_tc128 with mm_tc_v4's cure (loads ahead, 128 bits), same bits (session 231)
#   sh tools/nak/build2.sh mm_s8        # B1 step 3: int8 matmul on the tensor cores, 128x128 tiles, i32 out (session 225)
#   sh tools/nak/build2.sh attn_back_tile_tc # C-157: attn_back_tile_3's dQ / dK / dV on the tensor cores, f16 inputs (session 243)
#   sh tools/nak/build2.sh attn_back_tile_tc16 # C-157: attn_back_tile_tc in 16 KiB of shared memory, same arithmetic (session 244)
#   sh tools/nak/build2.sh kernel <name> <push-constant bytes>
#
# Needs (Ubuntu 24.04): the apt packages listed in build.sh, plus
#   python3 -m venv $NAK2_HOME/venv && $NAK2_HOME/venv/bin/pip install meson==1.8.3 mako pyyaml packaging
#   cargo install --locked bindgen-cli@0.71.1 cbindgen@0.29.0      (Mesa 26.3 needs Meson >= 1.7, bindgen >= 0.71.1, Rust >= 1.85)
set -eu
HERE=$(cd "$(dirname "$0")" && pwd)
NAK2_HOME=${NAK2_HOME:-$HOME/.cache/cyber-gpu-nak2}
MESA_VER=26.3.1
# Computed by us on the tarball from archive.mesa3d.org (20 Nov 2026); compare with the release announcement.
MESA_SHA256=4e4e438eac7b840a9383988cbb7c08dad8041da544b9319fc26d319fa7348961
SRC=$NAK2_HOME/mesa-$MESA_VER
INST=$NAK2_HOME/inst
CHIP=${NOUVEAU_CHIPSET:-174}   # 0x174 = GA104 (RTX 3070 Ti) -> SM 8.6
MESON=${MESON:-$NAK2_HOME/venv/bin/meson}

case "${1:-}" in
mesa)
    mkdir -p "$NAK2_HOME"; cd "$NAK2_HOME"
    [ -f mesa.tar.xz ] || curl -fsSL -o mesa.tar.xz "https://archive.mesa3d.org/mesa-$MESA_VER.tar.xz"
    echo "$MESA_SHA256  mesa.tar.xz" | sha256sum -c -
    [ -d "$SRC" ] || tar xf mesa.tar.xz
    cd "$SRC"
    PATH=$NAK2_HOME/venv/bin:$HOME/.cargo/bin:/usr/lib/llvm-18/bin:$PATH "$MESON" setup build -Dprefix="$INST" \
        -Dvulkan-drivers=nouveau -Dgallium-drivers= -Dtools=drm-shim -Dplatforms= \
        -Dglx=disabled -Degl=disabled -Dgles1=disabled -Dgles2=disabled -Dopengl=false \
        -Dllvm=enabled -Dbuildtype=release -Dvideo-codecs=
    PATH=$NAK2_HOME/venv/bin:$HOME/.cargo/bin:/usr/lib/llvm-18/bin:$PATH ninja -C build install
    ;;
mma16_f16)
    exec sh "$0" kernel mma16_f16 32
    ;;
mma16_s8)
    exec sh "$0" kernel mma16_s8 32
    ;;
mm_tc)
    exec sh "$0" kernel mm_tc 48
    ;;
mm_tc128)
    exec sh "$0" kernel mm_tc128 48
    ;;
mm_tc_gelu)
    exec sh "$0" kernel mm_tc_gelu 48
    ;;
mm_tc_pf)
    exec sh "$0" kernel mm_tc_pf 48
    ;;
mm_tc_v4)
    exec sh "$0" kernel mm_tc_v4 48
    ;;
mm_tc128_v4)
    exec sh "$0" kernel mm_tc128_v4 48
    ;;
mm_s8)
    exec sh "$0" kernel mm_s8 48
    ;;
attn_back_tile_tc)
    exec sh "$0" kernel attn_back_tile_tc 64
    ;;
attn_back_tile_tc16)
    exec sh "$0" kernel attn_back_tile_tc16 64
    ;;
kernel)
    NAME=${2:?nume}; PUSH=${3:?octeti push constants}
    OUT=$HERE/out2; TMP=$(mktemp -d); mkdir -p "$OUT"
    LIB=$(dirname "$(find -L "$INST" -name libnouveau_noop_drm_shim.so | head -1)")
    glslangValidator -V --target-env vulkan1.3 "$HERE/$NAME.comp" -o "$TMP/$NAME.spv"
    spirv-val --target-env vulkan1.3 "$TMP/$NAME.spv"
    cc -O1 -Wall -o "$TMP/nak_dump_cm" "$HERE/nak_dump_cm.c" -lvulkan
    VK_ICD_FILENAMES="$INST/share/vulkan/icd.d/nouveau_icd.x86_64.json" \
    LD_PRELOAD="$LIB/libnouveau_noop_drm_shim.so" NOUVEAU_CHIPSET=$CHIP NAK_DEBUG=print MESA_SHADER_CACHE_DISABLE=true \
        "$TMP/nak_dump_cm" "$TMP/$NAME.spv" "$PUSH" > "$TMP/nak.log" 2>&1 || { cat "$TMP/nak.log" >&2; exit 1; }
    grep -q "pipeline compiled" "$TMP/nak.log"
    grep -q "^Encoded shader:" "$TMP/nak.log" || { echo "NAK nu a tiparit codul (cache?)" >&2; exit 1; }
    grep "^coopmat:\|^device:" "$TMP/nak.log" > "$OUT/coopmat_shapes.sm86.txt"
    awk '/NAK IR after/{n=NR} {l[NR]=$0} END{for(i=n;i<=NR;i++) if (l[i] !~ /^pipeline compiled/) print l[i]}' \
        "$TMP/nak.log" > "$OUT/$NAME.sm86.txt"
    awk '/^Encoded shader:/{f=1;next} f&&/^ /{for(i=1;i<=NF;i++) print $i} f&&!/^ /{f=0}' "$TMP/nak.log" \
        | python3 -c "import sys,struct; sys.stdout.buffer.write(b''.join(struct.pack('<I',int(w,16)) for w in sys.stdin.read().split()))" \
        > "$OUT/$NAME.sm86.sass"
    [ -s "$OUT/$NAME.sm86.sass" ] || { echo "binar gol" >&2; exit 1; }
    sha256sum "$OUT/$NAME.sm86.sass"
    rm -rf "$TMP"
    ;;
*)
    echo "usage: $0 mesa|mma16_f16|mma16_s8|mm_tc|mm_tc128|mm_tc_gelu|mm_tc_pf|mm_tc_v4|mm_s8|attn_back_tile_tc|attn_back_tile_tc16|kernel <nume> <octeti>" >&2; exit 2
    ;;
esac
