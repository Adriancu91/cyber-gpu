#!/usr/bin/env sh
# Host tool (C-18: compiler = Mesa NAK, on the build machine, never in the OS).
# Builds Mesa's NVK/NAK + the nouveau drm-shim once, then compiles a GLSL
# compute shader for an NVIDIA chip WITHOUT a GPU and extracts the SM86 code.
#
#   sh tools/nak/build.sh mesa     # once: download + build Mesa 25.0.7 into $NAK_HOME (~15 min, 2 cores)
#   sh tools/nak/build.sh add64    # compile add64.comp for GA104 -> tools/nak/out/add64.sm86.{sass,txt}
#   sh tools/nak/build.sh matmul_naive   # the same for matmul_naive.comp (C-20 E, step 1)
#   sh tools/nak/build.sh matmul_smem    # step 3: shared-memory tiles (session 8)
#   sh tools/nak/build.sh matmul_1d      # step 4: 1D blocktiling, 8 results per thread (session 9)
#   sh tools/nak/build.sh matmul_2d      # step 5: 2D blocktiling, 8x8 results per thread (session 10)
#   sh tools/nak/build.sh matmul_vec4    # step 6: step 5 with 128-bit loads/stores, A transposed in shared (session 24)
#   sh tools/nak/build.sh q8_fused       # C-147 step 2: q8_matvec + bias / residual / GELU / K-V copy (session 120)
#   sh tools/nak/build.sh q8_matvec32    # C-147 step 3: q8_matvec with all 32 threads + 128-bit x loads (session 121)
#   sh tools/nak/build.sh layernorm256   # C-155 point 2: LayerNorm with 256 threads, one read of x (session 132)
#   sh tools/nak/build.sh argmax         # C-155 point 3: the greedy token on the board, 16-byte result (session 134)
#   sh tools/nak/build.sh q8_fused32     # C-155 point 4: q8_fused with q8_matvec32's work split (session 135)
#   sh tools/nak/build.sh mm_train       # C-160 point 4: training matmul, op(A)*op(B) (+bias), NN/NT/TN (session 142)
#   sh tools/nak/build.sh xent           # C-160 point 4, program 7: softmax + cross-entropy and its gradient (session 143)
#   sh tools/nak/build.sh ln_back        # C-160 point 4, program 4: LayerNorm backward, dx (session 144)
#   sh tools/nak/build.sh ln_back_res    # order 20:11 step 3: ln_back + the residual addition in one program (session 192)
#   sh tools/nak/build.sh colsum         # C-160 point 4, program 2 (+ dγ, dβ of program 4): column sums (session 144)
#   sh tools/nak/build.sh colsum_part    # C-160 point 7: colsum's first pass split over rows (session 159)
#   sh tools/nak/build.sh ln_fwd         # C-160 point 4, program 3: LayerNorm forward over rows, keeps μ and rstd (session 146)
#   sh tools/nak/build.sh gelu_tr        # C-160 point 4, program 5: GELU forward (mode 0) / backward (mode 1) (session 147)
#   sh tools/nak/build.sh attn_back      # C-160 point 4, program 6 (backward): causal attention backward, rows / columns (session 148)
#   sh tools/nak/build.sh attn_fwd       # C-160 point 4, program 6 (forward): causal attention over T positions, writes P (session 149)
#   sh tools/nak/build.sh emb_fwd        # C-160 point 4, program 8 (forward): x = emb[tok] + pos (session 150)
#   sh tools/nak/build.sh emb_back       # C-160 point 4, program 8 (backward): embedding gradients by segments, no atomics (session 150)
#   sh tools/nak/build.sh gnorm          # C-160 point 4, program 9: total gradient norm + clipping factor, no atomics (session 152)
#   sh tools/nak/build.sh opt_step       # C-160 point 4, program 10: SGD / AdamW step over all parameters (session 152)
#   sh tools/nak/build.sh attn_fwd128    # C-160 point 7: attn_fwd with blocks of 128 (session 162)
#   sh tools/nak/build.sh attn_back128   # C-160 point 7: attn_back with blocks of 128 (session 162)
#   sh tools/nak/build.sh attn_fwd_tile  # C-160 point 7: attention forward on 64-row tiles, 4x4 per thread (session 163)
#   sh tools/nak/build.sh attn_back_tile # C-160 point 7: attention backward mode 1 on 64-position tiles (session 164)
#   sh tools/nak/build.sh attn_back_tile0 # C-160 point 7: attention backward mode 0 on 64-row tiles (session 165)
#   sh tools/nak/build.sh attn_p         # C-160 point 7: attn_fwd without P·V, only P (session 168)
#   sh tools/nak/build.sh embed_dev      # C-157 point 7: embed with the token read from the argmax result (session 170)
#   sh tools/nak/build.sh attn_fwd_tile2 # C-157 point 2: attn_fwd_tile, same bits, shorter loops (session 172)
#   sh tools/nak/build.sh attn_fwd_tile3 # C-157 point 2: attn_fwd_tile2, same bits, branch-free loads (session 181)
#   sh tools/nak/build.sh attn_back_tile0_2 # C-157 point 2: attn_back_tile0, same bits, shorter loops (session 173)
#   sh tools/nak/build.sh attn_back_tile0_3 # C-157 point 2: attn_back_tile0_2, same bits, branch-free loads (session 179)
#   sh tools/nak/build.sh attn_back_tile_2 # C-157 point 2: attn_back_tile, same bits, shorter loops (session 173)
#   sh tools/nak/build.sh attn_back_tile_3 # C-157 point 2: attn_back_tile_2, same bits, branch-free loads (session 180)
#   sh tools/nak/build.sh attn_back_tile_4 # C-157 point 2: attn_back_tile_3, full diagonal tile of dK / dV unguarded (session 188)
#   sh tools/nak/build.sh attn_p_tile    # C-160 point 7: attn_fwd_tile2 without P·V, only P (session 176)
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
matmul_coalesced)
    exec sh "$0" kernel matmul_coalesced 40
    ;;
matmul_smem)
    exec sh "$0" kernel matmul_smem 40
    ;;
matmul_1d)
    exec sh "$0" kernel matmul_1d 40
    ;;
matmul_2d)
    exec sh "$0" kernel matmul_2d 40
    ;;
matmul_vec4)
    exec sh "$0" kernel matmul_vec4 40
    ;;
q8_fused)
    exec sh "$0" kernel q8_fused 72
    ;;
q8_matvec32)
    exec sh "$0" kernel q8_matvec32 32
    ;;
layernorm256)
    exec sh "$0" kernel layernorm256 44
    ;;
argmax)
    exec sh "$0" kernel argmax 20
    ;;
q8_fused32)
    exec sh "$0" kernel q8_fused32 72
    ;;
mm_train)
    exec sh "$0" kernel mm_train 48
    ;;
xent)
    exec sh "$0" kernel xent 40
    ;;
ln_back)
    exec sh "$0" kernel ln_back 56
    ;;
ln_back_res)
    exec sh "$0" kernel ln_back_res 64
    ;;
colsum)
    exec sh "$0" kernel colsum 60
    ;;
colsum_part)
    exec sh "$0" kernel colsum_part 64
    ;;
ln_fwd)
    exec sh "$0" kernel ln_fwd 60
    ;;
gelu_tr)
    exec sh "$0" kernel gelu_tr 32
    ;;
attn_back)
    exec sh "$0" kernel attn_back 64
    ;;
attn_fwd)
    exec sh "$0" kernel attn_fwd 44
    ;;
attn_fwd128)
    exec sh "$0" kernel attn_fwd128 44
    ;;
attn_fwd_tile)
    exec sh "$0" kernel attn_fwd_tile 44
    ;;
attn_p)
    exec sh "$0" kernel attn_p 44
    ;;
attn_fwd_tile2)
    exec sh "$0" kernel attn_fwd_tile2 44
    ;;
attn_fwd_tile3)
    exec sh "$0" kernel attn_fwd_tile3 44
    ;;
attn_p_tile)
    exec sh "$0" kernel attn_p_tile 44
    ;;
embed_dev)
    exec sh "$0" kernel embed_dev 44
    ;;
attn_back_tile)
    exec sh "$0" kernel attn_back_tile 64
    ;;
attn_back_tile0)
    exec sh "$0" kernel attn_back_tile0 64
    ;;
attn_back_tile0_2)
    exec sh "$0" kernel attn_back_tile0_2 64
    ;;
attn_back_tile0_3)
    exec sh "$0" kernel attn_back_tile0_3 64
    ;;
attn_back_tile_2)
    exec sh "$0" kernel attn_back_tile_2 64
    ;;
attn_back_tile_3)
    exec sh "$0" kernel attn_back_tile_3 64
    ;;
attn_back_tile_4)
    exec sh "$0" kernel attn_back_tile_4 64
    ;;
attn_back128)
    exec sh "$0" kernel attn_back128 64
    ;;
emb_fwd)
    exec sh "$0" kernel emb_fwd 40
    ;;
emb_back)
    exec sh "$0" kernel emb_back 48
    ;;
gnorm)
    exec sh "$0" kernel gnorm 40
    ;;
opt_step)
    exec sh "$0" kernel opt_step 80
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
    echo "usage: $0 mesa|add64|matmul_naive|matmul_coalesced|matmul_smem|matmul_1d|matmul_2d|matmul_vec4|q8_fused|q8_matvec32|layernorm256|argmax|kernel <nume> <octeti>" >&2; exit 2
    ;;
esac
