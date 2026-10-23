#!/usr/bin/env sh
# The same checks CI runs on the crates, in order. Stops at the first failure.
#   tools/check.sh
set -eu
cd "$(dirname "$0")/.."
echo "== 1. format";  cargo fmt --all -- --check
echo "== 2. clippy";  cargo clippy --workspace --all-targets -- -D warnings
echo "== 3. teste pe gazda (logica + dispozitive SIMULATE)"; cargo test --workspace --release
echo "== 4. build no_std x86_64-unknown-none"; cargo build --workspace --target x86_64-unknown-none --release
echo "== toate verificarile au trecut"
