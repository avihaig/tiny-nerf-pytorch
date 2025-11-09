#!/usr/bin/env bash
set -euo pipefail

mkdir -p data
OUT="data/tiny_nerf_data.npz"

if [ -f "$OUT" ]; then
  echo "[skip] $OUT already exists."
  exit 0
fi

echo "[info] Downloading tiny_nerf_data.npz ..."
URL1="http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/tiny_nerf_data.npz"
URL2="https://github.com/kunkun0w0/Clean-Torch-NeRFs/raw/main/tiny_nerf_data.npz"
URL3="https://github.com/volunt4s/TinyNeRF-pytorch/raw/main/tiny_nerf_data.npz"

download () {
  local url="$1"
  if command -v curl >/dev/null 2>&1; then
    curl -fL "$url" -o "$OUT"
  else
    wget -O "$OUT" "$url"
  fi
}

if ! download "$URL1"; then
  echo "[warn] Primary failed, trying mirrors..."
  download "$URL2" || download "$URL3"
fi

BYTES=$(wc -c < "$OUT")
echo "[ok] Downloaded to $OUT (${BYTES} bytes)"
