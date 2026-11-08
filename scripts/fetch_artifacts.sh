#!/usr/bin/env bash
# Fetches the two archives of checkpoints and per-sequence curves, checks each archive's sha256 and
# every file's sha256 (ARTIFACTS.sha256), and unpacks them into checkpoints/ and results/.
# Usage: bash scripts/fetch_artifacts.sh [<directory that already holds both archives>]
# Without an argument, it downloads both from the repository's v1.0 release (2.4 GB in all).
set -euo pipefail
cd "$(dirname "$0")/.."
src="${1:-}"
base="https://github.com/zypapers/mamba-conductance-erf/releases/download/v1.0"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
for pair in "checkpoints 95c4de64fa06ab19c469007bf660cc2367022f36df1a77d4bb83ded89d0e7ce0" "curves 855a91e1180d1cd67b71a53fa2b2f2902203342f92b3d3e09b703275578ac832"; do
  set -- $pair
  f="mamba-conductance-erf-$1.tar"
  if [ -n "$src" ]; then
    cp "$src/$f" "$tmp/$f"
  else
    curl -L --fail -o "$tmp/$f" "$base/$f"
  fi
  got=$(sha256sum "$tmp/$f" | cut -d' ' -f1)
  if [ "$got" != "$2" ]; then
    echo "$f: sha256 mismatch: got $got, expected $2" >&2
    exit 1
  fi
  tar -xf "$tmp/$f"
  rm "$tmp/$f"
done
sha256sum --quiet -c ARTIFACTS.sha256
echo "artifacts verified and unpacked"
