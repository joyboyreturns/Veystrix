#!/bin/bash
# Verify an official Veystrix download. Usage: tools/verify-download.sh <file.dmg|file.zip> [SHA256SUMS.txt]
set -euo pipefail
f="${1:?usage: verify-download.sh <file> [SHA256SUMS.txt]}"
sums="${2:-$(dirname "$f")/SHA256SUMS.txt}"
[[ -f "$f" && -f "$sums" ]] || { echo "need the file and SHA256SUMS.txt" >&2; exit 2; }
name="$(basename "$f")"
want="$(awk -v n="$name" '$2==n || $2=="*"n {print $1}' "$sums")"
[[ -n "$want" ]] || { echo "no checksum listed for $name" >&2; exit 2; }
have="$(shasum -a 256 "$f" | awk '{print $1}')"
if [[ "$have" == "$want" ]]; then echo "checksum OK: $name"; else echo "CHECKSUM MISMATCH for $name" >&2; exit 1; fi
if [[ "$f" == *.dmg ]]; then
  codesign --verify --strict --verbose=2 "$f" && echo "code signature OK"
  xcrun stapler validate "$f" && echo "notarization ticket OK"
fi
