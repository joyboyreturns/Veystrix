#!/bin/bash
# Verify an official Veystrix download. Usage: tools/verify-download.sh <file.dmg|file.zip> [SHA256SUMS.txt]
# The SHA-256 check is the one that matters for free (not notarized) builds.
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
  if sig="$(codesign -dv --verbose=2 "$f" 2>&1)" && grep -q "Authority=Developer ID" <<<"$sig"; then
    codesign --verify --strict "$f" && echo "Developer ID signature OK"
    xcrun stapler validate "$f" && echo "notarization ticket OK"
  else
    echo "note: this disk image has no Developer ID signature (free build, not notarized): the checksum above is the integrity check."
  fi
fi
