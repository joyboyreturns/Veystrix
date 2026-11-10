# Security policy

## Supported versions

Security fixes target the latest published release.

## Reporting a vulnerability

Use GitHub's **private vulnerability reporting** on this repository (Security tab →
Report a vulnerability). Do not open a public issue with exploit details, credentials,
private prompts or session files.

## Scope

The Veystrix app, its bundled engine helper, the update feed and the release artifacts
published here. Vulnerabilities in third-party components (MLX, Sparkle, Python
libraries) should also be reported to those projects.

## Integrity of releases

Official downloads are signed with a Developer ID, notarized by Apple, checksummed in
`SHA256SUMS.txt`, and updates are EdDSA-signed. See
[Verifying a download](docs/verifying-downloads.md).
