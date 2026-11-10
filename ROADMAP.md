# Roadmap

Plans are intentions, not promises. Items move only when they are verified.

## Before the first public release

- [x] Sparkle update signing key generated (2026-11-10); the private key is held by the owner.
- [ ] First free build (ad-hoc signed, not notarized) with checksums; first EdDSA-signed appcast entry;
      test an in-app update between two builds.
- [ ] *Later, optional:* Apple Developer ID signing, Hardened Runtime and notarization (needs a paid Apple
      Developer Program membership). The release pipeline is ready; Veystrix does not depend on it.
- [ ] Legal review of the license and the third-party notices.
- [ ] Full-routing validation of the shipped engine configuration on AC power under the
      published measurement standard.
- [ ] Safety validation of long-running sessions (cancellation, restart, long decode).

## After

- [ ] Real screenshots and a demonstration video.
- [ ] Verified results for 16K and longer contexts.
- [ ] Logit-level equivalence checks at full routing.
- [ ] Additional Apple Silicon configurations, as measured by the owner.
- [ ] Optional layer-major prefill promoted to default only after multi-turn, long-context
      and safety evidence agree.

## Not planned

Telemetry, mandatory accounts, activation servers, DRM, bundled third-party model weights.
