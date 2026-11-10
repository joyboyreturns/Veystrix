# Roadmap

Plans are intentions, not promises. Items move only when they are verified.

## Before the first public release

- [ ] Developer ID signing, Hardened Runtime and Apple notarization of the app and the
      engine helper (blocked until signing credentials are available).
- [ ] Sparkle update signing key generated and stored offline; first signed appcast entry.
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
