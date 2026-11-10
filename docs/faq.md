# FAQ

**Is Veystrix open source?** No. It is proprietary freeware. Some components inside it
are open source and keep their own licenses; see
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).

**Is it really free?** Yes. No payment, account or activation. The license may change
for future releases; a release you already installed keeps the terms you accepted.

**Does it work offline?** Yes, once the app and a model checkpoint are on your Mac.

**Does it include a model?** No. You choose and obtain one separately and must follow its
license. Veystrix never downloads one without your action.

**Can I fork or rebrand it?** Not under this license. See [LICENSE](../LICENSE). The
earlier MIT-licensed FlashNext code that Veystrix grew from remains available
under the MIT license it was published with; this does not extend to Veystrix-owned
components.

**Why does Veystrix refuse to start?** Its memory and swap guards found too little
available memory or too much swap. That protects your Mac. Close apps (Veystrix also purges the macOS file cache on its own when that helps), plug in, and try
again.

**How fast is it?** See [Verified performance](performance.md). It is much slower than a
cloud model: the 125B-parameter model streams from SSD on a 16 GB Mac.

**Where do I get the source code?** It is private.
