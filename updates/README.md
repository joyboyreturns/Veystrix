# Update feed

`appcast.xml` is the Sparkle 2 feed read by the Veystrix app over HTTPS. Entries are added
by the release process only, each with an EdDSA signature, a length, the minimum macOS
version and a link to the signed archive on the Releases page. The feed is empty until the
first signed release.
