# Updates

Veystrix uses the [Sparkle 2](https://sparkle-project.org) updater.

* **Check for Updates…** is in the Veystrix menu. Automatic checking is optional and
  off until you turn it on.
* The update feed is a public file on this repository over HTTPS:
  `https://raw.githubusercontent.com/joyboyreturns/Veystrix/main/updates/appcast.xml`
* Every update archive is signed with an EdDSA (Ed25519) key. The app contains only
  the matching public key and refuses an archive whose signature does not verify. The
  archive's code signature is also checked (Apple notarization applies only to releases
  announced as notarized; free builds are not).
* The app never installs an update with an older version number than the one you
  already have.
* An update never touches your model folder, cache or sessions. The engine's cache
  format has its own version marker; if a future version cannot read an old cache it
  starts a fresh one and leaves the old one in place.
* An interrupted update leaves the installed app unchanged; Sparkle swaps the new app
  in only after the archive is verified and extracted.
* Update checks send no prompts or file information. See [Privacy](privacy.md).

Release notes for each version are in the [Changelog](../CHANGELOG.md).
