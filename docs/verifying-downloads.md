# Verifying a download

Official builds come only from the
[Releases page](https://github.com/joyboyreturns/Veystrix/releases) of this
repository.

```bash
# 1. checksum (run in the folder that holds the download and SHA256SUMS.txt)
shasum -a 256 -c SHA256SUMS.txt --ignore-missing

# 2. the code signature is intact (after you copy it to Applications)
codesign --verify --deep --strict --verbose=2 /Applications/Veystrix.app
```

Free builds are signed **ad hoc** and are **not notarized**, so `spctl --assess` is expected to say
"rejected" and `xcrun stapler validate` is expected to fail: that is normal for a build without an Apple
Developer ID, and it is why the SHA-256 in step 1 is the check that matters. Each release's
`BUILD-INFO.txt` states the channel. If a future release is announced as Developer ID signed and
notarized, then `spctl --assess` should say "accepted, source=Notarized Developer ID" and
`xcrun stapler validate` should succeed; treat a failure of either on such a release as a bad download.

You can also run `tools/verify-download.sh <file>` from this repository.

Veystrix never asks for a license key, a payment, an account or an "activation
tool". A download page, video or message that offers one is not official; do not run it.
