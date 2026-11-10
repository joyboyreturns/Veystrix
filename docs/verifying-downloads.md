# Verifying a download

Official builds come only from the
[Releases page](https://github.com/joyboyreturns/Veystrix/releases) of this
repository.

```bash
# 1. checksum (run in the folder that holds the download and SHA256SUMS.txt)
shasum -a 256 -c SHA256SUMS.txt --ignore-missing

# 2. code signature and notarization of the app (after you copy it to Applications)
codesign --verify --deep --strict --verbose=2 /Applications/Veystrix.app
spctl --assess --type execute --verbose=2 /Applications/Veystrix.app
xcrun stapler validate /Applications/Veystrix.app
```

You can also run `tools/verify-download.sh <file>` from this repository.

Veystrix never asks for a license key, a payment, an account or an "activation
tool". A download page, video or message that offers one is not official; do not run it.
