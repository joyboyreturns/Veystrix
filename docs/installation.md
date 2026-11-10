# Installation and first run

> No public build exists yet. These steps describe the intended first release.
> Until the Releases page lists a build with checksums, do not install anything
> that claims to be Veystrix.

## 1. Download and verify

1. Download `Veystrix-<version>-arm64.dmg` and `SHA256SUMS.txt` from the
   [Releases page](https://github.com/joyboyreturns/Veystrix/releases).
2. Verify it: see [Verifying a download](verifying-downloads.md).
3. Open the DMG and drag **Veystrix** to **Applications**.
4. **First open (free build, not notarized).** Veystrix is free software and is not notarized by
   Apple, so macOS will say it cannot verify the app. This is expected for this build. Open it once
   so macOS blocks it, then go to **System Settings > Privacy & Security**, scroll to the message
   about Veystrix and choose **Open Anyway**, then confirm. (Terminal alternative, only after you
   verified the checksum: `xattr -dr com.apple.quarantine /Applications/Veystrix.app`.) The exact
   wording differs between macOS versions. Do this only for a file whose SHA-256 you verified.

## 2. First launch

The app opens to the Dashboard with the engine **stopped**. Veystrix never loads a
model on its own when you open it.

1. Open **Model**, choose the folder that holds your checkpoint, and read the
   validation result and the checkpoint's license note. Validation reads manifests and
   checks that required files exist; it does not read the weights.
2. Open **Settings** and review the port (default `8080`), the memory policy and the
   cache folder and quota.
3. Plug in AC power. Close memory-heavy apps.
4. Press **Start Engine**. The first load takes minutes and uses most of the Mac's
   memory. If a safety guard refuses to start, the Dashboard shows the exact reason
   (for example "not enough available memory"). Free memory and try again; do not
   look for a way around the guard.

## 3. Connect a coding agent

See [Harness integration](harness-integration.md). The local server address is shown
on the Dashboard.

## 4. Where Veystrix keeps things

| What | Where |
|---|---|
| App preferences | `~/Library/Application Support/Veystrix/` |
| Logs | `~/Library/Logs/Veystrix/` |
| Cache (SSD prefix cache) | `~/.flashnext/prefix-cache/` by default (the engine's existing location, kept so existing caches are reused); you can choose another folder in Settings |
| API key (if you create one) | `~/Library/Application Support/Veystrix/` with owner-only permissions |

Updating the app does not touch your model folder, your cache or your sessions.

## 5. Uninstall

Quit the app and drag it to the Trash. To remove its data, delete the folders above.
Your model files are wherever you put them and are never deleted by Veystrix.
