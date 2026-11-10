# Privacy and security

* **No telemetry.** Veystrix does not collect usage data and does not send your prompts,
  repositories, sessions or model data anywhere.
* **No account, no activation, no payment.**
* **The only network access Veystrix makes itself** is the optional update check to
  GitHub (see [Updates](updates.md)). You can turn it off in Settings.
* **Loopback by default.** The local API binds to `127.0.0.1`. Any process on your Mac
  can talk to a loopback server, so Veystrix also refuses browser requests from web
  pages unless you explicitly allow an origin. Binding to a non-loopback address requires
  an API key.
* **Secrets.** API keys are stored with owner-only file permissions and never written to
  logs or diagnostics. Diagnostics you export are sanitized (keys, home-folder paths
  and prompt text are removed).
* **Sessions and caches can contain your prompts and code.** They stay on your disk
  (Veystrix never uploads them). Do not publish them.
* **Agents you connect** (Hermes Agent, Qwen Code, and so on) have their own network
  behavior. Veystrix's integration guides turn off cloud fallbacks and auxiliary
  providers; check your agent's settings.
* **Model checkpoints** are separate software with their own licenses and provenance;
  Veystrix does not download one without your action.

Report a vulnerability: see [SECURITY.md](../SECURITY.md).
