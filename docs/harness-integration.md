# Using Veystrix with coding agents

Veystrix exposes an OpenAI-compatible server on your own Mac (`http://127.0.0.1:8080/v1`, loopback only). Any client
that speaks that protocol can use it. Nothing is sent to the internet by Veystrix.

| Client | Status |
|---|---|
| **Hermes Agent** (NousResearch) | Protocol tested offline with the real Hermes CLI against a mock server: streaming, reasoning, tool calls, multi-turn, errors, cancellation, no outbound traffic. **Not yet tested against the real model.** |
| Qwen Code | Provider settings and streaming keep-alive tested offline with the real client. Not yet tested against the real model. |
| OpenAI-compatible tools | Use base URL `http://127.0.0.1:8080/v1`, any API key (or the key you set), model id from `GET /v1/models`. |

The app's **Harnesses** screen shows ready-to-copy settings for each client.

## Things to know

* **Context:** Hermes needs a model window of at least 64K tokens; Veystrix reports 262,144. A prompt that is not
  already cached is refused above 32,768 tokens by the memory guard, so agents will compress earlier than the
  advertised window.
* **Stopping a request:** closing a client connection stops the engine's work. In Hermes' one-shot `-z` mode Ctrl-C
  only detaches; run `hermes gateway stop --yes` to cancel.
* **Speed:** the model streams from your SSD. Expect seconds per token, not tens of tokens per second. See
  [Verified performance](performance.md).
