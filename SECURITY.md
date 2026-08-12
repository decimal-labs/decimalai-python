# Security Policy

This repository is the DecimalAI Python SDK, published on PyPI as **`decimalai`**. It runs inside
your application, holds your DecimalAI API key, reads your agent's prompts and outputs, and sends
traces and manifests to `api.decimal.ai`. A report may concern this source tree, the published
package, or both — say which if you know.

## Reporting a vulnerability

**Please do not open a public issue for a security problem.**

Two ways to reach us, either is fine:

- **GitHub private vulnerability reporting** — **Security → Report a vulnerability** on this
  repository. That opens a private advisory only maintainers can see.
- **Email** — [hello@decimal.ai](mailto:hello@decimal.ai). A PGP key is available on request if you
  would rather not send details in cleartext.

Include what you have: what you found, how to reproduce it, the `decimalai` version and which extras
were installed, and what an attacker could actually do with it. If the problem is that the SDK sent
something it should not have, a redacted request body is the single most useful attachment.

## Scope

**In scope — this source tree**, everything under `decimalai/` (core, CLI, adapters, replay, export)
and `scripts/`. The parts most worth your attention:

- **API key handling.** Keys arrive from `DECIMAL_API_KEY`, from `decimalai.init(api_key=...)`, or
  from local config. Any path that writes a key to a log line, an exception message, a traceback, a
  trace payload, a file with loose permissions, or a host other than the configured `base_url` is a
  vulnerability.
- **What tracing captures and transmits.** The SDK is documented to send prompts, completions, tool
  calls, and manifest components — see the [security and data handling
  page](https://docs.decimal.ai/security) for the exact list. Capturing or transmitting anything
  *outside* that list, or ignoring a documented redaction/opt-out, is in scope.
- **Writing remote content to your disk.** `decimalai skills pull` writes registry `SKILL.md` files
  into your project. A skill name or payload that escapes the target directory, overwrites a file
  outside it, or clobbers something the CLI did not create is in scope.
- **What the SDK reads from your project.** It scans per-runtime skill directories for `SKILL.md`
  files. Reading beyond those documented directories — into `~`, into source files, into `.env` —
  is in scope, as is `include_global` behaving as if it were opted in.
- **Untrusted responses.** Anything in a server response that leads to code execution, arbitrary
  deserialization, or a filesystem write.
- **Transport.** TLS verification being skippable, downgraded, or off by default.

**In scope — the published artifact.** The wheel and sdist on PyPI as `decimalai`, including a
published artifact that does not match this source tree, a dependency pulled in only by the built
package, or anything about the PyPI project itself (name confusion, a typosquat you have found
impersonating it). Report those here even though the fix is not a code change.

**Out of scope**

- **Documented plaintext storage.** Prompt and output text is stored in plaintext by design, and the
  SDK does not redact for you — that is stated in the docs. Scrub client-side before you send. A
  report that the SDK failed to redact something it *claims* to redact is in scope.
- The DecimalAI hosted platform (`api.decimal.ai`, `app.decimal.ai`). Report those the same way, to
  the same address; they are just fixed elsewhere.
- Vulnerabilities in LangChain, LangGraph, OpenAI Agents, LlamaIndex, Pydantic AI, ADK, or any other
  framework we adapt to — unless our adapter is what makes the issue reachable, in which case please
  do tell us.
- Dependency CVEs with no reachable path through this package.
- Findings that require an already-compromised machine or an attacker who already has your API key.
- Scanner output with no demonstrated impact.

## What happens next

We are a small team, so rather than publish a response time we cannot hold to, here is what we
actually do:

- We acknowledge a report once we have read it, and we say plainly if triage is going to take a
  while.
- We tell you whether we consider it in scope and what we intend to do.
- We follow coordinated disclosure. We agree a timeline with you rather than impose one, and we will
  not ask you to stay quiet indefinitely.
- We are happy to credit you in the advisory, the `CHANGELOG.md` entry, and the release notes. Tell
  us how you would like to be named, or say that you would rather not be.

There is no paid bug bounty. That is a resourcing decision, not a judgment about the value of your
work.

## Safe harbour

If you make a good-faith effort to follow this policy, we will not pursue or support legal action
against you for your research. Good faith means avoiding privacy violations and service degradation,
only interacting with accounts and data you own or have permission to test — if you need a workspace
to demonstrate something, ask and we will help you set one up — and giving us a reasonable
opportunity to fix the issue before you disclose it publicly.

If you are not sure whether what you found is a security issue, email
[hello@decimal.ai](mailto:hello@decimal.ai) and ask. That is always the right call.
