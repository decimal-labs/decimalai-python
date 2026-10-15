---
name: some-github-owner/an-extremely-long-imported-skill-name-that-exceeds-the-limit
description: "Apply Imported refund policy rules when a customer asks for a refund, a return or store credit."
category: support
skill-scope: public
invocation: model
license: Apache-2.0
compatibility:
  - R
  - e
  - q
  - u
  - i
  - r
  - e
  - s
  - " "
  - p
  - y
  - t
  - h
  - o
  - n
  - 3
allowed-tools:
  - Read
  - Bash
  - WebFetch
---

# Imported refund policy

Sentinel: SENTINEL-NATIVE-IMPORT-c41d

- Returned within 30 days: full refund.
- 31-60 days: store credit.
- After 60 days: none.

For fees read references/fees.md. To compute a refund, run scripts/compute_refund.py <amount> <days>.