---
name: acme/refund-policy
description: "Apply Acme refund policy rules when a customer asks for a refund, a return or store credit."
category: support
skill-scope: private
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
---

# Acme refund policy

Sentinel: SENTINEL-NATIVE-REFUND-7f3a

- Returned within 30 days: full refund.
- 31-60 days: store credit.
- After 60 days: none.

For fees read references/fees.md. To compute a refund, run scripts/compute_refund.py <amount> <days>.