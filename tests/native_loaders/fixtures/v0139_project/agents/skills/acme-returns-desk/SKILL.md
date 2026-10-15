---
name: acme/returns-desk
description: "Apply Acme returns desk rules when a customer asks for a refund, a return or store credit."
category: support
skill-scope: private
invocation: user
disable-model-invocation: true
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
---

# Acme returns desk

Sentinel: SENTINEL-NATIVE-DESK-2b9c

- Returned within 30 days: full refund.
- 31-60 days: store credit.
- After 60 days: none.

For fees read references/fees.md. To compute a refund, run scripts/compute_refund.py <amount> <days>.