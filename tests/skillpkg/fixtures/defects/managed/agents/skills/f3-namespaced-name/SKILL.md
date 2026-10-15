---
name: acme/f3-namespaced-name
description: "Apply Acme refund rules when a customer asks for a refund, a return or store credit."
license: "Apache-2.0"
compatibility: "Requires python3"
allowed-tools: "Read"
metadata:
  decimalai-skill-id: "00000000-0000-0000-0000-00005eed0004"
  decimalai-version-id: "00000000-0000-0000-0000-0000f00d0004"
  decimalai-version: "3"
  decimalai-body-sha256: "2804b8c5e3955a593eefef55a8fbbdcb6ff702e84ad25d158a067606d171b76e"
  decimalai-name: "acme/f3-namespaced-name"
  decimalai-format: "1"
  decimalai-invocation: "model"
  decimalai-category: "support"
---
# Acme refund policy

- Returned within 30 days: full refund.
- 31-60 days: store credit.

Fees are in references/fees.md. To compute a refund, run scripts/compute_refund.py <amount> <days>.
