---
name: f7-disable-model-invocation-false
description: "Apply Acme refund rules when a customer asks for a refund, a return or store credit."
license: "Apache-2.0"
compatibility: "Requires python3"
allowed-tools: "Read"
disable-model-invocation: false
metadata:
  decimalai-skill-id: "00000000-0000-0000-0000-00005eed0009"
  decimalai-version-id: "00000000-0000-0000-0000-0000f00d0009"
  decimalai-version: "3"
  decimalai-body-sha256: "2804b8c5e3955a593eefef55a8fbbdcb6ff702e84ad25d158a067606d171b76e"
  decimalai-name: "acme/f7-disable-model-invocation-false"
  decimalai-format: "1"
  decimalai-invocation: "model"
  decimalai-category: "support"
---
# Acme refund policy

- Returned within 30 days: full refund.
- 31-60 days: store credit.

Fees are in references/fees.md. To compute a refund, run scripts/compute_refund.py <amount> <days>.
