#!/usr/bin/env python3
"""compute_refund.py <amount> <days>"""
import sys
amount, days = float(sys.argv[1]), int(sys.argv[2])
print('REFUND full refund: %.2f' % amount if days <= 30 else
      'REFUND store credit: %.2f' % amount if days <= 60 else 'REFUND none')
