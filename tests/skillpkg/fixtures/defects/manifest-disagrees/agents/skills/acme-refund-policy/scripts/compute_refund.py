#!/usr/bin/env python3
import sys
amount, days = float(sys.argv[1]), int(sys.argv[2])
print(amount if days <= 30 else 0.0)
