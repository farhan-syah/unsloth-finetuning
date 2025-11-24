#!/usr/bin/env python3
"""
Generate a text file containing all valid lm-eval task names.
This cached list is used for fast task validation in benchmark.py

Output: config/valid_lm_eval_tasks.txt (~350KB, 11,000+ tasks)
Loading time: <1ms (vs 10-20 seconds for lm_eval --tasks list)
"""

import subprocess
import json
import sys
from datetime import datetime

print("🔍 Fetching all available lm-eval tasks...")
print("   (This may take 10-20 seconds...)")

try:
    result = subprocess.run(
        ['python', '-m', 'lm_eval', '--tasks', 'list'],
        capture_output=True,
        text=True,
        timeout=60
    )

    # Parse task names from table format (first column only)
    tasks = []
    for line in result.stdout.strip().split('\n'):
        line = line.strip()
        if line.startswith('|') and '---' not in line and 'Group' not in line and 'Config Location' not in line:
            parts = line.split('|')
            if len(parts) >= 2:
                task_name = parts[1].strip()
                if task_name and task_name != 'Group':
                    tasks.append(task_name)

    print(f"✅ Found {len(tasks)} tasks")
    print(f"   Sample tasks: {', '.join(tasks[:5])}")

    # Save as plain text (one task per line, sorted)
    output_file = "config/valid_lm_eval_tasks.txt"
    with open(output_file, 'w') as f:
        f.write('\n'.join(sorted(tasks)))

    print(f"✅ Saved {len(tasks)} tasks to {output_file}")

except subprocess.TimeoutExpired:
    print("❌ ERROR: Command timed out after 60 seconds")
    sys.exit(1)
except Exception as e:
    print(f"❌ ERROR: {e}")
    sys.exit(1)
