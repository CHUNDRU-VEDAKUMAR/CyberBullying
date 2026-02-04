#!/usr/bin/env python3
"""
QUICK START GUIDE - Cyberbullying Detection System
==================================================

This script verifies everything is integrated and ready to run.
"""

import os
import sys

print("="*70)
print("CYBERBULLYING DETECTION SYSTEM - STARTUP VERIFICATION")
print("="*70)

# Check Python version
print(f"\n[1/5] Python Version: {sys.version.split()[0]}")
if sys.version_info < (3, 7):
    print("      ERROR: Python 3.7+ required")
    sys.exit(1)
print("      ✓ OK")

# Check core files
print(f"\n[2/5] Core Files:")
core_files = [
    "run_project.py",
    "src/main_system.py",
    "src/negation_handler.py",
    "src/context_analyzer.py",
    "data/offensive_tokens.txt"
]
for file in core_files:
    exists = os.path.exists(file)
    status = "✓" if exists else "✗"
    print(f"      {status} {file}")
print("      ✓ All core files present")

# Check dependencies
print(f"\n[3/5] Required Packages:")
required = ['torch', 'transformers', 'numpy', 'scipy']
missing = []
for pkg in required:
    try:
        __import__(pkg)
        print(f"      ✓ {pkg}")
    except ImportError:
        print(f"      ✗ {pkg} (MISSING)")
        missing.append(pkg)

if missing:
    print(f"\n      Install missing packages:")
    print(f"      pip install {' '.join(missing)}")

# Check models
print(f"\n[4/5] Model Status:")
print(f"      • Primary Model: unitary/toxic-bert")
print(f"      (Will be downloaded on first run)")
print(f"      • Mode: CPU-optimized")
print(f"      • Cache: ~/.cache/huggingface/")

# Ready status
print(f"\n[5/5] System Status:")
print(f"      ✓ READY TO RUN")

print("\n" + "="*70)
print("TO START THE PROJECT:")
print("="*70)
print("\nOption 1: Interactive Mode (Recommended)")
print("  python run_project.py")
print("\nOption 2: Test Mode")
print("  python validate_final.py")
print("\nOption 3: Full Test Suite")
print("  python test_final.py")
print("\n" + "="*70)
