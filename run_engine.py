#!/usr/bin/env python3
"""Standalone command-line RC3 inference; no training, Sionna, or hardware writes."""
import argparse
import json
from sanad_rc3_engine import SANADRC3Engine

def main():
    p=argparse.ArgumentParser(description="Frozen SANAD Network RC3 engine")
    p.add_argument("evidence",help="Precomputed 7-array × 300-report NPZ")
    p.add_argument("--stream",choices=("healthy","injected"),default="injected")
    a=p.parse_args()
    print(json.dumps(SANADRC3Engine().predict_npz(a.evidence,a.stream),indent=2))

if __name__=="__main__":
    main()
