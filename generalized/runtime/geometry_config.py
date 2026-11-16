"""Configuration-only generalization: number of stations and elements per station."""
import os
N_ARRAYS=int(os.environ.get("SANAD_STATIONS","7"))
N_ELEMENTS=int(os.environ.get("SANAD_ELEMENTS","8"))
if N_ARRAYS<4 or N_ELEMENTS<4:
    raise ValueError("SANAD_STATIONS and SANAD_ELEMENTS must be >= 4")
