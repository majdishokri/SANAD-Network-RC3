# Freeze policy

This repository is a standalone **RC3 research candidate**. `rc3.py` is the only public inference API; historical compatibility modules in `internal/` are never independent products. No RC0/RC1/RC2 training notebooks are included.

The fitted RC3 model is fixed (SHA-256 `0a90835bb57a56437fc09211902b581bb22377676f91dce68f309c92e3a95bf4`). Runtime threshold **0.50**, dataset development seeds **2500–2643**, independent test seeds **3000–3143**. The prospectively-tested result is **pending**.

`prospective/` contains the latest *SHA-corrected v2* experiment script and source lock; the original incomplete checksum is not retained as an executable workflow. The scoring sentinel is `>=0` consistent with the training labels.

Never silently swap a model pickle, modify feature arrays, tune a threshold on prospective results, or call development CV a prospective claim. For changes, create a new RC release. A Git commit SHA plus freeze manifest identifies the exact immutable snapshot; repository branch names alone are mutable without branch protection.
