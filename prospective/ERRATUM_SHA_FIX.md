# RC3 Frozen Prospective — checksum correction before new-seed evaluation

The v1 runner and lock file contained an invalid **50-character** value for `SOURCE_SHA256`: `121ce9aba417fb27b07d7f7f6e52633a8ef29a2c99aefddafb`.

The Drive-hosted RC0 source ZIP was downloaded and independently hashed; its correct **64-character** SHA-256 is `121ce9aba417fb27b07d7f7f6e52633a8ef29a2c99aefddc12a10428d02ddafb`. This matches the actual digest reported in the user's Colab error, so the archive did **not** fail integrity; the constant in the runner was erroneous.

The RC3 candidate model digest remains unchanged at `0a90835bb57a56437fc09211902b581bb22377676f91dce68f309c92e3a95bf4` (64 characters).

## Corrections in v2

1. Correct `SOURCE_SHA256` in runnable notebook, Python script, and lock manifest to the actual verified 64-character digest.
2. Keep the SHA verification strict, with explicit expected and actual SHA logging.
3. Repair an independent scoring mistake discovered before new-seed generation: `*_expected == -1` indicates healthy, whereas `>=0` denotes faulty antennas; v1 mistakenly used `>0`. The correction matches RC3 training label construction and leaves all predictor inputs and model outputs untouched.

Unchanged: model file and bytes, ZIP source file and bytes, 0.50 threshold, 3000–3143 seed range, simulation, feature extraction, scoring denominator, FAR constraints and fixed study ID. No prospective labels were inspected before this correction; v1 halted before Sionna seed generation.

Use the v2 notebook with a **new Colab runtime**. Preserve v1 and this erratum for reproducibility.
