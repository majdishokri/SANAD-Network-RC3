# Contributing to SANAD Network RC3

This repository is a **frozen research baseline with a public-facing release**, not an invitation to overwrite the trained classifier. External improvements should be proposed on a separate development branch and evaluated independently.

## Before proposing a change

1. Identify whether it is a documentation, performance-only, data-ingestion, mathematical/model, or experimental change. CPU acceleration must prove parity; model/threshold changes require a separate version.
2. Preserve frozen pretrained model SHA-256, 0.50 threshold, 7×8 original qualification geometry and study integrity. Never quietly relabel an experimental model as frozen RC3.
3. Run the public integrity, simulated qualification and accelerated artifact checks from README. Add deterministic unit tests for any new code.
4. For RF datasets, record publisher/DOI, data checksum, hardware topology, true timestamps, healthy/fault truth provenance, holdout boundaries and permission to use data. **Do not commit raw copyrighted RF or private telemetry.**
5. Keep performance claims strictly tied to the measured test. Label simulation, measured RF with synthetic fault, and documented physical hardware fault separately.

Do not submit credentials, private keys, arbitrary pickled/joblib models, network access tokens, access-controlled Drive links, user identifying information or proprietary evidence. Python pickle/joblib is capable of executing code during load, so models should be trusted and hash verified.

**License status:** public source availability does not currently carry an explicit software reuse license. Contact the repository owner about licensing before depending on redistribution or commercial use.
