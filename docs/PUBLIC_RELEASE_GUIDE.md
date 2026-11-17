# Public repository release guide

## Stable versus research

- Public main: human-facing documentation, **original frozen code and pretrained model** unchanged, and the previously verified accelerated CPU reference archive.
- Historical original Sionna freeze: commit **182e77b485cceb770ee6570ba332205c73823c91**, with its strict whole-tree integrity verifier.
- Historical accelerated freeze: commit **d19d9b8d598601e512757d8ec36468e95f1dc9f1**, containing the immutable package and 144-seed exact parity/timing report.
- Feature and experimental branches are **not automatically approved releases**. Their domain shift and sensitivity research do not overwrite the 0.50 frozen classifier.

The public main verifier checks the original pinned scientific and model files against the unchanged original freeze manifest. It deliberately permits new public README, .gitignore, security guidance, workflows and the accelerated release folder. A historical byte-identical whole-repository verification must be run against the exact original frozen commit rather than mutable main.

## Public-data and privacy boundaries

The original qualification record and some notebooks contain **links to external Google Drive folders**. Those identifiers are already visible in public history; repo editing alone does not revoke access. Review their sharing permissions and disable any unneeded public link access directly in Drive. Do not store credentials, private notes, personal files, signed customer agreements, personal data, or raw proprietary vehicle/RF recordings in Git.

An automatic scan checks the current repository and bundled ZIP for high-risk credential patterns and **reports file names only**, without printing any possible secrets. It is not a substitute for inspecting Git history, external Drive shares, or running GitHub secret scanning. If a secret was previously committed, assume it has been exposed: rotate it, revoke it at its issuer, and separately assess Git history cleanup.

Original RF datasets are hosted by their respective publishers and may have separate redistribution and commercial-use restrictions. Source links are not proof of permission to distribute raw IQ/CSI.

## Publication requirements

- Keep the model, freeze manifests, original simulated evidence and acceptance gates immutable.
- Label every metric as one of: held-out simulated, real source-processing, measured-channel digital injection, or independently labeled real hardware.
- Do not promote exploratory or threshold-selected experiments into blind qualification.
- Keep provenance, exact publisher checksums, actual elapsed-time units and physical array identity.
- Do not assign new license rights to external material. A software license must be explicitly chosen by the owner before this repository is described as open source.
- Report end-to-end RF acquisition separately from CPU-only model timings.

Run CI and verify all hashes before publishing a release tag. The accelerated reference ZIP is intentionally preserved, although the ordinary .gitignore ignores any newly generated arbitrary archives.
