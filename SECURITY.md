# Security policy

**Status:** research software only. There is no verified support for safety-critical, autonomous or direct RF-hardware control.

## Report a vulnerability privately

Do not publish exploit steps, sensitive evidence, API keys or private dataset URLs in public issues. Use GitHub private vulnerability reporting if the repository has it enabled, or contact the maintainer using an existing private channel. Public discussion should contain no credential material.

## Threat model and safe usage

- The distributed pretrained model is pinned by SHA-256. Never load third-party unverified Python joblib/pickle models; deserializing them may execute arbitrary code.
- Never paste cloud notebook, Google Drive, GitHub or radio management credentials into scripts or notebooks. Use local environment/secret managers for anything sensitive.
- The repository has no hardware actuation pathway. Predictions must not be attached directly to physical corrections, access control or safety interlocks without independent validation and authorization.
- The current code and accelerated archive are scanned for common credential signatures, but static scanning cannot establish absence of all secrets or exposures in Git history.
- Some original research artifacts reference external Drive folders. GitHub public visibility does not mean those folders should be publicly shared. Review Drive permissions; deleting a link in Git does not remove historical access.
- Treat raw RF, location traces and customer telemetry as potentially sensitive, and follow the source publisher's terms.

## Supported version

The public main branch is for the documentation/release surface; the frozen RC3 reference is the accelerated CPU V1 artifact. Prototype feature branches are unsupported, unqualified research.
