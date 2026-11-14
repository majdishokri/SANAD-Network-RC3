# Freeze policy — SANAD Network RC3, Sionna-qualified snapshot

`rc3.py` remains the only public inference API. The historical detector modules in `internal/rc0_runtime/` are read-only compatibility dependencies, not released alternative products. No training notebooks or independent RC0/RC1/RC2 releases are added.

**Model identity (unchanged):** SHA-256 `0a90835bb57a56437fc09211902b581bb22377676f91dce68f309c92e3a95bf4`; fixed threshold **0.50**; development seeds **2500–2643**; prospective seeds **3000–3143**.

**Evidence:** the pre-registered simulation study completed 144/144 prospective channel seeds on GPU Sionna. Three acceptance gates passed. See `qualification/SUMMARY_FROZEN.json`, `qualification/MANIFEST_FROZEN.json`, and `qualification/QUALIFICATION_REPORT.md`. Results apply to the simulated evaluation protocol only; no field calibration/actuation qualification is claimed.

**Transport:** quota-safe v3 changes Google Drive request pacing, caching, and retry only. It does not modify model, feature construction, labels, scoring, or acceptance criteria. Preserve the old frozen commit SHA `b96e016e696c818cb9fcb0e0de408284a49977c2` as baseline.

No new training, threshold change, feature modification, rescoring choice, or test-set selection on prospective outcomes. A future change must be a new release and independently qualified. A commit SHA is immutable, but Git branches are mutable unless protected.
