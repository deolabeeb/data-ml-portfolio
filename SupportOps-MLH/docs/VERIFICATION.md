# Verification of the mock-only candidate

23 October 2026, Windows / CPython 3.12.14, reused established project dependencies.

- Current original working-tree baseline: 192 passed, one existing deprecation warning, 35.09 seconds. This single pass does not resolve or promote the historically blocked S04 provider extension.
- Candidate: 139 passed in23.52seconds, one existing deprecation warning, no failures or skips. All 134 pre-existing mock regression cases retained; five new sample scope cases. Exact duration and XML receipt are retained in the programme report. Original test files are byte-identical.
- Six-choice scripted demo: 201, 200, 200, 404, 503, 200. Counts 0, 1, 2, then recovery3; failed turn does not advance count. TestClient in-process transport, socket guard and zero model calls. This was scripted execution, not owner participation or browser testing.
- Source/starter hashes unchanged. Candidate is a new mock-only derivative, not a byte-identical historical M1 snapshot. The provider-specific58 cases are outside this sample because their feature is excluded, not repaired or hidden in the original.
- First candidate run:137passed2failed because a new configuration error message broke a preserved prefix assertion. Implementation corrected; no old assertion weakened. Failed receipt retained internally.
- Pinned dependencies are the tested environment, not a security certification. Clean installation into a new environment, other OS/Python versions, live models, remote CI and production deployment remain unverified.

## Built package check

A wheel was built offline with bundled setuptools/wheel, installed without dependencies into an isolated target directory and imported under isolated Python mode. Browser HTML, own/foreign order reads and policy JSON fixtures passed a socket-blocked TestClient smoke probe. Established application dependencies were reused; this is not a clean dependency-install claim. The first wheel attempt in the project venv failed because setuptools was absent; bundled build tools succeeded without installing or changing anything globally. The wheel binary is internal evidence and is not included in the source ZIP.

## Reproduce

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\demo_m1.py --output artifacts\my-demo
```

Choose1,2,3,4,5,6,q. Use a fresh output directory. Read `evidence/demo.json` and `evidence/verification.json` for recorded synthetic results. No SQLite database, local environment or secret is included in the source archive. Public tokens in source are explicitly fictional demo fixtures.

Owner understanding and MLH acceptance are separate checks. The owner confirmed the starter was generated for him in ChatGPT; that origin question is resolved. Public release remains subject to exact publishing approval. Nothing has been published or submitted.
