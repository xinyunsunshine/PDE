# Public release checklist

Included in this repository:

- the framework-independent prompt discovery and prompt-pool API;
- PDE subclasses of the pinned RLinf runtime;
- one schema-valid example prompt pool;
- CPU contract and repository-structure tests;
- focused installation, architecture, citation, and license metadata.

The RLinf framework is an unmodified Git submodule pinned to commit
`fce5435df9472e2c61957e4f849fb903fc70827c`.

Required before the first result-bearing release:

- [ ] Publish the exact checkpoint URL, revision, license, and SHA-256.
- [ ] Publish the paper prompt pools and generation metadata.
- [ ] Record the LIBERO-PRO revision and released task IDs.
- [ ] Add one public-machine smoke-run transcript.
- [ ] Run `pytest -q`, `ruff check pde tests`, and the release audit.
- [ ] Tag the tested commit.

Checkpoints, datasets, logs, rollout videos, credentials, private endpoints,
cluster launch files, and externally licensed benchmark assets are excluded.
