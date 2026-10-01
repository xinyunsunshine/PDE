# Public release checklist

## Release manifest

Included in the initial release:

- `pde/`: stable discovery, artifact, sampling, and mixed-backprop interfaces;
- `rlinf/`: the VLA PPO stack and the historical PDE/HER integration;
- `config/libero_pro_*/pi05/`: pi0.5 LIBERO-PRO PDE and PPO configurations;
- `examples/pde/`: artifact examples and the microwave walkthrough inputs;
- `tests/unit_tests/test_pde_release.py`: CPU-only contract tests;
- license, citation, contribution guide, and upstream attribution.

Excluded from the repository:

- checkpoints, datasets, virtual environments, caches, logs, scheduler output,
  rollout dumps, raw robot recordings, and W&B state;
- unpublished benchmark assets or files whose redistribution terms are unclear;
- credentials, private endpoints, usernames, and cluster-specific launch files.

Published separately with stable URLs:

- weak pi0.5 SFT checkpoint and checksum;
- paper prompt pools for the released task split;
- task manifest, benchmark revision, and aggregate result files.

## Required before v0.1.0

- [ ] Replace the repository-owner placeholder in project metadata.
- [ ] Add the exact checkpoint URL, revision, license, and SHA-256.
- [ ] Add the paper prompt pools and their generation metadata.
- [ ] Record the LIBERO-PRO commit/release and released task IDs.
- [ ] Add one public-machine smoke-run transcript.
- [ ] Confirm redistribution rights for all retained assets and benchmark files.
- [ ] Run `python scripts/release_audit.py .` and the unit test suite. The audit
  intentionally fails while the repository-owner placeholder remains.
- [ ] Create the public GitHub repository, push `main`, and tag `v0.1.0`.

## Known boundary

The original research tree contains many exploratory configs and analyses. This
release snapshot keeps the upstream RLinf substrate but documents pi0.5 +
LIBERO-PRO as the supported PDE path. Additional models and environments should
be promoted only after their configs, artifacts, and smoke tests are complete.
