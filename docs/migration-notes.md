# Migration Notes

- Migrated `src/` to `apps/orchestrator/src/`.
- Migrated `services/trainer-service/app/` to `apps/executor-trainer/app/`.
- Extracted schemas to `packages/contracts/`.
- Fixed relative imports across all tiers.
- Verified orchestrator startup via `/health`.
