# Architecture: Executor + Orchestrator

## Overview
Clean separation between Orchestrator (JS) and Executor (Python).

## Layers
- **Orchestrator**: api -> application -> domain -> infrastructure.
- **Executor**: bootstrap -> pipeline -> model/adapters.

## Key Fixes
- pipeline.evaluation schema consistency.
- merged/full archive upload visibility.
- Thin entrypoint for runner.py.
