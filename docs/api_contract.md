# API Contract

## Callback Endpoints (Trainer Runtime -> Orchestrator)

### POST /api/jobs/status
Payload:
```json
{
  "job_id": "string",
  "status": "started|running|finished|failed",
  "stage": "training|merge_lora|...",
  "progress": 0.0,
  "message": "string"
}
```

### POST /api/jobs/logs
Payload:
```json
{
  "job_id": "string",
  "chunk": "string",
  "offset": 0
}
```

## Bootstrap Endpoint (Orchestrator -> Trainer Runtime)

### GET /api/v1/trainer/jobs/:jobId/bootstrap?token=...
Returns: `TrainerBootstrap` (see packages/contracts)
