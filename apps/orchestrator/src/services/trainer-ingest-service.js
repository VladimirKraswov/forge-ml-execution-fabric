const crypto = require('crypto');
const { db } = require('../db');
const { newId } = require('../utils/ids');
const { nowIso } = require('../utils/time');
const { parseJson, toJson } = require('../utils/json');
const { verifyCredential, markCredentialUsed, upsertResultSummary, recalculateJobProgress } = require('./job-service');
const { saveUploadedArtifactFile } = require('./artifact-storage-service');

const ARTIFACT_TYPE_MAP = {
  logs: 'logs',
  config: 'config',
  summary: 'summary',
  'train-metrics': 'train_metrics',
  'train-history': 'train_history',
  'eval-summary': 'eval_summary',
  'eval-details': 'eval_details',
  lora: 'lora_archive',
  merged: 'merged_archive',
  'full-archive': 'full_archive',
};

function sha256Text(value) {
  return crypto.createHash('sha256').update(String(value || '')).digest('hex');
}

function deepClone(value) {
  return JSON.parse(JSON.stringify(value == null ? {} : value));
}

function clampProgress(value, fallback = null) {
  if (value == null || value === '') return fallback;
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.max(0, Math.min(100, parsed));
}

function normalizeReportedStatus(value, fallback = 'running') {
  const raw = String(value || '').trim().toLowerCase();
  if (!raw) return fallback;
  if (['queued', 'pending'].includes(raw)) return 'queued';
  if (['started', 'starting'].includes(raw)) return 'started';
  if (['running'].includes(raw)) return 'running';
  if (['finished', 'success', 'completed', 'succeeded'].includes(raw)) return 'finished';
  if (['failed', 'error'].includes(raw)) return 'failed';
  if (['cancelled', 'canceled'].includes(raw)) return 'cancelled';
  return fallback;
}

async function assertTrainerRuntimeCredential(jobId, rawToken) {
  const credential = await verifyCredential(jobId, rawToken, 'report');
  if (!credential) {
    const error = new Error('Invalid trainer runtime token');
    error.statusCode = 403;
    throw error;
  }
  await markCredentialUsed(credential.id);
  return credential;
}

async function getTrainerContext(jobId) {
  const job = await db('jobs').where({ id: jobId }).first();
  if (!job) {
    const error = new Error('Job not found');
    error.statusCode = 404;
    throw error;
  }
  const attempt = await db('job_attempts').where({ id: job.latest_attempt_id }).first();
  if (!attempt) {
    const error = new Error('Job attempt not found');
    error.statusCode = 404;
    throw error;
  }
  const snapshot = job.current_config_snapshot_id
    ? await db('job_config_snapshots').where({ id: job.current_config_snapshot_id }).first()
    : null;

  const parsedSnapshot = snapshot ? parseJson(snapshot.snapshot_json, {}) : null;
  return { job, attempt, snapshot, parsedSnapshot };
}

async function insertReceiptIfNew(trx, { jobId, attemptId, endpointKind, uniqueKey, payload }) {
  const deliveryId = sha256Text(`${endpointKind}|${jobId}|${uniqueKey}`);
  const existing = await trx('ingest_receipts').where({ delivery_id: deliveryId }).first();
  if (existing) {
    return { duplicate: true, deliveryId };
  }

  await trx('ingest_receipts').insert({
    delivery_id: deliveryId,
    job_id: jobId,
    attempt_id: attemptId,
    endpoint_kind: endpointKind,
    sequence_no: null,
    payload_hash: sha256Text(JSON.stringify(payload || {})),
    received_at: nowIso(),
  });

  return { duplicate: false, deliveryId };
}

async function appendEvent(trx, {
  jobId,
  attemptId,
  stepKey = null,
  eventType,
  severity = 'info',
  deliveryId,
  eventTime,
  payload,
}) {
  await trx('job_events').insert({
    id: newId('evt'),
    job_id: jobId,
    attempt_id: attemptId,
    step_key: stepKey,
    event_type: eventType,
    severity,
    sequence_no: null,
    delivery_id: deliveryId,
    event_time: eventTime || nowIso(),
    received_at: nowIso(),
    payload_json: toJson(payload || {}),
  });
}

function stageToStepKey(stage) {
  const raw = String(stage || '').trim();
  if (!raw) return 'bootstrap';
  if (['bootstrap', 'hf_login'].includes(raw)) return 'bootstrap';
  if (['prepare_assets'].includes(raw)) return 'prepare_assets';
  if (['load_model', 'load_dataset', 'training', 'save_lora', 'train_completed'].includes(raw)) return 'training';
  if (['merge_lora'].includes(raw)) return 'merge_lora';
  if (['evaluation_prepare', 'evaluation', 'evaluation_completed'].includes(raw)) return 'evaluation';
  if (['publish_artifacts'].includes(raw)) return 'publish';
  if (['upload_artifacts', 'finished'].includes(raw)) return 'upload';
  return raw;
}

async function getCompiledSteps(snapshotId, trx = db) {
  if (!snapshotId) return [];
  return trx('job_pipeline_steps').where({ config_snapshot_id: snapshotId }).orderBy('order_index', 'asc');
}

async function getOrCreateStepRun(trx, { jobId, attemptId, snapshotId, stepKey }) {
  let row = await trx('job_step_runs').where({ attempt_id: attemptId, step_key: stepKey }).first();
  if (row) return row;

  const compiled = await getCompiledSteps(snapshotId, trx);
  const step = compiled.find((item) => item.step_key === stepKey) || null;
  const now = nowIso();
  const insert = {
    id: newId('srn'),
    job_id: jobId,
    attempt_id: attemptId,
    step_key: stepKey,
    step_kind: step?.step_kind || stepKey,
    status: 'pending',
    progress_current: null,
    progress_total: null,
    progress_unit: null,
    progress_percent: 0,
    message: null,
    metrics_json: toJson({}),
    started_at: null,
    finished_at: null,
    last_sequence_no: null,
    error_summary: null,
  };
  await trx('job_step_runs').insert(insert);
  return trx('job_step_runs').where({ id: insert.id }).first();
}

function derivePrimaryMetrics(resultPayload) {
  const trainingSummary = resultPayload?.training?.summary || {};
  const evaluationSummary = resultPayload?.evaluation?.summary || {};
  return {
    trainLoss: trainingSummary.train_loss ?? trainingSummary.final_loss ?? null,
    trainRuntime: trainingSummary.train_runtime ?? null,
    evalMae: evaluationSummary.mae ?? null,
    evalRmse: evaluationSummary.rmse ?? null,
    evalParseSuccessRate: evaluationSummary.parseSuccessRate ?? null,
  };
}

async function upsertExternalUploadArtifacts(trx, { jobId, attemptId, uploads }) {
  if (!uploads || typeof uploads !== 'object') return;

  for (const [key, value] of Object.entries(uploads)) {
    if (!value || typeof value !== 'object') continue;
    const uri = value.download_url || value.url || null;
    const storageKey = value.storage_key || value.archive_path || value.path || null;

    await trx('job_artifacts').insert({
      id: newId('art'),
      job_id: jobId,
      attempt_id: attemptId,
      step_key: null,
      artifact_type: String(key),
      role: 'external_upload',
      backend: uri ? 'external_url' : 'external_ref',
      uri: uri == null ? null : String(uri),
      storage_key: storageKey == null ? null : String(storageKey),
      content_type: null,
      format: null,
      size_bytes: null,
      checksum_sha256: null,
      metadata_json: toJson(value),
      is_primary: 0,
      previewable: 0,
      sync_status: 'uploaded',
      created_at: nowIso(),
    });
  }
}

async function handleTrainerStatus(body) {
  const jobId = String(body.job_id || '').trim();
  if (!jobId) throw new Error('job_id is required');

  const { job, attempt, snapshot } = await getTrainerContext(jobId);
  const stepKey = stageToStepKey(body.stage);
  const reportedStatus = normalizeReportedStatus(body.status, job.status || 'running');
  const uniqueKey = JSON.stringify({
    status: reportedStatus,
    stage: body.stage || null,
    progress: body.progress ?? null,
    message: body.message || null,
    timestamp: body.timestamp ?? null,
    logs: body.logs || null,
    extra: body.extra || null,
  });

  await db.transaction(async (trx) => {
    const receipt = await insertReceiptIfNew(trx, {
      jobId,
      attemptId: attempt.id,
      endpointKind: 'trainer_status',
      uniqueKey,
      payload: body,
    });
    if (receipt.duplicate) return;

    const stepRun = await getOrCreateStepRun(trx, { jobId, attemptId: attempt.id, snapshotId: snapshot?.id, stepKey });
    const now = nowIso();
    const progress = clampProgress(body.progress, null);

    await appendEvent(trx, {
      jobId,
      attemptId: attempt.id,
      stepKey,
      eventType: 'trainer.status',
      severity: reportedStatus === 'failed' ? 'error' : 'info',
      deliveryId: receipt.deliveryId,
      eventTime: body.timestamp ? new Date(Number(body.timestamp) * 1000).toISOString() : now,
      payload: body,
    });

    await trx('job_step_runs').where({ id: stepRun.id }).update({
      status: reportedStatus === 'finished' ? 'succeeded' : (reportedStatus === 'failed' ? 'failed' : 'running'),
      progress_percent: progress == null ? stepRun.progress_percent : progress,
      message: body.message == null ? stepRun.message : String(body.message),
      metrics_json: toJson({ ...(parseJson(stepRun.metrics_json, {}) || {}), ...(body.extra || {}) }),
      started_at: stepRun.started_at || now,
      finished_at: reportedStatus === 'finished' || reportedStatus === 'failed' ? now : stepRun.finished_at,
      error_summary: reportedStatus === 'failed' ? String(body.message || 'trainer reported failure') : stepRun.error_summary,
    });

    let jobProgress = clampProgress(body.progress, null);
    if (snapshot?.id) {
      jobProgress = await recalculateJobProgress(trx, jobId, attempt.id, snapshot.id);
    }

    await trx('job_attempts').where({ id: attempt.id }).update({
      status: reportedStatus,
      stage: String(body.stage || attempt.stage || 'bootstrap'),
      first_seen_at: attempt.first_seen_at || now,
      started_at: attempt.started_at || now,
      last_seen_at: now,
      finished_at: reportedStatus === 'finished' || reportedStatus === 'failed' ? now : attempt.finished_at,
      failure_reason: reportedStatus === 'failed' ? String(body.message || 'trainer reported failure') : attempt.failure_reason,
      updated_at: now,
    });

    await trx('jobs').where({ id: jobId }).update({
      status: reportedStatus,
      stage: String(body.stage || job.stage || 'bootstrap'),
      current_step_key: stepKey,
      headline: body.message ? String(body.message) : job.headline,
      terminal_reason: reportedStatus === 'failed' ? String(body.message || 'trainer reported failure') : job.terminal_reason,
      progress_percent: jobProgress == null ? job.progress_percent : jobProgress,
      started_at: job.started_at || now,
      finished_at: reportedStatus === 'finished' || reportedStatus === 'failed' ? now : job.finished_at,
      updated_at: now,
    });
  });

  return { ok: true };
}

async function handleTrainerProgress(body) {
  const jobId = String(body.job_id || '').trim();
  if (!jobId) throw new Error('job_id is required');

  const { job, attempt, snapshot } = await getTrainerContext(jobId);
  const stepKey = stageToStepKey(body.stage);
  const uniqueKey = JSON.stringify({
    stage: body.stage || null,
    progress: body.progress ?? null,
    message: body.message || null,
    timestamp: body.timestamp ?? null,
    extra: body.extra || null,
  });

  await db.transaction(async (trx) => {
    const receipt = await insertReceiptIfNew(trx, {
      jobId,
      attemptId: attempt.id,
      endpointKind: 'trainer_progress',
      uniqueKey,
      payload: body,
    });
    if (receipt.duplicate) return;

    const stepRun = await getOrCreateStepRun(trx, { jobId, attemptId: attempt.id, snapshotId: snapshot?.id, stepKey });
    const now = nowIso();
    const progress = clampProgress(body.progress, 0);

    await appendEvent(trx, {
      jobId,
      attemptId: attempt.id,
      stepKey,
      eventType: 'trainer.progress',
      severity: 'info',
      deliveryId: receipt.deliveryId,
      eventTime: body.timestamp ? new Date(Number(body.timestamp) * 1000).toISOString() : now,
      payload: body,
    });

    await trx('job_step_runs').where({ id: stepRun.id }).update({
      status: 'running',
      progress_percent: progress,
      message: body.message == null ? stepRun.message : String(body.message),
      metrics_json: toJson({ ...(parseJson(stepRun.metrics_json, {}) || {}), ...(body.extra || {}) }),
      started_at: stepRun.started_at || now,
      finished_at: stepRun.finished_at,
      error_summary: null,
    });

    let jobProgress = progress;
    if (snapshot?.id) {
      jobProgress = await recalculateJobProgress(trx, jobId, attempt.id, snapshot.id);
    }

    await trx('job_attempts').where({ id: attempt.id }).update({
      status: 'running',
      stage: String(body.stage || attempt.stage || 'bootstrap'),
      first_seen_at: attempt.first_seen_at || now,
      started_at: attempt.started_at || now,
      last_seen_at: now,
      updated_at: now,
    });

    await trx('jobs').where({ id: jobId }).update({
      status: 'running',
      stage: String(body.stage || job.stage || 'bootstrap'),
      current_step_key: stepKey,
      headline: body.message ? String(body.message) : job.headline,
      progress_percent: jobProgress,
      started_at: job.started_at || now,
      updated_at: now,
    });
  });

  return { ok: true };
}

async function handleTrainerLogs(body) {
  const jobId = String(body.job_id || '').trim();
  const chunk = String(body.chunk || '');
  if (!jobId) throw new Error('job_id is required');
  if (!chunk) return { ok: true };

  const { job, attempt } = await getTrainerContext(jobId);
  const offset = Number(body.offset || 0);
  const uniqueKey = `${offset}:${sha256Text(chunk)}`;

  await db.transaction(async (trx) => {
    const receipt = await insertReceiptIfNew(trx, {
      jobId,
      attemptId: attempt.id,
      endpointKind: 'trainer_logs',
      uniqueKey,
      payload: body,
    });
    if (receipt.duplicate) return;

    let stream = await trx('job_log_streams')
      .where({ job_id: jobId, attempt_id: attempt.id, step_key: null, stream_name: 'trainer.log' })
      .first();

    if (!stream) {
      const insert = {
        id: newId('lgs'),
        job_id: jobId,
        attempt_id: attempt.id,
        step_key: null,
        stream_name: 'trainer.log',
        created_at: nowIso(),
      };
      await trx('job_log_streams').insert(insert);
      stream = insert;
    }

    const maxChunkRow = await trx('job_log_chunks').where({ stream_id: stream.id }).max({ max: 'chunk_seq' }).first();
    const chunkSeq = Number(maxChunkRow?.max || 0) + 1;

    await trx('job_log_chunks').insert({
      id: newId('lgc'),
      stream_id: stream.id,
      chunk_seq: chunkSeq,
      offset_bytes: Number.isFinite(offset) ? offset : 0,
      size_bytes: Buffer.byteLength(chunk, 'utf-8'),
      encoding: 'utf-8',
      compression: null,
      text_payload: chunk,
      blob_key: null,
      emitted_at: nowIso(),
      received_at: nowIso(),
    });

    await appendEvent(trx, {
      jobId,
      attemptId: attempt.id,
      stepKey: null,
      eventType: 'trainer.logs',
      severity: 'info',
      deliveryId: receipt.deliveryId,
      eventTime: nowIso(),
      payload: {
        offset,
        sizeBytes: Buffer.byteLength(chunk, 'utf-8'),
      },
    });

    await trx('job_attempts').where({ id: attempt.id }).update({
      status: ['queued', 'started'].includes(attempt.status) ? 'running' : attempt.status,
      last_seen_at: nowIso(),
      updated_at: nowIso(),
    });

    await trx('jobs').where({ id: jobId }).update({
      status: ['queued', 'started'].includes(job.status) ? 'running' : job.status,
      updated_at: nowIso(),
    });
  });

  return { ok: true };
}

async function handleTrainerFinal(body) {
  const jobId = String(body.job_id || '').trim();
  if (!jobId) throw new Error('job_id is required');

  const { job, attempt, snapshot } = await getTrainerContext(jobId);
  const resultPayload = deepClone(body.result || {});
  const reportedStatus = normalizeReportedStatus(body.status || resultPayload.status, 'finished');
  const outcome = reportedStatus === 'failed' ? 'failed' : (reportedStatus === 'cancelled' ? 'cancelled' : 'finished');
  const uniqueKey = JSON.stringify({
    status: outcome,
    finishedAt: resultPayload.finished_at || body.timestamp || null,
    error: resultPayload.error || null,
    uploads: resultPayload.uploads || null,
  });

  await db.transaction(async (trx) => {
    const receipt = await insertReceiptIfNew(trx, {
      jobId,
      attemptId: attempt.id,
      endpointKind: 'trainer_final',
      uniqueKey,
      payload: body,
    });
    if (receipt.duplicate) return;

    await appendEvent(trx, {
      jobId,
      attemptId: attempt.id,
      stepKey: null,
      eventType: 'trainer.final',
      severity: outcome === 'failed' ? 'error' : 'info',
      deliveryId: receipt.deliveryId,
      eventTime: resultPayload.finished_at || nowIso(),
      payload: body,
    });

    const compiled = await getCompiledSteps(snapshot?.id, trx);
    for (const step of compiled) {
      if (!step.enabled) continue;
      const stepRun = await getOrCreateStepRun(trx, { jobId, attemptId: attempt.id, snapshotId: snapshot?.id, stepKey: step.step_key });
      const currentMetrics = parseJson(stepRun.metrics_json, {}) || {};
      let status = stepRun.status;
      if (outcome === 'finished' && ['pending', 'running'].includes(stepRun.status)) {
        status = 'succeeded';
      }
      if (outcome === 'failed' && step.step_key === stageToStepKey(job.stage)) {
        status = 'failed';
      }
      await trx('job_step_runs').where({ id: stepRun.id }).update({
        status,
        progress_percent: status === 'succeeded' ? 100 : stepRun.progress_percent,
        metrics_json: toJson(currentMetrics),
        started_at: stepRun.started_at || job.started_at || nowIso(),
        finished_at: status === 'succeeded' || status === 'failed' ? (stepRun.finished_at || nowIso()) : stepRun.finished_at,
      });
    }

    await upsertExternalUploadArtifacts(trx, {
      jobId,
      attemptId: attempt.id,
      uploads: resultPayload.uploads,
    });

    await upsertResultSummary(trx, jobId, attempt.id, {
      outcome,
      headline: resultPayload.error
        ? String(resultPayload.error)
        : (body.message || (outcome === 'failed' ? 'Trainer job failed' : 'Trainer job finished')),
      primaryMetrics: derivePrimaryMetrics(resultPayload),
      summary: resultPayload,
    });

    const progressPercent = snapshot?.id
      ? Math.max(await recalculateJobProgress(trx, jobId, attempt.id, snapshot.id), outcome === 'finished' ? 100 : 0)
      : (outcome === 'finished' ? 100 : job.progress_percent || 0);

    const finishedAt = resultPayload.finished_at || nowIso();
    await trx('job_attempts').where({ id: attempt.id }).update({
      status: outcome,
      stage: outcome === 'finished' ? 'finished' : String(job.stage || 'failed'),
      started_at: attempt.started_at || resultPayload.started_at || nowIso(),
      last_seen_at: nowIso(),
      finished_at: finishedAt,
      failure_reason: outcome === 'failed' ? String(resultPayload.error || body.message || 'trainer reported failure') : null,
      final_payload_received_at: nowIso(),
      updated_at: nowIso(),
    });

    await trx('jobs').where({ id: jobId }).update({
      status: outcome,
      stage: outcome === 'finished' ? 'finished' : String(job.stage || 'failed'),
      current_step_key: null,
      headline: outcome === 'failed'
        ? String(resultPayload.error || body.message || 'Trainer job failed')
        : 'Trainer job finished',
      terminal_reason: outcome === 'failed' ? String(resultPayload.error || body.message || 'trainer reported failure') : null,
      progress_percent: progressPercent,
      started_at: job.started_at || resultPayload.started_at || nowIso(),
      finished_at: finishedAt,
      updated_at: nowIso(),
    });
  });

  return { ok: true };
}

async function handleTrainerUpload({ artifactTypeParam, body, file, baseUrl }) {
  const jobId = String(body.job_id || '').trim();
  if (!jobId) throw new Error('job_id is required');
  if (!file?.path) throw new Error('file is required');

  const { attempt } = await getTrainerContext(jobId);
  const artifactType = ARTIFACT_TYPE_MAP[String(artifactTypeParam || '').trim()] || String(body.artifact_type || artifactTypeParam || 'unknown');
  const stored = await saveUploadedArtifactFile({
    sourcePath: file.path,
    originalName: file.originalname,
    jobId,
    artifactType,
  });

  const artifactId = newId('art');
  const downloadUrl = `${String(baseUrl || '').replace(/\/+$/, '')}/api/v1/trainer/jobs/${encodeURIComponent(jobId)}/artifacts/${artifactId}/download`;

  await db.transaction(async (trx) => {
    await trx('job_artifacts')
      .where({ job_id: jobId, attempt_id: attempt.id, artifact_type: artifactType })
      .update({ is_primary: 0 });

    await trx('job_artifacts').insert({
      id: artifactId,
      job_id: jobId,
      attempt_id: attempt.id,
      step_key: null,
      artifact_type: artifactType,
      role: 'uploaded',
      backend: 'local_fs',
      uri: downloadUrl,
      storage_key: stored.storageKey,
      content_type: file.mimetype || stored.contentType,
      format: null,
      size_bytes: stored.sizeBytes,
      checksum_sha256: stored.checksumSha256,
      metadata_json: toJson({
        originalFilename: file.originalname,
        uploadedArtifactType: body.artifact_type || artifactTypeParam,
      }),
      is_primary: 1,
      previewable: artifactType === 'logs' || artifactType === 'config' || artifactType === 'summary' ? 1 : 0,
      sync_status: 'stored',
      created_at: nowIso(),
    });

    await trx('job_events').insert({
      id: newId('evt'),
      job_id: jobId,
      attempt_id: attempt.id,
      step_key: null,
      event_type: 'trainer.upload.received',
      severity: 'info',
      sequence_no: null,
      delivery_id: sha256Text(`upload|${jobId}|${artifactType}|${stored.checksumSha256}`),
      event_time: nowIso(),
      received_at: nowIso(),
      payload_json: toJson({
        artifactId,
        artifactType,
        storageKey: stored.storageKey,
        sizeBytes: stored.sizeBytes,
      }),
    });
  });

  return {
    ok: true,
    artifact_id: artifactId,
    job_id: jobId,
    artifact_type: artifactType,
    storage_key: stored.storageKey,
    download_url: downloadUrl,
  };
}

module.exports = {
  assertTrainerRuntimeCredential,
  handleTrainerStatus,
  handleTrainerProgress,
  handleTrainerLogs,
  handleTrainerFinal,
  handleTrainerUpload,
};
