const crypto = require('crypto');
const { db } = require('../infrastructure/db/db');
const { newId } = require('../utils/ids');
const { nowIso } = require('../utils/time');
const { parseJson, toJson } = require('../utils/json');
const {
  getJobById,
  getAttemptById,
  getCurrentConfigSnapshot,
  getPipelineStepsForSnapshot,
  normalizeJobStatus,
  normalizeAttemptStatus,
  upsertResultSummary,
  recalculateJobProgress,
} = require('./job-service');

function payloadHash(payload) {
  return crypto.createHash('sha256').update(JSON.stringify(payload)).digest('hex');
}

async function assertRuntimeCredential(jobId, rawToken, attemptId = null) {
  const { verifyCredential, markCredentialUsed } = require('./job-service');
  const credential = await verifyCredential(jobId, rawToken, 'report');
  if (!credential) {
    const err = new Error('Invalid runtime report token');
    err.statusCode = 403;
    throw err;
  }

  if (attemptId && credential.bound_attempt_id && credential.bound_attempt_id !== attemptId) {
    const err = new Error('Report token is bound to another attempt');
    err.statusCode = 409;
    throw err;
  }

  await markCredentialUsed(credential.id, attemptId || null);
  return credential;
}

async function insertReceiptIfNew(trx, {
  deliveryId,
  jobId,
  attemptId,
  endpointKind,
  sequenceNo,
  payload,
}) {
  if (!deliveryId || !String(deliveryId).trim()) {
    const err = new Error('deliveryId is required');
    err.statusCode = 400;
    throw err;
  }

  const existing = await trx('ingest_receipts').where({ delivery_id: deliveryId }).first();
  if (existing) {
    return { duplicate: true };
  }

  await trx('ingest_receipts').insert({
    delivery_id: deliveryId,
    job_id: jobId,
    attempt_id: attemptId || null,
    endpoint_kind: endpointKind,
    sequence_no: sequenceNo == null ? null : Number(sequenceNo),
    payload_hash: payloadHash(payload),
    received_at: nowIso(),
  });

  return { duplicate: false };
}

async function appendEvent(trx, {
  jobId,
  attemptId,
  stepKey,
  eventType,
  severity = 'info',
  sequenceNo = null,
  deliveryId,
  eventTime,
  payload,
}) {
  await trx('job_events').insert({
    id: newId('evt'),
    job_id: jobId,
    attempt_id: attemptId || null,
    step_key: stepKey || null,
    event_type: eventType,
    severity,
    sequence_no: sequenceNo == null ? null : Number(sequenceNo),
    delivery_id: deliveryId,
    event_time: eventTime || nowIso(),
    received_at: nowIso(),
    payload_json: toJson(payload || {}),
  });
}

async function applyAttemptSequenceGuard(trx, attemptId, sequenceNo) {
  if (sequenceNo == null) {
    return { apply: true, current: null };
  }

  const attempt = await trx('job_attempts').where({ id: attemptId }).first();
  const current = attempt?.last_sequence_no == null ? null : Number(attempt.last_sequence_no);

  if (current != null && Number(sequenceNo) < current) {
    return { apply: false, current };
  }

  await trx('job_attempts').where({ id: attemptId }).update({
    last_sequence_no: Number(sequenceNo),
    last_seen_at: nowIso(),
    updated_at: nowIso(),
  });

  return { apply: true, current };
}

async function getOrCreateStepRun(trx, {
  jobId,
  attemptId,
  stepKey,
  stepKind,
}) {
  let row = await trx('job_step_runs')
    .where({ attempt_id: attemptId, step_key: stepKey })
    .first();

  if (!row) {
    const now = nowIso();
    const next = {
      id: newId('srn'),
      job_id: jobId,
      attempt_id: attemptId,
      step_key: stepKey,
      step_kind: stepKind,
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
    await trx('job_step_runs').insert(next);
    row = await trx('job_step_runs').where({ id: next.id }).first();
  }

  return row;
}

async function handleStatus(jobId, body) {
  const attemptId = String(body.attemptId || '').trim();
  if (!attemptId) {
    throw new Error('attemptId is required');
  }

  const job = await getJobById(jobId);
  if (!job) {
    const err = new Error('Job not found');
    err.statusCode = 404;
    throw err;
  }

  const attempt = await getAttemptById(attemptId);
  if (!attempt || attempt.jobId !== jobId) {
    const err = new Error('Attempt not found');
    err.statusCode = 404;
    throw err;
  }

  await db.transaction(async (trx) => {
    const receipt = await insertReceiptIfNew(trx, {
      deliveryId: body.deliveryId,
      jobId,
      attemptId,
      endpointKind: 'status',
      sequenceNo: body.sequenceNo,
      payload: body,
    });
    if (receipt.duplicate) return;

    await appendEvent(trx, {
      jobId,
      attemptId,
      stepKey: body.stepKey || null,
      eventType: 'runtime.status',
      severity: 'info',
      sequenceNo: body.sequenceNo,
      deliveryId: body.deliveryId,
      eventTime: body.eventTime || nowIso(),
      payload: body,
    });

    const guard = await applyAttemptSequenceGuard(trx, attemptId, body.sequenceNo);
    if (!guard.apply) return;

    const normalizedJobStatus = normalizeJobStatus(body.status, job.status);
    const normalizedAttemptStatus = normalizeAttemptStatus(normalizedJobStatus);
    const now = nowIso();

    await trx('job_attempts').where({ id: attemptId }).update({
      status: normalizedAttemptStatus,
      stage: String(body.stage || attempt.stage || 'bootstrap'),
      first_seen_at: attempt.firstSeenAt || now,
      started_at: ['starting', 'running', 'finalizing', 'succeeded', 'failed', 'cancelled'].includes(normalizedAttemptStatus)
        ? (attempt.startedAt || now)
        : attempt.startedAt,
      last_seen_at: now,
      updated_at: now,
    });

    const jobPatch = {
      status: normalizedJobStatus,
      stage: String(body.stage || job.stage || 'bootstrap'),
      headline: body.message ? String(body.message) : job.headline,
      current_step_key: body.stepKey || null,
      updated_at: now,
    };

    if (['starting', 'running', 'finalizing', 'succeeded', 'failed', 'cancelled'].includes(normalizedJobStatus)) {
      jobPatch.started_at = job.startedAt || now;
    }
    if (['succeeded', 'failed', 'cancelled', 'timed_out', 'lost'].includes(normalizedJobStatus)) {
      jobPatch.finished_at = now;
    }

    await trx('jobs').where({ id: jobId }).update(jobPatch);
  });

  return { ok: true };
}

async function handleProgress(jobId, body) {
  const attemptId = String(body.attemptId || '').trim();
  const stepKey = String(body.stepKey || '').trim();

  if (!attemptId || !stepKey) {
    throw new Error('attemptId and stepKey are required');
  }

  const [job, attempt, snapshot] = await Promise.all([
    getJobById(jobId),
    getAttemptById(attemptId),
    getCurrentConfigSnapshot(jobId),
  ]);

  if (!job) {
    const err = new Error('Job not found');
    err.statusCode = 404;
    throw err;
  }
  if (!attempt || attempt.jobId !== jobId) {
    const err = new Error('Attempt not found');
    err.statusCode = 404;
    throw err;
  }
  if (!snapshot) {
    throw new Error('Job config snapshot not found');
  }

  const compiledSteps = await getPipelineStepsForSnapshot(snapshot.id);
  const compiled = compiledSteps.find((step) => step.stepKey === stepKey);
  const stepKind = compiled?.stepKind || String(body.stepKind || 'execute');

  await db.transaction(async (trx) => {
    const receipt = await insertReceiptIfNew(trx, {
      deliveryId: body.deliveryId,
      jobId,
      attemptId,
      endpointKind: 'progress',
      sequenceNo: body.sequenceNo,
      payload: body,
    });
    if (receipt.duplicate) return;

    await appendEvent(trx, {
      jobId,
      attemptId,
      stepKey,
      eventType: 'runtime.progress',
      severity: 'info',
      sequenceNo: body.sequenceNo,
      deliveryId: body.deliveryId,
      eventTime: body.eventTime || nowIso(),
      payload: body,
    });

    const guard = await applyAttemptSequenceGuard(trx, attemptId, body.sequenceNo);
    if (!guard.apply) return;

    const existing = await getOrCreateStepRun(trx, {
      jobId,
      attemptId,
      stepKey,
      stepKind,
    });

    const progress = body.progress && typeof body.progress === 'object' ? body.progress : {};
    const metrics = body.metrics && typeof body.metrics === 'object' ? body.metrics : {};

    const nextStepStatus = String(body.stepStatus || existing.status || 'running');
    const progressPercent =
      progress.percent != null
        ? Number(progress.percent)
        : (
            progress.current != null &&
            progress.total != null &&
            Number(progress.total) > 0
              ? (Number(progress.current) / Number(progress.total)) * 100
              : (existing.progress_percent == null ? 0 : Number(existing.progress_percent))
          );

    await trx('job_step_runs')
      .where({ id: existing.id })
      .update({
        status: nextStepStatus,
        progress_current: progress.current == null ? existing.progress_current : Number(progress.current),
        progress_total: progress.total == null ? existing.progress_total : Number(progress.total),
        progress_unit: progress.unit == null ? existing.progress_unit : String(progress.unit),
        progress_percent: Math.max(0, Math.min(100, progressPercent)),
        message: body.message == null ? existing.message : String(body.message),
        metrics_json: toJson({
          ...(parseJson(existing.metrics_json, {}) || {}),
          ...metrics,
        }),
        started_at: existing.started_at || nowIso(),
        finished_at: ['succeeded', 'failed', 'skipped', 'cancelled'].includes(nextStepStatus)
          ? (existing.finished_at || nowIso())
          : existing.finished_at,
        last_sequence_no: body.sequenceNo == null ? existing.last_sequence_no : Number(body.sequenceNo),
        error_summary: body.errorSummary == null ? existing.error_summary : String(body.errorSummary),
      });

    const jobProgress = await recalculateJobProgress(trx, jobId, attemptId, snapshot.id);

    await trx('job_attempts').where({ id: attemptId }).update({
      status: 'running',
      stage: 'pipeline',
      first_seen_at: attempt.firstSeenAt || nowIso(),
      started_at: attempt.startedAt || nowIso(),
      last_seen_at: nowIso(),
      updated_at: nowIso(),
    });

    await trx('jobs').where({ id: jobId }).update({
      status: 'running',
      stage: 'pipeline',
      current_step_key: stepKey,
      headline: body.message ? String(body.message) : `Step ${stepKey} is running`,
      progress_percent: jobProgress,
      started_at: job.startedAt || nowIso(),
      updated_at: nowIso(),
    });
  });

  return { ok: true };
}

async function getOrCreateLogStream(trx, {
  jobId,
  attemptId,
  stepKey,
  streamName,
}) {
  let row = await trx('job_log_streams')
    .where({
      job_id: jobId,
      attempt_id: attemptId,
      step_key: stepKey || null,
      stream_name: streamName,
    })
    .first();

  if (!row) {
    const next = {
      id: newId('lgs'),
      job_id: jobId,
      attempt_id: attemptId,
      step_key: stepKey || null,
      stream_name: streamName,
      created_at: nowIso(),
    };
    await trx('job_log_streams').insert(next);
    row = await trx('job_log_streams').where({ id: next.id }).first();
  }

  return row;
}

async function handleLogs(jobId, body) {
  const attemptId = String(body.attemptId || '').trim();
  if (!attemptId) {
    throw new Error('attemptId is required');
  }

  const attempt = await getAttemptById(attemptId);
  if (!attempt || attempt.jobId !== jobId) {
    const err = new Error('Attempt not found');
    err.statusCode = 404;
    throw err;
  }

  const chunks = Array.isArray(body.chunks) && body.chunks.length
    ? body.chunks
    : [
        {
          streamName: String(body.streamName || 'stdout'),
          stepKey: body.stepKey ? String(body.stepKey) : null,
          chunkSeq: body.chunkSeq == null ? 1 : Number(body.chunkSeq),
          offsetBytes: body.offsetBytes == null ? 0 : Number(body.offsetBytes),
          encoding: 'utf-8',
          compression: null,
          payload: String(body.logs || ''),
          eventTime: body.eventTime || nowIso(),
        },
      ];

  await db.transaction(async (trx) => {
    const receipt = await insertReceiptIfNew(trx, {
      deliveryId: body.deliveryId,
      jobId,
      attemptId,
      endpointKind: 'logs',
      sequenceNo: body.sequenceNo,
      payload: body,
    });
    if (receipt.duplicate) return;

    await appendEvent(trx, {
      jobId,
      attemptId,
      stepKey: body.stepKey || null,
      eventType: 'runtime.logs',
      severity: 'info',
      sequenceNo: body.sequenceNo,
      deliveryId: body.deliveryId,
      eventTime: body.eventTime || nowIso(),
      payload: {
        chunkCount: chunks.length,
      },
    });

    for (const chunk of chunks) {
      const stream = await getOrCreateLogStream(trx, {
        jobId,
        attemptId,
        stepKey: chunk.stepKey ? String(chunk.stepKey) : null,
        streamName: String(chunk.streamName || 'stdout'),
      });

      const existing = await trx('job_log_chunks')
        .where({
          stream_id: stream.id,
          chunk_seq: Number(chunk.chunkSeq || 1),
        })
        .first();

      if (existing) continue;

      const textPayload = String(chunk.payload || '');
      await trx('job_log_chunks').insert({
        id: newId('lgc'),
        stream_id: stream.id,
        chunk_seq: Number(chunk.chunkSeq || 1),
        offset_bytes: Number(chunk.offsetBytes || 0),
        size_bytes: Buffer.byteLength(textPayload, 'utf8'),
        encoding: String(chunk.encoding || 'utf-8'),
        compression: chunk.compression == null ? null : String(chunk.compression),
        text_payload: textPayload,
        blob_key: null,
        emitted_at: chunk.eventTime || null,
        received_at: nowIso(),
      });
    }

    await trx('job_attempts').where({ id: attemptId }).update({
      last_seen_at: nowIso(),
      updated_at: nowIso(),
    });
  });

  return { ok: true };
}

async function handleArtifactsRegister(jobId, body) {
  const attemptId = String(body.attemptId || '').trim();
  if (!attemptId) {
    throw new Error('attemptId is required');
  }

  const attempt = await getAttemptById(attemptId);
  if (!attempt || attempt.jobId !== jobId) {
    const err = new Error('Attempt not found');
    err.statusCode = 404;
    throw err;
  }

  const artifacts = Array.isArray(body.artifacts) ? body.artifacts : [];
  if (!artifacts.length) {
    throw new Error('artifacts[] is required');
  }

  await db.transaction(async (trx) => {
    const receipt = await insertReceiptIfNew(trx, {
      deliveryId: body.deliveryId,
      jobId,
      attemptId,
      endpointKind: 'artifacts_register',
      sequenceNo: body.sequenceNo,
      payload: body,
    });
    if (receipt.duplicate) return;

    await appendEvent(trx, {
      jobId,
      attemptId,
      stepKey: body.stepKey || null,
      eventType: 'runtime.artifacts_registered',
      severity: 'info',
      sequenceNo: body.sequenceNo,
      deliveryId: body.deliveryId,
      eventTime: body.eventTime || nowIso(),
      payload: {
        artifactCount: artifacts.length,
      },
    });

    for (const artifact of artifacts) {
      await trx('job_artifacts').insert({
        id: newId('art'),
        job_id: jobId,
        attempt_id: attemptId,
        step_key: artifact.stepKey ? String(artifact.stepKey) : null,
        artifact_type: String(artifact.artifactType || 'unknown'),
        role: artifact.role == null ? null : String(artifact.role),
        backend: String(artifact.backend || 'external'),
        uri: artifact.uri == null ? null : String(artifact.uri),
        storage_key: artifact.storageKey == null ? null : String(artifact.storageKey),
        content_type: artifact.contentType == null ? null : String(artifact.contentType),
        format: artifact.format == null ? null : String(artifact.format),
        size_bytes: artifact.sizeBytes == null ? null : Number(artifact.sizeBytes),
        checksum_sha256: artifact.checksumSha256 == null ? null : String(artifact.checksumSha256),
        metadata_json: toJson(artifact.metadata || {}),
        is_primary: artifact.isPrimary ? 1 : 0,
        previewable: artifact.previewable ? 1 : 0,
        sync_status: String(artifact.syncStatus || 'declared'),
        created_at: nowIso(),
      });
    }
  });

  return { ok: true };
}

async function handleFinal(jobId, body) {
  const attemptId = String(body.attemptId || '').trim();
  if (!attemptId) {
    throw new Error('attemptId is required');
  }

  const [job, attempt, snapshot] = await Promise.all([
    getJobById(jobId),
    getAttemptById(attemptId),
    getCurrentConfigSnapshot(jobId),
  ]);

  if (!job) {
    const err = new Error('Job not found');
    err.statusCode = 404;
    throw err;
  }
  if (!attempt || attempt.jobId !== jobId) {
    const err = new Error('Attempt not found');
    err.statusCode = 404;
    throw err;
  }

  const outcome = normalizeJobStatus(body.reportedOutcome || body.status || 'succeeded', 'succeeded');
  const summaryPayload = body.resultSummary && typeof body.resultSummary === 'object'
    ? body.resultSummary
    : {};

  await db.transaction(async (trx) => {
    const receipt = await insertReceiptIfNew(trx, {
      deliveryId: body.deliveryId,
      jobId,
      attemptId,
      endpointKind: 'final',
      sequenceNo: body.sequenceNo,
      payload: body,
    });
    if (receipt.duplicate) return;

    await appendEvent(trx, {
      jobId,
      attemptId,
      stepKey: null,
      eventType: 'runtime.final',
      severity: outcome === 'failed' ? 'error' : 'info',
      sequenceNo: body.sequenceNo,
      deliveryId: body.deliveryId,
      eventTime: body.finishedAt || body.eventTime || nowIso(),
      payload: body,
    });

    if (Array.isArray(body.stepOutcomes)) {
      for (const item of body.stepOutcomes) {
        const stepKey = String(item.stepKey || '').trim();
        if (!stepKey) continue;

        let existing = await trx('job_step_runs').where({
          attempt_id: attemptId,
          step_key: stepKey,
        }).first();

        if (!existing) {
          existing = await getOrCreateStepRun(trx, {
            jobId,
            attemptId,
            stepKey,
            stepKind: String(item.stepKind || 'execute'),
          });
        }

        const status = String(item.status || 'succeeded');
        await trx('job_step_runs').where({ id: existing.id }).update({
          status,
          progress_percent: ['succeeded', 'skipped'].includes(status) ? 100 : existing.progress_percent,
          finished_at: existing.finished_at || nowIso(),
          metrics_json: toJson({
            ...(parseJson(existing.metrics_json, {}) || {}),
            ...(item.metrics && typeof item.metrics === 'object' ? item.metrics : {}),
          }),
          error_summary: item.errorSummary == null ? existing.error_summary : String(item.errorSummary),
          last_sequence_no: body.sequenceNo == null ? existing.last_sequence_no : Number(body.sequenceNo),
        });
      }
    }

    if (Array.isArray(body.artifacts) && body.artifacts.length) {
      for (const artifact of body.artifacts) {
        await trx('job_artifacts').insert({
          id: newId('art'),
          job_id: jobId,
          attempt_id: attemptId,
          step_key: artifact.stepKey ? String(artifact.stepKey) : null,
          artifact_type: String(artifact.artifactType || 'unknown'),
          role: artifact.role == null ? null : String(artifact.role),
          backend: String(artifact.backend || 'external'),
          uri: artifact.uri == null ? null : String(artifact.uri),
          storage_key: artifact.storageKey == null ? null : String(artifact.storageKey),
          content_type: artifact.contentType == null ? null : String(artifact.contentType),
          format: artifact.format == null ? null : String(artifact.format),
          size_bytes: artifact.sizeBytes == null ? null : Number(artifact.sizeBytes),
          checksum_sha256: artifact.checksumSha256 == null ? null : String(artifact.checksumSha256),
          metadata_json: toJson(artifact.metadata || {}),
          is_primary: artifact.isPrimary ? 1 : 0,
          previewable: artifact.previewable ? 1 : 0,
          sync_status: String(artifact.syncStatus || 'declared'),
          created_at: nowIso(),
        });
      }
    }

    if (Array.isArray(body.externalRefs)) {
      for (const ref of body.externalRefs) {
        if (String(ref.backend || '').trim() !== 'huggingface') continue;
        await trx('huggingface_sync_states').insert({
          id: newId('hfs'),
          job_id: jobId,
          attempt_id: attemptId,
          repo_id: String(ref.repoId || 'unknown/repo'),
          repo_type: String(ref.repoType || 'model'),
          requested_revision: ref.revision == null ? null : String(ref.revision),
          last_seen_revision: null,
          status: 'pending',
          manifest_json: toJson({
            source: 'runtime.final',
            externalRef: ref,
          }),
          last_error: null,
          last_synced_at: null,
          next_retry_at: null,
          retry_count: 0,
          created_at: nowIso(),
          updated_at: nowIso(),
        });
      }
    }

    await upsertResultSummary(trx, jobId, attemptId, {
      outcome,
      headline: summaryPayload.headline || body.message || (outcome === 'failed' ? 'Job failed' : 'Job succeeded'),
      primaryMetrics: summaryPayload.primaryMetrics || {},
      summary: summaryPayload.summary || summaryPayload,
    });

    const progressPercent = snapshot
      ? await recalculateJobProgress(trx, jobId, attemptId, snapshot.id)
      : 100;

    const now = nowIso();
    await trx('job_attempts').where({ id: attemptId }).update({
      status: normalizeAttemptStatus(outcome),
      stage: outcome === 'succeeded' ? 'done' : attempt.stage,
      exit_code: body.exitCode == null ? null : Number(body.exitCode),
      failure_reason: outcome === 'failed' ? String(body.error || body.failureReason || 'runtime reported failure') : null,
      final_payload_received_at: now,
      finished_at: body.finishedAt || now,
      last_seen_at: now,
      updated_at: now,
    });

    await trx('jobs').where({ id: jobId }).update({
      status: outcome,
      stage: outcome === 'succeeded' ? 'reconcile' : 'done',
      current_step_key: null,
      headline: summaryPayload.headline || body.message || (outcome === 'failed' ? 'Job failed' : 'Job completed'),
      terminal_reason: outcome === 'failed' ? String(body.error || body.failureReason || 'runtime reported failure') : null,
      progress_percent: Math.max(progressPercent, 100),
      started_at: job.startedAt || attempt.startedAt || now,
      finished_at: body.finishedAt || now,
      updated_at: now,
    });
  });

  return { ok: true };
}

module.exports = {
  assertRuntimeCredential,
  handleStatus,
  handleProgress,
  handleLogs,
  handleArtifactsRegister,
  handleFinal,
};
