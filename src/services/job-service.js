const crypto = require('crypto');
const { db } = require('../db');
const { newId } = require('../utils/ids');
const { nowIso, addMinutes, addHours } = require('../utils/time');
const { parseJson, toJson } = require('../utils/json');
const { hashToken, issueOpaqueToken } = require('../utils/crypto');
const { getRuntimeProfileById } = require('./runtime-profile-service');
const { CONFIG } = require('../config');

function normalizeJobStatus(value, fallback = 'running') {
  const raw = String(value || '').trim().toLowerCase();
  if (!raw) return fallback;
  if (['success', 'completed', 'finished', 'succeeded'].includes(raw)) return 'succeeded';
  if (['failed', 'error'].includes(raw)) return 'failed';
  if (['cancelled', 'canceled'].includes(raw)) return 'cancelled';
  if (raw === 'timed_out') return 'timed_out';
  if (raw === 'lost') return 'lost';
  if (['created', 'ready', 'starting', 'running', 'finalizing'].includes(raw)) return raw;
  return fallback;
}

function normalizeAttemptStatus(jobStatus) {
  switch (jobStatus) {
    case 'created':
    case 'ready':
      return 'issued';
    case 'starting':
      return 'starting';
    case 'running':
      return 'running';
    case 'finalizing':
      return 'finalizing';
    case 'succeeded':
      return 'succeeded';
    case 'failed':
      return 'failed';
    case 'cancelled':
      return 'cancelled';
    case 'timed_out':
      return 'timed_out';
    case 'lost':
      return 'lost';
    default:
      return 'running';
  }
}

function mapJob(row) {
  if (!row) return null;
  return {
    id: row.id,
    workspaceId: row.workspace_id,
    projectId: row.project_id,
    createdByUserId: row.created_by_user_id,
    name: row.name,
    jobKind: row.job_kind,
    status: row.status,
    stage: row.stage,
    desiredState: row.desired_state,
    runtimeProfileId: row.runtime_profile_id,
    currentConfigSnapshotId: row.current_config_snapshot_id,
    latestAttemptId: row.latest_attempt_id,
    currentStepKey: row.current_step_key,
    labels: parseJson(row.labels_json, {}),
    headline: row.headline,
    terminalReason: row.terminal_reason,
    progressPercent: Number(row.progress_percent || 0),
    createdAt: row.created_at,
    startedAt: row.started_at,
    finishedAt: row.finished_at,
    updatedAt: row.updated_at,
  };
}

function mapAttempt(row) {
  if (!row) return null;
  return {
    id: row.id,
    jobId: row.job_id,
    attemptNo: row.attempt_no,
    status: row.status,
    stage: row.stage,
    runtimeImage: row.runtime_image,
    executorVersion: row.executor_version,
    hostInfo: parseJson(row.host_info_json, {}),
    runtimeInfo: parseJson(row.runtime_info_json, {}),
    firstSeenAt: row.first_seen_at,
    configFetchedAt: row.config_fetched_at,
    startedAt: row.started_at,
    lastSeenAt: row.last_seen_at,
    finishedAt: row.finished_at,
    exitCode: row.exit_code,
    failureReason: row.failure_reason,
    finalPayloadReceivedAt: row.final_payload_received_at,
    lastSequenceNo: row.last_sequence_no == null ? null : Number(row.last_sequence_no),
    createdAt: row.created_at,
    updatedAt: row.updated_at,
  };
}

function mapSnapshot(row) {
  if (!row) return null;
  return {
    id: row.id,
    jobId: row.job_id,
    versionNo: row.version_no,
    configVersion: row.config_version,
    digestSha256: row.digest_sha256,
    compiledFromProfileId: row.compiled_from_profile_id,
    compiledFromProfileVersion: row.compiled_from_profile_version,
    snapshot: parseJson(row.snapshot_json, {}),
    createdByUserId: row.created_by_user_id,
    createdAt: row.created_at,
  };
}

function computeDigest(payload) {
  return crypto.createHash('sha256').update(JSON.stringify(payload)).digest('hex');
}

function defaultStepsForKind(jobKind) {
  switch (String(jobKind || '').trim()) {
    case 'train':
    case 'training':
      return [
        { key: 'train', kind: 'train', enabled: true, params: {} },
      ];
    default:
      return [
        { key: 'execute', kind: 'execute', enabled: true, params: {} },
      ];
  }
}

function titleCase(value) {
  return String(value || '')
    .replace(/[_-]+/g, ' ')
    .replace(/\b\w/g, (ch) => ch.toUpperCase());
}

function compileSteps(rawPipeline, jobKind) {
  const rawSteps = Array.isArray(rawPipeline?.steps) && rawPipeline.steps.length
    ? rawPipeline.steps
    : defaultStepsForKind(jobKind);

  return rawSteps.map((step, index) => {
    const key = String(step.key || `${step.kind || 'step'}_${index + 1}`).trim();
    return {
      stepKey: key,
      displayName: String(step.displayName || titleCase(key)),
      stepKind: String(step.kind || 'execute').trim(),
      enabled: step.enabled === false ? false : true,
      dependsOn: Array.isArray(step.dependsOn) ? step.dependsOn.map(String) : [],
      runIf: String(step.runIf || 'enabled'),
      orderIndex: index,
      weight: step.weight == null ? 1 : Number(step.weight),
      params: step.params && typeof step.params === 'object' ? step.params : {},
    };
  });
}

async function createCredential(trx, {
  jobId,
  attemptId,
  credentialType,
  scope,
  expiresAt,
}) {
  const rawToken = issueOpaqueToken(credentialType === 'config' ? 'cfg' : 'rpt');
  const row = {
    id: newId('cred'),
    job_id: jobId,
    attempt_id: attemptId || null,
    credential_type: credentialType,
    token_hash: hashToken(rawToken),
    scope_json: toJson(scope || {}),
    expires_at: expiresAt || null,
    revoked_at: null,
    first_used_at: null,
    bound_attempt_id: null,
    created_at: nowIso(),
  };
  await trx('runtime_callback_credentials').insert(row);
  return { rawToken, record: row };
}

async function issueConfigAccess(jobId, attemptId) {
  return db.transaction(async (trx) => {
    const expiresAt = addMinutes(new Date(), CONFIG.configTokenTtlMinutes);
    return createCredential(trx, {
      jobId,
      attemptId,
      credentialType: 'config',
      scope: { kind: 'config:read' },
      expiresAt,
    });
  });
}

async function issueReportAccess(trx, jobId, attemptId) {
  const expiresAt = addHours(new Date(), CONFIG.reportTokenTtlHours);
  return createCredential(trx, {
    jobId,
    attemptId,
    credentialType: 'report',
    scope: {
      kind: 'runtime:report',
      allow: ['status', 'progress', 'logs', 'final', 'artifacts:register'],
    },
    expiresAt,
  });
}


function isKnexExecutor(value) {
  return !!value && (
    typeof value === 'function' ||
    typeof value === 'object'
  ) && (
    typeof value.client === 'object' ||
    typeof value.transaction === 'function' ||
    typeof value.commit === 'function' ||
    typeof value.rollback === 'function'
  );
}

async function issueRuntimeReportAccess(trxOrJobId, maybeJobId, maybeAttemptId) {
  if (isKnexExecutor(trxOrJobId) && maybeAttemptId !== undefined) {
    return issueReportAccess(trxOrJobId, maybeJobId, maybeAttemptId);
  }

  const jobId = trxOrJobId;
  const attemptId = maybeJobId;
  return db.transaction(async (trx) => issueReportAccess(trx, jobId, attemptId));
}

async function getJobById(jobId) {
  const row = await db('jobs').where({ id: jobId }).first();
  return mapJob(row);
}

async function getAttemptById(attemptId) {
  const row = await db('job_attempts').where({ id: attemptId }).first();
  return mapAttempt(row);
}

async function getLatestAttemptForJob(jobId) {
  const row = await db('job_attempts')
    .where({ job_id: jobId })
    .orderBy('attempt_no', 'desc')
    .first();
  return mapAttempt(row);
}

async function getCurrentConfigSnapshot(jobId) {
  const job = await getJobById(jobId);
  if (!job?.currentConfigSnapshotId) return null;
  const row = await db('job_config_snapshots').where({ id: job.currentConfigSnapshotId }).first();
  return mapSnapshot(row);
}

async function getConfigSnapshots(jobId) {
  const rows = await db('job_config_snapshots')
    .where({ job_id: jobId })
    .orderBy('version_no', 'desc');
  return rows.map(mapSnapshot);
}

async function getPipelineStepsForSnapshot(snapshotId) {
  const rows = await db('job_pipeline_steps')
    .where({ config_snapshot_id: snapshotId })
    .orderBy('order_index', 'asc');

  return rows.map((row) => ({
    id: row.id,
    configSnapshotId: row.config_snapshot_id,
    stepKey: row.step_key,
    displayName: row.display_name,
    stepKind: row.step_kind,
    enabled: !!row.enabled,
    dependsOn: parseJson(row.depends_on_json, []),
    runIf: row.run_if,
    orderIndex: row.order_index,
    weight: row.weight == null ? null : Number(row.weight),
    params: parseJson(row.params_json, {}),
  }));
}

function buildSnapshotDocument({ jobId, jobName, jobKind, runtimeProfile, inputs, steps }) {
  return {
    configVersion: '1.0',
    jobId,
    jobName,
    jobKind,
    runtimeProfile: {
      id: runtimeProfile.id,
      profileKey: runtimeProfile.profileKey,
      version: runtimeProfile.version,
      title: runtimeProfile.title,
      runtimeImage: runtimeProfile.runtimeImage,
      baseModelFamily: runtimeProfile.baseModelFamily,
      capabilities: runtimeProfile.capabilities,
      resourceHints: runtimeProfile.resourceHints,
      launchHints: runtimeProfile.launchHints,
    },
    inputs: inputs || {},
    pipeline: {
      steps: steps.map((step) => ({
        key: step.stepKey,
        displayName: step.displayName,
        kind: step.stepKind,
        enabled: step.enabled,
        dependsOn: step.dependsOn,
        runIf: step.runIf,
        weight: step.weight,
        params: step.params,
      })),
    },
    outputs: {
      resultSummaryExpected: true,
      artifactsExpected: true,
    },
  };
}

async function createJob(payload, actor) {
  const runtimeProfileId = String(payload.runtimeProfileId || '').trim();
  if (!runtimeProfileId) {
    throw new Error('runtimeProfileId is required');
  }

  const runtimeProfile = await getRuntimeProfileById(runtimeProfileId);
  if (!runtimeProfile) {
    throw new Error('Runtime profile not found');
  }
  if (runtimeProfile.status !== 'active') {
    throw new Error('Runtime profile is not active');
  }

  const steps = compileSteps(payload.pipeline || {}, payload.jobKind || 'composite');
  const now = nowIso();
  const jobId = newId('job');
  const attemptId = newId('att');
  const snapshotId = newId('cfg');

  const snapshotDocument = buildSnapshotDocument({
    jobId,
    jobName: String(payload.name || jobId).trim(),
    jobKind: String(payload.jobKind || 'composite').trim(),
    runtimeProfile,
    inputs: payload.inputs || {},
    steps,
  });

  const digest = computeDigest(snapshotDocument);

  await db.transaction(async (trx) => {
    await trx('jobs').insert({
      id: jobId,
      workspace_id: payload.workspaceId || null,
      project_id: payload.projectId || null,
      created_by_user_id: actor?.sub || null,
      name: String(payload.name || jobId).trim(),
      job_kind: String(payload.jobKind || 'composite').trim(),
      status: 'ready',
      stage: 'config',
      desired_state: 'active',
      runtime_profile_id: runtimeProfile.id,
      current_config_snapshot_id: snapshotId,
      latest_attempt_id: attemptId,
      current_step_key: null,
      labels_json: toJson(payload.labels || {}),
      headline: 'Job is ready to be launched',
      terminal_reason: null,
      progress_percent: 0,
      created_at: now,
      started_at: null,
      finished_at: null,
      updated_at: now,
    });

    await trx('job_attempts').insert({
      id: attemptId,
      job_id: jobId,
      attempt_no: 1,
      status: 'issued',
      stage: 'config',
      runtime_image: runtimeProfile.runtimeImage,
      executor_version: null,
      host_info_json: toJson({}),
      runtime_info_json: toJson({}),
      first_seen_at: null,
      config_fetched_at: null,
      started_at: null,
      last_seen_at: null,
      finished_at: null,
      exit_code: null,
      failure_reason: null,
      final_payload_received_at: null,
      last_sequence_no: null,
      created_at: now,
      updated_at: now,
    });

    await trx('job_config_snapshots').insert({
      id: snapshotId,
      job_id: jobId,
      version_no: 1,
      config_version: '1.0',
      digest_sha256: digest,
      compiled_from_profile_id: runtimeProfile.id,
      compiled_from_profile_version: runtimeProfile.version,
      snapshot_json: toJson(snapshotDocument),
      created_by_user_id: actor?.sub || null,
      created_at: now,
    });

    for (const step of steps) {
      await trx('job_pipeline_steps').insert({
        id: newId('stp'),
        config_snapshot_id: snapshotId,
        step_key: step.stepKey,
        display_name: step.displayName,
        step_kind: step.stepKind,
        enabled: step.enabled ? 1 : 0,
        depends_on_json: toJson(step.dependsOn),
        run_if: step.runIf,
        order_index: step.orderIndex,
        weight: step.weight,
        params_json: toJson(step.params),
      });
    }

    await trx('job_events').insert({
      id: newId('evt'),
      job_id: jobId,
      attempt_id: attemptId,
      step_key: null,
      event_type: 'job.created',
      severity: 'info',
      sequence_no: 0,
      delivery_id: newId('delivery'),
      event_time: now,
      received_at: now,
      payload_json: toJson({
        jobId,
        attemptId,
        runtimeProfileId,
      }),
    });
  });

  return getJobView(jobId);
}

async function listJobs({ limit = 50, offset = 0 } = {}) {
  const rows = await db('jobs')
    .orderBy('created_at', 'desc')
    .limit(Math.max(1, Math.min(Number(limit || 50), 500)))
    .offset(Math.max(0, Number(offset || 0)));

  const results = [];
  for (const row of rows) {
    results.push(await getJobView(row.id));
  }
  return results;
}

async function getJobView(jobId) {
  const job = await getJobById(jobId);
  if (!job) return null;

  const [attempt, profile, resultSummary] = await Promise.all([
    job.latestAttemptId ? getAttemptById(job.latestAttemptId) : null,
    job.runtimeProfileId ? getRuntimeProfileById(job.runtimeProfileId) : null,
    getResultSummary(jobId),
  ]);

  return {
    ...job,
    latestAttempt: attempt,
    runtimeProfile: profile,
    resultSummary,
  };
}

async function getResultSummary(jobId) {
  const row = await db('job_result_summaries').where({ job_id: jobId }).first();
  if (!row) return null;
  return {
    jobId: row.job_id,
    attemptId: row.attempt_id,
    outcome: row.outcome,
    headline: row.headline,
    primaryMetrics: parseJson(row.primary_metrics_json, {}),
    summary: parseJson(row.summary_json, {}),
    createdAt: row.created_at,
    updatedAt: row.updated_at,
  };
}

async function getJobSteps(jobId) {
  const job = await getJobById(jobId);
  if (!job?.currentConfigSnapshotId) return [];

  const [compiled, runs] = await Promise.all([
    getPipelineStepsForSnapshot(job.currentConfigSnapshotId),
    job.latestAttemptId
      ? db('job_step_runs').where({ attempt_id: job.latestAttemptId })
      : [],
  ]);

  const runMap = new Map(
    runs.map((row) => [
      row.step_key,
      {
        id: row.id,
        status: row.status,
        progressCurrent: row.progress_current,
        progressTotal: row.progress_total,
        progressUnit: row.progress_unit,
        progressPercent: row.progress_percent,
        message: row.message,
        metrics: parseJson(row.metrics_json, {}),
        startedAt: row.started_at,
        finishedAt: row.finished_at,
        lastSequenceNo: row.last_sequence_no,
        errorSummary: row.error_summary,
      },
    ])
  );

  return compiled.map((step) => ({
    ...step,
    run: runMap.get(step.stepKey) || null,
  }));
}

async function getJobEvents(jobId, { limit = 500 } = {}) {
  const rows = await db('job_events')
    .where({ job_id: jobId })
    .orderBy('received_at', 'asc')
    .limit(Math.max(1, Math.min(Number(limit || 500), 5000)));

  return rows.map((row) => ({
    id: row.id,
    jobId: row.job_id,
    attemptId: row.attempt_id,
    stepKey: row.step_key,
    eventType: row.event_type,
    severity: row.severity,
    sequenceNo: row.sequence_no,
    deliveryId: row.delivery_id,
    eventTime: row.event_time,
    receivedAt: row.received_at,
    payload: parseJson(row.payload_json, {}),
  }));
}

async function getJobArtifacts(jobId) {
  const rows = await db('job_artifacts')
    .where({ job_id: jobId })
    .orderBy([
      { column: 'is_primary', order: 'desc' },
      { column: 'created_at', order: 'desc' },
    ]);

  return rows.map((row) => ({
    id: row.id,
    jobId: row.job_id,
    attemptId: row.attempt_id,
    stepKey: row.step_key,
    artifactType: row.artifact_type,
    role: row.role,
    backend: row.backend,
    uri: row.uri,
    storageKey: row.storage_key,
    contentType: row.content_type,
    format: row.format,
    sizeBytes: row.size_bytes,
    checksumSha256: row.checksum_sha256,
    metadata: parseJson(row.metadata_json, {}),
    isPrimary: !!row.is_primary,
    previewable: !!row.previewable,
    syncStatus: row.sync_status,
    createdAt: row.created_at,
  }));
}

async function getJobLogs(jobId, { stepKey = null, streamName = null, tailChunks = 50 } = {}) {
  let streamQuery = db('job_log_streams').where({ job_id: jobId });
  if (stepKey !== null) streamQuery = streamQuery.andWhere({ step_key: stepKey });
  if (streamName !== null) streamQuery = streamQuery.andWhere({ stream_name: streamName });

  const streams = await streamQuery.orderBy('created_at', 'asc');
  const result = [];

  for (const stream of streams) {
    const chunks = await db('job_log_chunks')
      .where({ stream_id: stream.id })
      .orderBy('chunk_seq', 'desc')
      .limit(Math.max(1, Math.min(Number(tailChunks || 50), 500)));

    result.push({
      streamId: stream.id,
      attemptId: stream.attempt_id,
      stepKey: stream.step_key,
      streamName: stream.stream_name,
      content: chunks.reverse().map((chunk) => chunk.text_payload || '').join(''),
      chunkCount: chunks.length,
    });
  }

  return result;
}

async function cloneJob(jobId, actor) {
  const job = await getJobById(jobId);
  if (!job) throw new Error('Job not found');

  const snapshot = await getCurrentConfigSnapshot(jobId);
  if (!snapshot) throw new Error('Job config snapshot not found');

  return createJob({
    workspaceId: job.workspaceId,
    projectId: job.projectId,
    name: `${job.name} (clone)`,
    jobKind: job.jobKind,
    runtimeProfileId: job.runtimeProfileId,
    inputs: snapshot.snapshot.inputs || {},
    pipeline: snapshot.snapshot.pipeline || {},
    labels: job.labels,
  }, actor);
}

async function retryJob(jobId, actor) {
  const job = await getJobById(jobId);
  if (!job) throw new Error('Job not found');

  const snapshot = await getCurrentConfigSnapshot(jobId);
  if (!snapshot) throw new Error('Job config snapshot not found');

  const previous = await getLatestAttemptForJob(jobId);
  const nextAttemptNo = Number(previous?.attemptNo || 0) + 1;
  const attemptId = newId('att');
  const now = nowIso();

  await db.transaction(async (trx) => {
    await trx('job_attempts').insert({
      id: attemptId,
      job_id: jobId,
      attempt_no: nextAttemptNo,
      status: 'issued',
      stage: 'config',
      runtime_image: snapshot.snapshot.runtimeProfile.runtimeImage,
      executor_version: null,
      host_info_json: toJson({}),
      runtime_info_json: toJson({}),
      first_seen_at: null,
      config_fetched_at: null,
      started_at: null,
      last_seen_at: null,
      finished_at: null,
      exit_code: null,
      failure_reason: null,
      final_payload_received_at: null,
      last_sequence_no: null,
      created_at: now,
      updated_at: now,
    });

    await trx('jobs').where({ id: jobId }).update({
      status: 'ready',
      stage: 'config',
      desired_state: 'active',
      latest_attempt_id: attemptId,
      current_step_key: null,
      headline: 'Retry attempt issued',
      terminal_reason: null,
      progress_percent: 0,
      started_at: null,
      finished_at: null,
      updated_at: now,
    });

    await trx('job_events').insert({
      id: newId('evt'),
      job_id: jobId,
      attempt_id: attemptId,
      step_key: null,
      event_type: 'job.retry_issued',
      severity: 'info',
      sequence_no: 0,
      delivery_id: newId('delivery'),
      event_time: now,
      received_at: now,
      payload_json: toJson({
        jobId,
        attemptId,
        previousAttemptId: previous?.id || null,
        requestedBy: actor?.sub || null,
      }),
    });
  });

  return getJobView(jobId);
}

async function cancelJob(jobId, actor) {
  const job = await getJobById(jobId);
  if (!job) throw new Error('Job not found');

  const now = nowIso();
  await db('jobs').where({ id: jobId }).update({
    desired_state: 'cancel_requested',
    headline: 'Cancellation requested',
    updated_at: now,
  });

  await db('job_events').insert({
    id: newId('evt'),
    job_id: jobId,
    attempt_id: job.latestAttemptId || null,
    step_key: null,
    event_type: 'job.cancel_requested',
    severity: 'warn',
    sequence_no: 0,
    delivery_id: newId('delivery'),
    event_time: now,
    received_at: now,
    payload_json: toJson({
      requestedBy: actor?.sub || null,
    }),
  });

  return getJobView(jobId);
}

async function verifyCredential(jobId, rawToken, credentialType) {
  const tokenHash = hashToken(rawToken);
  const row = await db('runtime_callback_credentials')
    .where({
      job_id: jobId,
      credential_type: credentialType,
      token_hash: tokenHash,
    })
    .whereNull('revoked_at')
    .first();

  if (!row) return null;

  if (row.expires_at && new Date(row.expires_at).getTime() < Date.now()) {
    return null;
  }

  return row;
}

async function markCredentialUsed(credentialId, boundAttemptId = null, executor = db) {
  const existing = await executor('runtime_callback_credentials').where({ id: credentialId }).first();
  if (!existing) return null;

  const patch = {};
  if (!existing.first_used_at) patch.first_used_at = nowIso();
  if (boundAttemptId && !existing.bound_attempt_id) patch.bound_attempt_id = boundAttemptId;

  if (Object.keys(patch).length > 0) {
    await executor('runtime_callback_credentials').where({ id: credentialId }).update(patch);
  }

  return executor('runtime_callback_credentials').where({ id: credentialId }).first();
}

async function buildLaunchSpec(jobId, publicBaseUrl) {
  const job = await getJobById(jobId);
  if (!job) throw new Error('Job not found');

  const attempt = await getLatestAttemptForJob(jobId);
  if (!attempt) throw new Error('Job attempt not found');

  const { rawToken } = await issueConfigAccess(jobId, attempt.id);
  const configUrl = `${String(publicBaseUrl || '').replace(/\/+$/, '')}/api/v1/runtime/jobs/${jobId}/config?token=${encodeURIComponent(rawToken)}`;

  const profile = await getRuntimeProfileById(job.runtimeProfileId);
  const shmSize = profile?.launchHints?.shmSize || '16g';

  return {
    jobId,
    attemptId: attempt.id,
    runtimeImage: attempt.runtimeImage,
    jobConfigUrl: configUrl,
    dockerRun: [
      'docker run --rm --gpus all \\',
      `  --shm-size ${shmSize} \\`,
      `  -e JOB_CONFIG_URL="${configUrl}" \\`,
      '  -e HF_TOKEN="$HF_TOKEN" \\',
      `  ${attempt.runtimeImage}`,
    ].join('\n'),
    environment: {
      JOB_CONFIG_URL: configUrl,
    },
  };
}

async function getRuntimeConfig(jobId, rawConfigToken, publicBaseUrl) {
  const credential = await verifyCredential(jobId, rawConfigToken, 'config');
  if (!credential) {
    const err = new Error('Invalid config token');
    err.statusCode = 403;
    throw err;
  }

  const job = await getJobById(jobId);
  if (!job) {
    const err = new Error('Job not found');
    err.statusCode = 404;
    throw err;
  }

  const attempt = await getLatestAttemptForJob(jobId);
  if (!attempt) {
    const err = new Error('Job attempt not found');
    err.statusCode = 404;
    throw err;
  }

  const snapshot = await getCurrentConfigSnapshot(jobId);
  if (!snapshot) {
    const err = new Error('Job config snapshot not found');
    err.statusCode = 404;
    throw err;
  }

  let reportAccess;
  await db.transaction(async (trx) => {
    await markCredentialUsed(credential.id, attempt.id, trx);
    reportAccess = await issueReportAccess(trx, jobId, attempt.id);

    const updates = {
      status: 'config_fetched',
      config_fetched_at: attempt.configFetchedAt || nowIso(),
      first_seen_at: attempt.firstSeenAt || nowIso(),
      last_seen_at: nowIso(),
      updated_at: nowIso(),
    };

    await trx('job_attempts').where({ id: attempt.id }).update(updates);

    if (job.status === 'ready') {
      await trx('jobs').where({ id: jobId }).update({
        status: 'starting',
        stage: 'bootstrap',
        headline: 'Runtime fetched config',
        updated_at: nowIso(),
      });
    }

    await trx('job_events').insert({
      id: newId('evt'),
      job_id: jobId,
      attempt_id: attempt.id,
      step_key: null,
      event_type: 'runtime.config_fetched',
      severity: 'info',
      sequence_no: 0,
      delivery_id: newId('delivery'),
      event_time: nowIso(),
      received_at: nowIso(),
      payload_json: toJson({
        attemptId: attempt.id,
        configSnapshotId: snapshot.id,
      }),
    });
  });

  const base = String(publicBaseUrl || '').replace(/\/+$/, '');
  return {
    configVersion: snapshot.configVersion,
    jobId,
    attemptId: attempt.id,
    configSnapshotId: snapshot.id,
    digestSha256: snapshot.digestSha256,
    reportToken: reportAccess.rawToken,
    config: snapshot.snapshot,
    endpoints: {
      status: `${base}/api/v1/runtime/jobs/${jobId}/status`,
      progress: `${base}/api/v1/runtime/jobs/${jobId}/progress`,
      logs: `${base}/api/v1/runtime/jobs/${jobId}/logs`,
      final: `${base}/api/v1/runtime/jobs/${jobId}/final`,
      artifactsRegister: `${base}/api/v1/runtime/jobs/${jobId}/artifacts/register`,
    },
  };
}

async function getJobHfSyncStates(jobId) {
  const rows = await db('huggingface_sync_states')
    .where({ job_id: jobId })
    .orderBy('created_at', 'desc');

  return rows.map((row) => ({
    id: row.id,
    jobId: row.job_id,
    attemptId: row.attempt_id,
    repoId: row.repo_id,
    repoType: row.repo_type,
    requestedRevision: row.requested_revision,
    lastSeenRevision: row.last_seen_revision,
    status: row.status,
    manifest: parseJson(row.manifest_json, {}),
    lastError: row.last_error,
    lastSyncedAt: row.last_synced_at,
    nextRetryAt: row.next_retry_at,
    retryCount: row.retry_count,
    createdAt: row.created_at,
    updatedAt: row.updated_at,
  }));
}

async function requestHfSync(jobId, body, actor) {
  const job = await getJobById(jobId);
  if (!job) throw new Error('Job not found');
  const now = nowIso();
  const attempt = await getLatestAttemptForJob(jobId);

  const row = {
    id: newId('hfs'),
    job_id: jobId,
    attempt_id: attempt?.id || null,
    repo_id: String(body.repoId || 'unknown/repo').trim(),
    repo_type: String(body.repoType || 'model').trim(),
    requested_revision: body.requestedRevision ? String(body.requestedRevision) : null,
    last_seen_revision: null,
    status: 'pending',
    manifest_json: toJson({
      requestedBy: actor?.sub || null,
      reason: body.reason || 'manual',
    }),
    last_error: null,
    last_synced_at: null,
    next_retry_at: null,
    retry_count: 0,
    created_at: now,
    updated_at: now,
  };

  await db('huggingface_sync_states').insert(row);

  await db('job_events').insert({
    id: newId('evt'),
    job_id: jobId,
    attempt_id: attempt?.id || null,
    step_key: null,
    event_type: 'hf_sync.requested',
    severity: 'info',
    sequence_no: 0,
    delivery_id: newId('delivery'),
    event_time: now,
    received_at: now,
    payload_json: toJson({
      syncStateId: row.id,
      repoId: row.repo_id,
      repoType: row.repo_type,
      reason: body.reason || 'manual',
    }),
  });

  return {
    requested: true,
    syncStateId: row.id,
  };
}

async function upsertResultSummary(trx, jobId, attemptId, payload) {
  const now = nowIso();
  const row = {
    job_id: jobId,
    attempt_id: attemptId,
    outcome: String(payload.outcome || 'succeeded'),
    headline: payload.headline ? String(payload.headline) : null,
    primary_metrics_json: toJson(payload.primaryMetrics || {}),
    summary_json: toJson(payload.summary || {}),
    created_at: now,
    updated_at: now,
  };

  const existing = await trx('job_result_summaries').where({ job_id: jobId }).first();
  if (existing) {
    await trx('job_result_summaries').where({ job_id: jobId }).update({
      attempt_id: row.attempt_id,
      outcome: row.outcome,
      headline: row.headline,
      primary_metrics_json: row.primary_metrics_json,
      summary_json: row.summary_json,
      updated_at: now,
    });
  } else {
    await trx('job_result_summaries').insert(row);
  }
}

async function recalculateJobProgress(trx, jobId, attemptId, configSnapshotId) {
  const compiled = await trx('job_pipeline_steps')
    .where({ config_snapshot_id: configSnapshotId })
    .orderBy('order_index', 'asc');
  if (!compiled.length) return 0;

  const runs = await trx('job_step_runs').where({ attempt_id: attemptId });
  const runMap = new Map(runs.map((row) => [row.step_key, row]));

  let totalWeight = 0;
  let totalProgress = 0;

  for (const step of compiled) {
    if (!step.enabled) continue;
    const weight = step.weight == null ? 1 : Number(step.weight);
    totalWeight += weight;

    const run = runMap.get(step.step_key);
    let progress = 0;
    if (run) {
      if (run.status === 'succeeded') progress = 100;
      else if (run.status === 'skipped') progress = 100;
      else if (run.status === 'failed') progress = run.progress_percent == null ? 0 : Number(run.progress_percent);
      else progress = run.progress_percent == null ? 0 : Number(run.progress_percent);
    }

    totalProgress += weight * progress;
  }

  if (totalWeight <= 0) return 0;
  return Math.max(0, Math.min(100, totalProgress / totalWeight));
}

module.exports = {
  normalizeJobStatus,
  normalizeAttemptStatus,
  getJobById,
  getAttemptById,
  getLatestAttemptForJob,
  getCurrentConfigSnapshot,
  getConfigSnapshots,
  getPipelineStepsForSnapshot,
  createJob,
  listJobs,
  getJobView,
  getJobSteps,
  getJobEvents,
  getJobLogs,
  getJobArtifacts,
  getResultSummary,
  cloneJob,
  retryJob,
  cancelJob,
  issueConfigAccess,
  issueRuntimeReportAccess,
  verifyCredential,
  markCredentialUsed,
  buildLaunchSpec,
  getRuntimeConfig,
  getJobHfSyncStates,
  requestHfSync,
  upsertResultSummary,
  recalculateJobProgress,
  issueReportAccess,
};
