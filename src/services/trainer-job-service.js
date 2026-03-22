'use strict';

const path = require('path');
const { spawn } = require('child_process');
const { db } = require('../db');
const { CONFIG } = require('../config');
const { newId } = require('../utils/ids');
const { nowIso } = require('../utils/time');
const { parseJson, toJson } = require('../utils/json');
const { buildPublicBaseUrl } = require('../utils/http');
const { getRuntimeProfileById } = require('./runtime-profile-service');
const {
  getJobView,
  verifyCredential,
  markCredentialUsed,
  issueConfigAccess,
  issueRuntimeReportAccess,
} = require('./job-service');

const TRAINER_JOB_KIND = 'trainer-service';
const TRAINER_RUNTIME_KIND = 'trainer-service/v1';

function deepClone(value) {
  return JSON.parse(JSON.stringify(value == null ? {} : value));
}

function asObject(value, fallback = {}) {
  return value && typeof value === 'object' && !Array.isArray(value) ? value : fallback;
}

function ensureArray(value) {
  return Array.isArray(value) ? value : [];
}

function sanitizeContainerName(value) {
  return String(value || '')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_.-]+/g, '-')
    .replace(/^-+|-+$/g, '') || 'trainer-job';
}

function normalizeJobId(inputJobId) {
  const raw = String(inputJobId || '').trim();
  return raw || newId('job');
}

function mergeObjects(baseValue, overrideValue) {
  return {
    ...asObject(baseValue, {}),
    ...asObject(overrideValue, {}),
  };
}

function normalizePipelineConfig(configInput) {
  const config = deepClone(configInput);

  const training = asObject(config.training, {});
  const postprocess = asObject(config.postprocess, {});
  const evaluation = asObject(config.evaluation, {});
  const upload = asObject(config.upload, {});
  const huggingface = asObject(config.huggingface, {});
  const pipeline = asObject(config.pipeline, {});

  const trainingStage = asObject(pipeline.training, {});
  const mergeStage = asObject(pipeline.merge, {});
  const evaluationStage = asObject(pipeline.evaluation, {});
  const publishStage = asObject(pipeline.publish, {});
  const uploadStage = asObject(pipeline.upload, {});
  const prepareAssetsStage = asObject(pipeline.prepare_assets, {});

  const evaluationDataset =
    evaluationStage.dataset && typeof evaluationStage.dataset === 'object' && !Array.isArray(evaluationStage.dataset)
      ? deepClone(evaluationStage.dataset)
      : (
          evaluation.dataset && typeof evaluation.dataset === 'object' && !Array.isArray(evaluation.dataset)
            ? deepClone(evaluation.dataset)
            : undefined
        );

  config.pipeline = {
    prepare_assets: {
      enabled: prepareAssetsStage.enabled !== false,
    },
    training: {
      ...training,
      ...trainingStage,
      enabled: trainingStage.enabled !== false,
    },
    merge: {
      ...postprocess,
      ...mergeStage,
      enabled: mergeStage.enabled !== false,
    },
    evaluation: {
      ...evaluation,
      ...evaluationStage,
      ...(evaluationDataset ? { dataset: evaluationDataset } : {}),
      enabled:
        evaluationStage.enabled != null
          ? Boolean(evaluationStage.enabled)
          : Boolean(evaluation.enabled),
    },
    publish: {
      ...huggingface,
      ...publishStage,
      enabled:
        publishStage.enabled != null
          ? Boolean(publishStage.enabled)
          : Boolean(huggingface.enabled),
    },
    upload: {
      ...upload,
      ...uploadStage,
      auth: {
        ...asObject(upload.auth, {}),
        ...asObject(uploadStage.auth, {}),
      },
      url_targets: {
        ...asObject(upload.url_targets, {}),
        ...asObject(uploadStage.url_targets, {}),
      },
      enabled:
        uploadStage.enabled != null
          ? Boolean(uploadStage.enabled)
          : Boolean(upload.enabled),
    },
  };

  return config;
}

function ensureTrainerConfigBase(rawConfig, jobId, jobName) {
  let config = deepClone(rawConfig);

  config.job_id = jobId;
  config.job_name = String(jobName || config.job_name || jobId).trim();
  config.mode = 'remote';

  config.outputs = asObject(config.outputs, {});
  config.outputs.base_dir = `/output/${jobId}`;

  config.reporting = asObject(config.reporting, {});
  config.upload = asObject(config.upload, {});
  config.upload.auth = asObject(config.upload.auth, {});
  config.upload.url_targets = asObject(config.upload.url_targets, {});
  config.huggingface = asObject(config.huggingface, {});
  config.evaluation = asObject(config.evaluation, {});
  config.training = asObject(config.training, {});
  config.postprocess = asObject(config.postprocess, {});
  config.pipeline = asObject(config.pipeline, {});

  config = normalizePipelineConfig(config);

  return config;
}

function ensureTrainerConfigBase(rawConfig, jobId, jobName) {
  let config = deepClone(rawConfig);

  config.job_id = jobId;
  config.job_name = String(jobName || config.job_name || jobId).trim();
  config.mode = 'remote';

  config.outputs = asObject(config.outputs, {});
  config.outputs.base_dir = `/output/${jobId}`;

  config.reporting = asObject(config.reporting, {});
  config.upload = asObject(config.upload, {});
  config.upload.auth = asObject(config.upload.auth, {});
  config.upload.url_targets = asObject(config.upload.url_targets, {});
  config.huggingface = asObject(config.huggingface, {});
  config.evaluation = asObject(config.evaluation, {});
  config.training = asObject(config.training, {});
  config.postprocess = asObject(config.postprocess, {});
  config.pipeline = asObject(config.pipeline, {});

  config = normalizePipelineConfig(config);

  return config;
}

function buildTrainerSteps(config) {
  const pipeline = asObject(config.pipeline, null);
  const postprocess = asObject(config.postprocess, {});
  const evaluation = asObject(config.evaluation, {});
  const upload = asObject(config.upload, {});
  const huggingface = asObject(config.huggingface, {});

  const bootstrapEnabled = true;
  const prepareAssetsEnabled = !pipeline || pipeline.prepare_assets?.enabled !== false;
  const trainingEnabled = !pipeline || pipeline.training?.enabled !== false;
  const mergeEnabled = trainingEnabled && (
    pipeline
      ? pipeline.merge?.enabled !== false && !!postprocess.merge_lora && !!postprocess.save_merged_16bit
      : !!postprocess.merge_lora && !!postprocess.save_merged_16bit
  );
  const evaluationEnabled = pipeline ? !!pipeline.evaluation?.enabled : !!evaluation.enabled;
  const publishEnabled = pipeline ? !!pipeline.publish?.enabled : !!huggingface.enabled;
  const uploadEnabled = pipeline ? !!pipeline.upload?.enabled : !!upload.enabled;

  return [
    {
      stepKey: 'bootstrap',
      displayName: 'Bootstrap',
      stepKind: 'bootstrap',
      enabled: bootstrapEnabled,
      dependsOn: [],
      runIf: 'always',
      orderIndex: 0,
      weight: 2,
      params: {},
    },
    {
      stepKey: 'prepare_assets',
      displayName: 'Prepare Assets',
      stepKind: 'prepare_assets',
      enabled: prepareAssetsEnabled,
      dependsOn: ['bootstrap'],
      runIf: 'enabled',
      orderIndex: 1,
      weight: 3,
      params: {},
    },
    {
      stepKey: 'training',
      displayName: 'Training',
      stepKind: 'training',
      enabled: trainingEnabled,
      dependsOn: prepareAssetsEnabled ? ['prepare_assets'] : ['bootstrap'],
      runIf: 'enabled',
      orderIndex: 2,
      weight: 75,
      params: {},
    },
    {
      stepKey: 'merge_lora',
      displayName: 'Merge LoRA',
      stepKind: 'merge_model',
      enabled: mergeEnabled,
      dependsOn: ['training'],
      runIf: 'enabled',
      orderIndex: 3,
      weight: 5,
      params: {},
    },
    {
      stepKey: 'evaluation',
      displayName: 'Evaluation',
      stepKind: 'evaluation',
      enabled: evaluationEnabled,
      dependsOn: mergeEnabled ? ['merge_lora'] : ['training'],
      runIf: 'enabled',
      orderIndex: 4,
      weight: 10,
      params: {},
    },
    {
      stepKey: 'publish',
      displayName: 'Publish Hugging Face',
      stepKind: 'publish_hf',
      enabled: publishEnabled,
      dependsOn: evaluationEnabled ? ['evaluation'] : (mergeEnabled ? ['merge_lora'] : ['training']),
      runIf: 'enabled',
      orderIndex: 5,
      weight: 3,
      params: {},
    },
    {
      stepKey: 'upload',
      displayName: 'Upload Artifacts',
      stepKind: 'upload_artifacts',
      enabled: uploadEnabled,
      dependsOn: publishEnabled ? ['publish'] : (evaluationEnabled ? ['evaluation'] : (mergeEnabled ? ['merge_lora'] : ['training'])),
      runIf: 'enabled',
      orderIndex: 6,
      weight: 2,
      params: {},
    },
  ];
}

function normalizeExecutor(payloadExecutor, runtimeProfile, jobId) {
  const input = asObject(payloadExecutor, {});
  const mounts = ensureArray(input.mounts).map((mount) => ({
    hostPath: String(mount.hostPath || '').trim(),
    containerPath: String(mount.containerPath || '').trim(),
    readOnly: !!mount.readOnly,
  })).filter((mount) => mount.hostPath && mount.containerPath);

  const outputHostDir = path.resolve(CONFIG.runtimeHostOutputRoot, jobId);
  const mergedMounts = [
    { hostPath: outputHostDir, containerPath: '/output', readOnly: false },
    ...mounts,
  ];

  return {
    image: String(input.image || runtimeProfile.runtimeImage || CONFIG.defaultRuntimeImage).trim(),
    gpus: input.gpus == null ? (runtimeProfile.launchHints?.gpus ?? 'all') : input.gpus,
    shmSize: String(input.shmSize || runtimeProfile.launchHints?.shmSize || '16g').trim(),
    mounts: mergedMounts,
    env: asObject(input.env, {}),
    network: input.network == null ? (CONFIG.runtimeDockerNetwork || null) : String(input.network || '').trim() || null,
    extraDockerArgs: ensureArray(input.extraDockerArgs).map((item) => String(item)),
    containerName: sanitizeContainerName(input.containerName || `trainer-${jobId}`),
    outputHostDir,
  };
}

function buildSnapshotDocument({ jobId, jobName, runtimeProfile, trainerConfigBase, executor, steps }) {
  return {
    configVersion: '1.0',
    runtimeKind: TRAINER_RUNTIME_KIND,
    jobId,
    jobName,
    runtimeProfile: {
      id: runtimeProfile.id,
      profileKey: runtimeProfile.profileKey,
      version: runtimeProfile.version,
      title: runtimeProfile.title,
      runtimeImage: runtimeProfile.runtimeImage,
      baseModelFamily: runtimeProfile.baseModelFamily,
      resourceHints: runtimeProfile.resourceHints,
      launchHints: runtimeProfile.launchHints,
    },
    trainerRuntime: {
      service: 'trainer-service',
      contractVersion: '1',
      trainerConfigBase,
      executor,
    },
    pipeline: {
      steps: steps.map((step) => ({
        key: step.stepKey,
        displayName: step.displayName,
        kind: step.stepKind,
        enabled: step.enabled,
        dependsOn: step.dependsOn,
        weight: step.weight,
      })),
    },
  };
}

function computeDigest(payload) {
  const crypto = require('crypto');
  return crypto.createHash('sha256').update(JSON.stringify(payload)).digest('hex');
}

async function createTrainerJob(payload, actor) {
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

  const inputConfig = asObject(payload.config, null);
  if (!inputConfig) {
    throw new Error('config object is required');
  }

  const jobId = normalizeJobId(payload.jobId);
  const existing = await db('jobs').where({ id: jobId }).first();
  if (existing) {
    throw new Error(`Job already exists: ${jobId}`);
  }

  const jobName = String(payload.name || inputConfig.job_name || jobId).trim();
  const trainerConfigBase = ensureTrainerConfigBase(inputConfig, jobId, jobName);
  const executor = normalizeExecutor(payload.executor, runtimeProfile, jobId);
  const steps = buildTrainerSteps(trainerConfigBase);

  const now = nowIso();
  const attemptId = newId('att');
  const snapshotId = newId('cfg');

  const snapshotDocument = buildSnapshotDocument({
    jobId,
    jobName,
    runtimeProfile,
    trainerConfigBase,
    executor,
    steps,
  });
  const digest = computeDigest(snapshotDocument);

  await db.transaction(async (trx) => {
    await trx('jobs').insert({
      id: jobId,
      workspace_id: payload.workspaceId == null ? null : String(payload.workspaceId),
      project_id: payload.projectId == null ? null : String(payload.projectId),
      created_by_user_id: actor?.sub || null,
      name: jobName,
      job_kind: TRAINER_JOB_KIND,
      status: 'queued',
      stage: 'bootstrap',
      desired_state: 'active',
      runtime_profile_id: runtimeProfile.id,
      current_config_snapshot_id: snapshotId,
      latest_attempt_id: attemptId,
      current_step_key: null,
      labels_json: toJson({ ...(asObject(payload.labels, {})), service: 'trainer-service' }),
      headline: 'Trainer job created',
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
      status: 'queued',
      stage: 'bootstrap',
      runtime_image: executor.image,
      executor_version: 'trainer-runtime-local-docker-v1',
      host_info_json: toJson({ outputHostDir: executor.outputHostDir }),
      runtime_info_json: toJson({ executor }),
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
      event_type: 'trainer.job.created',
      severity: 'info',
      sequence_no: 0,
      delivery_id: newId('delivery'),
      event_time: now,
      received_at: now,
      payload_json: toJson({
        jobId,
        jobName,
        runtimeProfileId,
        executor: {
          image: executor.image,
          gpus: executor.gpus,
          shmSize: executor.shmSize,
          containerName: executor.containerName,
        },
      }),
    });
  });

  return getJobView(jobId);
}

async function listTrainerJobs({ limit = 50, offset = 0 } = {}) {
  const rows = await db('jobs')
    .where({ job_kind: TRAINER_JOB_KIND })
    .orderBy('created_at', 'desc')
    .limit(Math.max(1, Math.min(Number(limit || 50), 500)))
    .offset(Math.max(0, Number(offset || 0)));

  const items = [];
  for (const row of rows) {
    items.push(await getJobView(row.id));
  }
  return items;
}

async function getTrainerSnapshot(jobId) {
  const job = await db('jobs').where({ id: jobId }).first();
  if (!job?.current_config_snapshot_id) return null;
  const snapshot = await db('job_config_snapshots').where({ id: job.current_config_snapshot_id }).first();
  if (!snapshot) return null;
  const parsed = parseJson(snapshot.snapshot_json, {});
  if (parsed.runtimeKind !== TRAINER_RUNTIME_KIND) return null;
  return {
    id: snapshot.id,
    jobId: snapshot.job_id,
    versionNo: snapshot.version_no,
    digestSha256: snapshot.digest_sha256,
    snapshot: parsed,
  };
}

function injectManagedCallbackConfig(snapshot, jobId, reportToken, baseUrl) {
  const trainerConfig = deepClone(snapshot.trainerRuntime?.trainerConfigBase || {});
  const root = String(baseUrl || '').replace(/\/+$/, '');

  trainerConfig.job_id = jobId;
  trainerConfig.job_name = snapshot.jobName || trainerConfig.job_name || jobId;
  trainerConfig.mode = 'remote';

  trainerConfig.upload = asObject(trainerConfig.upload, {});
  trainerConfig.upload.auth = asObject(trainerConfig.upload.auth, {});
  trainerConfig.upload.url_targets = {
    ...asObject(trainerConfig.upload.url_targets, {}),
    logs_url: `${root}/api/jobs/upload/logs`,
    effective_config_url: `${root}/api/jobs/upload/config`,
    summary_url: `${root}/api/jobs/upload/summary`,
    train_metrics_url: `${root}/api/jobs/upload/train-metrics`,
    train_history_url: `${root}/api/jobs/upload/train-history`,
    eval_summary_url: `${root}/api/jobs/upload/eval-summary`,
    eval_details_url: `${root}/api/jobs/upload/eval-details`,
    lora_archive_url: `${root}/api/jobs/upload/lora`,
    merged_archive_url: `${root}/api/jobs/upload/merged`,
    full_archive_url: `${root}/api/jobs/upload/full-archive`,
  };
  trainerConfig.upload.auth.bearer_token = reportToken;

  return {
    job_id: jobId,
    job_name: snapshot.jobName || trainerConfig.job_name || jobId,
    callback_auth_token: reportToken,
    status_url: `${root}/api/jobs/status`,
    progress_url: `${root}/api/jobs/progress`,
    final_url: `${root}/api/jobs/final`,
    logs_url: `${root}/api/jobs/logs`,
    config: trainerConfig,
  };
}

async function buildTrainerBootstrapPayload(jobId, rawConfigToken, baseUrl) {
  const credential = await verifyCredential(jobId, rawConfigToken, 'config');
  if (!credential) {
    const error = new Error('Invalid config token');
    error.statusCode = 403;
    throw error;
  }

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

  const snapshot = await getTrainerSnapshot(jobId);
  if (!snapshot) {
    const error = new Error('Trainer snapshot not found');
    error.statusCode = 404;
    throw error;
  }

  let reportAccess;
  await db.transaction(async (trx) => {
    await markCredentialUsed(credential.id, attempt.id, trx);
    reportAccess = await issueRuntimeReportAccess(trx, jobId, attempt.id);

    const now = nowIso();
    await trx('job_attempts').where({ id: attempt.id }).update({
      status: 'started',
      stage: 'bootstrap',
      config_fetched_at: attempt.config_fetched_at || now,
      first_seen_at: attempt.first_seen_at || now,
      last_seen_at: now,
      updated_at: now,
    });

    await trx('jobs').where({ id: jobId }).update({
      status: 'started',
      stage: 'bootstrap',
      headline: 'Runtime fetched trainer config',
      started_at: job.started_at || now,
      updated_at: now,
    });

    await trx('job_events').insert({
      id: newId('evt'),
      job_id: jobId,
      attempt_id: attempt.id,
      step_key: 'bootstrap',
      event_type: 'trainer.bootstrap.served',
      severity: 'info',
      sequence_no: 0,
      delivery_id: newId('delivery'),
      event_time: now,
      received_at: now,
      payload_json: toJson({
        configSnapshotId: snapshot.id,
        attemptId: attempt.id,
      }),
    });
  });

  return injectManagedCallbackConfig(snapshot.snapshot, jobId, reportAccess.rawToken, baseUrl);
}

async function buildTrainerLaunchSpec(jobId, publicBaseUrl) {
  const job = await db('jobs').where({ id: jobId }).first();
  if (!job) throw new Error('Job not found');
  if (job.job_kind !== TRAINER_JOB_KIND) throw new Error('Job is not a trainer job');

  const attempt = await db('job_attempts').where({ id: job.latest_attempt_id }).first();
  if (!attempt) throw new Error('Job attempt not found');

  const snapshot = await getTrainerSnapshot(jobId);
  if (!snapshot) throw new Error('Trainer snapshot not found');

  const { rawToken } = await issueConfigAccess(jobId, attempt.id);
  const baseUrl = String(publicBaseUrl || '').replace(/\/+$/, '');
  const jobConfigUrl = `${baseUrl}/api/v1/trainer/jobs/${jobId}/bootstrap?token=${encodeURIComponent(rawToken)}`;
  const executor = asObject(snapshot.snapshot.trainerRuntime?.executor, {});
  const mountArgs = ensureArray(executor.mounts).map((mount) => {
    const mode = mount.readOnly ? ':ro' : '';
    return `-v ${mount.hostPath}:${mount.containerPath}${mode}`;
  });

  const lines = [
    'docker run --rm -d \\',
    `  --name ${executor.containerName} \\`,
  ];

  if (executor.gpus != null && String(executor.gpus) !== '') {
    lines.push(`  --gpus ${String(executor.gpus)} \\`);
  }
  if (executor.shmSize) {
    lines.push(`  --shm-size ${executor.shmSize} \\`);
  }
  if (executor.network) {
    lines.push(`  --network ${executor.network} \\`);
  }
  for (const mountArg of mountArgs) {
    lines.push(`  ${mountArg} \\`);
  }
  lines.push(`  -e JOB_CONFIG_URL="${jobConfigUrl}" \\`);
  lines.push('  -e HF_TOKEN="$HF_TOKEN" \\');
  for (const item of ensureArray(executor.extraDockerArgs)) {
    lines.push(`  ${item} \\`);
  }
  lines.push(`  ${executor.image}`);

  return {
    jobId,
    attemptId: attempt.id,
    runtimeImage: executor.image,
    containerName: executor.containerName,
    jobConfigUrl,
    hostOutputDir: executor.outputHostDir,
    dockerRun: lines.join('\n'),
    environment: {
      JOB_CONFIG_URL: jobConfigUrl,
      HF_TOKEN: '$HF_TOKEN',
    },
    mounts: executor.mounts,
    executor,
  };
}

function dockerRunPromise(binary, args) {
  return new Promise((resolve, reject) => {
    const child = spawn(binary, args, { stdio: ['ignore', 'pipe', 'pipe'] });
    let stdout = '';
    let stderr = '';
    child.stdout.on('data', (chunk) => { stdout += chunk.toString('utf-8'); });
    child.stderr.on('data', (chunk) => { stderr += chunk.toString('utf-8'); });
    child.on('error', reject);
    child.on('close', (code) => {
      if (code === 0) {
        resolve({ stdout: stdout.trim(), stderr: stderr.trim() });
        return;
      }
      reject(new Error(stderr.trim() || stdout.trim() || `docker exited with code ${code}`));
    });
  });
}

function buildDockerArgs(spec, launchBody = {}) {
  const executor = asObject(spec.executor, {});
  const body = asObject(launchBody, {});
  const env = { JOB_CONFIG_URL: spec.jobConfigUrl };

  for (const key of ensureArray(body.inheritEnv)) {
    const envKey = String(key || '').trim();
    if (!envKey) continue;
    if (process.env[envKey] != null) env[envKey] = process.env[envKey];
  }
  for (const [key, value] of Object.entries(asObject(executor.env, {}))) {
    env[String(key)] = value == null ? '' : String(value);
  }
  for (const [key, value] of Object.entries(asObject(body.env, {}))) {
    env[String(key)] = value == null ? '' : String(value);
  }

  const args = ['run', '--rm', '-d', '--label', `forge.job_id=${spec.jobId}`, '--name', executor.containerName];
  if (executor.gpus != null && String(executor.gpus) !== '') {
    args.push('--gpus', String(executor.gpus));
  }
  if (executor.shmSize) {
    args.push('--shm-size', String(executor.shmSize));
  }
  if (executor.network) {
    args.push('--network', String(executor.network));
  }
  for (const mount of ensureArray(executor.mounts)) {
    args.push('-v', `${mount.hostPath}:${mount.containerPath}${mount.readOnly ? ':ro' : ''}`);
  }
  for (const [key, value] of Object.entries(env)) {
    args.push('-e', `${key}=${String(value)}`);
  }
  for (const item of ensureArray(executor.extraDockerArgs)) {
    args.push(String(item));
  }
  args.push(executor.image);
  return { args, envKeys: Object.keys(env).sort() };
}

async function launchTrainerJob(jobId, body, actor, req) {
  const spec = await buildTrainerLaunchSpec(jobId, buildPublicBaseUrl(req, CONFIG.publicBaseUrl));
  const executor = asObject(spec.executor, {});
  await db('jobs').where({ id: jobId }).first();

  await require('fs/promises').mkdir(executor.outputHostDir, { recursive: true });

  const { args, envKeys } = buildDockerArgs(spec, body || {});
  const now = nowIso();

  try {
    const result = await dockerRunPromise(CONFIG.runtimeDockerBin, args);
    const containerId = String(result.stdout || '').split(/\s+/).filter(Boolean)[0] || null;

    const attemptRow = await db('job_attempts').where({ id: spec.attemptId }).first();
    const runtimeInfo = parseJson(attemptRow?.runtime_info_json, {});

    await db.transaction(async (trx) => {
      await trx('job_attempts').where({ id: spec.attemptId }).update({
        status: 'started',
        stage: 'bootstrap',
        started_at: attemptRow?.started_at || now,
        last_seen_at: now,
        runtime_info_json: toJson({
          ...runtimeInfo,
          containerId,
          lastLaunch: {
            launchedAt: now,
            envKeys,
            args,
          },
        }),
        updated_at: now,
      });

      await trx('jobs').where({ id: jobId }).update({
        status: 'started',
        stage: 'bootstrap',
        headline: 'Trainer container launched',
        started_at: now,
        updated_at: now,
      });

      await trx('job_events').insert({
        id: newId('evt'),
        job_id: jobId,
        attempt_id: spec.attemptId,
        step_key: 'bootstrap',
        event_type: 'trainer.launch.started',
        severity: 'info',
        sequence_no: 0,
        delivery_id: newId('delivery'),
        event_time: now,
        received_at: now,
        payload_json: toJson({
          containerId,
          containerName: spec.containerName,
          runtimeImage: spec.runtimeImage,
          hostOutputDir: spec.hostOutputDir,
        }),
      });
    });

    return {
      launched: true,
      jobId,
      attemptId: spec.attemptId,
      containerId,
      containerName: spec.containerName,
      jobConfigUrl: spec.jobConfigUrl,
      hostOutputDir: spec.hostOutputDir,
    };
  } catch (error) {
    await db.transaction(async (trx) => {
      await trx('job_attempts').where({ id: spec.attemptId }).update({
        status: 'failed',
        stage: 'bootstrap',
        failure_reason: String(error.message || error),
        finished_at: now,
        updated_at: now,
      });

      await trx('jobs').where({ id: jobId }).update({
        status: 'failed',
        stage: 'bootstrap',
        headline: 'Trainer launch failed',
        terminal_reason: String(error.message || error),
        finished_at: now,
        updated_at: now,
      });

      await trx('job_events').insert({
        id: newId('evt'),
        job_id: jobId,
        attempt_id: spec.attemptId,
        step_key: 'bootstrap',
        event_type: 'trainer.launch.failed',
        severity: 'error',
        sequence_no: 0,
        delivery_id: newId('delivery'),
        event_time: now,
        received_at: now,
        payload_json: toJson({ error: String(error.message || error) }),
      });
    });
    throw error;
  }
}

async function stopTrainerJob(jobId, actor) {
  const job = await db('jobs').where({ id: jobId }).first();
  if (!job) throw new Error('Job not found');
  if (job.job_kind !== TRAINER_JOB_KIND) throw new Error('Job is not a trainer job');

  const attempt = await db('job_attempts').where({ id: job.latest_attempt_id }).first();
  if (!attempt) throw new Error('Job attempt not found');

  const runtimeInfo = parseJson(attempt.runtime_info_json, {});
  const containerId = runtimeInfo.containerId || runtimeInfo.lastLaunch?.containerId || null;
  const now = nowIso();

  await db('jobs').where({ id: jobId }).update({
    desired_state: 'cancel_requested',
    headline: 'Stop requested',
    updated_at: now,
  });

  await db('job_events').insert({
    id: newId('evt'),
    job_id: jobId,
    attempt_id: attempt.id,
    step_key: null,
    event_type: 'trainer.stop.requested',
    severity: 'warn',
    sequence_no: 0,
    delivery_id: newId('delivery'),
    event_time: now,
    received_at: now,
    payload_json: toJson({ requestedBy: actor?.sub || null, containerId }),
  });

  if (!containerId) {
    return { requested: true, stopped: false, reason: 'container id is not known yet' };
  }

  await dockerRunPromise(CONFIG.runtimeDockerBin, ['rm', '-f', containerId]);

  await db.transaction(async (trx) => {
    await trx('job_attempts').where({ id: attempt.id }).update({
      status: 'cancelled',
      stage: attempt.stage || 'bootstrap',
      finished_at: now,
      updated_at: now,
    });
    await trx('jobs').where({ id: jobId }).update({
      status: 'cancelled',
      finished_at: now,
      updated_at: now,
    });
    await trx('job_events').insert({
      id: newId('evt'),
      job_id: jobId,
      attempt_id: attempt.id,
      step_key: null,
      event_type: 'trainer.stop.completed',
      severity: 'warn',
      sequence_no: 0,
      delivery_id: newId('delivery'),
      event_time: now,
      received_at: now,
      payload_json: toJson({ containerId }),
    });
  });

  return { requested: true, stopped: true, containerId };
}



module.exports = {
  TRAINER_JOB_KIND,
  TRAINER_RUNTIME_KIND,
  createTrainerJob,
  listTrainerJobs,
  getTrainerSnapshot,
  buildTrainerBootstrapPayload,
  buildTrainerLaunchSpec,
  launchTrainerJob,
  stopTrainerJob,
  buildTrainerSteps,
};