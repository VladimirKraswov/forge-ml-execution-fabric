#!/usr/bin/env node
'use strict';

const fs = require('fs');
const fsp = require('fs/promises');
const path = require('path');
const os = require('os');
const http = require('http');
const https = require('https');
const { spawn } = require('child_process');

// ============================================================================
// Configuration
// ============================================================================

const ORCHESTRATOR_PORT = 18787;
const DATASETS_PORT = 18888;

// Hugging Face dataset details
const HF_DATASET = 'itkacademy/500-500-500-ru';
const HF_REPO_TARGET = 'XProger/test';
const HF_TOKEN = process.env.HF_TOKEN || '';

// Docker image for executor
const EXECUTOR_IMAGE = 'xproger/itk-executor-trainer:latest';

// Runtime profile (assumes trainer-service profile exists)
const RUNTIME_PROFILE_KEY = 'trainer-service-qwen7b';

// ============================================================================
// Helper functions (mostly reused from check-trainer-runtime-hf-e2e.js)
// ============================================================================

function color(code, text) {
  if (!process.stdout.isTTY) return text;
  return `\u001b[${code}m${text}\u001b[0m`;
}
function info(text) { console.log(color('36', `• ${text}`)); }
function ok(text) { console.log(color('32', `✔ ${text}`)); }
function warn(text) { console.log(color('33', `! ${text}`)); }
function fail(text) { console.error(color('31', `✖ ${text}`)); }

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function ensureFile(filePath) {
  if (!fs.existsSync(filePath)) throw new Error(`Missing file: ${filePath}`);
}

function safeJsonParse(text) {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}

function truncate(text, max = 3000) {
  const value = String(text || '');
  if (value.length <= max) return value;
  return `${value.slice(0, max)}\n...<truncated ${value.length - max} chars>`;
}

async function requestRaw(method, urlString, { headers = {}, body = null, timeoutMs = 15000 } = {}) {
  return new Promise((resolve, reject) => {
    const url = new URL(urlString);
    const lib = url.protocol === 'https:' ? https : http;

    const req = lib.request(
      {
        protocol: url.protocol,
        hostname: url.hostname,
        port: url.port,
        path: `${url.pathname}${url.search}`,
        method,
        headers: {
          connection: 'close',
          ...headers,
        },
        agent: false,
      },
      (res) => {
        const chunks = [];
        res.on('data', (chunk) => chunks.push(chunk));
        res.on('end', () => {
          const buffer = Buffer.concat(chunks);
          resolve({
            status: res.statusCode || 0,
            headers: res.headers,
            buffer,
            text: buffer.toString('utf-8'),
          });
        });
      }
    );

    req.setTimeout(timeoutMs, () => {
      req.destroy(new Error(`Request timeout after ${timeoutMs}ms`));
    });

    req.on('error', reject);

    if (body) req.write(body);
    req.end();
  });
}

async function requestJson(method, urlString, {
  headers = {},
  json = undefined,
  timeoutMs = 15000,
  expectedStatus = null,
} = {}) {
  const body = json === undefined ? null : Buffer.from(JSON.stringify(json), 'utf-8');

  const response = await requestRaw(method, urlString, {
    headers: {
      ...(body
        ? {
            'content-type': 'application/json',
            'content-length': String(body.length),
          }
        : {}),
      ...headers,
    },
    body,
    timeoutMs,
  });

  const payload = response.text ? safeJsonParse(response.text) : null;

  if (expectedStatus != null && response.status !== expectedStatus) {
    throw new Error(`${method} ${urlString} -> HTTP ${response.status}: ${truncate(response.text)}`);
  }
  if (response.status >= 400) {
    throw new Error(`${method} ${urlString} -> HTTP ${response.status}: ${truncate(response.text)}`);
  }

  return payload;
}

async function requestJsonRetry(method, urlString, {
  headers = {},
  json = undefined,
  timeoutMs = 8000,
  expectedStatus = null,
  attempts = 3,
  delayMs = 1200,
} = {}) {
  let lastError = null;

  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    try {
      return await requestJson(method, urlString, {
        headers,
        json,
        timeoutMs,
        expectedStatus,
      });
    } catch (error) {
      lastError = error;
      const message = String(error && error.message ? error.message : error);

      const retryable =
        /timeout/i.test(message) ||
        /ECONNRESET/i.test(message) ||
        /socket hang up/i.test(message) ||
        /EPIPE/i.test(message) ||
        /ETIMEDOUT/i.test(message) ||
        /HTTP 5\d\d/i.test(message);

      if (!retryable || attempt >= attempts) {
        throw error;
      }
      await sleep(delayMs);
    }
  }

  throw lastError;
}

async function runCommand(command, args, {
  cwd = process.cwd(),
  env = process.env,
  verbose = false,
  label = command,
} = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { cwd, env, stdio: ['ignore', 'pipe', 'pipe'] });
    let stdout = '';
    let stderr = '';

    child.stdout.on('data', (chunk) => {
      const text = chunk.toString('utf-8');
      stdout += text;
      if (verbose) process.stdout.write(color('90', `[${label}] ${text}`));
    });

    child.stderr.on('data', (chunk) => {
      const text = chunk.toString('utf-8');
      stderr += text;
      if (verbose) process.stderr.write(color('90', `[${label}:err] ${text}`));
    });

    child.on('error', reject);
    child.on('close', (code) => {
      if (code === 0) {
        resolve({ stdout: stdout.trim(), stderr: stderr.trim() });
        return;
      }
      reject(new Error(`${label} failed (exit ${code})\nSTDOUT:\n${truncate(stdout)}\nSTDERR:\n${truncate(stderr)}`));
    });
  });
}

async function stopChild(child, signal = 'SIGTERM') {
  if (!child || child.killed || child.exitCode != null) return;
  try { child.kill(signal); } catch {}
  const deadline = Date.now() + 10000;
  while (Date.now() < deadline) {
    if (child.exitCode != null) return;
    await sleep(250);
  }
  try { child.kill('SIGKILL'); } catch {}
}

function getContainerHostAlias() {
  return 'host.docker.internal';
}

function buildContainerBaseUrl(port) {
  return `http://${getContainerHostAlias()}:${port}`;
}

function remapUrlForHost(urlString, { backendPort, datasetsPort }) {
  const src = new URL(urlString);
  const pathAndQuery = `${src.pathname}${src.search}${src.hash}`;

  if (src.hostname === 'host.docker.internal') {
    if (src.port && String(src.port) === String(datasetsPort)) {
      return `http://127.0.0.1:${datasetsPort}${pathAndQuery}`;
    }
    return `http://127.0.0.1:${src.port || backendPort}${pathAndQuery}`;
  }

  if (src.port && String(src.port) === String(backendPort)) {
    return `http://127.0.0.1:${backendPort}${pathAndQuery}`;
  }
  if (src.port && String(src.port) === String(datasetsPort)) {
    return `http://127.0.0.1:${datasetsPort}${pathAndQuery}`;
  }

  return urlString;
}

async function waitForHealth(baseUrl, timeoutMs) {
  const startedAt = Date.now();
  let lastError = null;

  while (Date.now() - startedAt < timeoutMs) {
    try {
      const payload = await requestJson('GET', `${baseUrl}/health`, { timeoutMs: 2000 });
      if (payload?.ok) return payload;
    } catch (error) {
      lastError = error;
    }
    await sleep(500);
  }

  throw new Error(`Backend did not become healthy in time${lastError ? `: ${lastError.message}` : ''}`);
}

// ============================================================================
// Backend helpers (spawn, API)
// ============================================================================

async function spawnBackend(projectRoot, workRoot, args) {
  const dataRoot = path.join(workRoot, 'backend-data');
  const artifactsRoot = path.join(workRoot, 'artifacts');
  const tmpUploadsRoot = path.join(workRoot, 'tmp-uploads');
  const runtimeOutputRoot = path.join(workRoot, 'runtime-output');
  const publicBaseUrl = buildContainerBaseUrl(args.port);
  const externalBaseUrl = `http://127.0.0.1:${args.port}`;

  await Promise.all([
    fsp.mkdir(dataRoot, { recursive: true }),
    fsp.mkdir(artifactsRoot, { recursive: true }),
    fsp.mkdir(tmpUploadsRoot, { recursive: true }),
    fsp.mkdir(runtimeOutputRoot, { recursive: true }),
  ]);

  const env = {
    ...process.env,
    SVC_HOST: args.host,
    SVC_PORT: String(args.port),
    APP_PUBLIC_BASE_URL: publicBaseUrl,
    DATA_ROOT: dataRoot,
    DB_FILE: path.join(dataRoot, 'orchestrator.sqlite'),
    ARTIFACTS_ROOT: artifactsRoot,
    TMP_UPLOADS_ROOT: tmpUploadsRoot,
    RUNTIME_HOST_OUTPUT_ROOT: runtimeOutputRoot,
    JWT_SECRET: 'forge-e2e-secret',
    ADMIN_USERNAME: 'admin',
    ADMIN_PASSWORD: 'admin123456',
  };

  // Путь к серверу в новой структуре
  const serverPath = path.join(projectRoot, 'apps/orchestrator/src/server.js');
  const child = spawn(process.execPath, [serverPath], {
    cwd: projectRoot,
    env,
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  let stdout = '';
  let stderr = '';

  child.stdout.on('data', (chunk) => {
    const text = chunk.toString('utf-8');
    stdout += text;
    if (args.verbose) process.stdout.write(color('90', `[backend] ${text}`));
  });

  child.stderr.on('data', (chunk) => {
    const text = chunk.toString('utf-8');
    stderr += text;
    if (args.verbose) process.stderr.write(color('90', `[backend:err] ${text}`));
  });

  try {
    await waitForHealth(externalBaseUrl, 30000);
  } catch (error) {
    try { child.kill('SIGTERM'); } catch {}
    throw new Error(`${error.message}\nSTDOUT:\n${truncate(stdout)}\nSTDERR:\n${truncate(stderr)}`);
  }

  return {
    child,
    externalBaseUrl,
    publicBaseUrl,
    dataRoot,
    artifactsRoot,
    runtimeOutputRoot,
    logs: () => ({ stdout, stderr }),
  };
}

async function login(baseUrl) {
  const payload = await requestJson('POST', `${baseUrl}/api/v1/auth/login`, {
    json: { username: 'admin', password: 'admin123456' },
    expectedStatus: 200,
  });
  if (!payload?.token) throw new Error('JWT token not returned by login');
  return payload.token;
}

async function getRuntimeProfileId(baseUrl, jwt, profileKey) {
  const profiles = await requestJson('GET', `${baseUrl}/api/v1/runtime-profiles`, {
    headers: { authorization: `Bearer ${jwt}` },
  });
  if (!Array.isArray(profiles) || profiles.length === 0) {
    throw new Error('No runtime profiles returned');
  }
  const profile = profiles.find(p => p.profileKey === profileKey && p.status === 'active');
  if (!profile) throw new Error(`Runtime profile ${profileKey} not found or not active`);
  return profile.id;
}

async function createJob(baseUrl, jwt, payload) {
  return requestJson('POST', `${baseUrl}/api/v1/trainer/jobs`, {
    headers: { authorization: `Bearer ${jwt}` },
    json: payload,
    expectedStatus: 201,
    timeoutMs: 60000,
  });
}

async function getLaunchSpec(baseUrl, jwt, jobId) {
  return requestJson('GET', `${baseUrl}/api/v1/trainer/jobs/${encodeURIComponent(jobId)}/launch-spec`, {
    headers: { authorization: `Bearer ${jwt}` },
    timeoutMs: 30000,
  });
}

async function launchJob(baseUrl, jwt, jobId) {
  return requestJson('POST', `${baseUrl}/api/v1/trainer/jobs/${encodeURIComponent(jobId)}/launch`, {
    headers: { authorization: `Bearer ${jwt}` },
    json: { inheritEnv: ['HF_TOKEN'] },
    expectedStatus: 202,
    timeoutMs: 60000,
  });
}

async function getJob(baseUrl, jwt, jobId) {
  return requestJsonRetry('GET', `${baseUrl}/api/v1/trainer/jobs/${encodeURIComponent(jobId)}`, {
    headers: { authorization: `Bearer ${jwt}` },
    timeoutMs: 8000,
    attempts: 2,
    delayMs: 1000,
  });
}

async function getJobResult(baseUrl, jwt, jobId) {
  return requestJsonRetry('GET', `${baseUrl}/api/v1/jobs/${encodeURIComponent(jobId)}/result`, {
    headers: { authorization: `Bearer ${jwt}` },
    timeoutMs: 8000,
    attempts: 2,
    delayMs: 1000,
  });
}

async function getJobArtifacts(baseUrl, jwt, jobId) {
  return requestJsonRetry('GET', `${baseUrl}/api/v1/jobs/${encodeURIComponent(jobId)}/artifacts`, {
    headers: { authorization: `Bearer ${jwt}` },
    timeoutMs: 8000,
    attempts: 2,
    delayMs: 1000,
  });
}

async function getJobLogs(baseUrl, jwt, jobId) {
  return requestJsonRetry('GET', `${baseUrl}/api/v1/jobs/${encodeURIComponent(jobId)}/logs`, {
    headers: { authorization: `Bearer ${jwt}` },
    timeoutMs: 8000,
    attempts: 2,
    delayMs: 1000,
  });
}

async function getJobEvents(baseUrl, jwt, jobId) {
  return requestJsonRetry('GET', `${baseUrl}/api/v1/jobs/${encodeURIComponent(jobId)}/events?limit=500`, {
    headers: { authorization: `Bearer ${jwt}` },
    timeoutMs: 8000,
    attempts: 2,
    delayMs: 1000,
  });
}

async function waitForJobTerminal(baseUrl, jwt, jobId, timeoutMs, {
  containerName = '',
  verbose = false,
} = {}) {
  const startedAt = Date.now();
  let lastStatus = '';
  let lastStage = '';
  let lastProgress = null;
  let consecutivePollErrors = 0;

  while (Date.now() - startedAt < timeoutMs) {
    try {
      const job = await getJob(baseUrl, jwt, jobId);
      consecutivePollErrors = 0;

      const status = String(job?.status || '').toLowerCase();
      const stage = String(job?.stage || '');
      const progress =
        job?.progressPercent != null
          ? Number(job.progressPercent)
          : job?.progress != null
            ? Number(job.progress)
            : null;

      if (status !== lastStatus || stage !== lastStage || progress !== lastProgress) {
        info(
          `Job ${jobId}: status=${status || '<empty>'} stage=${stage || '<empty>'} progress=${progress == null ? 'n/a' : progress}`
        );
        lastStatus = status;
        lastStage = stage;
        lastProgress = progress;
      }

      if (['finished', 'failed', 'cancelled', 'succeeded'].includes(status)) {
        return job;
      }

      if (containerName) {
        const inspect = await getDockerInspect(containerName, verbose);
        const state = inspect?.State || null;

        if (inspect?.missing) {
          const logs = await getDockerLogs(containerName, verbose, 300);
          throw new Error(
            `Trainer container ${containerName} disappeared before backend received terminal state.\n` +
            `Docker inspect: ${truncate(JSON.stringify(inspect, null, 2), 2000)}\n` +
            `Docker logs:\n${truncate(logs, 6000)}`
          );
        }

        if (state && ['exited', 'dead'].includes(String(state.Status || '').toLowerCase())) {
          const logs = await getDockerLogs(containerName, verbose, 300);
          throw new Error(
            `Trainer container ${containerName} stopped before backend received terminal state.\n` +
            `Docker state: ${truncate(JSON.stringify(state, null, 2), 2000)}\n` +
            `Docker logs:\n${truncate(logs, 6000)}`
          );
        }
      }
    } catch (error) {
      consecutivePollErrors += 1;
      warn(`Job polling failed (${consecutivePollErrors}): ${error.message}`);

      const health = await probeUrl(`${baseUrl}/health`, 4000).catch((healthError) => ({
        ok: false,
        error: String(healthError.message || healthError),
      }));
      info(`Backend health probe during polling: ${JSON.stringify(health)}`);

      try {
        const result = await getJobResult(baseUrl, jwt, jobId);
        const outcome = String(
          result?.outcome ||
          result?.status ||
          result?.summary?.status ||
          ''
        ).toLowerCase();

        if (['success', 'succeeded', 'finished'].includes(outcome)) {
          info(`Terminal state derived from result endpoint: ${outcome}`);
          return {
            status: 'finished',
            stage: 'finished',
            derivedFrom: 'result',
            result,
          };
        }

        if (['failed', 'error', 'cancelled'].includes(outcome)) {
          return {
            status: outcome === 'error' ? 'failed' : outcome,
            stage: outcome,
            derivedFrom: 'result',
            result,
          };
        }
      } catch {
        // ignore fallback errors
      }

      if (consecutivePollErrors >= 2 && containerName) {
        const inspect = await getDockerInspect(containerName, verbose);
        const dockerState = inspect?.State || inspect?.state || null;
        if (dockerState) {
          info(`Docker state: ${JSON.stringify(dockerState)}`);
        } else if (inspect?.missing) {
          const dockerLogs = await getDockerLogs(containerName, verbose, 300);
          throw new Error(
            `Container ${containerName} disappeared while job is still non-terminal.\n` +
            `Docker inspect: ${truncate(JSON.stringify(inspect, null, 2), 2000)}\n` +
            `Docker logs:\n${truncate(dockerLogs, 6000)}`
          );
        }
      }
    }

    await sleep(5000);
  }

  const health = await probeUrl(`${baseUrl}/health`, 5000).catch((error) => ({
    ok: false,
    error: String(error.message || error),
  }));
  const logs = await getJobLogs(baseUrl, jwt, jobId).catch((error) => ({
    error: String(error.message || error),
  }));

  const events = await getJobEvents(baseUrl, jwt, jobId).catch((error) => ({
    error: String(error.message || error),
  }));
  const result = await getJobResult(baseUrl, jwt, jobId).catch((error) => ({
    error: String(error.message || error),
  }));
  const inspect = await getDockerInspect(containerName, verbose);
  const dockerLogs = await getDockerLogs(containerName, verbose, 300);

  throw new Error(
    `Job ${jobId} did not reach terminal state in time\n` +
    `Backend health: ${truncate(JSON.stringify(health, null, 2), 2000)}\n` +
    `Result endpoint: ${truncate(JSON.stringify(result, null, 2), 3000)}\n` +
    `Recent logs: ${truncate(JSON.stringify(logs, null, 2), 4000)}\n` +
    `Recent events: ${truncate(JSON.stringify(events, null, 2), 4000)}\n` +
    `Docker inspect: ${truncate(JSON.stringify(inspect, null, 2), 3000)}\n` +
    `Docker logs:\n${truncate(dockerLogs, 6000)}`
  );
}

// ============================================================================
// Docker helpers
// ============================================================================

async function getDockerInspect(containerRef, verbose = false) {
  if (!containerRef) return null;
  try {
    const { stdout } = await runCommand(
      'docker',
      ['inspect', containerRef],
      { verbose, label: 'docker-inspect' }
    );
    const parsed = safeJsonParse(stdout);
    return Array.isArray(parsed) ? parsed[0] || null : parsed;
  } catch (error) {
    const message = String(error.message || error);
    return {
      missing: /No such object|no such object|No such container/i.test(message),
      error: message,
    };
  }
}

async function getDockerLogs(containerRef, verbose = false, tail = 200) {
  if (!containerRef) return '';
  try {
    const { stdout, stderr } = await runCommand(
      'docker',
      ['logs', '--tail', String(tail), containerRef],
      { verbose, label: 'docker-logs' }
    );
    return [stdout, stderr].filter(Boolean).join('\n');
  } catch (error) {
    return `Failed to read docker logs: ${String(error.message || error)}`;
  }
}

async function checkDockerImageExists(image) {
  try {
    await runCommand('docker', ['inspect', image], { verbose: false });
    return true;
  } catch {
    return false;
  }
}

async function buildAndPushExecutorImage(trainerServiceDir, imageTag, verbose) {
  info(`Building executor image ${imageTag} from ${trainerServiceDir}`);
  await runCommand(
    'docker',
    [
      'build',
      '-t', imageTag,
      '-f', path.join(trainerServiceDir, 'docker', 'Dockerfile'),
      trainerServiceDir,
    ],
    {
      cwd: trainerServiceDir,
      verbose,
      label: 'docker-build',
    }
  );
  ok(`Image built: ${imageTag}`);

  info(`Pushing ${imageTag} to Docker Hub...`);
  await runCommand('docker', ['push', imageTag], { verbose, label: 'docker-push' });
  ok(`Image pushed: ${imageTag}`);
}

// ============================================================================
// Bootstrap and config building
// ============================================================================

async function probeUrl(urlString, timeoutMs = 10000) {
  const startedAt = Date.now();
  try {
    const response = await requestRaw('GET', urlString, { timeoutMs });
    return {
      ok: response.status >= 200 && response.status < 300,
      status: response.status,
      ms: Date.now() - startedAt,
      bodyPreview: String(response.text || '').slice(0, 500),
    };
  } catch (error) {
    return {
      ok: false,
      ms: Date.now() - startedAt,
      error: String(error && error.message ? error.message : error),
    };
  }
}

async function tryFetchBootstrapForHost(launchSpecUrl, args) {
  const bootstrapUrlForHost = remapUrlForHost(launchSpecUrl, {
    backendPort: args.port,
    datasetsPort: args.datasetsPort,
  });

  info(`Bootstrap URL from launch spec: ${launchSpecUrl}`);
  info(`Bootstrap URL for host fetch: ${bootstrapUrlForHost}`);

  const probe = await probeUrl(bootstrapUrlForHost, 10000);
  info(`Bootstrap probe: ${JSON.stringify(probe)}`);

  if (!probe.ok) {
    warn(`Host bootstrap probe failed, but container may still access it: ${probe.error || `HTTP ${probe.status}`}`);
    return {
      ok: false,
      url: bootstrapUrlForHost,
      probe,
      payload: null,
    };
  }

  try {
    const payload = await requestJson('GET', bootstrapUrlForHost, { timeoutMs: 15000 });
    return {
      ok: true,
      url: bootstrapUrlForHost,
      probe,
      payload,
    };
  } catch (error) {
    warn(`Host bootstrap fetch failed, but container launch will continue: ${error.message}`);
    return {
      ok: false,
      url: bootstrapUrlForHost,
      probe,
      payload: null,
      error: error.message,
    };
  }
}

async function downloadArtifact(downloadUrl, destination, jwt) {
  const response = await requestRaw('GET', downloadUrl, {
    headers: { authorization: `Bearer ${jwt}` },
    timeoutMs: 60000,
  });
  if (response.status >= 400) {
    throw new Error(`Artifact download failed: HTTP ${response.status}: ${truncate(response.text)}`);
  }
  await fsp.writeFile(destination, response.buffer);
}

async function hfListRepoFiles(repo, token) {
  const response = await requestRaw('GET', `https://huggingface.co/api/models/${repo}`, {
    headers: token ? { authorization: `Bearer ${token}` } : {},
    timeoutMs: 30000,
  });
  if (response.status >= 400) {
    throw new Error(`HF API returned HTTP ${response.status}: ${truncate(response.text)}`);
  }
  const payload = safeJsonParse(response.text);
  const siblings = Array.isArray(payload?.siblings)
    ? payload.siblings.map((item) => item.rfilename).filter(Boolean)
    : [];
  return { payload, siblings };
}

function hasAny(files, patterns) {
  return patterns.some((pattern) => {
    if (pattern instanceof RegExp) return files.some((file) => pattern.test(file));
    return files.includes(pattern);
  });
}

async function waitForHfArtifacts(repo, token, timeoutMs) {
  const startedAt = Date.now();
  let lastFiles = [];

  while (Date.now() - startedAt < timeoutMs) {
    try {
      const { siblings } = await hfListRepoFiles(repo, token);
      lastFiles = siblings;

      const hasModel = hasAny(siblings, [/\.safetensors$/, 'config.json', 'tokenizer.json']);
      const hasMetadata = hasAny(siblings, [
        'artifacts/result/job-result.json',
        'artifacts/train/train_summary.json',
      ]);

      if (hasModel && hasMetadata) {
        return siblings;
      }
    } catch {
      // publish may still be in progress
    }

    await sleep(5000);
  }

  throw new Error(
    `Expected model files and metadata did not appear in HF repo in time. Last seen files: ${lastFiles.slice(0, 50).join(', ')}`
  );
}

// ============================================================================
// Build job payload
// ============================================================================

function buildTrainerJobPayload({
  runtimeProfileId,
  jobId,
  runtimeImage,
  datasetBaseUrl,
  hfRepo,
  logicalBaseModelId,
}) {
  // Используем URL локального fixture сервера
  const trainUrl = `${datasetBaseUrl}/datasets/train.json`;
  const valUrl = `${datasetBaseUrl}/datasets/val.json`;
  const evalUrl = `${datasetBaseUrl}/datasets/eval.jsonl`;

  const model = {
    source: 'local',
    local_path: '/app',
    trust_remote_code: false,
    load_in_4bit: true,
    dtype: 'bfloat16',
    max_seq_length: 128,
  };

  if (logicalBaseModelId) {
    model.repo_id = logicalBaseModelId;
    model.base_model = logicalBaseModelId;
    model.base_model_name_or_path = logicalBaseModelId;
  }

  const training = {
    method: 'qlora',
    max_seq_length: 128,
    per_device_train_batch_size: 1,
    gradient_accumulation_steps: 1,
    num_train_epochs: 1,
    learning_rate: 0.0001,
    warmup_ratio: 0.03,
    logging_steps: 1,
    save_steps: 1,
    eval_steps: 1,
    bf16: true,
    packing: false,
    save_total_limit: 1,
    optim: 'adamw_8bit',
  };

  const postprocess = {
    merge_lora: true,
    save_merged_16bit: true,
    run_awq_quantization: false,
  };

  const evaluationDataset = {
    source: 'url',
    url: evalUrl,
    format: 'jsonl',
    question_field: 'question',
    answer_field: 'candidate_answer',
    score_field: 'reference_score',
    max_score_field: 'max_score',
    tags_field: 'hash_tags',
  };

  return {
    runtimeProfileId,
    jobId,
    name: `trainer-e2e-${jobId}`,
    labels: { e2e: true, target: 'huggingface', repo: hfRepo },
    config: {
      job_id: jobId,
      job_name: `trainer-e2e-${jobId}`,
      mode: 'remote',
      model,
      dataset: {
        source: 'url',
        train_url: trainUrl,
        val_url: valUrl,
        format: 'instruction_output',
        input_field: 'input',
        output_field: 'output',
      },
      training,
      lora: {
        r: 8,
        lora_alpha: 16,
        lora_dropout: 0.0,
        bias: 'none',
        use_gradient_checkpointing: 'unsloth',
        random_state: 3407,
        target_modules: ['q_proj', 'v_proj'],
      },
      outputs: {
        base_dir: `/output/${jobId}`,
      },
      postprocess,
      evaluation: {
        enabled: true,
        target: 'merged',
        max_samples: 1,
        max_new_tokens: 32,
        temperature: 0,
        do_sample: false,
        dataset: evaluationDataset,
      },
      upload: {
        enabled: true,
        target: 'url',
        timeout_sec: 300,
      },
      huggingface: {
        enabled: true,
        push_lora: false,
        push_merged: true,
        repo_id_merged: hfRepo,
        repo_id_metadata: hfRepo,
        private: false,
        commit_message: `trainer-runtime e2e ${jobId}`,
      },
      pipeline: {
        prepare_assets: { enabled: true },
        training: { enabled: true, ...training },
        merge: { enabled: true, ...postprocess },
        evaluation: {
          enabled: true,
          target: 'merged',
          max_samples: 1,
          max_new_tokens: 32,
          temperature: 0,
          do_sample: false,
          dataset: evaluationDataset,
        },
        publish: {
          enabled: true,
          push_lora: false,
          push_merged: true,
          repo_id_merged: hfRepo,
          repo_id_metadata: hfRepo,
          private: false,
          commit_message: `trainer-runtime e2e ${jobId}`,
        },
        upload: { enabled: true, target: 'url', timeout_sec: 300 },
      },
    },
    executor: {
      image: runtimeImage,
      gpus: 'all',
      shmSize: '16g',
      extraDockerArgs: ['--add-host=host.docker.internal:host-gateway'],
    },
  };
}

// ============================================================================
// Main
// ============================================================================

async function main() {
  const args = {
    projectRoot: process.argv[2] || '.', // should be path to forge-ml-execution-fabric
    trainerServiceDir: './apps/executor-trainer', // Исправлено: правильный путь к executor'у
    runtimeImage: EXECUTOR_IMAGE,
    hfRepo: HF_REPO_TARGET,
    baseImage: 'igortet/model-qwen-7b',
    imageTag: EXECUTOR_IMAGE,
    host: '0.0.0.0',
    port: ORCHESTRATOR_PORT,
    datasetsPort: DATASETS_PORT,
    timeoutMinutes: 90,
    hfWaitSeconds: 180,
    jobId: `job_e2e_${Date.now()}`,
    logicalBaseModelId: 'Qwen/Qwen2.5-7B-Instruct',
    keepWorkdir: false,
    verbose: true,
    skipBuild: false,
  };

  const hfToken = String(process.env.HF_TOKEN || '').trim();
  if (!hfToken) {
    throw new Error('HF_TOKEN environment variable is required');
  }

  const projectRoot = path.resolve(args.projectRoot);
  // Проверяем, что сервер существует в новой структуре
  const serverPath = path.join(projectRoot, 'apps/orchestrator/src/server.js');
  ensureFile(serverPath);
  // Также проверяем package.json в корне (для установки зависимостей, если нужно)
  ensureFile(path.join(projectRoot, 'package.json'));

  const workRoot = await fsp.mkdtemp(path.join(os.tmpdir(), 'forge-trainer-e2e-'));
  info(`Workdir: ${workRoot}`);

  // Ensure executor image exists or build it
  const imageExists = await checkDockerImageExists(EXECUTOR_IMAGE);
  if (!imageExists) {
    if (!args.trainerServiceDir) {
      throw new Error(`Image ${EXECUTOR_IMAGE} not found and --trainer-service-dir not provided to build it.`);
    }
    // Проверяем, что директория с исходниками executor'а существует
    const executorSourceDir = path.resolve(projectRoot, args.trainerServiceDir);
    if (!fs.existsSync(executorSourceDir)) {
      throw new Error(`Executor source directory not found: ${executorSourceDir}`);
    }
    await buildAndPushExecutorImage(executorSourceDir, EXECUTOR_IMAGE, args.verbose);
  } else {
    info(`Executor image ${EXECUTOR_IMAGE} already exists, using it.`);
  }

  let backend = null;

  try {
    info('Starting backend');
    backend = await spawnBackend(projectRoot, workRoot, args);
    ok(`Backend is up on ${backend.externalBaseUrl}`);
    info(`Backend public base URL for containers: ${backend.publicBaseUrl}`);

    const jwt = await login(backend.externalBaseUrl);
    ok('Admin login works');

    const runtimeProfileId = await getRuntimeProfileId(backend.externalBaseUrl, jwt, RUNTIME_PROFILE_KEY);
    ok(`Runtime profile resolved: ${runtimeProfileId}`);

    const jobId = args.jobId;
    const jobPayload = buildTrainerJobPayload({
      runtimeProfileId,
      jobId,
      runtimeImage: EXECUTOR_IMAGE,
      datasetBaseUrl: '', // not used
      hfRepo: HF_REPO_TARGET,
      logicalBaseModelId: args.logicalBaseModelId,
    });

    info(`Creating trainer job ${jobId}`);
    const created = await createJob(backend.externalBaseUrl, jwt, jobPayload);
    if (created?.id !== jobId) throw new Error('Created job id mismatch');
    ok('Trainer job created');

    const launchSpec = await getLaunchSpec(backend.externalBaseUrl, jwt, jobId);
    if (!launchSpec?.jobConfigUrl) throw new Error('Launch spec did not return jobConfigUrl');
    ok('Launch spec generated');

    // Host-side bootstrap check (non-fatal)
    const bootstrapCheck = await tryFetchBootstrapForHost(launchSpec.jobConfigUrl, args);
    if (bootstrapCheck.ok) {
      ok('Bootstrap config fetch works from host');
      const bootstrap = bootstrapCheck.payload;
      if (!bootstrap?.config?.upload?.url_targets?.summary_url || !bootstrap?.status_url) {
        throw new Error('Bootstrap payload is incomplete');
      }
      ok('Bootstrap endpoint returns managed callback/upload URLs');
    } else {
      warn('Skipping strict host bootstrap validation; continuing with real container launch');
    }

    // The executor will be started manually by the test, but we can also simulate by using the launch spec.
    // In this test, we will NOT actually run the container automatically; we will output the command for manual verification.
    // However, to keep the test fully automated, we could run it. But given the instructions, we just need to log.
    // We'll output the command and then wait for the user to run it? That would not be automated.
    // Instead, we can run it using the launch spec.
    // Note: In the new architecture, the orchestrator does not run the container; it only provides the spec.
    // But for E2E test, we need to run it to verify end-to-end. So we'll run it here.

    info(`Launching trainer container using: ${launchSpec.dockerRun}`);
    const launch = await launchJob(backend.externalBaseUrl, jwt, jobId);
    if (!launch?.launched) throw new Error('Launch endpoint did not confirm launch');
    ok(`Trainer container launched: ${launch.containerId || launch.containerName}`);

    const launchedContainerName = launch.containerName || launch.containerId || '';
    if (launchedContainerName) {
      info(`Launched container: ${launchedContainerName}`);
    }

    const terminalJob = await waitForJobTerminal(
      backend.externalBaseUrl,
      jwt,
      jobId,
      args.timeoutMinutes * 60 * 1000,
      {
        containerName: launchedContainerName,
        verbose: args.verbose,
      }
    );

    const terminalStatus = String(terminalJob.status || '').toLowerCase();
    if (!['finished', 'success', 'succeeded'].includes(terminalStatus)) {
      const logs = await getJobLogs(backend.externalBaseUrl, jwt, jobId).catch(() => []);
      const events = await getJobEvents(backend.externalBaseUrl, jwt, jobId).catch(() => []);
      throw new Error(
        `Job finished unsuccessfully: status=${terminalJob.status} stage=${terminalJob.stage} reason=${terminalJob.terminalReason || '<none>'}\n` +
        `Recent logs: ${truncate(JSON.stringify(logs, null, 2), 4000)}\n` +
        `Recent events: ${truncate(JSON.stringify(events, null, 2), 4000)}`
      );
    }
    ok('Job reached finished state');

    const resultSummary = await getJobResult(backend.externalBaseUrl, jwt, jobId);
    const outcome = String(
      resultSummary?.outcome ||
      resultSummary?.status ||
      resultSummary?.summary?.status ||
      ''
    ).toLowerCase();

    if (!resultSummary?.summary && !resultSummary?.result_json && !resultSummary?.resultJson) {
      throw new Error(`Unexpected job result summary: ${truncate(JSON.stringify(resultSummary, null, 2), 4000)}`);
    }
    if (outcome && !['success', 'succeeded', 'finished'].includes(outcome)) {
      throw new Error(`Job result outcome is not successful: ${truncate(JSON.stringify(resultSummary, null, 2), 4000)}`);
    }
    ok('Backend stored final result summary');

    const artifacts = await getJobArtifacts(backend.externalBaseUrl, jwt, jobId);
    const artifactTypes = new Set(
      (Array.isArray(artifacts) ? artifacts : []).map((item) => item.artifactType || item.artifact_type)
    );

    const mustHaveArtifacts = [
      'logs',
      'config',
      'summary',
      'train_metrics',
      'train_history',
      'eval_summary',
      'eval_details',
      'merged_archive',
      'full_archive',
    ];

    const missingArtifactTypes = mustHaveArtifacts.filter((item) => !artifactTypes.has(item));
    if (missingArtifactTypes.length) {
      const uploadErrors =
        resultSummary?.summary?.upload_errors ||
        resultSummary?.summary?.result?.upload_errors ||
        resultSummary?.summary?.uploadErrors ||
        {};

      throw new Error(
        `Missing stored artifact types: ${missingArtifactTypes.join(', ')}\n` +
        `Upload errors: ${JSON.stringify(uploadErrors, null, 2)}\n` +
        `All artifacts: ${JSON.stringify(artifacts, null, 2)}`
      );
    }
    ok('Backend stored expected uploaded artifacts');

    const downloadableArtifact = (Array.isArray(artifacts) ? artifacts : []).find((item) => {
      const t = item.artifactType || item.artifact_type;
      return t === 'summary' || t === 'config' || t === 'logs';
    });

    if (!downloadableArtifact?.downloadUrl && !downloadableArtifact?.download_url && !downloadableArtifact?.uri) {
      throw new Error('No downloadable artifact found');
    }

    const downloadUrl =
      downloadableArtifact.downloadUrl ||
      downloadableArtifact.download_url ||
      downloadableArtifact.uri;

    const artifactDownloadPath = path.join(workRoot, 'downloaded-artifact.bin');
    await downloadArtifact(downloadUrl, artifactDownloadPath, jwt);
    const downloadedStat = await fsp.stat(artifactDownloadPath);
    if (downloadedStat.size <= 0) throw new Error('Downloaded artifact is empty');
    ok('Artifact download endpoint works');

    info(`Waiting for HF repo ${HF_REPO_TARGET} to reflect merged model + metadata`);
    const hfFiles = await waitForHfArtifacts(HF_REPO_TARGET, hfToken, args.hfWaitSeconds * 1000);
    ok('HF repo contains merged model files and metadata artifacts');

    const uploads = resultSummary.summary?.uploads || resultSummary.summary?.result?.uploads || null;
    if (uploads && typeof uploads === 'object') {
      info(`Reported uploads keys: ${Object.keys(uploads).sort().join(', ')}`);
    }

    console.log('\n=== SUCCESS ===');
    console.log(`Job ID: ${jobId}`);
    console.log(`Runtime image: ${EXECUTOR_IMAGE}`);
    console.log(`Backend: ${backend.externalBaseUrl}`);
    console.log(`HF repo: https://huggingface.co/${HF_REPO_TARGET}`);
    console.log(`HF files sample: ${hfFiles.slice(0, 20).join(', ')}`);
    console.log(`Workdir: ${workRoot}`);
    if (!args.keepWorkdir) {
      console.log('Temporary workdir will be deleted on exit.');
    }
  } finally {
    if (backend?.child) {
      await stopChild(backend.child);
    }
    if (!args.keepWorkdir) {
      try {
        await fsp.rm(workRoot, { recursive: true, force: true });
      } catch {}
    }
  }
}

main().catch((error) => {
  fail(String(error.message || error));
  process.exitCode = 1;
});