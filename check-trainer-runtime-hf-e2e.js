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
// CLI
// ============================================================================

function printHelp() {
  console.log(`Usage:
  node ./check-trainer-runtime-hf-e2e.js [options]

Required:
  --project-root PATH           Backend project root (contains package.json and src/server.js)
  --hf-repo OWNER/REPO          Target Hugging Face model repo, e.g. XProger/test_2

One of:
  --trainer-service-dir PATH    Build runtime image from trainer-service sources
  --runtime-image IMAGE         Use already built trainer runtime image

Options:
  --base-image IMAGE            Docker BASE_IMAGE for trainer-service build (default: igortet/model-qwen-7b)
  --image-tag IMAGE             Built image tag (default: forge-trainer-e2e:<timestamp>)
  --host HOST                   Backend listen host (default: 0.0.0.0)
  --port N                      Backend listen port (default: 18787)
  --datasets-port N             Fixture dataset server port (default: 18888)
  --timeout-minutes N           Max whole-job wait time (default: 90)
  --hf-wait-seconds N           Wait after finish for HF files (default: 180)
  --job-id ID                   Explicit job id (default: auto)
  --logical-base-model-id ID    Optional logical HF base model id for metadata, e.g. Qwen/Qwen2.5-7B-Instruct
  --keep-workdir                Keep temp workdir and backend data
  --verbose                     Stream docker/backend output
  --skip-build                  Skip docker build even if --trainer-service-dir is set; requires --runtime-image
  --help, -h                    Show help

Environment:
  HF_TOKEN                      Required. Used by trainer-service publish step and HF verification API.

What this script validates end-to-end:
  1) backend boot + auth + runtime profile
  2) trainer job creation + bootstrap + docker launch
  3) real trainer-service pipeline: assets -> training -> merge -> evaluation -> upload -> HF publish
  4) backend job result, logs, stored artifacts and artifact download
  5) Hugging Face repo contains merged model files and metadata artifacts
`);
}

function parseArgs(argv) {
  const args = {
    projectRoot: null,
    trainerServiceDir: null,
    runtimeImage: null,
    hfRepo: '',
    baseImage: 'igortet/model-qwen-7b',
    imageTag: `forge-trainer-e2e:${Date.now()}`,
    host: '0.0.0.0',
    port: 18787,
    datasetsPort: 18888,
    timeoutMinutes: 90,
    hfWaitSeconds: 180,
    jobId: '',
    logicalBaseModelId: '',
    keepWorkdir: false,
    verbose: false,
    skipBuild: false,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    switch (token) {
      case '--project-root':
        args.projectRoot = path.resolve(argv[++i] || '.');
        break;
      case '--trainer-service-dir':
        args.trainerServiceDir = path.resolve(argv[++i] || '.');
        break;
      case '--runtime-image':
        args.runtimeImage = String(argv[++i] || '').trim();
        break;
      case '--hf-repo':
        args.hfRepo = String(argv[++i] || '').trim();
        break;
      case '--base-image':
        args.baseImage = String(argv[++i] || '').trim();
        break;
      case '--image-tag':
        args.imageTag = String(argv[++i] || '').trim();
        break;
      case '--host':
        args.host = String(argv[++i] || '').trim() || args.host;
        break;
      case '--port':
        args.port = Number(argv[++i] || args.port);
        break;
      case '--datasets-port':
        args.datasetsPort = Number(argv[++i] || args.datasetsPort);
        break;
      case '--timeout-minutes':
        args.timeoutMinutes = Number(argv[++i] || args.timeoutMinutes);
        break;
      case '--hf-wait-seconds':
        args.hfWaitSeconds = Number(argv[++i] || args.hfWaitSeconds);
        break;
      case '--job-id':
        args.jobId = String(argv[++i] || '').trim();
        break;
      case '--logical-base-model-id':
        args.logicalBaseModelId = String(argv[++i] || '').trim();
        break;
      case '--keep-workdir':
        args.keepWorkdir = true;
        break;
      case '--verbose':
        args.verbose = true;
        break;
      case '--skip-build':
        args.skipBuild = true;
        break;
      case '--help':
      case '-h':
        printHelp();
        process.exit(0);
        break;
      default:
        throw new Error(`Unknown argument: ${token}`);
    }
  }

  if (!args.projectRoot) throw new Error('--project-root is required');
  if (!args.hfRepo) throw new Error('--hf-repo is required');
  if (!args.runtimeImage && !args.trainerServiceDir) {
    throw new Error('Provide --trainer-service-dir or --runtime-image');
  }
  if (args.skipBuild && !args.runtimeImage) {
    throw new Error('--skip-build requires --runtime-image');
  }

  if (!Number.isFinite(args.port) || args.port <= 0) args.port = 18787;
  if (!Number.isFinite(args.datasetsPort) || args.datasetsPort <= 0) args.datasetsPort = 18888;
  if (!Number.isFinite(args.timeoutMinutes) || args.timeoutMinutes <= 0) args.timeoutMinutes = 90;
  if (!Number.isFinite(args.hfWaitSeconds) || args.hfWaitSeconds <= 0) args.hfWaitSeconds = 180;

  return args;
}

// ============================================================================
// Logging
// ============================================================================

function color(code, text) {
  if (!process.stdout.isTTY) return text;
  return `\u001b[${code}m${text}\u001b[0m`;
}
function info(text) { console.log(color('36', `• ${text}`)); }
function ok(text) { console.log(color('32', `✔ ${text}`)); }
function warn(text) { console.log(color('33', `! ${text}`)); }
function fail(text) { console.error(color('31', `✖ ${text}`)); }

// ============================================================================
// Utils
// ============================================================================

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

// ============================================================================
// Network helpers
// ============================================================================

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

function requestRaw(method, urlString, { headers = {}, body = null, timeoutMs = 15000 } = {}) {
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
// Process helpers
// ============================================================================

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

// ============================================================================
// Backend + fixture bootstrap
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

  const child = spawn(process.execPath, ['src/server.js'], {
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

async function startFixtureServer(port, datasetRoot, verbose = false) {
  await fsp.mkdir(datasetRoot, { recursive: true });

  const trainSamples = [
    { input: 'Назови столицу Франции.', output: 'Столица Франции — Париж.' },
    { input: 'Сколько будет 2+2?', output: '2+2=4.' },
  ];
  const valSamples = [
    { input: 'Назови столицу Италии.', output: 'Столица Италии — Рим.' },
  ];
  const evalSamples = [
    {
      id: 'eval_1',
      question: 'Сколько будет 2+2?',
      candidate_answer: '4',
      reference_score: 5,
      max_score: 5,
      hash_tags: ['math', 'sanity'],
    },
  ];

  await fsp.writeFile(path.join(datasetRoot, 'train.json'), JSON.stringify(trainSamples, null, 2), 'utf-8');
  await fsp.writeFile(path.join(datasetRoot, 'val.json'), JSON.stringify(valSamples, null, 2), 'utf-8');
  await fsp.writeFile(
    path.join(datasetRoot, 'eval.jsonl'),
    evalSamples.map((row) => JSON.stringify(row)).join('\n') + '\n',
    'utf-8'
  );

  const server = http.createServer(async (req, res) => {
    try {
      const url = new URL(req.url, `http://${req.headers.host}`);
      const routeMap = {
        '/datasets/train.json': { file: 'train.json', contentType: 'application/json; charset=utf-8' },
        '/datasets/val.json': { file: 'val.json', contentType: 'application/json; charset=utf-8' },
        '/datasets/eval.jsonl': { file: 'eval.jsonl', contentType: 'application/x-ndjson; charset=utf-8' },
        '/health': { json: { ok: true } },
      };

      const route = routeMap[url.pathname];
      if (!route) {
        res.statusCode = 404;
        res.end('not found');
        return;
      }

      if (route.json) {
        res.statusCode = 200;
        res.setHeader('content-type', 'application/json; charset=utf-8');
        res.end(JSON.stringify(route.json));
        return;
      }

      const abs = path.join(datasetRoot, route.file);
      const data = await fsp.readFile(abs);
      res.statusCode = 200;
      res.setHeader('content-type', route.contentType);
      res.end(data);
    } catch (error) {
      res.statusCode = 500;
      res.end(String(error.message || error));
    }
  });

  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(port, '0.0.0.0', resolve);
  });

  const externalBaseUrl = `http://127.0.0.1:${port}`;
  const publicBaseUrl = buildContainerBaseUrl(port);

  if (verbose) info(`Fixture dataset server listening on 0.0.0.0:${port}`);
  return { server, externalBaseUrl, publicBaseUrl };
}

// ============================================================================
// API helpers
// ============================================================================

async function login(baseUrl) {
  const payload = await requestJson('POST', `${baseUrl}/api/v1/auth/login`, {
    json: { username: 'admin', password: 'admin123456' },
    expectedStatus: 200,
  });
  if (!payload?.token) throw new Error('JWT token not returned by login');
  return payload.token;
}

async function getRuntimeProfileId(baseUrl, jwt) {
  const profiles = await requestJson('GET', `${baseUrl}/api/v1/runtime-profiles`, {
    headers: { authorization: `Bearer ${jwt}` },
  });
  if (!Array.isArray(profiles) || profiles.length === 0) {
    throw new Error('No runtime profiles returned');
  }
  const active = profiles.find((item) => item.status === 'active') || profiles[0];
  if (!active?.id) throw new Error('Runtime profile is missing id');
  return active.id;
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

// ============================================================================
// E2E payload
// ============================================================================

function buildTrainerJobPayload({
  runtimeProfileId,
  jobId,
  runtimeImage,
  datasetBaseUrl,
  hfRepo,
  logicalBaseModelId,
}) {
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
    url: `${datasetBaseUrl}/datasets/eval.jsonl`,
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
        train_url: `${datasetBaseUrl}/datasets/train.json`,
        val_url: `${datasetBaseUrl}/datasets/val.json`,
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

      // ВАЖНО: pipeline.* должен соответствовать pydantic stage schemas,
      // а не быть просто { enabled: true }.
      pipeline: {
        prepare_assets: { enabled: true },

        training: {
          enabled: true,
          ...training,
        },

        merge: {
          enabled: true,
          ...postprocess,
        },

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

        upload: {
          enabled: true,
          target: 'url',
          timeout_sec: 300,
        },
      }
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
// Build runtime image
// ============================================================================

async function buildRuntimeImage(args) {
  if (args.runtimeImage && (args.skipBuild || !args.trainerServiceDir)) {
    return args.runtimeImage;
  }

  const trainerDir = args.trainerServiceDir;
  ensureFile(path.join(trainerDir, 'docker', 'Dockerfile'));
  ensureFile(path.join(trainerDir, 'requirements.txt'));

  info(`Building trainer runtime image ${args.imageTag}`);
  await runCommand(
    'docker',
    [
      'build',
      '--build-arg', `BASE_IMAGE=${args.baseImage}`,
      '-t', args.imageTag,
      '-f', path.join(trainerDir, 'docker', 'Dockerfile'),
      trainerDir,
    ],
    {
      cwd: trainerDir,
      verbose: args.verbose,
      label: 'docker-build',
    }
  );
  ok(`Runtime image built: ${args.imageTag}`);
  return args.imageTag;
}

// ============================================================================
// Bootstrap checks
// ============================================================================

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

// ============================================================================
// Waiters
// ============================================================================

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
          // Контейнер мог умереть и auto-remove из-за --rm.
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
// Artifact + HF checks
// ============================================================================

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
// Main
// ============================================================================

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const hfToken = String(process.env.HF_TOKEN || '').trim();
  if (!hfToken) {
    throw new Error('HF_TOKEN environment variable is required');
  }

  const projectRoot = path.resolve(args.projectRoot);
  ensureFile(path.join(projectRoot, 'package.json'));
  ensureFile(path.join(projectRoot, 'src', 'server.js'));

  const workRoot = await fsp.mkdtemp(path.join(os.tmpdir(), 'forge-trainer-e2e-'));
  const datasetRoot = path.join(workRoot, 'datasets');

  let backend = null;
  let fixture = null;
  let runtimeImage = args.runtimeImage;

  try {
    info(`Workdir: ${workRoot}`);

    runtimeImage = await buildRuntimeImage(args);

    info('Starting fixture dataset server');
    fixture = await startFixtureServer(args.datasetsPort, datasetRoot, args.verbose);
    ok('Fixture dataset server is up');

    info('Starting backend');
    backend = await spawnBackend(projectRoot, workRoot, args);
    ok(`Backend is up on ${backend.externalBaseUrl}`);
    info(`Backend public base URL for containers: ${backend.publicBaseUrl}`);
    info(`Dataset base URL for containers: ${fixture.publicBaseUrl}`);

    const jwt = await login(backend.externalBaseUrl);
    ok('Admin login works');

    const runtimeProfileId = await getRuntimeProfileId(backend.externalBaseUrl, jwt);
    ok(`Runtime profile resolved: ${runtimeProfileId}`);

    const jobId = args.jobId || `job_e2e_${Date.now()}`;
    const jobPayload = buildTrainerJobPayload({
      runtimeProfileId,
      jobId,
      runtimeImage,
      datasetBaseUrl: fixture.publicBaseUrl,
      hfRepo: args.hfRepo,
      logicalBaseModelId: args.logicalBaseModelId,
    });

    info(`Creating trainer job ${jobId}`);
    const created = await createJob(backend.externalBaseUrl, jwt, jobPayload);
    if (created?.id !== jobId) throw new Error('Created job id mismatch');
    ok('Trainer job created');

    const launchSpec = await getLaunchSpec(backend.externalBaseUrl, jwt, jobId);
    if (!launchSpec?.jobConfigUrl) throw new Error('Launch spec did not return jobConfigUrl');
    ok('Launch spec generated');

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

    info('Launching trainer container');
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

    info(`Waiting for HF repo ${args.hfRepo} to reflect merged model + metadata`);
    const hfFiles = await waitForHfArtifacts(args.hfRepo, hfToken, args.hfWaitSeconds * 1000);
    ok('HF repo contains merged model files and metadata artifacts');

    const uploads = resultSummary.summary?.uploads || resultSummary.summary?.result?.uploads || null;
    if (uploads && typeof uploads === 'object') {
      info(`Reported uploads keys: ${Object.keys(uploads).sort().join(', ')}`);
    }

    console.log('\n=== SUCCESS ===');
    console.log(`Job ID: ${jobId}`);
    console.log(`Runtime image: ${runtimeImage}`);
    console.log(`Backend: ${backend.externalBaseUrl}`);
    console.log(`HF repo: https://huggingface.co/${args.hfRepo}`);
    console.log(`HF files sample: ${hfFiles.slice(0, 20).join(', ')}`);
    console.log(`Workdir: ${workRoot}`);
    if (!args.keepWorkdir) {
      console.log('Temporary workdir will be deleted on exit.');
    }
  } finally {
    if (fixture?.server) {
      await new Promise((resolve) => fixture.server.close(resolve));
    }
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