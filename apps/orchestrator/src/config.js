const path = require('path');
require('dotenv').config();

function asInt(value, fallback) {
  const parsed = Number.parseInt(String(value ?? ''), 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function normalizeOrigin(value) {
  const raw = String(value || '*').trim();
  if (!raw || raw === '*') return '*';
  try {
    return new URL(raw).origin;
  } catch {
    return '*';
  }
}

const dataRoot = path.resolve(process.env.DATA_ROOT || '.forge');
const dbFile = path.resolve(process.env.DB_FILE || path.join(dataRoot, 'orchestrator.sqlite'));

const CONFIG = {
  host: process.env.SVC_HOST || '0.0.0.0',
  port: asInt(process.env.SVC_PORT, 8787),
  publicBaseUrl: String(process.env.APP_PUBLIC_BASE_URL || '').trim(),
  corsOrigin: normalizeOrigin(process.env.CORS_ORIGIN || '*'),

  dataRoot,
  dbFile,
  artifactsRoot: path.resolve(process.env.ARTIFACTS_ROOT || path.join(dataRoot, 'artifacts')),
  tmpUploadsRoot: path.resolve(process.env.TMP_UPLOADS_ROOT || path.join(dataRoot, 'tmp-uploads')),

  jwtSecret: String(process.env.JWT_SECRET || 'change-me-now'),
  adminUsername: String(process.env.ADMIN_USERNAME || 'admin'),
  adminPassword: String(process.env.ADMIN_PASSWORD || 'admin123456'),

  configTokenTtlMinutes: asInt(process.env.CONFIG_TOKEN_TTL_MINUTES, 120),
  reportTokenTtlHours: asInt(process.env.REPORT_TOKEN_TTL_HOURS, 24),

  defaultRuntimeImage: String(
    process.env.DEFAULT_RUNTIME_IMAGE || 'igortet/itk-ai-trainer-service:qwen-7b'
  ).trim(),
  runtimeDockerBin: String(process.env.RUNTIME_DOCKER_BIN || 'docker').trim(),
  runtimeDockerNetwork: String(process.env.RUNTIME_DOCKER_NETWORK || '').trim(),
  runtimeHostOutputRoot: path.resolve(
    process.env.RUNTIME_HOST_OUTPUT_ROOT || path.join(dataRoot, 'runtime-output')
  ),

  maxUploadBytes: asInt(process.env.MAX_UPLOAD_BYTES, 20 * 1024 * 1024 * 1024),
};

module.exports = { CONFIG };
