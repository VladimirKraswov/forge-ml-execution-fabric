const fs = require('fs');
const fsp = require('fs/promises');
const path = require('path');
const crypto = require('crypto');
const { CONFIG } = require('../config');

function sanitizeSegment(value) {
  return String(value || '')
    .trim()
    .replace(/[^a-zA-Z0-9._-]+/g, '_')
    .replace(/^_+|_+$/g, '') || 'file';
}

function inferContentType(filename, fallback = 'application/octet-stream') {
  const ext = path.extname(String(filename || '')).toLowerCase();
  switch (ext) {
    case '.json':
      return 'application/json';
    case '.jsonl':
      return 'application/x-ndjson';
    case '.csv':
      return 'text/csv';
    case '.log':
    case '.txt':
      return 'text/plain; charset=utf-8';
    case '.gz':
    case '.tgz':
      return 'application/gzip';
    case '.tar':
      return 'application/x-tar';
    case '.md':
      return 'text/markdown; charset=utf-8';
    default:
      return fallback;
  }
}

async function ensureArtifactRoots() {
  await fsp.mkdir(CONFIG.artifactsRoot, { recursive: true });
  await fsp.mkdir(CONFIG.tmpUploadsRoot, { recursive: true });
}

async function computeSha256(filePath) {
  return new Promise((resolve, reject) => {
    const hash = crypto.createHash('sha256');
    const stream = fs.createReadStream(filePath);
    stream.on('data', (chunk) => hash.update(chunk));
    stream.on('end', () => resolve(hash.digest('hex')));
    stream.on('error', reject);
  });
}

async function moveFile(sourcePath, targetPath) {
  await fsp.mkdir(path.dirname(targetPath), { recursive: true });
  try {
    await fsp.rename(sourcePath, targetPath);
  } catch (error) {
    if (error && error.code === 'EXDEV') {
      await fsp.copyFile(sourcePath, targetPath);
      await fsp.unlink(sourcePath);
      return;
    }
    throw error;
  }
}

async function saveUploadedArtifactFile({
  sourcePath,
  originalName,
  jobId,
  artifactType,
}) {
  await ensureArtifactRoots();

  const safeName = sanitizeSegment(originalName || `${artifactType}.bin`);
  const storageKey = path.posix.join(
    'jobs',
    sanitizeSegment(jobId),
    sanitizeSegment(artifactType),
    `${Date.now()}_${crypto.randomBytes(6).toString('hex')}__${safeName}`
  );
  const absolutePath = path.join(CONFIG.artifactsRoot, storageKey);

  await moveFile(sourcePath, absolutePath);
  const stat = await fsp.stat(absolutePath);
  const checksumSha256 = await computeSha256(absolutePath);

  return {
    storageKey,
    absolutePath,
    sizeBytes: stat.size,
    checksumSha256,
    filename: safeName,
    contentType: inferContentType(safeName),
  };
}

function resolveArtifactAbsolutePath(storageKey) {
  if (!storageKey) return null;
  return path.join(CONFIG.artifactsRoot, storageKey);
}

module.exports = {
  sanitizeSegment,
  inferContentType,
  ensureArtifactRoots,
  saveUploadedArtifactFile,
  resolveArtifactAbsolutePath,
};
