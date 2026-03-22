const express = require('express');
const multer = require('multer');
const path = require('path');
const { CONFIG } = require('../../config');
const { buildPublicBaseUrl } = require('../../utils/http');
const {
  assertTrainerRuntimeCredential,
  handleTrainerStatus,
  handleTrainerProgress,
  handleTrainerLogs,
  handleTrainerFinal,
  handleTrainerUpload,
} = require('../../ingest/trainer/trainer-ingest-service');

const router = express.Router();
const upload = multer({
  dest: path.join(CONFIG.tmpUploadsRoot, 'incoming'),
  limits: {
    fileSize: CONFIG.maxUploadBytes,
  },
});

function extractBearer(req) {
  const header = String(req.headers.authorization || '');
  if (!header.startsWith('Bearer ')) return null;
  return header.slice('Bearer '.length).trim();
}

async function runtimeAuth(req, res, next) {
  try {
    const token = extractBearer(req);
    const jobId = String(req.body?.job_id || req.query?.job_id || '').trim();
    if (!token) return res.status(401).json({ error: 'Authorization bearer token is required' });
    if (!jobId) return res.status(400).json({ error: 'job_id is required' });
    req.runtimeCredential = await assertTrainerRuntimeCredential(jobId, token);
    next();
  } catch (error) {
    res.status(error.statusCode || 403).json({ error: String(error.message || error) });
  }
}

router.post('/status', runtimeAuth, async (req, res) => {
  try {
    res.json(await handleTrainerStatus(req.body || {}));
  } catch (error) {
    res.status(error.statusCode || 400).json({ error: String(error.message || error) });
  }
});

router.post('/progress', runtimeAuth, async (req, res) => {
  try {
    res.json(await handleTrainerProgress(req.body || {}));
  } catch (error) {
    res.status(error.statusCode || 400).json({ error: String(error.message || error) });
  }
});

router.post('/logs', runtimeAuth, async (req, res) => {
  try {
    res.json(await handleTrainerLogs(req.body || {}));
  } catch (error) {
    res.status(error.statusCode || 400).json({ error: String(error.message || error) });
  }
});

router.post('/final', runtimeAuth, async (req, res) => {
  try {
    res.json(await handleTrainerFinal(req.body || {}));
  } catch (error) {
    res.status(error.statusCode || 400).json({ error: String(error.message || error) });
  }
});

router.post('/upload/:artifactType', upload.single('file'), async (req, res) => {
  try {
    const token = extractBearer(req);
    const jobId = String(req.body?.job_id || '').trim();
    if (!token) return res.status(401).json({ error: 'Authorization bearer token is required' });
    if (!jobId) return res.status(400).json({ error: 'job_id is required' });
    await assertTrainerRuntimeCredential(jobId, token);

    const baseUrl = buildPublicBaseUrl(req, process.env.APP_PUBLIC_BASE_URL || '');
    res.status(201).json(await handleTrainerUpload({
      artifactTypeParam: req.params.artifactType,
      body: req.body || {},
      file: req.file,
      baseUrl,
    }));
  } catch (error) {
    res.status(error.statusCode || 400).json({ error: String(error.message || error) });
  }
});

module.exports = router;
