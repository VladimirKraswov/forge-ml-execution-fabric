const express = require('express');
const { buildPublicBaseUrl } = require('../utils/http');
const { getRuntimeConfig } = require('../services/job-service');
const {
  assertRuntimeCredential,
  handleStatus,
  handleProgress,
  handleLogs,
  handleArtifactsRegister,
  handleFinal,
} = require('../services/runtime-ingest-service');

const router = express.Router();

function extractBearer(req) {
  const header = String(req.headers.authorization || '');
  if (!header.startsWith('Bearer ')) return null;
  return header.slice('Bearer '.length).trim();
}

async function runtimeAuth(req, res, next) {
  try {
    const token = extractBearer(req);
    const attemptId = String(req.body?.attemptId || '').trim() || null;
    if (!token) {
      return res.status(401).json({ error: 'Runtime bearer token is required' });
    }

    req.runtimeCredential = await assertRuntimeCredential(req.params.jobId, token, attemptId);
    next();
  } catch (error) {
    res.status(error.statusCode || 403).json({ error: String(error.message || error) });
  }
}

router.get('/:jobId/config', async (req, res) => {
  try {
    const token = String(req.query.token || '').trim();
    if (!token) {
      return res.status(401).json({ error: 'Config token is required' });
    }

    const baseUrl = buildPublicBaseUrl(req, process.env.APP_PUBLIC_BASE_URL || '');
    const payload = await getRuntimeConfig(req.params.jobId, token, baseUrl);
    res.json(payload);
  } catch (error) {
    res.status(error.statusCode || 500).json({ error: String(error.message || error) });
  }
});

router.post('/:jobId/status', runtimeAuth, async (req, res) => {
  try {
    res.json(await handleStatus(req.params.jobId, req.body || {}));
  } catch (error) {
    res.status(error.statusCode || 400).json({ error: String(error.message || error) });
  }
});

router.post('/:jobId/progress', runtimeAuth, async (req, res) => {
  try {
    res.json(await handleProgress(req.params.jobId, req.body || {}));
  } catch (error) {
    res.status(error.statusCode || 400).json({ error: String(error.message || error) });
  }
});

router.post('/:jobId/logs', runtimeAuth, async (req, res) => {
  try {
    res.json(await handleLogs(req.params.jobId, req.body || {}));
  } catch (error) {
    res.status(error.statusCode || 400).json({ error: String(error.message || error) });
  }
});

router.post('/:jobId/artifacts/register', runtimeAuth, async (req, res) => {
  try {
    res.json(await handleArtifactsRegister(req.params.jobId, req.body || {}));
  } catch (error) {
    res.status(error.statusCode || 400).json({ error: String(error.message || error) });
  }
});

router.post('/:jobId/final', runtimeAuth, async (req, res) => {
  try {
    res.json(await handleFinal(req.params.jobId, req.body || {}));
  } catch (error) {
    res.status(error.statusCode || 400).json({ error: String(error.message || error) });
  }
});

module.exports = router;
