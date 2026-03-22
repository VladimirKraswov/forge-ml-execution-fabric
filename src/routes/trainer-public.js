const express = require('express');
const { buildPublicBaseUrl } = require('../utils/http');
const { buildTrainerBootstrapPayload } = require('../services/trainer-job-service');

const router = express.Router();

router.get('/:jobId/bootstrap', async (req, res) => {
  try {
    const token = String(req.query.token || '').trim();
    if (!token) return res.status(401).json({ error: 'token is required' });
    const baseUrl = buildPublicBaseUrl(req, process.env.APP_PUBLIC_BASE_URL || '');
    res.json(await buildTrainerBootstrapPayload(req.params.jobId, token, baseUrl));
  } catch (error) {
    res.status(error.statusCode || 500).json({ error: String(error.message || error) });
  }
});

module.exports = router;
