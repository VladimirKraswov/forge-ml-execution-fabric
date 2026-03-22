const express = require('express');
const {
  createJob,
  listJobs,
  getJobView,
  buildLaunchSpec,
  getConfigSnapshots,
  getJobSteps,
  getJobEvents,
  getJobLogs,
  getJobArtifacts,
  getResultSummary,
  cloneJob,
  retryJob,
  cancelJob,
  getJobHfSyncStates,
  requestHfSync,
} = require('../services/job-service');
const { buildPublicBaseUrl } = require('../utils/http');

const router = express.Router();

router.get('/', async (req, res) => {
  try {
    const limit = Number(req.query.limit || 50);
    const offset = Number(req.query.offset || 0);
    res.json(await listJobs({ limit, offset }));
  } catch (error) {
    res.status(500).json({ error: String(error.message || error) });
  }
});

router.post('/', async (req, res) => {
  try {
    const created = await createJob(req.body || {}, req.user || null);
    res.status(201).json(created);
  } catch (error) {
    res.status(400).json({ error: String(error.message || error) });
  }
});

router.get('/:jobId', async (req, res) => {
  try {
    const job = await getJobView(req.params.jobId);
    if (!job) {
      return res.status(404).json({ error: 'Job not found' });
    }
    res.json(job);
  } catch (error) {
    res.status(500).json({ error: String(error.message || error) });
  }
});

router.post('/:jobId/retry', async (req, res) => {
  try {
    res.json(await retryJob(req.params.jobId, req.user || null));
  } catch (error) {
    const message = String(error.message || error);
    res.status(message.includes('not found') ? 404 : 400).json({ error: message });
  }
});

router.post('/:jobId/clone', async (req, res) => {
  try {
    const created = await cloneJob(req.params.jobId, req.user || null);
    res.status(201).json(created);
  } catch (error) {
    const message = String(error.message || error);
    res.status(message.includes('not found') ? 404 : 400).json({ error: message });
  }
});

router.post('/:jobId/cancel', async (req, res) => {
  try {
    res.json(await cancelJob(req.params.jobId, req.user || null));
  } catch (error) {
    const message = String(error.message || error);
    res.status(message.includes('not found') ? 404 : 400).json({ error: message });
  }
});

router.get('/:jobId/launch-spec', async (req, res) => {
  try {
    const baseUrl = buildPublicBaseUrl(req, process.env.APP_PUBLIC_BASE_URL || '');
    res.json(await buildLaunchSpec(req.params.jobId, baseUrl));
  } catch (error) {
    const message = String(error.message || error);
    res.status(message.includes('not found') ? 404 : 400).json({ error: message });
  }
});

router.get('/:jobId/config-snapshots', async (req, res) => {
  try {
    res.json(await getConfigSnapshots(req.params.jobId));
  } catch (error) {
    res.status(500).json({ error: String(error.message || error) });
  }
});

router.get('/:jobId/steps', async (req, res) => {
  try {
    res.json(await getJobSteps(req.params.jobId));
  } catch (error) {
    res.status(500).json({ error: String(error.message || error) });
  }
});

router.get('/:jobId/events', async (req, res) => {
  try {
    res.json(await getJobEvents(req.params.jobId, { limit: Number(req.query.limit || 500) }));
  } catch (error) {
    res.status(500).json({ error: String(error.message || error) });
  }
});

router.get('/:jobId/logs', async (req, res) => {
  try {
    res.json(await getJobLogs(req.params.jobId, {
      stepKey: req.query.stepKey == null ? null : String(req.query.stepKey),
      streamName: req.query.streamName == null ? null : String(req.query.streamName),
      tailChunks: Number(req.query.tailChunks || 50),
    }));
  } catch (error) {
    res.status(500).json({ error: String(error.message || error) });
  }
});

router.get('/:jobId/artifacts', async (req, res) => {
  try {
    res.json(await getJobArtifacts(req.params.jobId));
  } catch (error) {
    res.status(500).json({ error: String(error.message || error) });
  }
});

router.get('/:jobId/result', async (req, res) => {
  try {
    const result = await getResultSummary(req.params.jobId);
    if (!result) {
      return res.status(404).json({ error: 'Result summary not found' });
    }
    res.json(result);
  } catch (error) {
    res.status(500).json({ error: String(error.message || error) });
  }
});

router.get('/:jobId/hf-sync', async (req, res) => {
  try {
    res.json(await getJobHfSyncStates(req.params.jobId));
  } catch (error) {
    res.status(500).json({ error: String(error.message || error) });
  }
});

router.post('/:jobId/hf-sync', async (req, res) => {
  try {
    res.status(202).json(await requestHfSync(req.params.jobId, req.body || {}, req.user || null));
  } catch (error) {
    const message = String(error.message || error);
    res.status(message.includes('not found') ? 404 : 400).json({ error: message });
  }
});

module.exports = router;
