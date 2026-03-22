const path = require('path');
const express = require('express');
const { buildPublicBaseUrl } = require('../../utils/http');
const { getJobView } = require('../../application/jobs/get-job-view');
const {
  createTrainerJob,
  listTrainerJobs,
  buildTrainerBootstrapPayload,
  buildTrainerLaunchSpec,
  launchTrainerJob,
  stopTrainerJob,
} = require('../../application/trainer');
const { resolveArtifactAbsolutePath } = require('../../infrastructure/storage/artifact-storage-service');
const { db } = require('../../infrastructure/db/db');

const router = express.Router();

router.get('/', async (req, res) => {
  try {
    const limit = Number(req.query.limit || 50);
    const offset = Number(req.query.offset || 0);
    res.json(await listTrainerJobs({ limit, offset }));
  } catch (error) {
    res.status(500).json({ error: String(error.message || error) });
  }
});

router.post('/', async (req, res) => {
  try {
    const created = await createTrainerJob(req.body || {}, req.user || null);
    res.status(201).json(created);
  } catch (error) {
    res.status(400).json({ error: String(error.message || error) });
  }
});

router.get('/:jobId', async (req, res) => {
  try {
    const job = await getJobView(req.params.jobId);
    if (!job) return res.status(404).json({ error: 'Job not found' });
    res.json(job);
  } catch (error) {
    res.status(500).json({ error: String(error.message || error) });
  }
});

router.get('/:jobId/launch-spec', async (req, res) => {
  try {
    const baseUrl = buildPublicBaseUrl(req, process.env.APP_PUBLIC_BASE_URL || '');
    res.json(await buildTrainerLaunchSpec(req.params.jobId, baseUrl));
  } catch (error) {
    const message = String(error.message || error);
    res.status(message.includes('not found') ? 404 : 400).json({ error: message });
  }
});

router.post('/:jobId/launch', async (req, res) => {
  try {
    res.status(202).json(await launchTrainerJob(req.params.jobId, req.body || {}, req.user || null, req));
  } catch (error) {
    res.status(400).json({ error: String(error.message || error) });
  }
});

router.post('/:jobId/stop', async (req, res) => {
  try {
    res.json(await stopTrainerJob(req.params.jobId, req.user || null));
  } catch (error) {
    const message = String(error.message || error);
    res.status(message.includes('not found') ? 404 : 400).json({ error: message });
  }
});

router.get('/:jobId/artifacts/:artifactId/download', async (req, res) => {
  try {
    const row = await db('job_artifacts')
      .where({ id: req.params.artifactId, job_id: req.params.jobId })
      .first();
    if (!row) return res.status(404).json({ error: 'Artifact not found' });

    const absolutePath = resolveArtifactAbsolutePath(row.storage_key);
    if (!absolutePath) return res.status(404).json({ error: 'Artifact file is not available' });

    return res.download(absolutePath, path.basename(absolutePath));
  } catch (error) {
    res.status(500).json({ error: String(error.message || error) });
  }
});

module.exports = router;
