const express = require('express');
const {
  listRuntimeProfiles,
  getRuntimeProfileById,
  createRuntimeProfile,
  patchRuntimeProfile,
} = require('../services/runtime-profile-service');

const router = express.Router();

router.get('/', async (_req, res) => {
  try {
    res.json(await listRuntimeProfiles());
  } catch (error) {
    res.status(500).json({ error: String(error.message || error) });
  }
});

router.post('/', async (req, res) => {
  try {
    const created = await createRuntimeProfile(req.body || {});
    res.status(201).json(created);
  } catch (error) {
    res.status(400).json({ error: String(error.message || error) });
  }
});

router.get('/:profileId', async (req, res) => {
  try {
    const profile = await getRuntimeProfileById(req.params.profileId);
    if (!profile) {
      return res.status(404).json({ error: 'Runtime profile not found' });
    }
    res.json(profile);
  } catch (error) {
    res.status(500).json({ error: String(error.message || error) });
  }
});

router.patch('/:profileId', async (req, res) => {
  try {
    const updated = await patchRuntimeProfile(req.params.profileId, req.body || {});
    res.json(updated);
  } catch (error) {
    const message = String(error.message || error);
    res.status(message.includes('not found') ? 404 : 400).json({ error: message });
  }
});

module.exports = router;
