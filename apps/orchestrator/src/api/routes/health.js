const express = require('express');
const { db } = require('../../infrastructure/db/db');

const router = express.Router();

router.get('/', async (_req, res) => {
  const [jobCountRow, profileCountRow] = await Promise.all([
    db('jobs').count({ count: '*' }).first(),
    db('runtime_profiles').count({ count: '*' }).first(),
  ]);

  res.json({
    ok: true,
    service: 'forge-ml-execution-fabric-orchestrator',
    storage: 'sqlite',
    time: new Date().toISOString(),
    counts: {
      jobs: Number(jobCountRow?.count || 0),
      runtimeProfiles: Number(profileCountRow?.count || 0),
    },
  });
});

module.exports = router;
