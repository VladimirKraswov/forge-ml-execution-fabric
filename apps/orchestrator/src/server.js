const fs = require('fs');
const path = require('path');
const express = require('express');
const cors = require('cors');

const { CONFIG } = require('./config');
const { initDb } = require('./infrastructure/db/db');
const { seedAdminUser } = require('./infrastructure/auth/auth-service');
const { seedDefaultRuntimeProfiles } = require('./services/runtime-profile-service');
const { ensureArtifactRoots } = require('./infrastructure/storage/artifact-storage-service');
const { authRequired } = require('./api/middleware/auth');

const healthRoute = require('./api/routes/health');
const authRoute = require('./api/routes/auth');
const runtimeProfilesRoute = require('./api/routes/runtime-profiles');
const jobsRoute = require('./api/routes/jobs');
const runtimeJobsRoute = require('./api/routes/runtime-jobs');
const trainerJobsRoute = require('./api/routes/trainer-jobs');
const trainerPublicRoute = require('./api/routes/trainer-public');
const trainerCallbacksRoute = require('./api/routes/trainer-callbacks');

async function main() {
  fs.mkdirSync(path.dirname(CONFIG.dbFile), { recursive: true });
  fs.mkdirSync(CONFIG.runtimeHostOutputRoot, { recursive: true });
  await ensureArtifactRoots();

  await initDb();
  await seedAdminUser();
  await seedDefaultRuntimeProfiles();

  const app = express();
  app.set('trust proxy', true);
  app.use(cors({
    origin: CONFIG.corsOrigin === '*' ? true : CONFIG.corsOrigin,
    credentials: false,
  }));
  app.use(express.json({ limit: '25mb' }));

  app.get('/', (_req, res) => {
    res.json({
      ok: true,
      service: 'forge-ml-execution-fabric-orchestrator',
      version: '1.1.0',
      docsHint: 'Use /health, /api/v1/*, /api/jobs/*',
    });
  });

  app.use('/health', healthRoute);

  app.use('/api/v1/auth', authRoute);

  app.use('/api/jobs', trainerCallbacksRoute);
  app.use('/api/v1/runtime/jobs', runtimeJobsRoute);
  app.use('/api/v1/trainer/jobs', trainerPublicRoute);

  app.use('/api/v1/runtime-profiles', authRequired, runtimeProfilesRoute);
  app.use('/api/v1/jobs', authRequired, jobsRoute);
  app.use('/api/v1/trainer/jobs', authRequired, trainerJobsRoute);

  app.use((err, _req, res, _next) => {
    console.error(err);
    res.status(err.statusCode || 500).json({
      error: String(err.message || err),
    });
  });

  app.listen(CONFIG.port, CONFIG.host, () => {
    console.log(`Forge ML Execution Fabric Orchestrator listening on http://${CONFIG.host}:${CONFIG.port}`);
    console.log(`SQLite DB: ${CONFIG.dbFile}`);
    console.log(`Artifacts root: ${CONFIG.artifactsRoot}`);
    console.log(`Runtime output root: ${CONFIG.runtimeHostOutputRoot}`);
  });
}

main().catch((error) => {
  console.error('Fatal startup error');
  console.error(error);
  process.exit(1);
});
