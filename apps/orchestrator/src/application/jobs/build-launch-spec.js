const { buildLaunchSpec: original } = require('../../services/job-service');
async function buildLaunchSpec(jobId, baseUrl) { return original(jobId, baseUrl); }
module.exports = { buildLaunchSpec };
