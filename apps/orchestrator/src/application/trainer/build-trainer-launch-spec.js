async function buildTrainerLaunchSpec(jobId, baseUrl) {
  const { buildTrainerLaunchSpec: original } = require('../../services/trainer-job-service');
  return original(jobId, baseUrl);
}
module.exports = { buildTrainerLaunchSpec };
