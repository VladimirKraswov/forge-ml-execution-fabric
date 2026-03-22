async function buildTrainerBootstrapPayload(jobId, token, baseUrl) {
  const { buildTrainerBootstrapPayload: original } = require('../../services/trainer-job-service');
  return original(jobId, token, baseUrl);
}
module.exports = { buildTrainerBootstrapPayload };
