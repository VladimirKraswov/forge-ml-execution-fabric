async function launchTrainerJob(jobId, body, actor, req) {
  const { launchTrainerJob: original } = require('../../services/trainer-job-service');
  return original(jobId, body, actor, req);
}
module.exports = { launchTrainerJob };
