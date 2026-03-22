async function stopTrainerJob(jobId, actor) {
  const { stopTrainerJob: original } = require('../../services/trainer-job-service');
  return original(jobId, actor);
}
module.exports = { stopTrainerJob };
