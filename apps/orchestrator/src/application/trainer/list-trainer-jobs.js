async function listTrainerJobs(args) {
  const { listTrainerJobs: original } = require('../../services/trainer-job-service');
  return original(args);
}
module.exports = { listTrainerJobs };
