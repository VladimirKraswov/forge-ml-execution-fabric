const { createJob: original } = require('../../services/job-service');
async function createJob(payload, actor) { return original(payload, actor); }
module.exports = { createJob };
