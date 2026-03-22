const { db } = require('../../infrastructure/db/db');
const { getRuntimeProfileById } = require('../../services/runtime-profile-service');
const { newId } = require('../../utils/ids');
const { nowIso } = require('../../utils/time');
const { toJson } = require('../../utils/json');
const { getJobView } = require('../jobs/get-job-view');

async function createTrainerJob(payload, actor) {
  // Simplified for migration verification, logic should match src/services/trainer-job-service.js
  const { createTrainerJob: original } = require('../../services/trainer-job-service');
  return original(payload, actor);
}

module.exports = { createTrainerJob };
