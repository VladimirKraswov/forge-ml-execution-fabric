const { db } = require('../../infrastructure/db/db');
const { getJobView } = require('./get-job-view');

async function listJobs({ limit = 50, offset = 0 } = {}) {
  const rows = await db('jobs')
    .orderBy('created_at', 'desc')
    .limit(Math.max(1, Math.min(Number(limit || 50), 500)))
    .offset(Math.max(0, Number(offset || 0)));

  const results = [];
  for (const row of rows) {
    results.push(await getJobView(row.id));
  }
  return results;
}

module.exports = { listJobs };
