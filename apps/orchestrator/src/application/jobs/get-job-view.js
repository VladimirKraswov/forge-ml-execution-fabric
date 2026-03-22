const { getJobById } = require('../../infrastructure/repositories/jobs-repo');
const { getAttemptById } = require('../../infrastructure/repositories/attempts-repo');
const { getRuntimeProfileById } = require('../../services/runtime-profile-service');
const { db } = require('../../infrastructure/db/db');
const { parseJson } = require('../../utils/json');

async function getResultSummary(jobId) {
  const row = await db('job_result_summaries').where({ job_id: jobId }).first();
  if (!row) return null;
  return {
    jobId: row.job_id,
    attemptId: row.attempt_id,
    outcome: row.outcome,
    headline: row.headline,
    primaryMetrics: parseJson(row.primary_metrics_json, {}),
    summary: parseJson(row.summary_json, {}),
    createdAt: row.created_at,
    updatedAt: row.updated_at,
  };
}

async function getJobView(jobId) {
  const job = await getJobById(jobId);
  if (!job) return null;

  const [attempt, profile, resultSummary] = await Promise.all([
    job.latestAttemptId ? getAttemptById(job.latestAttemptId) : null,
    job.runtimeProfileId ? getRuntimeProfileById(job.runtimeProfileId) : null,
    getResultSummary(jobId),
  ]);

  return {
    ...job,
    latestAttempt: attempt,
    runtimeProfile: profile,
    resultSummary,
  };
}

module.exports = { getJobView };
