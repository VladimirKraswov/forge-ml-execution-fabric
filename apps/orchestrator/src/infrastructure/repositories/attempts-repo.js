const { db } = require('../db/db');
const { parseJson } = require('../../utils/json');

function mapAttempt(row) {
  if (!row) return null;
  return {
    id: row.id,
    jobId: row.job_id,
    attemptNo: row.attempt_no,
    status: row.status,
    stage: row.stage,
    runtimeImage: row.runtime_image,
    executorVersion: row.executor_version,
    hostInfo: parseJson(row.host_info_json, {}),
    runtimeInfo: parseJson(row.runtime_info_json, {}),
    firstSeenAt: row.first_seen_at,
    configFetchedAt: row.config_fetched_at,
    startedAt: row.started_at,
    lastSeenAt: row.last_seen_at,
    finishedAt: row.finished_at,
    exitCode: row.exit_code,
    failureReason: row.failure_reason,
    finalPayloadReceivedAt: row.final_payload_received_at,
    lastSequenceNo: row.last_sequence_no == null ? null : Number(row.last_sequence_no),
    createdAt: row.created_at,
    updatedAt: row.updated_at,
  };
}

async function getAttemptById(attemptId) {
  const row = await db('job_attempts').where({ id: attemptId }).first();
  return mapAttempt(row);
}

async function getLatestAttemptForJob(jobId) {
  const row = await db('job_attempts')
    .where({ job_id: jobId })
    .orderBy('attempt_no', 'desc')
    .first();
  return mapAttempt(row);
}

module.exports = { getAttemptById, getLatestAttemptForJob, mapAttempt };
