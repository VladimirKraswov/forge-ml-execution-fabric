const { db } = require('../db/db');
const { parseJson } = require('../../utils/json');

function mapJob(row) {
  if (!row) return null;
  return {
    id: row.id,
    workspaceId: row.workspace_id,
    projectId: row.project_id,
    createdByUserId: row.created_by_user_id,
    name: row.name,
    jobKind: row.job_kind,
    status: row.status,
    stage: row.stage,
    desiredState: row.desired_state,
    runtimeProfileId: row.runtime_profile_id,
    currentConfigSnapshotId: row.current_config_snapshot_id,
    latestAttemptId: row.latest_attempt_id,
    currentStepKey: row.current_step_key,
    labels: parseJson(row.labels_json, {}),
    headline: row.headline,
    terminalReason: row.terminal_reason,
    progressPercent: Number(row.progress_percent || 0),
    createdAt: row.created_at,
    startedAt: row.started_at,
    finishedAt: row.finished_at,
    updatedAt: row.updated_at,
  };
}

async function getJobById(jobId) {
  const row = await db('jobs').where({ id: jobId }).first();
  return mapJob(row);
}

module.exports = { getJobById, mapJob };
