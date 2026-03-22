const { db } = require('../../infrastructure/db/db');
const { getJobById } = require('../../infrastructure/repositories/jobs-repo');
const { getLatestAttemptForJob } = require('../../infrastructure/repositories/attempts-repo');
const { newId } = require('../../utils/ids');
const { nowIso } = require('../../utils/time');
const { toJson, parseJson } = require('../../utils/json');
const { getJobView } = require('./get-job-view');

async function retryJob(jobId, actor) {
  const job = await getJobById(jobId);
  if (!job) throw new Error('Job not found');

  const snapshotRow = await db('job_config_snapshots').where({ id: job.currentConfigSnapshotId }).first();
  if (!snapshotRow) throw new Error('Job config snapshot not found');
  const snapshot = parseJson(snapshotRow.snapshot_json, {});

  const previous = await getLatestAttemptForJob(jobId);
  const nextAttemptNo = Number(previous?.attemptNo || 0) + 1;
  const attemptId = newId('att');
  const now = nowIso();

  await db.transaction(async (trx) => {
    await trx('job_attempts').insert({
      id: attemptId,
      job_id: jobId,
      attempt_no: nextAttemptNo,
      status: 'issued',
      stage: 'config',
      runtime_image: snapshot.runtimeProfile.runtimeImage,
      executor_version: null,
      host_info_json: toJson({}),
      runtime_info_json: toJson({}),
      first_seen_at: null,
      config_fetched_at: null,
      started_at: null,
      last_seen_at: null,
      finished_at: null,
      exit_code: null,
      failure_reason: null,
      final_payload_received_at: null,
      last_sequence_no: null,
      created_at: now,
      updated_at: now,
    });

    await trx('jobs').where({ id: jobId }).update({
      status: 'ready',
      stage: 'config',
      desired_state: 'active',
      latest_attempt_id: attemptId,
      current_step_key: null,
      headline: 'Retry attempt issued',
      terminal_reason: null,
      progress_percent: 0,
      started_at: null,
      finished_at: null,
      updated_at: now,
    });
  });

  return getJobView(jobId);
}

module.exports = { retryJob };
